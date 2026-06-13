"""
Model loading and inference logic.

Loaded once at startup via FastAPI lifespan; all endpoints share one instance.
Supports CUDA, Apple MPS (Metal), and CPU — auto-detected at startup.
"""

import math
import logging
import os
import torch
import numpy as np

from model import PINN
from schemas import MATERIALS

log = logging.getLogger(__name__)

T_AMB = 293.15   # [K]


# ── Device detection ──────────────────────────────────────────────────────────

def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Normalizer (reconstructed from checkpoint) ────────────────────────────────

class Normalizer:
    """
    Mirrors sampler.Normalizer but reconstructed from the bounds stored
    in the checkpoint — no need for config.yaml at inference time.
    """

    def __init__(self, bounds: dict[str, tuple[float, float]]) -> None:
        self.bounds = bounds

    def norm(self, val: torch.Tensor, key: str) -> torch.Tensor:
        lo, hi = self.bounds[key]
        return (val - lo) / (hi - lo)

    def norm_log(self, val: torch.Tensor, key: str) -> torch.Tensor:
        lo, hi = self.bounds[key]
        log_lo, log_hi = math.log(lo), math.log(hi)
        return (torch.log(val.clamp_min(1e-30)) - log_lo) / (log_hi - log_lo)


# ── Analytical reference solution ─────────────────────────────────────────────

def _gaussian_delta_np(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def analytical_delta_T(
    x: np.ndarray,
    t: float,
    alpha: float,
    rho_c: float,
    l: float,
    intensity: float,
    x0: float,
    v: float,
    N_terms: int = 150,
) -> np.ndarray:
    """
    Fourier eigenfunction series — exact solution for insulated-end BC.

    ΔT(x,t) = (i·t)/(ρc·l)  +  Σ_{n≥1} aₙ(t)·cos(nπx/l)

    aₙ(t) = (2i/ρcl) · e^{−μₙt} · Iₙ(t)
    Iₙ(t) = [e^{μₙτ}(μₙcos(bτ+c)+b·sin(bτ+c))/(μₙ²+b²)]₀ᵗ
    """
    i_eff = intensity / rho_c

    delta_T = np.full_like(x, i_eff * t / l, dtype=np.float64)

    for n in range(1, N_terms + 1):
        mu_n  = alpha * (n * math.pi / l) ** 2
        b     = n * math.pi * v / l
        c     = n * math.pi * x0 / l
        denom = mu_n ** 2 + b ** 2

        if mu_n * t > 690.0:
            break

        f_t = mu_n * math.cos(b * t + c) + b * math.sin(b * t + c)
        f_0 = mu_n * math.cos(c)         + b * math.sin(c)
        a_n = (2.0 * i_eff / l) * (f_t - math.exp(-mu_n * t) * f_0) / denom
        delta_T = delta_T + a_n * np.cos(n * math.pi * x / l)

    return delta_T


# ── Predictor ─────────────────────────────────────────────────────────────────

class Predictor:
    """Thread-safe inference wrapper around the trained PINN."""

    def __init__(self, checkpoint_path: str) -> None:
        self.device = _pick_device()
        log.info("Inference device: %s", self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        m_cfg = ckpt["model_cfg"]
        # The Fourier σ is a buffer restored by load_state_dict, so the value
        # passed here is irrelevant; fall back across the config key variants.
        sigma0 = m_cfg.get("fourier_sigma_start", m_cfg.get("fourier_sigma", 1.0))
        self.model = PINN(
            fourier_m=m_cfg["fourier_m"],
            fourier_sigma=sigma0,
            hidden_layers=m_cfg["hidden_layers"],
            hidden_size=m_cfg["hidden_size"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.normalizer = Normalizer(ckpt["normalizer_bounds"])
        self.T_amb = ckpt["physics_cfg"].get("T_amb", T_AMB)

        log.info("Model loaded from %s", checkpoint_path)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _pi_groups(
        self, alpha: float, l: float, i_eff: float, x0: float, v: float, t_total: float
    ) -> tuple[float, float, float, float]:
        """Physical params → (Fo, x0_norm, beta, T_c) — see sampler.compute_pi_groups."""
        Fo      = alpha * t_total / (l ** 2)
        x0_norm = x0 / l
        beta    = v * t_total / l
        T_c     = i_eff * t_total / l
        return Fo, x0_norm, beta, T_c

    def _build_pi_norm(
        self, Fo: float, x0_norm: float, beta: float, n: int
    ) -> torch.Tensor:
        """Build (n, 3) normalised π-group tensor [Fo_n, x0_n, β_n]."""
        dev = self.device

        def full(val: float) -> torch.Tensor:
            return torch.full((n,), val, dtype=torch.float32, device=dev)

        fo_n   = self.normalizer.norm_log(full(Fo),      "Fo")
        x0_n   = self.normalizer.norm(    full(x0_norm), "x0_frac")
        beta_n = self.normalizer.norm(    full(beta),    "beta")
        return torch.stack([fo_n, x0_n, beta_n], dim=1)

    def _coords_norm(
        self, x_phys: np.ndarray, t_phys: float, l: float, t_total: float
    ) -> torch.Tensor:
        x_n = torch.tensor(x_phys / l,       dtype=torch.float32, device=self.device)
        t_n = torch.full_like(x_n, t_phys / t_total)
        return torch.stack([x_n, t_n], dim=1)

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_point(
        self,
        material: str,
        length: float,
        intensity: float,
        x0: float,
        velocity: float,
        t_total: float,
        x_query: float,
        t_query: float,
    ) -> dict:
        """Single-point temperature prediction + analytical reference."""
        mat    = MATERIALS[material]
        alpha  = mat["alpha"]
        rho_c  = mat["rho_c"]
        i_eff  = intensity / rho_c

        Fo, x0_norm, beta, T_c = self._pi_groups(alpha, length, i_eff, x0, velocity, t_total)
        coords = self._coords_norm(np.array([x_query]), t_query, length, t_total)
        pi     = self._build_pi_norm(Fo, x0_norm, beta, 1)

        # Network outputs dimensionless u; rescale to physical ΔT.
        dT_pinn = float(self.model(coords, pi).item()) * T_c

        dT_ref = float(
            analytical_delta_T(
                np.array([x_query]), t_query,
                alpha, rho_c, length, intensity, x0, velocity,
            )[0]
        )

        rel_err = abs(dT_pinn - dT_ref) / (abs(dT_ref) + 1e-8)

        return {
            "temperature_K":      self.T_amb + dT_pinn,
            "temperature_C":      self.T_amb + dT_pinn - 273.15,
            "delta_T":            dT_pinn,
            "analytical_K":       self.T_amb + dT_ref,
            "analytical_delta_T": dT_ref,
            "relative_error":     rel_err,
            "T_amb":              self.T_amb,
            "device":             str(self.device),
        }

    @torch.no_grad()
    def predict_heatmap(
        self,
        material: str,
        length: float,
        intensity: float,
        x0: float,
        velocity: float,
        t_total: float,
        nx: int = 60,
        nt: int = 60,
    ) -> dict:
        """
        Evaluate PINN + analytical on an (nt × nx) grid.
        Returns data ready for JSON serialisation.
        """
        mat   = MATERIALS[material]
        alpha = mat["alpha"]
        rho_c = mat["rho_c"]
        i_eff = intensity / rho_c

        Fo, x0_norm, beta, T_c = self._pi_groups(alpha, length, i_eff, x0, velocity, t_total)

        x_phys = np.linspace(0.0, length,  nx)   # (nx,)
        t_phys = np.linspace(0.0, t_total, nt)   # (nt,)

        # Build flattened grid: (nt*nx, 2) coords and (nt*nx, 3) π-groups
        x_grid, t_grid = np.meshgrid(x_phys, t_phys)   # both (nt, nx)
        x_flat = x_grid.ravel()                          # (nt*nx,)
        t_flat = t_grid.ravel()

        x_n = torch.tensor(x_flat / length,  dtype=torch.float32, device=self.device)
        t_n = torch.tensor(t_flat / t_total, dtype=torch.float32, device=self.device)
        coords = torch.stack([x_n, t_n], dim=1)                    # (N, 2)
        pi     = self._build_pi_norm(Fo, x0_norm, beta, len(x_flat))

        # Network outputs dimensionless u; rescale to physical ΔT.
        dT_pinn_flat = self.model(coords, pi).squeeze().cpu().numpy() * T_c
        dT_pinn = dT_pinn_flat.reshape(nt, nx)

        # Analytical solution row-by-row (varies with t)
        dT_ref = np.zeros((nt, nx))
        for i, t_val in enumerate(t_phys):
            if t_val == 0.0:
                continue   # IC: ΔT = 0 everywhere
            dT_ref[i] = analytical_delta_T(
                x_phys, t_val, alpha, rho_c, length, intensity, x0, velocity
            )

        burner_pos = [min(x0 + velocity * t, length) for t in t_phys.tolist()]

        return {
            "x_grid":             x_phys.tolist(),
            "t_grid":             t_phys.tolist(),
            "delta_T_pinn":       dT_pinn.tolist(),
            "delta_T_analytical": dT_ref.tolist(),
            "burner_positions":   burner_pos,
            "T_amb":              self.T_amb,
        }


# ── Singleton factory (used by FastAPI lifespan) ──────────────────────────────

_predictor: Predictor | None = None


def load_predictor(checkpoint_path: str | None = None) -> Predictor:
    global _predictor
    if checkpoint_path is None:
        checkpoint_path = os.environ.get(
            "MODEL_PATH",
            "/workspace/checkpoints/model_final.pt",
        )
    _predictor = Predictor(checkpoint_path)
    return _predictor


def get_predictor() -> Predictor:
    if _predictor is None:
        raise RuntimeError("Predictor not initialised — call load_predictor() first.")
    return _predictor
