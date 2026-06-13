"""
Physics-parameter sampling and collocation-point generation.

Parameter vector fed into the network (all normalised to [0, 1]):
    [α_n, l_n, i_eff_n, x0_n, v_n]

where:
    α_n     = linear norm of thermal diffusivity α
    l_n     = linear norm of pipe length l
    i_eff_n = linear norm of effective source strength i_eff = intensity/(ρc)
    x0_n    = x0 / l  (burner start position as fraction of pipe length)
    v_n     = linear norm of burner velocity v
"""

import torch
import numpy as np
from scipy.stats import qmc


# ── Normalizer ────────────────────────────────────────────────────────────────

class Normalizer:
    """Linear normalisation of physical scalars to [0, 1]."""

    def __init__(self, cfg: dict) -> None:
        p = cfg["physics"]
        self.bounds: dict[str, tuple[float, float]] = {
            "alpha":   tuple(p["alpha_range"]),
            "rho_c":   tuple(p["rho_c_range"]),
            "l":       tuple(p["length_range"]),
            "intensity": tuple(p["intensity_range"]),
            "x0_frac": tuple(p["x0_fraction_range"]),
            "v":       tuple(p["velocity_range"]),
            "t_total": tuple(p["t_total_range"]),
            # i_eff = intensity / rho_c
            "i_eff": (
                p["intensity_range"][0] / p["rho_c_range"][1],
                p["intensity_range"][1] / p["rho_c_range"][0],
            ),
        }

    def norm(self, val: torch.Tensor, key: str) -> torch.Tensor:
        lo, hi = self.bounds[key]
        return (val - lo) / (hi - lo)

    def denorm(self, val: torch.Tensor, key: str) -> torch.Tensor:
        lo, hi = self.bounds[key]
        return val * (hi - lo) + lo


# ── Parameter sampling ────────────────────────────────────────────────────────

def sample_params(n: int, cfg: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """
    Sample n physics-parameter sets via Latin Hypercube Sampling.

    Returns a dict of (n,) float32 tensors on `device`.
    Enforces x0 + v·t_total ≤ l so the burner stays inside the pipe.
    """
    p = cfg["physics"]

    # LHS over 5 dimensions: alpha, rho_c, l, intensity, x0_frac
    lhs = qmc.LatinHypercube(d=5, seed=None)
    u = torch.tensor(lhs.random(n=n), dtype=torch.float32, device=device)

    def scale(col: int, lo: float, hi: float) -> torch.Tensor:
        return u[:, col] * (hi - lo) + lo

    alpha     = scale(0, *p["alpha_range"])
    rho_c     = scale(1, *p["rho_c_range"])
    l         = scale(2, *p["length_range"])
    intensity = scale(3, *p["intensity_range"])
    x0_frac   = scale(4, *p["x0_fraction_range"])
    x0        = x0_frac * l

    t_lo, t_hi = p["t_total_range"]
    t_total = torch.rand(n, device=device) * (t_hi - t_lo) + t_lo

    # Constrain velocity so burner stays in pipe: v ≤ (l − x0) / t_total
    v_lo = torch.full((n,), p["velocity_range"][0], device=device)
    v_hi = torch.clamp(
        (l - x0) / t_total,
        min=p["velocity_range"][0],
        max=p["velocity_range"][1],
    )
    v = torch.rand(n, device=device) * (v_hi - v_lo) + v_lo

    i_eff = intensity / rho_c   # effective source strength [K·m/s]

    return {
        "alpha":     alpha,
        "rho_c":     rho_c,
        "l":         l,
        "intensity": intensity,
        "x0":        x0,
        "v":         v,
        "t_total":   t_total,
        "i_eff":     i_eff,
    }


# ── Collocation-point construction ───────────────────────────────────────────

def _to_column(scalar: torch.Tensor, n: int) -> torch.Tensor:
    """Expand a 0-dim or (1,) tensor to a (n,) tensor."""
    return scalar.reshape(1).expand(n).clone()


def _params_norm_for_set(
    raw_k: dict[str, torch.Tensor],
    normalizer: Normalizer,
    n_pts: int,
) -> torch.Tensor:
    """
    Build a (n_pts, 5) normalised parameter tensor for one parameter set.
    Column order: [α_n, l_n, i_eff_n, x0_n, v_n].
    """
    alpha_n  = normalizer.norm(_to_column(raw_k["alpha"],  n_pts), "alpha")
    l_n      = normalizer.norm(_to_column(raw_k["l"],      n_pts), "l")
    i_eff_n  = normalizer.norm(_to_column(raw_k["i_eff"],  n_pts), "i_eff")
    x0_n     = _to_column(raw_k["x0"] / raw_k["l"],               n_pts)
    v_n      = normalizer.norm(_to_column(raw_k["v"],      n_pts), "v")
    return torch.stack([alpha_n, l_n, i_eff_n, x0_n, v_n], dim=1)


def rad_resample_pde(
    model,
    cfg: dict,
    normalizer: Normalizer,
    device: torch.device,
) -> tuple:
    """
    RAD — Residual-based Adaptive Distribution (arXiv:2207.10289).

    Steps:
      1. Sample pool_factor × n_pde candidate points uniformly.
      2. Evaluate |PDE residual| via finite differences (4 forward passes, no_grad).
      3. Draw n_pde points with probability ∝ |residual| — concentrates near
         the moving heat source without any manual tuning.

    Returns (coords_pde, params_pde, raw_pde) — same format as build_batch's
    PDE tuple, so it can be passed directly as pde_override.
    """
    from physics import pde_residuals_fd

    s_cfg       = cfg["sampling"]
    n_p         = s_cfg["n_params_per_step"]
    n_pde       = s_cfg["n_pde"]
    pool_factor = int(s_cfg.get("rad_pool_factor", 5))
    pps_cand    = (n_pde * pool_factor) // n_p   # candidates per param set
    pps_target  = n_pde // n_p                   # selected per param set

    raw_sets = sample_params(n_p, cfg, device)
    dtype    = next(model.parameters()).dtype

    coords_out, params_out = [], []
    raw_acc: dict[str, list[torch.Tensor]] = {k: [] for k in raw_sets}

    model.eval()
    for k in range(n_p):
        raw_k = {key: raw_sets[key][k] for key in raw_sets}

        xn = torch.rand(pps_cand, device=device, dtype=dtype)
        tn = torch.rand(pps_cand, device=device, dtype=dtype)
        coords_cand = torch.stack([xn, tn], dim=1)
        params_cand = _params_norm_for_set(raw_k, normalizer, pps_cand).to(dtype)
        raw_cand    = {key: _to_column(raw_k[key], pps_cand).to(dtype) for key in raw_sets}

        resids = pde_residuals_fd(model, coords_cand, params_cand, raw_cand)  # (pps_cand,)
        probs  = resids.float() / (resids.float().sum() + 1e-8)
        idx    = torch.multinomial(probs, pps_target, replacement=False)

        coords_out.append(coords_cand[idx])
        params_out.append(params_cand[idx])
        for key in raw_sets:
            raw_acc[key].append(raw_cand[key][idx])

    model.train()
    raw_pde = {key: torch.cat(raw_acc[key]) for key in raw_sets}
    return torch.cat(coords_out), torch.cat(params_out), raw_pde


def build_batch(
    cfg: dict,
    normalizer: Normalizer,
    device: torch.device,
    pde_override: tuple | None = None,
) -> tuple:
    """
    Build one full training batch.

    Strategy: sample n_params_per_step physics-parameter sets, generate
    n_pde // n_p interior points per set, similarly for BC and IC.
    All tensors are concatenated into single flat arrays.

    Returns
    -------
    (
        (coords_pde, params_pde, raw_pde),   ← for pde_loss
        (coords_bc,  params_bc),              ← for bc_loss
        (coords_ic,  params_ic),              ← for ic_loss
    )

    coords_*   : (N, 2)  float32  –  [x_norm, t_norm]
    params_*   : (N, 5)  float32  –  normalised physics params
    raw_pde    : dict of (N_pde,) float32 tensors  –  physical values

    pde_override : if provided (from rad_resample_pde), skip PDE point generation
                   and use these points instead.  BC and IC are always freshly sampled.
    """
    n_p     = cfg["sampling"]["n_params_per_step"]
    n_bc    = cfg["sampling"]["n_bc"]
    n_ic    = cfg["sampling"]["n_ic"]
    pps_bc  = max(n_bc // n_p, 2)
    pps_ic  = max(n_ic // n_p, 2)

    raw_sets = sample_params(n_p, cfg, device)   # each value: (n_p,)

    coords_bc, params_bc = [], []
    coords_ic, params_ic = [], []

    if pde_override is None:
        n_pde   = cfg["sampling"]["n_pde"]
        pps_pde = n_pde // n_p
        coords_pde, params_pde = [], []
        raw_pde_acc: dict[str, list[torch.Tensor]] = {k: [] for k in raw_sets}

    for k in range(n_p):
        raw_k = {key: raw_sets[key][k] for key in raw_sets}   # 0-dim tensors

        if pde_override is None:
            # ── PDE interior: 70% uniform + 30% near burner trajectory ──────
            n_unif   = int(pps_pde * 0.70)
            n_burner = pps_pde - n_unif

            xn_unif = torch.rand(n_unif, device=device)
            tn_unif = torch.rand(n_unif, device=device)

            tn_burner = torch.rand(n_burner, device=device)
            sigma_n   = 1.0 / 20.0
            x_b_norm  = (raw_k["x0"] + raw_k["v"] * tn_burner * raw_k["t_total"]) / raw_k["l"]
            xn_burner = (x_b_norm + torch.randn(n_burner, device=device) * sigma_n
                         ).clamp(0.0, 1.0)

            xn = torch.cat([xn_unif, xn_burner])
            tn = torch.cat([tn_unif, tn_burner])
            coords_pde.append(torch.stack([xn, tn], dim=1))
            params_pde.append(_params_norm_for_set(raw_k, normalizer, pps_pde))
            for key in raw_sets:
                raw_pde_acc[key].append(_to_column(raw_k[key], pps_pde))

        # ── BC: x_norm ∈ {0, 1} (equal split), t_norm ∈ (0,1) ───────────
        half  = pps_bc // 2
        x_bc  = torch.cat([
            torch.zeros(half,          device=device),
            torch.ones(pps_bc - half,  device=device),
        ])
        t_bc  = torch.rand(pps_bc, device=device)
        coords_bc.append(torch.stack([x_bc, t_bc], dim=1))
        params_bc.append(_params_norm_for_set(raw_k, normalizer, pps_bc))

        # ── IC: t_norm = 0, x_norm ∈ (0,1) ──────────────────────────────
        x_ic = torch.rand(pps_ic, device=device)
        t_ic = torch.zeros(pps_ic, device=device)
        coords_ic.append(torch.stack([x_ic, t_ic], dim=1))
        params_ic.append(_params_norm_for_set(raw_k, normalizer, pps_ic))

    if pde_override is None:
        raw_pde   = {key: torch.cat(raw_pde_acc[key]) for key in raw_sets}
        pde_tuple = (torch.cat(coords_pde), torch.cat(params_pde), raw_pde)
    else:
        pde_tuple = pde_override

    return (
        pde_tuple,
        (torch.cat(coords_bc),  torch.cat(params_bc)),
        (torch.cat(coords_ic),  torch.cat(params_ic)),
    )
