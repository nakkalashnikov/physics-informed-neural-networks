"""
PDE residual, boundary-condition, and initial-condition losses.
Analytical reference solution via Fourier eigenfunction expansion.

Heat equation (1D rod, moving point source, insulated ends):

    ∂ΔT/∂t  =  α · ∂²ΔT/∂x²  +  i_eff · δ(x − x_b(t))

where:
    ΔT      = T − T_amb                        temperature rise [K]
    α       = k/(ρc)                           thermal diffusivity [m²/s]
    i_eff   = i/(ρc)                           effective source strength [K·m/s]
    x_b(t)  = x0 + v·t                         burner position [m]

Boundary conditions (insulated ends):
    ∂ΔT/∂x = 0  at  x = 0  and  x = l

Initial condition (hard-enforced in model.py, kept here for diagnostics):
    ΔT(x, 0) = 0
"""

import math
import torch
import torch.nn as nn
import numpy as np


# ── Delta-function approximation ─────────────────────────────────────────────

def _gaussian_delta(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Gaussian approximation of δ(x − μ) with width σ."""
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (
        sigma * math.sqrt(2.0 * math.pi)
    )


# ── PINN loss terms ───────────────────────────────────────────────────────────

def pde_loss(
    model: nn.Module,
    coords_norm: torch.Tensor,   # (N, 2)  normalised [x_n, t_n] ∈ [0,1]
    params_norm: torch.Tensor,   # (N, 5)  normalised physics params
    raw: dict,                   # physical scalars per point, each (N,) tensor
    epsilon: float = 1.0,        # causal weight strength ε (Wang et al. 2022)
    n_bins: int = 10,            # time windows for causal weighting
) -> torch.Tensor:
    """
    Causal PDE residual loss (Wang et al., JMLR 2024).

    Points are grouped into n_bins uniform windows along t_norm ∈ [0, 1].
    Each bin k gets a causal weight:

        w_k = exp(−ε · Σ_{j < k} L_j)

    where L_j is the mean squared residual in bin j.  While early-time
    residuals are large the later bins are suppressed, forcing the network
    to master the initial-value problem before fitting late times.

    Derivative chain rule for normalised coordinates:
        ∂ΔT/∂t_phys  =  ∂ΔT/∂t_norm  ·  (1/t_total)
        ∂²ΔT/∂x_phys² =  ∂²ΔT/∂x_norm²  ·  (1/l²)
    """
    coords = coords_norm.detach().requires_grad_(True)

    dT = model(coords, params_norm)               # (N, 1)

    grad1 = torch.autograd.grad(
        dT.sum(), coords, create_graph=True
    )[0]                                          # (N, 2)
    dT_dxn = grad1[:, 0:1]                        # ∂ΔT/∂x_norm
    dT_dtn = grad1[:, 1:2]                        # ∂ΔT/∂t_norm

    dT_dxnxn = torch.autograd.grad(
        dT_dxn.sum(), coords, create_graph=True
    )[0][:, 0:1]                                  # ∂²ΔT/∂x_norm²

    # Physical parameters  (N, 1)
    l       = raw["l"].unsqueeze(1)
    t_total = raw["t_total"].unsqueeze(1)
    alpha   = raw["alpha"].unsqueeze(1)
    i_eff   = raw["i_eff"].unsqueeze(1)
    x0      = raw["x0"].unsqueeze(1)
    v       = raw["v"].unsqueeze(1)

    dT_dt  = dT_dtn  / t_total          # ∂ΔT/∂t  [K/s]
    dT_dxx = dT_dxnxn / (l ** 2)        # ∂²ΔT/∂x² [K/m²]

    x_phys = coords[:, 0:1] * l         # [m]
    t_phys = coords[:, 1:2] * t_total   # [s]

    x_b    = x0 + v * t_phys
    sigma_g = l / 50.0
    Q      = i_eff * _gaussian_delta(x_phys, x_b, sigma_g)   # [K/s]

    residual = dT_dt - alpha * dT_dxx - Q          # (N, 1)
    residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Causal time-weighting ─────────────────────────────────────────────
    t_vals = coords_norm[:, 1].detach()            # (N,) — original (no grad)

    bin_losses: list[torch.Tensor] = []
    for k in range(n_bins):
        lo = k / n_bins
        hi = (k + 1) / n_bins
        mask = (t_vals >= lo) & (t_vals < hi)
        if k == n_bins - 1:
            mask = mask | (t_vals >= hi)           # include t_norm = 1.0
        if mask.any():
            bin_losses.append((residual[mask] ** 2).mean())
        else:
            bin_losses.append(residual.new_zeros(()))

    causal_loss = residual.new_zeros(())
    cumulative  = 0.0
    for k, bl in enumerate(bin_losses):
        w = math.exp(-epsilon * cumulative)
        causal_loss = causal_loss + w * bl
        cumulative += bl.item()                    # detached: weight = constant

    return causal_loss / n_bins


def bc_loss(
    model: nn.Module,
    coords_norm: torch.Tensor,   # (N, 2)  x_norm ∈ {0, 1}
    params_norm: torch.Tensor,   # (N, 5)
) -> torch.Tensor:
    """
    Neumann BC: ∂ΔT/∂x = 0 at x = 0 and x = l.

    Uses one-sided finite differences instead of autograd to avoid
    MPS numerical instability with create_graph=True.
    Finite difference is O(eps) accurate and fully differentiable
    w.r.t. model parameters via standard backprop.
    """
    eps = 1e-3   # in normalised x coords

    left_mask  = coords_norm[:, 0] < 0.5
    right_mask = ~left_mask

    losses = []

    if left_mask.any():
        c_left  = coords_norm[left_mask]
        p_left  = params_norm[left_mask]
        c_shift = c_left.clone()
        c_shift[:, 0] = eps
        dTdx = (model(c_shift, p_left) - model(c_left, p_left)) / eps
        losses.append((dTdx ** 2).mean())

    if right_mask.any():
        c_right = coords_norm[right_mask]
        p_right = params_norm[right_mask]
        c_shift = c_right.clone()
        c_shift[:, 0] = 1.0 - eps
        dTdx = (model(c_right, p_right) - model(c_shift, p_right)) / eps
        losses.append((dTdx ** 2).mean())

    if not losses:
        return coords_norm.new_tensor(0.0)

    return torch.stack(losses).mean()


def pde_residuals_fd(
    model: nn.Module,
    coords_norm: torch.Tensor,   # (N, 2)
    params_norm: torch.Tensor,   # (N, 5)
    raw: dict,
    eps_x: float = 1e-3,
    eps_t: float = 1e-4,
) -> torch.Tensor:
    """
    Absolute PDE residuals via finite differences — for RAD sampling only.

    Uses 4 forward passes (center, x±ε, t+ε) inside torch.no_grad(), so no
    computation graph is built. This makes it ~40× cheaper in memory than
    autograd at the large pool sizes RAD needs (pool_factor × n_pde points).

    Accuracy is sufficient for RAD: we only need to know *where* residuals
    are large, not their exact values.

    Returns: (N,) tensor of |residual| values.
    """
    with torch.no_grad():
        l       = raw["l"].unsqueeze(1)
        t_total = raw["t_total"].unsqueeze(1)
        alpha   = raw["alpha"].unsqueeze(1)
        i_eff   = raw["i_eff"].unsqueeze(1)
        x0      = raw["x0"].unsqueeze(1)
        v       = raw["v"].unsqueeze(1)

        dT_c = model(coords_norm, params_norm)

        c_xp = coords_norm.clone(); c_xp[:, 0] = (c_xp[:, 0] + eps_x).clamp(0.0, 1.0)
        c_xm = coords_norm.clone(); c_xm[:, 0] = (c_xm[:, 0] - eps_x).clamp(0.0, 1.0)
        c_tp = coords_norm.clone(); c_tp[:, 1] = (c_tp[:, 1] + eps_t).clamp(0.0, 1.0)

        dT_xp = model(c_xp, params_norm)
        dT_xm = model(c_xm, params_norm)
        dT_tp = model(c_tp, params_norm)

        dT_dxx_n = (dT_xp - 2.0 * dT_c + dT_xm) / (eps_x ** 2)   # ∂²/∂x_norm²
        dT_dt_n  = (dT_tp - dT_c) / eps_t                          # ∂/∂t_norm

        dT_dt  = dT_dt_n  / t_total      # [K/s]
        dT_dxx = dT_dxx_n / (l ** 2)     # [K/m²]

        x_phys  = coords_norm[:, 0:1] * l
        t_phys  = coords_norm[:, 1:2] * t_total
        x_b     = x0 + v * t_phys
        sigma_g = l / 50.0
        Q       = i_eff * _gaussian_delta(x_phys, x_b, sigma_g)

        residual = (dT_dt - alpha * dT_dxx - Q).squeeze(1)
        return residual.abs()


def ic_loss(
    model: nn.Module,
    coords_norm: torch.Tensor,   # (N, 2)  t_norm = 0
    params_norm: torch.Tensor,   # (N, 5)
) -> torch.Tensor:
    """
    IC diagnostic: ΔT(x, 0) = 0.
    With hard IC (model output = t_norm * NN), this is always ≈ 0.
    Kept for monitoring purposes; lambda_ic = 0 in config.
    """
    dT = model(coords_norm, params_norm)
    dT = torch.nan_to_num(dT, nan=0.0, posinf=0.0, neginf=0.0)
    return (dT ** 2).mean()


def total_loss(
    model: nn.Module,
    batch: tuple,
    weights: dict,
    epsilon: float = 1.0,
    n_bins: int = 10,
) -> tuple[torch.Tensor, float, float, float]:
    """
    Weighted sum of all three loss terms.

    Args:
        batch:   output of sampler.build_batch()
        weights: {'pde': λ_pde, 'bc': λ_bc, 'ic': λ_ic}
        epsilon: causal weight strength for pde_loss
        n_bins:  time bins for causal weighting

    Returns:
        (total_loss, l_pde_scalar, l_bc_scalar, l_ic_scalar)
    """
    (c_pde, p_pde, raw_pde), (c_bc, p_bc), (c_ic, p_ic) = batch

    l_pde = pde_loss(model, c_pde, p_pde, raw_pde, epsilon=epsilon, n_bins=n_bins)
    l_bc  = bc_loss(model, c_bc, p_bc)
    l_ic  = ic_loss(model, c_ic, p_ic)

    total = weights["pde"] * l_pde + weights["bc"] * l_bc + weights["ic"] * l_ic
    return total, l_pde.item(), l_bc.item(), l_ic.item()


# ── Analytical reference solution (NumPy) ─────────────────────────────────────

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
    Temperature rise ΔT = T − T_amb via Fourier eigenfunction expansion.

    Exact series solution (insulated-end BC):

        ΔT(x,t) = (i·t)/(ρc·l)                           [n=0 mode]
                 + Σ_{n=1}^{N} aₙ(t) · cos(nπx/l)        [higher modes]

    where:
        aₙ(t) = (2i/ρcl) · exp(−μₙt) · Iₙ(t)
        μₙ    = α(nπ/l)²
        Iₙ(t) = ∫₀ᵗ exp(μₙτ) cos(nπ(x₀+vτ)/l) dτ

    The integral Iₙ(t) is evaluated analytically:
        ∫ exp(aτ)·cos(bτ+c) dτ
            = exp(aτ)·(a·cos(bτ+c) + b·sin(bτ+c)) / (a²+b²)

    with  a = μₙ,  b = nπv/l,  c = nπx₀/l.
    """
    i_eff = intensity / rho_c   # [K·m/s]

    # n=0: uniform temperature rise (total energy conservation)
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
