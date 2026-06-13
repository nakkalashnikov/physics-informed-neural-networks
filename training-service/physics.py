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

# Dimensionless source width: σ* = σ_g / l = 1/50  (parameter-independent).
SIGMA_NORM = 1.0 / 50.0


def pde_loss(
    model: nn.Module,
    coords_norm: torch.Tensor,   # (N, 2)  dimensionless [x*, t*] ∈ [0,1]
    pi_norm: torch.Tensor,       # (N, 3)  normalised π-groups [Fo_n, x0_n, β_n]
    raw: dict,                   # raw π-values per point: {Fo, x0_norm, beta}, each (N,)
    epsilon: float = 1.0,        # causal weight strength ε (Wang et al. 2022)
    n_bins: int = 10,            # time windows for causal weighting
) -> torch.Tensor:
    """
    Causal PDE residual loss (Wang et al., JMLR 2024) on the dimensionless PDE

        u_t* = Fo · u_x*x*  +  S(x*, t*)

    where u = ΔT / T_c, x* = x/l, t* = t/t_total, Fo = α·t_total/l², and the
    source S = δ_gauss(x*; x_b*, σ*) with x_b* = x0_norm + β·t* and σ* = 1/50.
    Because S has a parameter-independent peak (≈ 19.95), the residual is O(1)
    for every parameter set and the loss weights all of them equally.

    Points are grouped into n_bins uniform windows along t* ∈ [0, 1]; each bin k
    is weighted by w_k = exp(−ε · Σ_{j<k} L_j) so early times are mastered first.
    """
    coords = coords_norm.detach().requires_grad_(True)

    u = model(coords, pi_norm)                    # (N, 1)  dimensionless

    grad1 = torch.autograd.grad(
        u.sum(), coords, create_graph=True
    )[0]                                          # (N, 2)
    u_xn = grad1[:, 0:1]                          # ∂u/∂x*
    u_tn = grad1[:, 1:2]                          # ∂u/∂t*

    u_xnxn = torch.autograd.grad(
        u_xn.sum(), coords, create_graph=True
    )[0][:, 0:1]                                  # ∂²u/∂x*²

    Fo      = raw["Fo"].unsqueeze(1)
    x0_norm = raw["x0_norm"].unsqueeze(1)
    beta    = raw["beta"].unsqueeze(1)

    x_norm = coords[:, 0:1]
    t_norm = coords[:, 1:2]
    x_b    = x0_norm + beta * t_norm                       # burner position in x*
    S      = _gaussian_delta(x_norm, x_b, SIGMA_NORM)      # dimensionless source

    residual = u_tn - Fo * u_xnxn - S              # (N, 1)
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
    coords_norm: torch.Tensor,   # (N, 2)  x* ∈ {0, 1}
    params_norm: torch.Tensor,   # (N, 3)  π-groups
) -> torch.Tensor:
    """
    Neumann BC: ∂u/∂x* = 0 at x* = 0 and x* = 1  (same as ∂ΔT/∂x = 0).

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
    coords_norm: torch.Tensor,   # (N, 2)  [x*, t*]
    pi_norm: torch.Tensor,       # (N, 3)  [Fo_n, x0_n, β_n]
    raw: dict,                   # {Fo, x0_norm, beta}, each (N,)
    eps_x: float = 1e-3,
    eps_t: float = 1e-4,
) -> torch.Tensor:
    """
    Absolute dimensionless PDE residuals via finite differences — RAD only.

    Uses 4 forward passes (center, x±ε, t+ε) inside torch.no_grad(), so no
    computation graph is built (~40× cheaper in memory than autograd at the
    large pool sizes RAD needs). Accuracy is sufficient for RAD: we only need
    to know *where* residuals are large, not their exact values.

    residual = u_t* − Fo·u_x*x* − S(x*, t*)

    Returns: (N,) tensor of |residual| values.
    """
    with torch.no_grad():
        Fo      = raw["Fo"].unsqueeze(1)
        x0_norm = raw["x0_norm"].unsqueeze(1)
        beta    = raw["beta"].unsqueeze(1)

        c_xp = coords_norm.clone(); c_xp[:, 0] = (c_xp[:, 0] + eps_x).clamp(0.0, 1.0)
        c_xm = coords_norm.clone(); c_xm[:, 0] = (c_xm[:, 0] - eps_x).clamp(0.0, 1.0)
        c_tp = coords_norm.clone(); c_tp[:, 1] = (c_tp[:, 1] + eps_t).clamp(0.0, 1.0)

        u_c  = model(coords_norm, pi_norm)
        u_xp = model(c_xp,        pi_norm)
        u_xm = model(c_xm,        pi_norm)
        u_tp = model(c_tp,        pi_norm)

        u_xx = (u_xp - 2.0 * u_c + u_xm) / (eps_x ** 2)   # ∂²u/∂x*²
        u_t  = (u_tp - u_c) / eps_t                        # ∂u/∂t*

        x_norm = coords_norm[:, 0:1]
        t_norm = coords_norm[:, 1:2]
        x_b    = x0_norm + beta * t_norm
        S      = _gaussian_delta(x_norm, x_b, SIGMA_NORM)

        residual = (u_t - Fo * u_xx - S).squeeze(1)
        return residual.abs()


def ic_loss(
    model: nn.Module,
    coords_norm: torch.Tensor,   # (N, 2)  t* = 0
    params_norm: torch.Tensor,   # (N, 3)  π-groups
) -> torch.Tensor:
    """
    IC diagnostic: u(x*, 0) = 0.
    With hard IC (model output = t* · NN), this is always ≈ 0.
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
    (c_pde, pi_pde, raw_pde), (c_bc, pi_bc), (c_ic, pi_ic) = batch

    l_pde = pde_loss(model, c_pde, pi_pde, raw_pde, epsilon=epsilon, n_bins=n_bins)
    l_bc  = bc_loss(model, c_bc, pi_bc)
    l_ic  = ic_loss(model, c_ic, pi_ic)

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
