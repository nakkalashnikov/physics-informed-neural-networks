"""
Physics-parameter sampling, dimensionless reduction, and collocation points.

── Dimensionless formulation (Buckingham-π) ──────────────────────────────────
The physical problem has 5 free parameters (α, l, i_eff, x0, v) plus t_total.
Non-dimensionalising with

    x* = x / l          t* = t / t_total          u = ΔT / T_c
    T_c = i_eff · t_total / l        (characteristic temperature rise)

collapses the PDE to

    u_t* = Fo · u_x*x*  +  S(x*, t*)

which depends on exactly THREE dimensionless groups:

    Fo       = α · t_total / l²        Fourier number (diffusion vs. time)
    x0_norm  = x0 / l                  burner start (fraction of length)
    β        = v · t_total / l         burner travel (fraction of length)

The source S = δ_gauss(x*; x_b*, σ*) with σ* = 1/50 has a parameter-independent
peak ≈ 19.95, so the residual is O(1) for every parameter set — the loss no
longer ignores low-amplitude (small i_eff / large l) cases.

The network input is therefore (x*, t*, Fo_n, x0_n, β_n) and it outputs the
dimensionless u.  The physical temperature is recovered as ΔT = T_c · u, applied
by the caller (validation / inference), never inside the network.
"""

import math

import torch


# ── Normalizer ────────────────────────────────────────────────────────────────

class Normalizer:
    """
    Linear normalisation of scalars to [0, 1], plus log-normalisation for the
    Fourier number (which spans ~4 orders of magnitude across the parameter
    space and would otherwise crush the small-Fo end toward 0).
    """

    def __init__(self, cfg: dict) -> None:
        p = cfg["physics"]

        alpha_lo, alpha_hi = p["alpha_range"]
        l_lo,     l_hi     = p["length_range"]
        t_lo,     t_hi     = p["t_total_range"]

        # Fourier number extremes:  Fo = α·t_total/l²
        #   min → small α, short time, long pipe ;  max → large α, long time, short pipe
        fo_lo = alpha_lo * t_lo / (l_hi ** 2)
        fo_hi = alpha_hi * t_hi / (l_lo ** 2)

        self.bounds: dict[str, tuple[float, float]] = {
            "alpha":     tuple(p["alpha_range"]),
            "rho_c":     tuple(p["rho_c_range"]),
            "l":         tuple(p["length_range"]),
            "intensity": tuple(p["intensity_range"]),
            "x0_frac":   tuple(p["x0_fraction_range"]),
            "v":         tuple(p["velocity_range"]),
            "t_total":   tuple(p["t_total_range"]),
            "i_eff": (
                p["intensity_range"][0] / p["rho_c_range"][1],
                p["intensity_range"][1] / p["rho_c_range"][0],
            ),
            # ── Dimensionless π-groups (network inputs) ──
            "Fo":   (fo_lo, fo_hi),   # log-normalised
            "beta": (0.0, 1.0),       # β = v·t_total/l ∈ (0, 1) by the in-pipe constraint
        }

    def norm(self, val: torch.Tensor, key: str) -> torch.Tensor:
        lo, hi = self.bounds[key]
        return (val - lo) / (hi - lo)

    def denorm(self, val: torch.Tensor, key: str) -> torch.Tensor:
        lo, hi = self.bounds[key]
        return val * (hi - lo) + lo

    def norm_log(self, val: torch.Tensor, key: str) -> torch.Tensor:
        """Log-space normalisation to [0, 1] — for quantities spanning decades."""
        lo, hi = self.bounds[key]
        log_lo, log_hi = math.log(lo), math.log(hi)
        return (torch.log(val.clamp_min(1e-30)) - log_lo) / (log_hi - log_lo)


# ── Parameter sampling ────────────────────────────────────────────────────────

def sample_params(n: int, cfg: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """
    Sample n physics-parameter sets uniformly on the GPU.

    Returns a dict of (n,) float32 tensors on `device`.
    Velocity is constrained so x0 + v·t_total ≤ l (burner stays inside the pipe).
    """
    p = cfg["physics"]

    u = torch.rand(n, 5, device=device)

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

    v_lo = torch.full((n,), p["velocity_range"][0], device=device)
    v_hi = torch.clamp(
        (l - x0) / t_total,
        min=p["velocity_range"][0],
        max=p["velocity_range"][1],
    )
    v = torch.rand(n, device=device) * (v_hi - v_lo) + v_lo

    i_eff = intensity / rho_c   # [K·m/s]

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


# ── Dimensionless reduction ───────────────────────────────────────────────────

def compute_pi_groups(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Map physical parameters → dimensionless π-groups and the temperature scale.

    Returns (each same shape as the inputs):
        Fo       = α·t_total/l²       diffusion number
        x0_norm  = x0/l               burner start fraction
        beta     = v·t_total/l        burner travel fraction
        T_c      = i_eff·t_total/l    characteristic ΔT  (output rescale factor)
    """
    l = raw["l"]
    return {
        "Fo":      raw["alpha"] * raw["t_total"] / (l ** 2),
        "x0_norm": raw["x0"] / l,
        "beta":    raw["v"] * raw["t_total"] / l,
        "T_c":     raw["i_eff"] * raw["t_total"] / l,
    }


# ── Collocation-point construction ───────────────────────────────────────────

def _to_column(scalar: torch.Tensor, n: int) -> torch.Tensor:
    """Expand a 0-dim or (1,) tensor to a (n,) tensor."""
    return scalar.reshape(1).expand(n).clone()


def _pi_norm_for_set(
    pi_k: dict[str, torch.Tensor],
    normalizer: Normalizer,
    n_pts: int,
) -> torch.Tensor:
    """
    Build a (n_pts, 3) normalised π-group tensor for one parameter set.
    Column order: [Fo_n, x0_n, β_n].
    """
    fo_n   = normalizer.norm_log(_to_column(pi_k["Fo"],      n_pts), "Fo")
    x0_n   = normalizer.norm(    _to_column(pi_k["x0_norm"], n_pts), "x0_frac")
    beta_n = normalizer.norm(    _to_column(pi_k["beta"],    n_pts), "beta")
    return torch.stack([fo_n, x0_n, beta_n], dim=1)


def rad_resample_pde(
    model,
    cfg: dict,
    normalizer: Normalizer,
    device: torch.device,
) -> tuple:
    """
    RAD — Residual-based Adaptive Distribution (arXiv:2207.10289), dimensionless.

      1. Sample pool_factor × n_pde candidate points uniformly in (x*, t*).
      2. Evaluate |dimensionless residual| via finite differences (no_grad).
      3. Draw n_pde points with probability ∝ |residual| — concentrates near
         the moving heat source automatically.

    Returns (coords_pde, pi_pde, raw_pde) — same format as build_batch's PDE
    tuple, where raw_pde holds the raw π-values (Fo, x0_norm, beta) per point.
    """
    from physics import pde_residuals_fd

    s_cfg       = cfg["sampling"]
    n_p         = s_cfg["n_params_per_step"]
    n_pde       = s_cfg["n_pde"]
    pool_factor = int(s_cfg.get("rad_pool_factor", 5))
    pps_cand    = (n_pde * pool_factor) // n_p
    pps_target  = n_pde // n_p

    raw_sets = sample_params(n_p, cfg, device)
    pi_sets  = compute_pi_groups(raw_sets)
    dtype    = next(model.parameters()).dtype

    raw_keys = ("Fo", "x0_norm", "beta")
    coords_out, pi_out = [], []
    raw_acc: dict[str, list[torch.Tensor]] = {k: [] for k in raw_keys}

    model.eval()
    for k in range(n_p):
        pi_k = {key: pi_sets[key][k] for key in pi_sets}

        xn = torch.rand(pps_cand, device=device, dtype=dtype)
        tn = torch.rand(pps_cand, device=device, dtype=dtype)
        coords_cand = torch.stack([xn, tn], dim=1)
        pi_cand     = _pi_norm_for_set(pi_k, normalizer, pps_cand).to(dtype)
        raw_cand    = {key: _to_column(pi_k[key], pps_cand).to(dtype) for key in raw_keys}

        resids = pde_residuals_fd(model, coords_cand, pi_cand, raw_cand)   # (pps_cand,)
        probs  = resids.float() / (resids.float().sum() + 1e-8)
        idx    = torch.multinomial(probs, pps_target, replacement=False)

        coords_out.append(coords_cand[idx])
        pi_out.append(pi_cand[idx])
        for key in raw_keys:
            raw_acc[key].append(raw_cand[key][idx])

    model.train()
    raw_pde = {key: torch.cat(raw_acc[key]) for key in raw_keys}
    return torch.cat(coords_out), torch.cat(pi_out), raw_pde


def build_batch(
    cfg: dict,
    normalizer: Normalizer,
    device: torch.device,
    pde_override: tuple | None = None,
) -> tuple:
    """
    Build one full training batch in dimensionless coordinates.

    Returns
    -------
    (
        (coords_pde, pi_pde, raw_pde),   ← for pde_loss
        (coords_bc,  pi_bc),             ← for bc_loss
        (coords_ic,  pi_ic),             ← for ic_loss
    )

    coords_* : (N, 2)  – [x*, t*] ∈ [0, 1]
    pi_*     : (N, 3)  – normalised [Fo_n, x0_n, β_n]
    raw_pde  : dict of (N_pde,) tensors – raw π-values {Fo, x0_norm, beta}

    pde_override : if provided (from rad_resample_pde), skip PDE point generation
                   and use it instead. BC/IC are always freshly sampled.
    """
    n_p    = cfg["sampling"]["n_params_per_step"]
    n_bc   = cfg["sampling"]["n_bc"]
    n_ic   = cfg["sampling"]["n_ic"]
    pps_bc = max(n_bc // n_p, 2)
    pps_ic = max(n_ic // n_p, 2)

    raw_sets = sample_params(n_p, cfg, device)
    pi_sets  = compute_pi_groups(raw_sets)

    coords_bc, pi_bc = [], []
    coords_ic, pi_ic = [], []

    raw_keys = ("Fo", "x0_norm", "beta")
    if pde_override is None:
        n_pde   = cfg["sampling"]["n_pde"]
        pps_pde = n_pde // n_p
        coords_pde, pi_pde = [], []
        raw_pde_acc: dict[str, list[torch.Tensor]] = {k: [] for k in raw_keys}

    for k in range(n_p):
        pi_k = {key: pi_sets[key][k] for key in pi_sets}

        if pde_override is None:
            # ── PDE interior: 70% uniform + 30% near burner trajectory ──────
            n_unif   = int(pps_pde * 0.70)
            n_burner = pps_pde - n_unif

            xn_unif = torch.rand(n_unif, device=device)
            tn_unif = torch.rand(n_unif, device=device)

            tn_burner = torch.rand(n_burner, device=device)
            sigma_n   = 1.0 / 20.0
            x_b_norm  = pi_k["x0_norm"] + pi_k["beta"] * tn_burner   # burner pos in x*
            xn_burner = (x_b_norm + torch.randn(n_burner, device=device) * sigma_n
                         ).clamp(0.0, 1.0)

            xn = torch.cat([xn_unif, xn_burner])
            tn = torch.cat([tn_unif, tn_burner])
            coords_pde.append(torch.stack([xn, tn], dim=1))
            pi_pde.append(_pi_norm_for_set(pi_k, normalizer, pps_pde))
            for key in raw_keys:
                raw_pde_acc[key].append(_to_column(pi_k[key], pps_pde))

        # ── BC: x* ∈ {0, 1} (equal split), t* ∈ (0, 1) ───────────────────
        half = pps_bc // 2
        x_bc = torch.cat([
            torch.zeros(half,         device=device),
            torch.ones(pps_bc - half, device=device),
        ])
        t_bc = torch.rand(pps_bc, device=device)
        coords_bc.append(torch.stack([x_bc, t_bc], dim=1))
        pi_bc.append(_pi_norm_for_set(pi_k, normalizer, pps_bc))

        # ── IC: t* = 0, x* ∈ (0, 1) (diagnostic; hard-enforced in model) ──
        x_ic = torch.rand(pps_ic, device=device)
        t_ic = torch.zeros(pps_ic, device=device)
        coords_ic.append(torch.stack([x_ic, t_ic], dim=1))
        pi_ic.append(_pi_norm_for_set(pi_k, normalizer, pps_ic))

    if pde_override is None:
        raw_pde   = {key: torch.cat(raw_pde_acc[key]) for key in raw_keys}
        pde_tuple = (torch.cat(coords_pde), torch.cat(pi_pde), raw_pde)
    else:
        pde_tuple = pde_override

    return (
        pde_tuple,
        (torch.cat(coords_bc), torch.cat(pi_bc)),
        (torch.cat(coords_ic), torch.cat(pi_ic)),
    )
