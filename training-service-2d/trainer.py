"""
Hybrid-loss training loop for the PI-DeepONet.

    L = lambda_data * L_data  +  lambda_pde * L_pde  +  lambda_bc * L_bc
    all terms are RELATIVE (divided by per-trajectory label scale) so the ~7-order Q* spread
    across the parameter space does not let loud samples dominate (see data.py).

Adam + cosine LR, RFF sigma curriculum, gradient-norm clipping, non-finite-gradient guard
(MPS/sharp-source safety, per project memory). AMP/compile/FP64 stay off.
"""

from __future__ import annotations

import math
import time

import torch

from data import Batch, BatchPrefetcher, CachedBatcher, SourcePrefetcher, build_batch
from model import build_model
from physics import bc_flux_residual, bc_insulation_residual, pde_residual
from validation.report import pi_table_sweep


def select_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _t(arr, device, grad=False):
    x = torch.as_tensor(arr, dtype=torch.float32, device=device)
    if grad:
        x.requires_grad_(True)
    return x


def _sigma_at(step: int, total: int, cfg: dict) -> float:
    # Ramp the RFF bandwidth to its target over the first `ramp_frac` of the run, then HOLD.
    # A full-length linear ramp spends equal time at every bandwidth and only reaches the
    # useful high-frequency band when the cosine LR has already decayed to ~0 — wasting most
    # of the budget. Reaching sigma_end early (25%) leaves the bulk of steps at full resolution
    # with a still-useful LR (Wang et al., JMLR 2024, frequency-progressive training).
    a = float(cfg["trunk"]["rff_sigma_start"])
    b = float(cfg["trunk"]["rff_sigma_end"])
    frac = float(cfg["trunk"].get("rff_sigma_ramp_frac", 0.25))
    ramp = max(int(total * frac), 1)
    return a + (b - a) * min(step / ramp, 1.0)


def _make_lr_lambda(total: int, warmup: int, lr_peak: float, lr_min: float):
    """Linear warmup (0 -> peak) then cosine decay (peak -> lr_min).

    Warmup tames the huge early PDE-residual gradients (the pde 6000% spike at step 0);
    standard PINN practice (Wang et al., JMLR 2024). Returns a multiplicative factor on lr_peak.
    """
    min_factor = lr_min / lr_peak

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_factor + (1.0 - min_factor) * cos

    return lr_lambda


def _clipped_mse(r2: torch.Tensor, gamma: float) -> torch.Tensor:
    """Mean squared relative residual with each point capped at gamma * median(r^2).

    A handful of points landing on the razor-thin moving source have second
    derivatives (hence residuals) orders of magnitude above the rest; their squared
    term then dominates the mean and steers the whole gradient. grad-norm clipping
    only shortens that gradient vector — it does NOT fix its direction, so the
    descent still chases the spike. Capping each point at gamma * median bounds any
    single point's contribution (and zeroes its gradient when clipped), so the bulk
    of the field can keep descending. BRDR-style robust residual (Jiang et al. 2025).
    The cap is detached (a threshold, not part of the graph). gamma <= 0 -> plain
    MSE = exact previous behavior, for clean A/B.
    """
    if gamma and gamma > 0.0:
        cap = gamma * r2.detach().median()
        r2 = torch.minimum(r2, cap)
    return r2.mean()


def compute_losses(model, batch: Batch, device, sigma_factor: float,
                   clip_gamma: float = 0.0) -> dict:
    """Return dict of the three relative loss components (+ total)."""
    # --- data loss ---
    db = _t(batch.data_branch, device)
    dc = _t(batch.data_coords, device)
    du = _t(batch.data_u, device)
    ds = _t(batch.data_scale, device)
    u_pred = model(db, dc)
    L_data = (((u_pred - du) / ds) ** 2).mean()

    # --- pde residual ---
    ib = _t(batch.int_branch, device)
    ic = _t(batch.int_coords, device, grad=True)
    Fo = _t(batch.int_Fo, device)
    AR = _t(batch.int_AR, device)
    isc = _t(batch.int_scale, device)
    R = pde_residual(model, ib, ic, Fo, AR)
    L_pde = _clipped_mse((R / isc) ** 2, clip_gamma)

    # --- flux BC at yhat=0 ---
    bb = _t(batch.bc0_branch, device)
    bc = _t(batch.bc0_coords, device, grad=True)
    Q = _t(batch.bc0_Q, device)
    xb = _t(batch.bc0_xb, device)
    w = _t(batch.bc0_w, device)
    bfs = _t(batch.bc0_fluxscale, device)
    R_bc0 = bc_flux_residual(model, bb, bc, Q, xb, w, sigma_factor)
    L_bc0 = _clipped_mse((R_bc0 / bfs) ** 2, clip_gamma)

    # --- insulated edges ---
    xb_ = _t(batch.insx_branch, device)
    xc_ = _t(batch.insx_coords, device, grad=True)
    xsc = _t(batch.insx_scale, device)
    R_insx = bc_insulation_residual(model, xb_, xc_, axis=0)
    L_insx = _clipped_mse((R_insx / xsc) ** 2, clip_gamma)

    yb_ = _t(batch.insy_branch, device)
    yc_ = _t(batch.insy_coords, device, grad=True)
    ysc = _t(batch.insy_scale, device)
    R_insy = bc_insulation_residual(model, yb_, yc_, axis=1)
    L_insy = _clipped_mse((R_insy / ysc) ** 2, clip_gamma)

    # flux BC (yhat=0) and insulation (other edges) returned SEPARATELY: the analytic labels
    # come from a cosine basis (insulated wall + delta source, u_y(0)=0) while this flux term
    # demands u_y(0)=-Q*ghat — inconsistent at the hot boundary (see check_consistency.py).
    # lambda_flux=0 drops the flux term to test whether that conflict drives the plateau.
    return {"data": L_data, "pde": L_pde, "flux": L_bc0, "ins": L_insx + L_insy}


def train(cfg: dict, device: torch.device, total_steps: int | None = None,
          use_prefetch: bool | None = None, log_every: int = 50,
          checkpoint_path: str = "checkpoint.pt", cache_path: str | None = None,
          profile: bool = False, n_traj: int | None = None, n_int: int | None = None,
          validate_every: int = 0, val_n: int = 16) -> str:
    tr = cfg["training"]
    total = int(total_steps if total_steps is not None else tr["total_steps"])
    seed = int(tr["seed"])
    torch.manual_seed(seed)

    # Batch-size overrides (throughput sweep): bigger batch amortizes per-step launch
    # overhead (launch-bound small model) AND raises epochs/sec to fight undertraining.
    if n_traj is not None:
        cfg["batch"]["n_traj_per_batch"] = int(n_traj)
    if n_int is not None:
        cfg["batch"]["n_interior"] = int(n_int)
    print(f"batch: n_traj={cfg['batch']['n_traj_per_batch']} n_int={cfg['batch']['n_interior']}")

    model = build_model(cfg).to(device)
    lam = cfg["loss"]
    sf = float(cfg["physics"]["sigma_factor"])
    clip_gamma = float(lam.get("residual_clip_gamma", 0.0))
    lam_flux = float(lam.get("lambda_flux", lam["lambda_bc"]))   # flux BC at yhat=0 (own weight)
    if clip_gamma > 0:
        print(f"residual clip: gamma={clip_gamma:g} (per-point cap at gamma*median)")
    print(f"loss weights: data={lam['lambda_data']:g} pde={lam['lambda_pde']:g} "
          f"flux={lam_flux:g} ins(bc)={lam['lambda_bc']:g}")

    lr_peak = float(tr["lr"])
    warmup = int(tr.get("warmup_steps", 2000))
    opt = torch.optim.Adam(model.parameters(), lr=lr_peak)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _make_lr_lambda(total, warmup, lr_peak, float(tr["lr_min"])))
    clip = float(tr["grad_clip_norm"])

    import numpy as np
    if cache_path is not None:                       # cached labels -> cheap on-the-fly batches
        base = CachedBatcher(cfg, seed, path=cache_path)
        on_cuda = device.type == "cuda"
        prefetch = on_cuda if use_prefetch is None else use_prefetch
        pf = SourcePrefetcher(base) if prefetch else None   # overlap CPU build w/ GPU
        source = pf if prefetch else base
        rng = None
    else:
        on_cuda = device.type == "cuda"
        prefetch = on_cuda if use_prefetch is None else use_prefetch
        pf = BatchPrefetcher(cfg, seed) if prefetch else None
        source = pf
        rng = None if prefetch else np.random.default_rng(seed)

    def _sync():
        if profile and device.type == "cuda":
            torch.cuda.synchronize()

    prof = {"get": 0.0, "fwd": 0.0, "bwd": 0.0, "n": 0}

    t0 = time.time()
    for step in range(total):
        sigma = _sigma_at(step, total, cfg)
        model.set_sigma(sigma)

        _sync(); _p0 = time.perf_counter()
        batch = source.get() if source is not None else build_batch(cfg, rng)
        _sync(); _p1 = time.perf_counter()

        loss_parts = compute_losses(model, batch, device, sf, clip_gamma)
        loss = (float(lam["lambda_data"]) * loss_parts["data"]
                + float(lam["lambda_pde"]) * loss_parts["pde"]
                + lam_flux * loss_parts["flux"]
                + float(lam["lambda_bc"]) * loss_parts["ins"])
        _sync(); _p2 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # non-finite-gradient guard: clip returns the total grad-norm (a fused reduction);
        # if any grad is NaN/Inf the norm is non-finite. One check instead of a 40-tensor
        # Python loop with a sync per param (which dominated the launch-bound step overhead).
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if not torch.isfinite(total_norm):
            opt.zero_grad(set_to_none=True)
            if step % log_every == 0:
                print(f"step {step}: non-finite grad — skipped")
            continue

        opt.step()
        sched.step()
        _sync(); _p3 = time.perf_counter()

        if validate_every > 0 and step > 0 and (step % validate_every == 0 or step == total - 1):
            model.eval()
            with torch.no_grad():
                sw = pi_table_sweep(model, cfg, device, n_cases=val_n)
            model.train()
            model.set_sigma(sigma)
            elapsed_h = (time.time() - t0) / 3600
            print(f"  ══ VAL step {step:6d} ({100.0*(step+1)/total:.0f}%) │ σ {sigma:.2f} │ "
                  f"L2 median {sw['median']:.1%}  p90 {sw['p90']:.1%}  max {sw['max']:.1%}  "
                  f"│ {elapsed_h:.2f}h elapsed")

        if profile:
            prof["get"] += _p1 - _p0; prof["fwd"] += _p2 - _p1
            prof["bwd"] += _p3 - _p2; prof["n"] += 1

        if step % log_every == 0 or step == total - 1:
            rate = (step + 1) / (time.time() - t0)
            eta_h = (total - step - 1) / max(rate, 1e-9) / 3600.0
            prog = 100.0 * (step + 1) / total
            # Loss components are mean-squared RELATIVE residuals; sqrt → RMS relative
            # error, shown as a percentage so the numbers are readable at a glance
            # (e.g. data 69% means the predicted field is ~69% off on average).
            data_pct = 100.0 * loss_parts["data"].item() ** 0.5
            pde_pct  = 100.0 * loss_parts["pde"].item() ** 0.5
            flux_pct = 100.0 * loss_parts["flux"].item() ** 0.5
            ins_pct  = 100.0 * loss_parts["ins"].item() ** 0.5
            print(f"step {step:6d} {prog:3.0f}% │ σ {sigma:.2f} │ "
                  f"data {data_pct:5.1f}%  pde {pde_pct:6.1f}%  flux {flux_pct:6.1f}%  ins {ins_pct:5.1f}% │ "
                  f"lr {sched.get_last_lr()[0]:.1e} │ {rate:.1f} it/s │ ETA {eta_h:4.1f}h")

            if profile and prof["n"] > 0:
                n = prof["n"]
                g, f, b = prof["get"] / n * 1e3, prof["fwd"] / n * 1e3, prof["bwd"] / n * 1e3
                tot = g + f + b
                print(f"           profile (ms/step): get {g:5.1f} ({g/tot:4.0%})  "
                      f"fwd+resid {f:5.1f} ({f/tot:4.0%})  bwd+step {b:5.1f} ({b/tot:4.0%})  "
                      f"| total {tot:5.1f}")
                prof.update(get=0.0, fwd=0.0, bwd=0.0, n=0)

    if pf:
        pf.close()
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": total}, checkpoint_path)
    print(f"saved {checkpoint_path}")
    return checkpoint_path
