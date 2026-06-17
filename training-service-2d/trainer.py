"""
Hybrid-loss training loop for the PI-DeepONet.

    L = lambda_data * L_data  +  lambda_pde * L_pde  +  lambda_bc * L_bc
    all terms are RELATIVE (divided by per-trajectory label scale) so the ~7-order Q* spread
    across the parameter space does not let loud samples dominate (see data.py).

Adam + cosine LR, RFF sigma curriculum, gradient-norm clipping, non-finite-gradient guard
(MPS/sharp-source safety, per project memory). AMP/compile/FP64 stay off.
"""

from __future__ import annotations

import time

import torch

from data import Batch, BatchPrefetcher, CachedBatcher, SourcePrefetcher, build_batch
from model import build_model
from physics import bc_flux_residual, bc_insulation_residual, pde_residual


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
    a = float(cfg["trunk"]["rff_sigma_start"])
    b = float(cfg["trunk"]["rff_sigma_end"])
    return a + (b - a) * min(step / max(total, 1), 1.0)


def compute_losses(model, batch: Batch, device, sigma_factor: float) -> dict:
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
    L_pde = ((R / isc) ** 2).mean()

    # --- flux BC at yhat=0 ---
    bb = _t(batch.bc0_branch, device)
    bc = _t(batch.bc0_coords, device, grad=True)
    Q = _t(batch.bc0_Q, device)
    xb = _t(batch.bc0_xb, device)
    w = _t(batch.bc0_w, device)
    bfs = _t(batch.bc0_fluxscale, device)
    R_bc0 = bc_flux_residual(model, bb, bc, Q, xb, w, sigma_factor)
    L_bc0 = ((R_bc0 / bfs) ** 2).mean()

    # --- insulated edges ---
    xb_ = _t(batch.insx_branch, device)
    xc_ = _t(batch.insx_coords, device, grad=True)
    xsc = _t(batch.insx_scale, device)
    R_insx = bc_insulation_residual(model, xb_, xc_, axis=0)
    L_insx = ((R_insx / xsc) ** 2).mean()

    yb_ = _t(batch.insy_branch, device)
    yc_ = _t(batch.insy_coords, device, grad=True)
    ysc = _t(batch.insy_scale, device)
    R_insy = bc_insulation_residual(model, yb_, yc_, axis=1)
    L_insy = ((R_insy / ysc) ** 2).mean()

    L_bc = L_bc0 + L_insx + L_insy
    return {"data": L_data, "pde": L_pde, "bc": L_bc}


def train(cfg: dict, device: torch.device, total_steps: int | None = None,
          use_prefetch: bool | None = None, log_every: int = 50,
          checkpoint_path: str = "checkpoint.pt", cache_path: str | None = None,
          profile: bool = False, n_traj: int | None = None, n_int: int | None = None) -> str:
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

    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=float(tr["lr_min"]))
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
        model.set_sigma(_sigma_at(step, total, cfg))

        _sync(); _p0 = time.perf_counter()
        batch = source.get() if source is not None else build_batch(cfg, rng)
        _sync(); _p1 = time.perf_counter()

        loss_parts = compute_losses(model, batch, device, sf)
        loss = (float(lam["lambda_data"]) * loss_parts["data"]
                + float(lam["lambda_pde"]) * loss_parts["pde"]
                + float(lam["lambda_bc"]) * loss_parts["bc"])
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

        if profile:
            prof["get"] += _p1 - _p0; prof["fwd"] += _p2 - _p1
            prof["bwd"] += _p3 - _p2; prof["n"] += 1

        if step % log_every == 0 or step == total - 1:
            rate = (step + 1) / (time.time() - t0)
            # Loss components are mean-squared RELATIVE residuals; sqrt → RMS relative
            # error, shown as a percentage so the numbers are readable at a glance
            # (e.g. data 69% means the predicted field is ~69% off on average).
            data_pct = 100.0 * loss_parts["data"].item() ** 0.5
            pde_pct  = 100.0 * loss_parts["pde"].item() ** 0.5
            bc_pct   = 100.0 * loss_parts["bc"].item() ** 0.5
            print(f"step {step:6d} | "
                  f"data {data_pct:5.1f}% | pde {pde_pct:5.1f}% | bc {bc_pct:5.1f}% | "
                  f"loss {loss.item():.3e} | lr {sched.get_last_lr()[0]:.1e} | {rate:.1f} it/s")

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
