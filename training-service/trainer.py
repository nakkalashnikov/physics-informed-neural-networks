"""
Training loop: Adam + cosine LR annealing with three improvements:

  1. Hard IC  — model output = t_norm * NN, so ΔT(x,0) ≡ 0 by construction.
               lambda_ic = 0; ic_loss is monitored but not trained on.

  2. Causal PDE loss  — time bins are weighted by w_k = exp(-ε·Σ_{j<k} L_j)
                        so the network masters early times before late times
                        (Wang et al., JMLR 2024).

  3. Adaptive Fourier sigma  — bandwidth rises linearly from sigma_start to
                               sigma_end over the full Adam phase, giving the
                               network a smooth-to-sharp frequency curriculum.

L-BFGS phase removed: it was overfitting to a fixed batch and raised
validation error from 186% → 289% in the original run.
"""

import os
import json
import logging
import threading
import queue
import torch
import numpy as np
from tqdm import tqdm

from model import PINN, build_model
from physics import total_loss, analytical_delta_T
from sampler import (
    Normalizer, build_batch, sample_params, rad_resample_pde, compute_pi_groups,
)

log = logging.getLogger(__name__)


# ── Validation ────────────────────────────────────────────────────────────────

class _BatchPrefetcher:
    """
    Generates batches in a background thread so CPU sampling is overlapped
    with GPU compute. The queue holds ready-made batches — the main loop
    just calls .get() with zero wait.

    Memory cost: queue_size × ~2 MB per batch (negligible on 16+ GB VRAM).
    """

    def __init__(self, cfg: dict, normalizer, device: torch.device, queue_size: int = 4):
        self._cfg       = cfg
        self._norm      = normalizer
        self._device    = device
        self._queue     = queue.Queue(maxsize=queue_size)
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                batch = build_batch(self._cfg, self._norm, self._device)
                self._queue.put(batch)
            except Exception:
                pass  # keep running even on transient errors

    def get(self) -> tuple:
        return self._queue.get()

    def stop(self) -> None:
        self._stop.set()


def _cast_batch(batch: tuple, dtype: torch.dtype) -> tuple:
    """Recursively cast all tensors in a batch tuple to the given dtype."""
    def _cast(x):
        if isinstance(x, torch.Tensor):
            return x.to(dtype)
        if isinstance(x, dict):
            return {k: _cast(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(_cast(v) for v in x)
        return x
    return _cast(batch)


def validate(
    model: PINN,
    normalizer: Normalizer,
    cfg: dict,
    device: torch.device,
    n_test: int = 8,
) -> float:
    """
    Mean relative L2 error  ‖ΔT_PINN − ΔT_ref‖ / ‖ΔT_ref‖
    averaged over n_test random parameter sets.
    """
    model.eval()
    dtype = next(model.parameters()).dtype
    raw = sample_params(n_test, cfg, device)
    pi  = compute_pi_groups(raw)
    x_pts_norm = np.linspace(0.0, 1.0, 100)
    n = 100
    errors: list[float] = []

    with torch.no_grad():
        for k in range(n_test):
            alpha  = raw["alpha"][k].item()
            rho_c  = raw["rho_c"][k].item()
            l      = raw["l"][k].item()
            intens = raw["intensity"][k].item()
            x0     = raw["x0"][k].item()
            v      = raw["v"][k].item()
            t_tot  = raw["t_total"][k].item()

            t_q    = t_tot * 0.7
            t_norm = torch.full((n,), t_q / t_tot, device=device, dtype=dtype)
            x_norm = torch.tensor(x_pts_norm, dtype=dtype, device=device)
            coords = torch.stack([x_norm, t_norm], dim=1)

            # Network input: the three normalised π-groups for this parameter set.
            fo_n   = normalizer.norm_log(
                torch.full((n,), pi["Fo"][k].item(),      device=device, dtype=dtype), "Fo")
            x0_n   = normalizer.norm(
                torch.full((n,), pi["x0_norm"][k].item(), device=device, dtype=dtype), "x0_frac")
            beta_n = normalizer.norm(
                torch.full((n,), pi["beta"][k].item(),    device=device, dtype=dtype), "beta")
            pi_norm = torch.stack([fo_n, x0_n, beta_n], dim=1)

            # Network outputs dimensionless u; rescale to physical ΔT.
            T_c     = pi["T_c"][k].item()
            u_pred  = model(coords, pi_norm).squeeze().cpu().numpy()
            dT_pred = T_c * u_pred

            x_phys = x_pts_norm * l
            dT_ref = analytical_delta_T(x_phys, t_q, alpha, rho_c, l, intens, x0, v)

            rel_err = np.linalg.norm(dT_pred - dT_ref) / (np.linalg.norm(dT_ref) + 1e-8)
            errors.append(rel_err)

    model.train()
    mean_err = float(np.mean(errors))
    log.info("  Validation L2 relative error: %.4f", mean_err)
    return mean_err


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save(
    model: PINN,
    normalizer: Normalizer,
    cfg: dict,
    tag: str | int,
    history: dict,
    ckpt_dir: str,
) -> None:
    path = os.path.join(ckpt_dir, f"model_{tag}.pt")
    torch.save(
        {
            "model_state_dict":  model.state_dict(),
            "model_cfg":         cfg["model"],
            "normalizer_bounds": normalizer.bounds,
            "physics_cfg":       cfg["physics"],
        },
        path,
    )
    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    log.info("  Checkpoint saved → %s", path)


# ── Main training function ────────────────────────────────────────────────────

def train(cfg: dict, device: torch.device, resume: str | None = None) -> None:
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %d trainable parameters", n_params)

    use_fp64 = bool(cfg.get("use_fp64", False))
    dtype    = torch.float64 if use_fp64 else torch.float32
    if use_fp64:
        model = model.double()
        log.info("FP64 mode enabled (double precision)")

    use_amp = bool(cfg.get("use_amp", False)) and device.type == "cuda" and not use_fp64
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    if cfg.get("use_compile", False) and hasattr(torch, "compile"):
        # mode="default": kernel/operator fusion without CUDA Graphs.
        # "reduce-overhead" (CUDA Graphs) conflicts with create_graph=True in pde_loss
        # (donated-buffer aliasing) and with multiple model() calls in pde_residuals_fd
        # (output buffer overwrite). Both issues disappear with mode="default".
        model = torch.compile(model, mode="default")
        log.info("torch.compile enabled (mode=default, no CUDA Graphs)")

    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        # infer step from filename, e.g. model_55000.pt → 55000
        import re
        m = re.search(r"model_(\d+)\.pt", resume)
        start_step = int(m.group(1)) if m else 0
        log.info("Resumed from %s  (start_step=%d)", resume, start_step)

    normalizer = Normalizer(cfg)
    t_cfg = cfg["training"]
    m_cfg = cfg["model"]

    weights = {
        "pde": float(t_cfg["lambda_pde"]),
        "bc":  float(t_cfg["lambda_bc"]),
        "ic":  float(t_cfg["lambda_ic"]),
    }

    causal_epsilon = float(t_cfg.get("causal_epsilon", 1.0))
    causal_n_bins  = int(t_cfg.get("causal_n_bins", 10))

    sigma_start = float(m_cfg.get("fourier_sigma_start", m_cfg.get("fourier_sigma", 1.0)))
    sigma_end   = float(m_cfg.get("fourier_sigma_end",   m_cfg.get("fourier_sigma", 3.5)))

    history: dict = {
        "step": [], "loss": [], "pde": [], "bc": [], "ic": [],
        "sigma": [],
        "val": [],  # list of [step, error] pairs
    }

    use_rad          = bool(cfg["sampling"].get("rad_enabled", False))
    rad_update_every = int(cfg["sampling"].get("rad_update_every", 50))
    rad_cache: tuple | None = None

    log.info("=" * 60)
    log.info("Adam  (%d steps)  |  sigma %.1f → %.1f  |  causal ε=%.1f",
             t_cfg["adam_steps"], sigma_start, sigma_end, causal_epsilon)
    log.info("Hard IC: on   |   FP64: %s  |   AMP: %s  |   RAD: %s",
             "on" if use_fp64 else "off",
             "on" if use_amp else "off",
             f"on (every {rad_update_every} steps)" if use_rad else "off")
    log.info("=" * 60)

    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["adam_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_cfg["adam_steps"],
        eta_min=t_cfg["adam_lr_min"],
    )
    if start_step > 0:
        # Fast-forward the scheduler step-by-step so it uses the correct
        # closed-form cosine value at start_step.
        # CosineAnnealingLR's get_lr() is recursive (depends on group['lr']),
        # so last_epoch= alone won't produce the right LR when starting mid-schedule.
        for _ in range(start_step):
            scheduler.step()
        log.info("Scheduler fast-forwarded to step %d  (lr=%.2e)",
                 start_step, optimizer.param_groups[0]["lr"])

    model.train()
    use_prefetch = bool(cfg.get("use_prefetch", True)) and device.type == "cuda"
    prefetcher   = _BatchPrefetcher(cfg, normalizer, device) if use_prefetch else None
    if use_prefetch:
        log.info("Async batch prefetcher enabled (queue_size=4)")

    pbar = tqdm(range(start_step, t_cfg["adam_steps"]), desc="Adam", dynamic_ncols=True)

    for step in pbar:

        # ── Adaptive Fourier sigma ────────────────────────────────────────
        progress = step / max(t_cfg["adam_steps"] - 1, 1)
        sigma = sigma_start + (sigma_end - sigma_start) * progress
        model.set_sigma(sigma)

        if use_rad and step % rad_update_every == 0:
            rad_cache = rad_resample_pde(model, cfg, normalizer, device)

        # Get pre-built batch from prefetcher (or build inline as fallback).
        # When RAD is active, replace only the PDE tuple — BC/IC come from prefetch.
        if prefetcher is not None:
            _raw = prefetcher.get()
            batch = (_raw[0] if rad_cache is None else rad_cache, _raw[1], _raw[2])
        else:
            batch = build_batch(cfg, normalizer, device, pde_override=rad_cache)

        if use_fp64:
            batch = _cast_batch(batch, dtype)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss, l_pde, l_bc, l_ic = total_loss(
                model, batch, weights,
                epsilon=causal_epsilon,
                n_bins=causal_n_bins,
            )

        if not torch.isfinite(loss):
            log.warning("Step %d: non-finite loss (%.3e) — skipping", step, loss.item())
            if use_amp:
                scaler.update()
            scheduler.step()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grads_ok = all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        if not grads_ok:
            log.warning("Step %d: NaN/Inf gradient — skipping", step)
            optimizer.zero_grad()
            if use_amp:
                scaler.update()
            scheduler.step()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg["grad_clip_norm"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % t_cfg["log_every"] == 0:
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{loss.item():.3e}",
                pde=f"{l_pde:.3e}",
                bc=f"{l_bc:.3e}",
                sigma=f"{sigma:.2f}",
                lr=f"{lr:.1e}",
            )
            history["step"].append(step)
            history["loss"].append(loss.item())
            history["pde"].append(l_pde)
            history["bc"].append(l_bc)
            history["ic"].append(l_ic)
            history["sigma"].append(sigma)

        if step > 0 and step % t_cfg["save_every"] == 0:
            _save(model, normalizer, cfg, step, history, t_cfg["checkpoint_dir"])

        if step > 0 and step % t_cfg["validate_every"] == 0:
            err = validate(model, normalizer, cfg, device)
            history["val"].append([step, err])

    if prefetcher is not None:
        prefetcher.stop()

    # ── Final save ────────────────────────────────────────────────────────────
    final_err = validate(model, normalizer, cfg, device)
    history["val"].append(["final", final_err])
    _save(model, normalizer, cfg, "final", history, t_cfg["checkpoint_dir"])
    log.info("Training complete. Final validation L2 error: %.4f", final_err)
