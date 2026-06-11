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
import torch
import numpy as np
from tqdm import tqdm

from model import PINN, build_model
from physics import total_loss, analytical_delta_T
from sampler import Normalizer, build_batch, sample_params

log = logging.getLogger(__name__)


# ── Validation ────────────────────────────────────────────────────────────────

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
    raw = sample_params(n_test, cfg, device)
    x_pts_norm = np.linspace(0.0, 1.0, 100)
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
            i_eff  = raw["i_eff"][k].item()

            t_q    = t_tot * 0.7
            t_norm = torch.full((100,), t_q / t_tot, device=device)
            x_norm = torch.tensor(x_pts_norm, dtype=torch.float32, device=device)
            coords = torch.stack([x_norm, t_norm], dim=1)

            n = 100
            alpha_n  = normalizer.norm(torch.full((n,), alpha,  device=device), "alpha")
            l_n      = normalizer.norm(torch.full((n,), l,      device=device), "l")
            i_eff_n  = normalizer.norm(torch.full((n,), i_eff,  device=device), "i_eff")
            x0_n     = torch.full((n,), x0 / l, device=device)
            v_n      = normalizer.norm(torch.full((n,), v,      device=device), "v")
            params   = torch.stack([alpha_n, l_n, i_eff_n, x0_n, v_n], dim=1)

            dT_pred = model(coords, params).squeeze().cpu().numpy()

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

def train(cfg: dict, device: torch.device) -> None:
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %d trainable parameters", n_params)

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

    log.info("=" * 60)
    log.info("Adam  (%d steps)  |  sigma %.1f → %.1f  |  causal ε=%.1f",
             t_cfg["adam_steps"], sigma_start, sigma_end, causal_epsilon)
    log.info("Hard IC: on   |   L-BFGS: removed")
    log.info("=" * 60)

    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["adam_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_cfg["adam_steps"],
        eta_min=t_cfg["adam_lr_min"],
    )

    model.train()
    pbar = tqdm(range(t_cfg["adam_steps"]), desc="Adam", dynamic_ncols=True)

    for step in pbar:

        # ── Adaptive Fourier sigma ────────────────────────────────────────
        progress = step / max(t_cfg["adam_steps"] - 1, 1)
        sigma = sigma_start + (sigma_end - sigma_start) * progress
        model.set_sigma(sigma)

        batch = build_batch(cfg, normalizer, device)

        optimizer.zero_grad()
        loss, l_pde, l_bc, l_ic = total_loss(
            model, batch, weights,
            epsilon=causal_epsilon,
            n_bins=causal_n_bins,
        )

        if not torch.isfinite(loss):
            log.warning("Step %d: non-finite loss (%.3e) — skipping", step, loss.item())
            scheduler.step()
            continue

        loss.backward()

        grads_ok = all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        if not grads_ok:
            log.warning("Step %d: NaN/Inf gradient — skipping", step)
            optimizer.zero_grad()
            scheduler.step()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg["grad_clip_norm"])
        optimizer.step()
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

    # ── Final save ────────────────────────────────────────────────────────────
    final_err = validate(model, normalizer, cfg, device)
    history["val"].append(["final", final_err])
    _save(model, normalizer, cfg, "final", history, t_cfg["checkpoint_dir"])
    log.info("Training complete. Final validation L2 error: %.4f", final_err)
