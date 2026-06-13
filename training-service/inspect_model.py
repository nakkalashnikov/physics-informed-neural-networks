"""
Inspect a trained PINN checkpoint — compare predictions vs analytical solution.

Usage (from training-service/ folder):
    .venv/bin/python inspect_model.py
    .venv/bin/python inspect_model.py --ckpt checkpoints/model_final.pt
    .venv/bin/python inspect_model.py --ckpt checkpoints/model_35000.pt --plot
"""

import argparse
import sys
import os

import numpy as np
import torch
import yaml

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",  default="checkpoints/model_final.pt",
                    help="Path to checkpoint .pt file")
parser.add_argument("--plot",  action="store_true",
                    help="Show matplotlib plot (needs display)")
parser.add_argument("--n",     type=int, default=6,
                    help="Number of random parameter sets to test")
args = parser.parse_args()

if not os.path.exists(args.ckpt):
    print(f"ERROR: checkpoint not found: {args.ckpt}")
    print("Available checkpoints:")
    for f in sorted(os.listdir("checkpoints")):
        print(f"  checkpoints/{f}")
    sys.exit(1)

# ── Load checkpoint ───────────────────────────────────────────────────────────
print(f"\nLoading: {args.ckpt}")
ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

model_cfg    = ckpt["model_cfg"]
physics_cfg  = ckpt["physics_cfg"]
norm_bounds  = ckpt["normalizer_bounds"]

# Rebuild config dict that sampler/model expect
cfg = {
    "model":   model_cfg,
    "physics": physics_cfg,
}

# ── Rebuild model ─────────────────────────────────────────────────────────────
from model import build_model
from sampler import Normalizer, sample_params, compute_pi_groups
from physics import analytical_delta_T

model = build_model(cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded  ({n_params:,} parameters)")

# ── Use the Normalizer that matches the checkpoint's physics bounds ───────────
normalizer = Normalizer(cfg)

device = torch.device("cpu")

# ── Sample random parameter sets ─────────────────────────────────────────────
torch.manual_seed(0)
raw = sample_params(args.n, cfg, device)
pi  = compute_pi_groups(raw)

print(f"\n{'─'*70}")
print(f"{'Set':>3}  {'α':>10}  {'l':>5}  {'i_eff':>9}  {'v':>6}  "
      f"{'t_tot':>6}  {'RelErr':>8}")
print(f"{'─'*70}")

all_errors = []
results = []

x_pts = np.linspace(0.0, 1.0, 200)   # normalised x grid

with torch.no_grad():
    for k in range(args.n):
        alpha  = raw["alpha"][k].item()
        rho_c  = raw["rho_c"][k].item()
        l      = raw["l"][k].item()
        intens = raw["intensity"][k].item()
        x0     = raw["x0"][k].item()
        v      = raw["v"][k].item()
        t_tot  = raw["t_total"][k].item()
        i_eff  = raw["i_eff"][k].item()

        # Evaluate at 70% of simulation time
        t_q    = t_tot * 0.7
        n      = len(x_pts)

        x_norm = torch.tensor(x_pts, dtype=torch.float32)
        t_norm = torch.full((n,), t_q / t_tot)
        coords = torch.stack([x_norm, t_norm], dim=1)

        fo_n    = normalizer.norm_log(torch.full((n,), pi["Fo"][k].item()),      "Fo")
        x0_n    = normalizer.norm(    torch.full((n,), pi["x0_norm"][k].item()), "x0_frac")
        beta_n  = normalizer.norm(    torch.full((n,), pi["beta"][k].item()),    "beta")
        pi_norm = torch.stack([fo_n, x0_n, beta_n], dim=1)

        T_c     = pi["T_c"][k].item()
        dT_pred = model(coords, pi_norm).squeeze().numpy() * T_c

        x_phys  = x_pts * l
        dT_ref  = analytical_delta_T(x_phys, t_q, alpha, rho_c, l, intens, x0, v)

        rel_err = np.linalg.norm(dT_pred - dT_ref) / (np.linalg.norm(dT_ref) + 1e-8)
        all_errors.append(rel_err)
        results.append((x_pts * l, dT_pred, dT_ref, rel_err, k))

        print(f"{k:>3}  {alpha:>10.2e}  {l:>5.2f}  {i_eff:>9.4f}  {v:>6.4f}  "
              f"{t_tot:>6.1f}  {rel_err:>8.2%}")

print(f"{'─'*70}")
print(f"Mean relative L2 error: {np.mean(all_errors):.2%}")
print(f"Best:  {min(all_errors):.2%}   Worst: {max(all_errors):.2%}")

# ── Extra: range check (is model outputting all zeros?) ──────────────────────
print(f"\n--- Output range check ---")
for x_phys, dT_pred, dT_ref, err, k in results:
    pred_range = dT_pred.max() - dT_pred.min()
    ref_range  = dT_ref.max()  - dT_ref.min()
    print(f"Set {k}: PINN range={pred_range:+.3f} K   Analytical range={ref_range:+.3f} K   "
          f"{'⚠ MODEL OUTPUT NEAR ZERO' if pred_range < 0.01 * ref_range else 'OK'}")

# ── Optional plot ─────────────────────────────────────────────────────────────
if args.plot:
    try:
        import matplotlib.pyplot as plt

        n_plot = min(args.n, 6)
        cols   = 3
        rows   = (n_plot + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
        axes = np.array(axes).flatten()

        for idx, (x_phys, dT_pred, dT_ref, err, k) in enumerate(results[:n_plot]):
            ax = axes[idx]
            ax.plot(x_phys, dT_ref,  "b-",  lw=2,   label="Analytical")
            ax.plot(x_phys, dT_pred, "r--", lw=1.5, label="PINN")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("ΔT [K]")
            ax.set_title(f"Set {k}  —  err={err:.1%}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for ax in axes[n_plot:]:
            ax.set_visible(False)

        fig.suptitle(f"{args.ckpt}  |  mean err={np.mean(all_errors):.2%}", y=1.01)
        plt.tight_layout()
        plt.savefig("inspect_output.png", dpi=120, bbox_inches="tight")
        print("\nPlot saved → inspect_output.png")
        plt.show()

    except ImportError:
        print("matplotlib not installed — skipping plot")
