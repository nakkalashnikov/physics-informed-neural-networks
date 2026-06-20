"""
Validate a trained PI-DeepONet checkpoint.

Usage (from training-service-2d/):
    python validate_checkpoint.py --ckpt checkpoint.pt --device cpu [--n 64]

Runs two independent checks:
  1. acceptance_card — model vs EXACT Fourier across n random (pi, trajectory) cases:
     relative-L2 median/p90/max + intrinsic energy balance → green/yellow/red verdict.
  2. FDM cross-check — model vs an ADI Crank-Nicolson solver (a completely different
     numerical route) on a few cases — the strongest independent confirmation.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from model import build_model
from nondim import sample_pi_groups
from trajectory import sample_trajectory
from analytic import fourier_field_u
from validation.report import acceptance_card
from validation.intrinsic import predict_field
from validation.fdm import crank_nicolson_2d


def _l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoint.pt")
    ap.add_argument("--device", default="cpu", help="cpu|mps|cuda (cpu recommended on Mac)")
    ap.add_argument("--n", type=int, default=None, help="override pi_table_n_cases")
    ap.add_argument("--fdm-cases", type=int, default=3, help="independent FDM cross-checks")
    ap.add_argument("--plot", action="store_true", help="save a model-vs-Fourier field heatmap")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["cfg"]

    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.ckpt}  | step={ck['step']}  | {n_params:,} params  | device={device}\n")

    # ── 1. Acceptance card (model vs exact Fourier) ──────────────────────────
    print("── Acceptance card (model vs exact Fourier) ─────────────────────────")
    card = acceptance_card(model, cfg, device, n_cases=args.n)
    th = cfg["validation"]["thresholds"]
    print(f"  L2 p90      : {card['l2_p90']:.4f}   (green<{th['l2_green']}  yellow<{th['l2_yellow']})  → {card['l2_tier'].upper()}")
    print(f"  L2 max      : {card['l2_max']:.4f}")
    print(f"  energy max  : {card['energy_max']:.4f}   (green<{th['energy_green']}  yellow<{th['energy_yellow']})  → {card['energy_tier'].upper()}")
    print(f"  VERDICT     : {card['verdict'].upper()}\n")

    # ── 2. Independent FDM cross-check (model vs ADI Crank-Nicolson) ──────────
    print(f"── Independent FDM cross-check (model vs ADI solver, {args.fdm_cases} cases) ──")
    rng = np.random.default_rng(2024)
    t_snaps = np.array([0.5, 1.0])
    fdm_errs = []
    for i in range(args.fdm_cases):
        pi = sample_pi_groups(cfg, rng)
        traj = sample_trajectory(cfg, rng)
        x, y, u_fdm = crank_nicolson_2d(traj, pi, cfg, t_snaps)
        u_pred = predict_field(model, traj, pi, cfg, x, y, t_snaps, device)
        e_fdm = _l2(u_pred, u_fdm)
        # also model vs Fourier on the same grid for context
        u_an = fourier_field_u(x, y, t_snaps, traj, pi, cfg)
        e_an = _l2(u_pred, u_an)
        fdm_errs.append(e_fdm)
        print(f"  case {i}: model-vs-FDM L2={e_fdm:.4f}   model-vs-Fourier L2={e_an:.4f}")
    print(f"  FDM cross-check median L2: {np.median(fdm_errs):.4f}\n")

    # ── 3. Optional field visualisation ──────────────────────────────────────
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(7)
        pi = sample_pi_groups(cfg, rng)
        traj = sample_trajectory(cfg, rng)
        nx, ny = 120, 80
        x = np.linspace(0, 1, nx); y = np.linspace(0, 1, ny); t = np.array([1.0])
        u_pred = predict_field(model, traj, pi, cfg, x, y, t, device)[0]   # (ny, nx)
        u_true = fourier_field_u(x, y, t, traj, pi, cfg)[0]
        err = _l2(u_pred, u_true)

        vmax = float(max(abs(u_true).max(), abs(u_pred).max()))
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for a, fld, ttl in [
            (ax[0], u_true, "Fourier (exact)"),
            (ax[1], u_pred, "PI-DeepONet"),
            (ax[2], np.abs(u_pred - u_true), "|difference|"),
        ]:
            im = a.imshow(fld, origin="lower", extent=[0, 1, 0, 1], aspect="auto",
                          vmin=0 if "diff" in ttl else -vmax, vmax=vmax, cmap="inferno")
            a.set_title(ttl); a.set_xlabel("x̂"); a.set_ylabel("ŷ")
            fig.colorbar(im, ax=a, fraction=0.046)
        fig.suptitle(f"t=1.0  |  relative L2 = {err:.1%}", y=1.02)
        plt.tight_layout()
        out = "validation_field.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"Saved field comparison → {out}  (L2={err:.1%})\n")

    print("Done.")


if __name__ == "__main__":
    main()
