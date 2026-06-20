"""
DIAGNOSTIC: can the model memorise ONE field?  (capacity vs spectral-bias test)

We pick a single (pi, trajectory) case, build its EXACT Fourier field as the target,
and train the network to fit just that one field — pure supervised MSE, no physics.
A single case needs no generalisation, so:

  * loss → ~0    → the architecture CAN represent the field. Capacity is fine; the
                   failure in full training is generalisation / loss-balance / optimisation.
  * loss plateaus high → the architecture CANNOT represent this field at all. That is a
                   REPRESENTATIONAL limit (spectral bias / RFF frequency too low), and no
                   amount of training data or epochs fixes it.

Try it twice and compare:
    python overfit_one.py --sigma-end 3.5    # the configured value
    python overfit_one.py --sigma-end 15     # crank the trunk frequencies up
If 3.5 plateaus but 15 fits → the bottleneck is RFF frequency, NOT layers/neurons.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from model import build_model
from nondim import PiGroups, normalize
from trajectory import sample_trajectory, sample_at_nodes
from analytic import fourier_field_u
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--sigma-end", type=float, default=3.5, help="trunk RFF final bandwidth")
    ap.add_argument("--w", type=float, default=0.05, help="source width S/l (smaller = sharper)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = yaml.safe_load(open("config.yaml"))

    # ── 1. Fix ONE problem instance ──────────────────────────────────────────
    # One pi-group set (choose w on the CLI to control source sharpness) and one
    # fixed random trajectory. Everything below is this single case, forever.
    pi = PiGroups(Fo=5e-3, AR=0.15, w=args.w)
    traj = sample_trajectory(cfg, np.random.default_rng(0))

    # ── 2. Build the EXACT target field on a grid ────────────────────────────
    nx, ny = 80, 60
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    t = np.array([0.25, 0.5, 0.75, 1.0])
    u_true = fourier_field_u(x, y, t, traj, pi, cfg)          # (nt, ny, nx) exact

    # Flatten to a point cloud of (x*, y_hat, t*) -> u targets
    X, Y, T = np.meshgrid(x, y, t, indexing="xy")            # each (ny, nx, nt)
    # reorder u_true (nt,ny,nx) to match meshgrid (ny,nx,nt)
    u_flat = np.transpose(u_true, (1, 2, 0)).reshape(-1)     # (ny*nx*nt,)
    coords = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1).astype(np.float32)

    coords_t = torch.as_tensor(coords, device=device)
    target_t = torch.as_tensor(u_flat.astype(np.float32), device=device).unsqueeze(1)

    # ── 3. The branch input is CONSTANT (one fixed case) ─────────────────────
    # Same vector for every point: [trajectory samples ++ normalised pi-groups].
    k = int(cfg["trajectory"]["k_sensors"])
    bvec = np.concatenate([sample_at_nodes(traj, k), np.array(normalize(pi, cfg))]).astype(np.float32)
    branch_t = torch.as_tensor(bvec, device=device).unsqueeze(0).expand(coords_t.shape[0], -1)

    # ── 4. Model + optimiser ─────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    sigma_start = float(cfg["trunk"]["rff_sigma_start"])
    sigma_end = args.sigma_end
    tgt_norm = float(torch.linalg.norm(target_t))

    print(f"Overfit ONE case | {n_params:,} params | w={args.w} (sigma_star={pi.sigma_star(6.0):.4f}) "
          f"| sigma {sigma_start}->{sigma_end} | {args.steps} steps\n")

    # ── 5. Train (pure supervised MSE on the single field) ───────────────────
    for step in range(args.steps):
        # ramp the trunk RFF bandwidth, exactly like real training does
        frac = step / max(args.steps - 1, 1)
        model.set_sigma(sigma_start + (sigma_end - sigma_start) * frac)

        pred = model(branch_t, coords_t)
        loss = ((pred - target_t) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 200 == 0 or step == args.steps - 1:
            # relative L2 = ||pred - target|| / ||target||  — same metric as validation
            rel = float(torch.linalg.norm(pred - target_t)) / (tgt_norm + 1e-12)
            print(f"step {step:5d} | rel-L2 {rel:6.1%} | sigma {model.trunk.fourier.sigma.item():.2f}")

    rel = float(torch.linalg.norm(model(branch_t, coords_t) - target_t)) / (tgt_norm + 1e-12)
    print(f"\nFINAL rel-L2 = {rel:.1%}")
    print("→ near 0%  : architecture CAN represent the field (capacity OK)")
    print("→ stays high: REPRESENTATIONAL limit (spectral bias / RFF too low)")


if __name__ == "__main__":
    main()
