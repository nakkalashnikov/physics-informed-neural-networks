"""CLI entry point for training the PI-DeepONet."""

from __future__ import annotations

import argparse
import json

import yaml

from trainer import select_device, train


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PI-DeepONet 2D moving heat source")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (auto if omitted)")
    ap.add_argument("--steps", type=int, default=None, help="override total_steps")
    ap.add_argument("--out", default="checkpoint.pt")
    ap.add_argument("--no-prefetch", action="store_true")
    ap.add_argument("--cache", default=None, help="path to precomputed labels.npz (skips online Fourier)")
    ap.add_argument("--profile", action="store_true", help="print per-stage ms/step breakdown (get/fwd/bwd)")
    ap.add_argument("--n-traj", type=int, default=None, help="override batch.n_traj_per_batch")
    ap.add_argument("--n-int", type=int, default=None, help="override batch.n_interior")
    ap.add_argument("--validate-every", type=int, default=0,
                    help="run L2 validation every N steps (0=disabled)")
    ap.add_argument("--val-n", type=int, default=16,
                    help="number of cases for mid-training validation (default 16)")
    ap.add_argument("--lambda-pde", type=float, default=None,
                    help="override loss.lambda_pde (e.g. 0.0 disables PDE loss for diagnostics)")
    ap.add_argument("--lambda-data", type=float, default=None,
                    help="override loss.lambda_data")
    ap.add_argument("--lambda-bc", type=float, default=None,
                    help="override loss.lambda_bc (insulation edges)")
    ap.add_argument("--lambda-flux", type=float, default=None,
                    help="override loss.lambda_flux (flux BC at yhat=0; 0 disables)")
    ap.add_argument("--residual-clip-gamma", type=float, default=None,
                    help="override loss.residual_clip_gamma (per-point residual cap; 0 disables)")
    ap.add_argument("--sigma-end", type=float, default=None,
                    help="override trunk.rff_sigma_end (trunk RFF bandwidth — sharpness ceiling)")
    ap.add_argument("--sigma-start", type=float, default=None,
                    help="override trunk.rff_sigma_start")
    ap.add_argument("--sigma-bands", type=str, default=None,
                    help='override trunk.rff_sigma_bands, e.g. "[2,6,12]"')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.lambda_pde is not None:
        cfg["loss"]["lambda_pde"] = args.lambda_pde
    if args.lambda_data is not None:
        cfg["loss"]["lambda_data"] = args.lambda_data
    if args.lambda_bc is not None:
        cfg["loss"]["lambda_bc"] = args.lambda_bc
    if args.lambda_flux is not None:
        cfg["loss"]["lambda_flux"] = args.lambda_flux
    if args.residual_clip_gamma is not None:
        cfg["loss"]["residual_clip_gamma"] = args.residual_clip_gamma
    if args.sigma_end is not None:
        cfg["trunk"]["rff_sigma_end"] = args.sigma_end
    if args.sigma_start is not None:
        cfg["trunk"]["rff_sigma_start"] = args.sigma_start
    if args.sigma_bands is not None:
        cfg["trunk"]["rff_sigma_bands"] = [float(b) for b in json.loads(args.sigma_bands)]

    device = select_device(args.device)
    print(f"device: {device}")
    train(cfg, device, total_steps=args.steps,
          use_prefetch=False if args.no_prefetch else None,
          checkpoint_path=args.out, cache_path=args.cache, profile=args.profile,
          n_traj=args.n_traj, n_int=args.n_int,
          validate_every=args.validate_every, val_n=args.val_n)


if __name__ == "__main__":
    main()
