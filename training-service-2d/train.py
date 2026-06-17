"""CLI entry point for training the PI-DeepONet."""

from __future__ import annotations

import argparse

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
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = select_device(args.device)
    print(f"device: {device}")
    train(cfg, device, total_steps=args.steps,
          use_prefetch=False if args.no_prefetch else None,
          checkpoint_path=args.out, cache_path=args.cache, profile=args.profile)


if __name__ == "__main__":
    main()
