"""
Entry point.  Usage:
    python train.py [--config config.yaml]
"""

import argparse
import logging
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _get_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg_device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PINN for 1D heat equation")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint, e.g. checkpoints/model_55000.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = _get_device(cfg.get("device", "auto"))
    log.info("Device: %s", device)

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")  # enables TF32 on Ampere+

    from trainer import train
    train(cfg, device, resume=args.resume)


if __name__ == "__main__":
    main()
