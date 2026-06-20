"""Validate the inverse PI-DeepONet: noise robustness (real thermocouples are noisy) and the
recovered-vs-true scatter. Clean accuracy is ~1%, but the honest result is how it degrades under
measurement noise — that's the report figure.

    python validate_inverse.py --ckpt models/ckpt_inverse.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from model_inverse import build_inverse_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/ckpt_inverse.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--noise", default="0,0.01,0.02,0.05,0.10,0.20",
                    help="comma list of noise-to-signal ratios to sweep")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    model = build_inverse_model(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    d = np.load(cfg["data"]["cache"].replace(".npz", "_val.npz"))
    meas = torch.as_tensor(d["meas"], dtype=torch.float32, device=dev)     # (N, K*M)
    mat = torch.as_tensor(d["mat"], dtype=torch.float32, device=dev)
    Y = torch.as_tensor(d["tgt"], dtype=torch.float32, device=dev)         # (N, x0, v)
    margin = float(cfg["trajectory"]["x_margin"]); rng_x = 1.0 - 2.0 * margin
    rms = meas.pow(2).mean(1, keepdim=True).sqrt()                         # per-sample signal scale
    g = torch.Generator(device=dev).manual_seed(0)

    print(f"val {meas.shape[0]} cases | K={cfg['inverse']['n_sensors']} sensors\n")
    print(f"{'noise':>7} │ {'x0 err':>8} {'v err':>8} {'traj relL2':>11}   (median)")
    print("─" * 44)
    for eps in [float(x) for x in args.noise.split(",")]:
        noisy = meas + eps * rms * torch.randn(meas.shape, generator=g, device=dev)
        X = torch.cat([noisy, mat], dim=1)
        with torch.no_grad():
            rec = model.recover(X)
            Q = int(cfg["inverse"]["n_query"]); t_q = torch.linspace(0, 1, Q, device=dev).reshape(-1, 1)
            xb_pred = model(X, t_q); xb_true = Y[:, 0:1] + Y[:, 1:2] * t_q.squeeze(1)[None, :]
            relL2 = ((xb_pred - xb_true).pow(2).mean(1).sqrt()
                     / (xb_true.pow(2).mean(1).sqrt() + 1e-9)).median().item()
        x0e = ((rec[:, 0] - Y[:, 0]).abs() / rng_x).median().item()
        ve = ((rec[:, 1] - Y[:, 1]).abs() / (2 * rng_x)).median().item()
        print(f"{eps:6.0%} │ {x0e:7.1%} {ve:7.1%} {relL2:10.1%}")


if __name__ == "__main__":
    main()
