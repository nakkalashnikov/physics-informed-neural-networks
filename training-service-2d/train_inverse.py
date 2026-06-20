"""Train the inverse PI-DeepONet — mode A (supervised on the source trajectory).

Loss = ‖xb_pred(t) − xb_true(t)‖ over query times (xb_true = x0 + v·t). No forward model needed, so
it trains in minutes (even on Mac/CPU). Reports trajectory rel-L2 + per-param (x0, v) recovery error
on a held-out set. Mode B (physics-consistency via analytic_torch) is a separate add-on.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from model_inverse import build_inverse_model


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load(cfg: dict, split: str):
    d = np.load(cfg["data"]["cache"].replace(".npz", f"_{split}.npz"))
    X = np.concatenate([d["meas"], d["mat"]], axis=1).astype(np.float32)   # [measurements ; material]
    return (torch.as_tensor(X), torch.as_tensor(d["traj_q"], dtype=torch.float32),   # X, xb(t_q) target
            torch.as_tensor(d["t_q"], dtype=torch.float32).reshape(-1, 1))           # query grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_inverse.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default="ckpt_inverse.pt")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dev = _device(args.device)
    print("device:", dev)

    Xtr, Ttr, t_q = _load(cfg, "train"); Xtr, Ttr, t_q = Xtr.to(dev), Ttr.to(dev), t_q.to(dev)
    Xva, Tva, _ = _load(cfg, "val");     Xva, Tva = Xva.to(dev), Tva.to(dev)
    arbitrary = bool(cfg["trajectory"].get("arbitrary", False))
    print(f"train {Xtr.shape[0]} | val {Xva.shape[0]} | branch_in {Xtr.shape[1]} | "
          f"Q={t_q.shape[0]} | {'arbitrary' if arbitrary else 'linear'} trajectories")

    model = build_inverse_model(cfg).to(dev)
    tr = cfg["training"]
    total = int(args.steps or tr["steps"]); bs = int(tr["batch"])
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]),
                           weight_decay=float(tr["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total, eta_min=float(tr["lr_min"]))

    margin = float(cfg["trajectory"]["x_margin"]); rng_x = 1.0 - 2.0 * margin
    N = Xtr.shape[0]

    def evaluate():
        model.eval()
        with torch.no_grad():
            xb_pred = model(Xva, t_q)                                   # (Nval, Q)
            relL2 = ((xb_pred - Tva).pow(2).mean(1).sqrt()
                     / (Tva.pow(2).mean(1).sqrt() + 1e-9)).median().item()
            # readout: x0 = xb(0), v = (xb(1)-xb(0)) — informative for the linear family
            x0e = ((xb_pred[:, 0] - Tva[:, 0]).abs() / rng_x).median().item()
            ve = (((xb_pred[:, -1] - xb_pred[:, 0]) - (Tva[:, -1] - Tva[:, 0])).abs()
                  / (2 * rng_x)).median().item()
        model.train()
        return relL2, x0e, ve

    for step in range(total):
        idx = torch.randint(0, N, (bs,), device=dev)
        loss = (model(Xtr[idx], t_q) - Ttr[idx]).pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()

        if step % int(tr["val_every"]) == 0 or step == total - 1:
            rl2, x0e, ve = evaluate()
            print(f"step {step:6d} | loss {loss.item():.2e} | "
                  f"traj relL2 {rl2:5.1%} │ x0 err {x0e:5.1%}  v(end) err {ve:5.1%} │ "
                  f"lr {sched.get_last_lr()[0]:.1e}")

    torch.save({"model": model.state_dict(), "cfg": cfg}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
