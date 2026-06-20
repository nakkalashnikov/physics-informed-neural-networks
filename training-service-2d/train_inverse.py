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
    return torch.as_tensor(X), torch.as_tensor(d["tgt"], dtype=torch.float32)   # X, (x0,v)


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

    Xtr, Ytr = _load(cfg, "train"); Xtr, Ytr = Xtr.to(dev), Ytr.to(dev)
    Xva, Yva = _load(cfg, "val");   Xva, Yva = Xva.to(dev), Yva.to(dev)
    print(f"train {Xtr.shape[0]} | val {Xva.shape[0]} | branch_in {Xtr.shape[1]}")

    model = build_inverse_model(cfg).to(dev)
    tr = cfg["training"]
    total = int(args.steps or tr["steps"]); bs = int(tr["batch"])
    Q = int(cfg["inverse"]["n_query"])
    t_q = torch.linspace(0, 1, Q, device=dev).reshape(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]),
                           weight_decay=float(tr["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total, eta_min=float(tr["lr_min"]))

    margin = float(cfg["trajectory"]["x_margin"])
    rng_x = 1.0 - 2.0 * margin                 # x0 range; v range ~ 2*rng_x
    N = Xtr.shape[0]

    def evaluate():
        model.eval()
        with torch.no_grad():
            xb_pred = model(Xva, t_q)                                   # (Nval, Q)
            xb_true = Yva[:, 0:1] + Yva[:, 1:2] * t_q.squeeze(1)[None, :]
            relL2 = (xb_pred - xb_true).pow(2).mean(1).sqrt() / (xb_true.pow(2).mean(1).sqrt() + 1e-9)
            rec = model.recover(Xva)
            x0e = (rec[:, 0] - Yva[:, 0]).abs() / rng_x
            ve = (rec[:, 1] - Yva[:, 1]).abs() / (2 * rng_x)
        model.train()
        return relL2.median().item(), x0e.median().item(), ve.median().item()

    for step in range(total):
        idx = torch.randint(0, N, (bs,), device=dev)
        xb_pred = model(Xtr[idx], t_q)                                  # (bs, Q)
        xb_true = Ytr[idx, 0:1] + Ytr[idx, 1:2] * t_q.squeeze(1)[None, :]
        loss = (xb_pred - xb_true).pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()

        if step % int(tr["val_every"]) == 0 or step == total - 1:
            rl2, x0e, ve = evaluate()
            print(f"step {step:6d} | loss {loss.item():.2e} | "
                  f"traj relL2 {rl2:5.1%} │ x0 err {x0e:5.1%}  v err {ve:5.1%} │ "
                  f"lr {sched.get_last_lr()[0]:.1e}")

    torch.save({"model": model.state_dict(), "cfg": cfg}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
