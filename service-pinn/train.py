"""Train a single-instance PINN for the 2D moving heat source (linear constant-velocity).

Loss = lambda_data * data_anchors + lambda_pde * PDE_residual + lambda_bc * insulation_BC, all
RELATIVE (divided by the instance's label RMS). The PDE is IN the loss -> genuine physics-informed.
Hard IC via ×t*. Reuses core (multi-scale trunk, physics residuals, analytic reference).

    python train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import yaml

from core.analytic import fourier_field_u, fourier_labels_u
from core.nondim import PiGroups
from core.physics import bc_insulation_residual, pde_residual
from model import build_pinn


def _device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sigma_at(step, total, t):
    a, b = float(t["rff_sigma_start"]), float(t["rff_sigma_end"])
    ramp = max(int(total * float(t.get("rff_sigma_ramp_frac", 0.5))), 1)
    return a + (b - a) * min(step / ramp, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default="checkpoints/ckpt_pinn.pt")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dev = _device(args.device)
    print("device:", dev)

    inst = cfg["instance"]
    Fo, AR, w = float(inst["Fo"]), float(inst["AR"]), float(inst["w"])
    x0, v = float(inst["x0"]), float(inst["v"])
    pi = PiGroups(Fo=Fo, AR=AR, w=w)
    traj = lambda t: x0 + v * np.atleast_1d(np.asarray(t, float))
    sf = float(cfg["physics"]["sigma_factor"])
    Q_star = pi.Q_star
    print(f"instance: Fo={Fo:g} AR={AR} w={w} | x0={x0} v={v} | Q*={Q_star:.3f}")

    # --- precompute anchors (data) and the validation field, both ONCE (fixed instance) ---
    rng = np.random.default_rng(int(cfg["training"]["seed"]))
    na = int(cfg["batch"]["n_anchor"])
    a_coords = rng.random((na, 3))
    a_u = fourier_labels_u(a_coords, traj, pi, cfg).reshape(-1, 1)
    scale = max(float(np.sqrt(np.mean(a_u ** 2))), 1e-3)
    A = torch.tensor(a_coords, dtype=torch.float32, device=dev)
    AU = torch.tensor(a_u, dtype=torch.float32, device=dev)

    gx = np.linspace(0, 1, 60); gy = np.linspace(0, 1, 40); gt = np.array([0.25, 0.5, 0.75, 1.0])
    Ufield = fourier_field_u(gx, gy, gt, traj, pi, cfg)                     # (t,y,x)
    GX, GY, GT = np.meshgrid(gx, gy, gt, indexing="xy")
    Vc = torch.tensor(np.stack([GX.ravel(), GY.ravel(), GT.ravel()], 1), dtype=torch.float32, device=dev)
    Vu = torch.tensor(np.transpose(Ufield, (1, 2, 0)).reshape(-1, 1), dtype=torch.float32, device=dev)

    model = build_pinn(cfg).to(dev)
    lam, tr, bt = cfg["loss"], cfg["training"], cfg["batch"]
    total = int(args.steps or tr["total_steps"])
    n_int, n_ins, n_data = int(bt["n_interior"]), int(bt["n_bc_ins"]), int(bt["n_data"])
    src_frac = float(bt.get("source_frac", 0.5)); band = float(bt.get("source_band_sigmas", 3.0))
    sig_star = w / sf
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]))
    warm = int(tr.get("warmup_steps", 2000))

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / max(warm, 1)
        p = (s - warm) / max(total - warm, 1)
        return float(tr["lr_min"]) / float(tr["lr"]) + (1 - float(tr["lr_min"]) / float(tr["lr"])) * \
            0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    clip = float(tr["grad_clip_norm"])

    def _t(a, grad=False):
        x = torch.as_tensor(a, dtype=torch.float32, device=dev)
        return x.requires_grad_(True) if grad else x

    t0 = time.time()
    for step in range(total):
        model.set_sigma(_sigma_at(step, total, cfg["trunk"]))

        # interior collocation, source-concentrated in x
        it = rng.random(n_int); ix = rng.random(n_int)
        ns = int(round(src_frac * n_int))
        ix[:ns] = np.clip(traj(it[:ns]) + rng.normal(0, band * sig_star, ns), 0, 1)
        ic = np.stack([ix, rng.random(n_int), it], 1)
        Fo_t = _t(np.full((n_int, 1), Fo)); AR_t = _t(np.full((n_int, 1), AR))
        R = pde_residual(model, None, _t(ic, grad=True), Fo_t, AR_t)
        L_pde = ((R / scale) ** 2).mean()

        # insulation edges (x=0, x=1, ŷ=1)
        xe = rng.random((n_ins, 3)); xe[: n_ins // 2, 0] = 0.0; xe[n_ins // 2:, 0] = 1.0
        ye = rng.random((n_ins, 3)); ye[:, 1] = 1.0
        L_bc = (((bc_insulation_residual(model, None, _t(xe, grad=True), axis=0)) / scale) ** 2).mean() \
            + (((bc_insulation_residual(model, None, _t(ye, grad=True), axis=1)) / scale) ** 2).mean()

        # data anchors
        sel = torch.randint(0, na, (n_data,), device=dev)
        L_data = (((model(None, A[sel]) - AU[sel]) / scale) ** 2).mean()

        loss = float(lam["lambda_data"]) * L_data + float(lam["lambda_pde"]) * L_pde \
            + float(lam["lambda_bc"]) * L_bc
        opt.zero_grad(set_to_none=True); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if torch.isfinite(gnorm):
            opt.step()
        sched.step()

        if step % int(tr["validate_every"]) == 0 or step == total - 1:
            model.eval()
            with torch.no_grad():
                up = model(None, Vc)
                rel = (float(torch.linalg.norm(up - Vu)) / (float(torch.linalg.norm(Vu)) + 1e-9))
            model.train()
            print(f"step {step:6d} ({100*(step+1)//total:3d}%) | rel-L2 {rel:6.1%} | "
                  f"data {100*L_data.item()**.5:5.1f}%  pde {100*L_pde.item()**.5:6.1f}%  "
                  f"bc {100*L_bc.item()**.5:5.1f}% | {(time.time()-t0)/60:.1f}m")

    torch.save({"model": model.state_dict(), "cfg": cfg}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
