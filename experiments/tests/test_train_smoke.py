"""Smoke test: the full hybrid-loss pipeline can LEARN (overfit one fixed batch)."""

import numpy as np
import torch
import yaml

from data import build_batch
from model import build_model
from trainer import compute_losses

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)


def test_overfit_single_batch():
    cfg = dict(CFG)
    cfg["batch"] = {"n_traj_per_batch": 2, "n_interior": 256, "n_bc0": 64,
                    "n_bc_ins": 64, "n_data": 256}
    device = torch.device("cpu")
    torch.manual_seed(0)
    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    batch = build_batch(cfg, np.random.default_rng(0))
    sf = cfg["physics"]["sigma_factor"]

    first = None
    for step in range(250):
        parts = compute_losses(model, batch, device, sf)
        loss = 10 * parts["data"] + parts["pde"] + 10 * parts["bc"]
        if step == 0:
            first = parts["data"].item()
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    last = parts["data"].item()
    assert last < 0.2 * first, f"data loss did not collapse: {first:.3e} -> {last:.3e}"
