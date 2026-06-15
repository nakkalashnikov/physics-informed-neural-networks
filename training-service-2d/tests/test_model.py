"""Tests for model.py — hard IC, output shapes, sigma curriculum, O(1) output."""

import torch
import yaml

from model import build_model

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

K = CFG["trajectory"]["k_sensors"]


def _inputs(n, t_value=None):
    branch = torch.rand(n, K + 3)
    coords = torch.rand(n, 3)
    if t_value is not None:
        coords[:, 2] = t_value
    return branch, coords


def test_hard_ic_exact():
    torch.manual_seed(0)
    model = build_model(CFG)
    branch, coords = _inputs(64, t_value=0.0)
    u = model(branch, coords)
    assert torch.allclose(u, torch.zeros_like(u), atol=1e-12)


def test_output_shape():
    model = build_model(CFG)
    branch, coords = _inputs(128)
    u = model(branch, coords)
    assert u.shape == (128, 1)


def test_output_finite_and_scaled():
    torch.manual_seed(1)
    model = build_model(CFG)
    branch, coords = _inputs(256)
    u = model(branch, coords)
    assert torch.isfinite(u).all()
    assert u.abs().max() < 100.0  # O(1)-ish at init, not exploding


def test_set_sigma():
    model = build_model(CFG)
    model.set_sigma(3.5)
    assert float(model.trunk.fourier.sigma) == 3.5


def test_param_count_reasonable():
    model = build_model(CFG)
    n = sum(p.numel() for p in model.parameters())
    assert 1e5 < n < 5e6, f"param count {n}"
