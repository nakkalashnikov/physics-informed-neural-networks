"""Tests for the validation suite — energy balance, invariants, residual map, ensemble,
defect error map, Rosenthal sanity."""

import numpy as np
import torch
import yaml

from analytic import fourier_field_u
from model import build_model
from nondim import PiGroups
from trajectory import sample_trajectory
from validation.ensemble import predict_ensemble
from validation.intrinsic import (
    check_invariants,
    energy_balance,
    error_map_from_defect,
    predict_field,
    residual_map,
)
from validation.rosenthal import thin_plate_rosenthal

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)
DEVICE = torch.device("cpu")


def _small_cfg():
    cfg = yaml.safe_load(open("config.yaml"))
    cfg["trunk"]["width"] = 32
    cfg["trunk"]["n_pirate_blocks"] = 1
    cfg["branch"]["hidden"] = [32]
    cfg["branch"]["out_dim"] = 32
    cfg["trunk"]["out_dim"] = 32
    return cfg


def test_energy_balance_on_analytic_is_zero():
    """The exact field conserves energy => residual ~ 0 (validates the metric itself)."""
    pi = PiGroups(Fo=5e-3, AR=0.15, w=0.1)
    traj = sample_trajectory(CFG, np.random.default_rng(0))
    x = np.linspace(0, 1, 80); y = np.linspace(0, 1, 60); t = np.array([0.3, 0.7, 1.0])
    u = fourier_field_u(x, y, t, traj, pi, CFG)
    res = energy_balance(u, x, y, t)
    assert res.max() < 5e-3


def test_invariants_on_analytic():
    pi = PiGroups(Fo=5e-3, AR=0.15, w=0.1)
    traj = sample_trajectory(CFG, np.random.default_rng(1))
    x = np.linspace(0, 1, 60); y = np.linspace(0, 1, 50); t = np.array([0.0, 0.5, 1.0])
    u = fourier_field_u(x, y, t, traj, pi, CFG)
    inv = check_invariants(u, t)
    assert inv["mono_energy"] and inv["ic"] and inv["nonneg"]


def test_residual_map_mechanics():
    cfg = _small_cfg()
    torch.manual_seed(0)
    model = build_model(cfg)
    pi = PiGroups(Fo=5e-3, AR=0.15, w=0.1)
    traj = sample_trajectory(cfg, np.random.default_rng(2))
    rm = residual_map(model, traj, pi, cfg, DEVICE, nx=20, ny=15, nt=8)
    assert set(rm) == {"mean", "max", "max_bc"}
    assert all(np.isfinite(v) for v in rm.values())


def test_predict_field_shape_and_hard_ic():
    cfg = _small_cfg()
    model = build_model(cfg)
    pi = PiGroups(Fo=5e-3, AR=0.15, w=0.1)
    traj = sample_trajectory(cfg, np.random.default_rng(3))
    x = np.linspace(0, 1, 10); y = np.linspace(0, 1, 8); t = np.array([0.0, 0.5])
    u = predict_field(model, traj, pi, cfg, x, y, t, DEVICE)
    assert u.shape == (2, 8, 10)
    assert np.allclose(u[0], 0.0, atol=1e-6)   # hard IC at t*=0


def test_ensemble_std():
    cfg = _small_cfg()
    models = [build_model(cfg) for _ in range(3)]
    for i, m in enumerate(models):
        torch.manual_seed(i)
        for p in m.parameters():
            with torch.no_grad():
                p.add_(0.01 * torch.randn_like(p))
    pi = PiGroups(Fo=5e-3, AR=0.15, w=0.1)
    traj = sample_trajectory(cfg, np.random.default_rng(4))
    x = np.linspace(0, 1, 10); y = np.linspace(0, 1, 8); t = np.array([0.5, 1.0])
    mean, std = predict_ensemble(models, traj, pi, cfg, x, y, t, DEVICE)
    assert mean.shape == std.shape == (2, 8, 10)
    assert std.min() >= 0.0 and std.max() > 0.0


def test_defect_error_map_runs():
    cfg = _small_cfg()
    cfg["fdm"] = {"nx": 41, "ny": 21, "nt": 60, "r_target": 0.4}
    model = build_model(cfg)
    pi = PiGroups(Fo=8e-3, AR=0.2, w=0.12)
    traj = sample_trajectory(cfg, np.random.default_rng(5))
    e = error_map_from_defect(model, traj, pi, cfg, DEVICE, np.array([0.5, 1.0]))
    assert e.shape == (2, 21, 41) and np.isfinite(e).all()


def test_rosenthal_physical():
    x = np.linspace(0.1, 0.9, 80); y = np.linspace(0.0, 0.3, 40)
    dT = thin_plate_rosenthal(x, y, t=0.0, P=100.0, k=20.0, h=0.05, alpha=5e-5, v=0.02, x0=0.5)
    assert np.isfinite(dT).all() and dT.min() >= 0.0
    # hotter behind the source (xi<0) than equally far ahead, on the centerline y~0
    j = np.argmin(np.abs(y - 0.0))
    i_behind = np.argmin(np.abs(x - 0.45))
    i_ahead = np.argmin(np.abs(x - 0.55))
    assert dT[j, i_behind] > dT[j, i_ahead]
