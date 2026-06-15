"""Tests for validation/fdm.py — the INDEPENDENT cross-check of the Fourier formula.

If these pass, the load-bearing analytic solution is corroborated by a completely different
numerical method. Tolerances are grid-limited (~few %), not formula-limited.
"""

import numpy as np
import pytest
import yaml

from analytic import fourier_field_u
from nondim import PiGroups
from trajectory import sample_trajectory
from validation.fdm import crank_nicolson_2d

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)


def _l2(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_fdm_matches_analytic_constant_source():
    pi = PiGroups(Fo=5e-3, AR=0.15, w=0.10)
    traj = lambda t: np.full_like(np.atleast_1d(np.asarray(t, float)), 0.5)
    t_snaps = np.array([0.3, 0.7, 1.0])
    x, y, u_fdm = crank_nicolson_2d(traj, pi, CFG, t_snaps)
    u_an = fourier_field_u(x, y, t_snaps, traj, pi, CFG)
    err = _l2(u_fdm, u_an)
    assert err < 0.03, f"FDM vs analytic L2={err:.4f}"


def test_fdm_matches_analytic_moving_source():
    pi = PiGroups(Fo=8e-3, AR=0.2, w=0.12)
    traj = sample_trajectory(CFG, np.random.default_rng(11))
    t_snaps = np.array([0.5, 1.0])
    x, y, u_fdm = crank_nicolson_2d(traj, pi, CFG, t_snaps)
    u_an = fourier_field_u(x, y, t_snaps, traj, pi, CFG)
    err = _l2(u_fdm, u_an)
    assert err < 0.05, f"FDM vs analytic (moving) L2={err:.4f}"


def test_fdm_energy_conservation():
    """Independent of analytic: FDM field must conserve energy (mean u == t*)."""
    pi = PiGroups(Fo=5e-3, AR=0.15, w=0.10)
    traj = sample_trajectory(CFG, np.random.default_rng(3))
    t_snaps = np.array([0.4, 1.0])
    x, y, u = crank_nicolson_2d(traj, pi, CFG, t_snaps)
    for i, t in enumerate(t_snaps):
        mean_u = np.trapezoid(np.trapezoid(u[i], x, axis=1), y)
        assert mean_u == pytest.approx(t, rel=0.03)
