"""Tests for analytic.py — energy conservation (the correctness gate), mean-mode, IC, BC sign."""

import numpy as np
import pytest
import yaml

from analytic import fourier_field_u, fourier_labels_u
from nondim import PiGroups
from trajectory import sample_trajectory

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)


def _const_traj(x0):
    return lambda t: np.full_like(np.atleast_1d(np.asarray(t, float)), x0)


def test_energy_conservation_mean_is_t():
    """mean(u) over the domain must equal t* exactly (A_00 = t*) -> exact energy conservation.
    This is the load-bearing correctness check (independent of any FD solver)."""
    pi = PiGroups(Fo=5e-3, AR=0.15, w=0.08)
    traj = sample_trajectory(CFG, np.random.default_rng(0))
    x = np.linspace(0, 1, 80)
    y = np.linspace(0, 1, 60)
    for t in [0.1, 0.5, 1.0]:
        u = fourier_field_u(x, y, np.array([t]), traj, pi, CFG)[0]  # (y, x)
        mean_u = np.trapezoid(np.trapezoid(u, x, axis=1), y)        # domain area = 1
        assert mean_u == pytest.approx(t, rel=2e-3)


def test_initial_condition_zero():
    pi = PiGroups(Fo=1e-2, AR=0.1, w=0.05)
    traj = _const_traj(0.5)
    x = np.linspace(0, 1, 40)
    y = np.linspace(0, 1, 30)
    u0 = fourier_field_u(x, y, np.array([0.0]), traj, pi, CFG)[0]
    assert np.allclose(u0, 0.0, atol=1e-9)


def test_hotter_near_source_edge():
    """Heat enters at y_hat=0; that edge must be hotter than the insulated far edge y_hat=1."""
    pi = PiGroups(Fo=1e-3, AR=0.1, w=0.06)
    traj = _const_traj(0.5)
    x = np.linspace(0, 1, 60)
    y = np.array([0.0, 1.0])
    u = fourier_field_u(x, y, np.array([0.3]), traj, pi, CFG)[0]  # (2, x)
    # near the source location x*=0.5, bottom edge hotter than top edge
    i_mid = np.argmin(np.abs(x - 0.5))
    assert u[0, i_mid] > u[1, i_mid]


def test_nonnegative_up_to_gibbs():
    """Pure heating => u >= 0 physically. The truncated cosine series shows small Gibbs ringing
    near the sharp boundary flux at y_hat=0 (a flux BC converges slowly in an interior cosine
    basis). Tolerate ringing up to a few % of the peak; this is a spectral artifact, not a sign
    error (energy conservation, the real correctness gate, holds)."""
    pi = PiGroups(Fo=3e-3, AR=0.2, w=0.1)
    traj = sample_trajectory(CFG, np.random.default_rng(5))
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 40)
    u = fourier_field_u(x, y, np.array([0.7]), traj, pi, CFG)[0]
    assert u.min() > -0.06 * u.max()


def test_labels_match_field():
    """Scattered-label evaluator must agree with the field evaluator at the same points."""
    pi = PiGroups(Fo=5e-3, AR=0.15, w=0.08)
    traj = sample_trajectory(CFG, np.random.default_rng(2))
    x = np.linspace(0.05, 0.95, 5)
    y = np.linspace(0.05, 0.95, 4)
    t = 0.6
    field = fourier_field_u(x, y, np.array([t]), traj, pi, CFG, M=48, N=32)[0]  # (y,x)
    pts = np.array([[xi, yi, t] for yi in y for xi in x])
    lab = fourier_labels_u(pts, traj, pi, CFG, M=48, N=32, ntau=int(CFG["analytic"]["quad_nt"]))
    assert np.allclose(lab, field.reshape(-1), rtol=5e-2, atol=5e-3)
