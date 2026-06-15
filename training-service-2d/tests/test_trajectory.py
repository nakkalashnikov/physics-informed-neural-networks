"""Tests for trajectory.py — bounds, continuity, speed cap, reproducibility."""

import numpy as np

from trajectory import sample_at_nodes, sample_trajectory, spline_interp

CFG = {
    "trajectory": {
        "k_sensors": 101, "n_fourier_modes": 5, "spectrum_decay": 2.0,
        "sigma0": 1.0, "x_margin": 0.05, "speed_max_star": 4.0,
    }
}
MARGIN = CFG["trajectory"]["x_margin"]
SPEED = CFG["trajectory"]["speed_max_star"]


def test_in_bounds():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, 500)
    for _ in range(200):
        traj = sample_trajectory(CFG, rng)
        x = traj(t)
        # grid-based min-max normalization can overshoot by O(grid_spacing^2); negligible
        assert x.min() >= MARGIN - 1e-3
        assert x.max() <= 1.0 - MARGIN + 1e-3


def test_speed_cap():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 1, 2000)
    dt = t[1] - t[0]
    for _ in range(200):
        traj = sample_trajectory(CFG, rng)
        v = np.abs(np.gradient(traj(t), dt))
        assert v.max() <= SPEED * 1.05  # small numerical slack


def test_continuity():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, 5000)
    traj = sample_trajectory(CFG, rng)
    x = traj(t)
    jumps = np.abs(np.diff(x))
    # No teleport: max step bounded by speed*dt with slack
    assert jumps.max() <= SPEED * (t[1] - t[0]) * 5


def test_reproducible():
    t = np.linspace(0, 1, 100)
    a = sample_trajectory(CFG, np.random.default_rng(7))(t)
    b = sample_trajectory(CFG, np.random.default_rng(7))(t)
    assert np.allclose(a, b)


def test_nodes_and_spline():
    rng = np.random.default_rng(4)
    traj = sample_trajectory(CFG, rng)
    k = CFG["trajectory"]["k_sensors"]
    nodes = sample_at_nodes(traj, k)
    assert nodes.shape == (k,)
    t_nodes = np.linspace(0, 1, k)
    sp = spline_interp(t_nodes, nodes)
    # spline reproduces nodes and stays close to the true trajectory between nodes
    assert np.allclose(sp(t_nodes), nodes, atol=1e-9)
    t_fine = np.linspace(0, 1, 777)
    assert np.max(np.abs(sp(t_fine) - traj(t_fine))) < 1e-2
