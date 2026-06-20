"""
Random continuous burner-trajectory generator x_b*(t*) in [0,1] time, [margin,1-margin] space.

The burner moves arbitrarily along x (the only constraints: bounded position, bounded speed,
continuous — no teleport). We draw smooth random trajectories from a random Fourier series with
a smoothness-decaying spectrum, then affine-map into the allowed band and enforce the speed cap.

A trajectory is returned as a vectorized callable xb_star(t_star: np.ndarray) -> np.ndarray, plus
a C1 spline interpolant of its k node samples for use inside the physics residual.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline


def sample_trajectory(cfg: dict, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
    """Draw one smooth random trajectory xb_star(t_star), t_star in [0,1] -> [margin,1-margin].

    Construction:
      raw(t) = sum_{j=1..J} a_j sin(2*pi*j*t) + b_j cos(2*pi*j*t),  a_j,b_j ~ N(0,(sigma0/j^p)^2)
    then min-max normalize raw to [margin, 1-margin]; finally rescale time-amplitude (shrink toward
    the mid-line) if the max |d/dt| exceeds speed_max_star, so the speed cap is met without clipping
    position (clipping would break C1 continuity).
    """
    tc = cfg["trajectory"]
    margin = float(tc["x_margin"])

    if tc.get("linear", False):
        # Restricted family: straight-line constant-velocity source. The trajectory FUNCTION
        # space collapses to two numbers (x0, v=x1-x0), so the operator's generalisation problem
        # becomes low-dimensional — vs the ~infinite-dim random-Fourier space that plateaued at
        # ~76%. Draw start & end in the allowed band; |v|=|x1-x0| <= 1-2*margin << speed cap, so
        # the line is in-bounds and speed-legal by construction (no clipping needed).
        x0, x1 = rng.uniform(margin, 1.0 - margin, size=2)
        return lambda t: x0 + (x1 - x0) * np.atleast_1d(np.asarray(t, dtype=float))

    J = int(tc["n_fourier_modes"])
    p = float(tc["spectrum_decay"])
    sigma0 = float(tc["sigma0"])
    speed_max = float(tc["speed_max_star"])

    j = np.arange(1, J + 1)
    amp = sigma0 / j**p
    a = rng.normal(0.0, amp)
    b = rng.normal(0.0, amp)

    def raw(t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        ang = 2.0 * np.pi * np.outer(t, j)          # (T, J)
        return (np.sin(ang) @ a) + (np.cos(ang) @ b)

    # Min-max normalize over a dense grid -> [margin, 1-margin]
    grid = np.linspace(0.0, 1.0, 1024)
    r = raw(grid)
    lo, hi = r.min(), r.max()
    span = hi - lo
    if span < 1e-9:                                  # degenerate (near-constant) draw
        center = 0.5
        return lambda t: np.full_like(np.atleast_1d(np.asarray(t, float)), center)

    def mapped(t: np.ndarray) -> np.ndarray:
        return margin + (1.0 - 2.0 * margin) * (raw(t) - lo) / span

    # Enforce speed cap by shrinking amplitude toward the midline (preserves continuity).
    dt = grid[1] - grid[0]
    vmax = np.abs(np.gradient(mapped(grid), dt)).max()
    shrink = 1.0 if vmax <= speed_max else speed_max / vmax
    mid = 0.5  # shrink toward domain mid-line to respect speed cap

    def traj(t: np.ndarray) -> np.ndarray:
        m = mapped(t)
        return mid + shrink * (m - mid)

    return traj


def sample_at_nodes(traj_fn: Callable[[np.ndarray], np.ndarray], k: int) -> np.ndarray:
    """Sample the trajectory at k uniform t* nodes -> branch input vector (length k)."""
    t_nodes = np.linspace(0.0, 1.0, k)
    return traj_fn(t_nodes)


def spline_interp(t_nodes: np.ndarray, x_nodes: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """C1 cubic interpolant of the k samples; used for the source term in the physics residual."""
    cs = CubicSpline(t_nodes, x_nodes, bc_type="natural")
    return lambda t: cs(np.atleast_1d(np.asarray(t, dtype=float)))
