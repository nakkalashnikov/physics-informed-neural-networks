"""
Tier-1 INDEPENDENT numerical reference: ADI (Peaceman-Rachford) solver for the dimensionless PDE.

Derived by a completely different route than analytic.py (finite differences, not eigenfunctions),
so agreement between the two is a genuine cross-check of the load-bearing Fourier formula. Implicit
(ADI) rather than explicit because thin plates (AR -> 0.02) make the y-diffusivity Fo/AR^2 huge and
explicit time-stepping unstable / infeasible.

Solves:  u_t* = Fo u_x*x*  +  (Fo/AR^2) u_yhat_yhat   on [0,1]^2,
  u_yhat(yhat=0) = -Q_star * ghat(x* - xb*(t*)),  insulated on the other three edges,  u(.,.,0)=0.

Also exposes solve_defect() for the Tier-2B error map (same operator, source = -R).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from core.nondim import PiGroups


def _neumann_d2(U: np.ndarray, axis: int, d: float) -> np.ndarray:
    """Homogeneous-Neumann (insulated both ends) 2nd difference along `axis`, spacing d."""
    out = np.zeros_like(U)
    if axis == 1:  # x (columns)
        out[:, 1:-1] = (U[:, :-2] - 2 * U[:, 1:-1] + U[:, 2:]) / d**2
        out[:, 0] = (2 * U[:, 1] - 2 * U[:, 0]) / d**2
        out[:, -1] = (2 * U[:, -2] - 2 * U[:, -1]) / d**2
    else:  # y (rows)
        out[1:-1, :] = (U[:-2, :] - 2 * U[1:-1, :] + U[2:, :]) / d**2
        out[0, :] = (2 * U[1, :] - 2 * U[0, :]) / d**2
        out[-1, :] = (2 * U[-2, :] - 2 * U[-1, :]) / d**2
    return out


def _implicit_operator(n: int, coeff: float, d: float):
    """Factorize (I - coeff * D_homogNeumann) as a sparse tridiagonal LU. coeff = (dt/2)*diffusivity."""
    s = coeff / d**2
    main = np.full(n, 1.0 + 2.0 * s)
    upper = np.full(n - 1, -s)
    lower = np.full(n - 1, -s)
    upper[0] = -2.0 * s   # row 0: ghost = interior neighbor (insulated)
    lower[-1] = -2.0 * s  # row n-1: same
    M = sp.diags([lower, main, upper], offsets=[-1, 0, 1], format="csc")
    return splu(M)


def _solve_pde(
    source_fn: Callable[[float], np.ndarray] | None,
    pi: PiGroups,
    cfg: dict,
    t_snapshots: np.ndarray,
    forcing_field: np.ndarray | None = None,
    t_grid_forcing: np.ndarray | None = None,
) -> np.ndarray:
    """Core ADI stepper. Returns u[len(t_snapshots), ny, nx].

    source_fn(t) -> (nx,) boundary-flux row source for u, OR None.
    forcing_field: optional volumetric source -R(t,y,x) for the defect equation (Tier 2B).
    """
    f = cfg["fdm"]
    nx, ny, nt = int(f["nx"]), int(f["ny"]), int(f["nt"])
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    dt = 1.0 / nt

    ax = pi.Fo                 # x diffusivity
    ay = pi.Fo / pi.AR**2      # y diffusivity

    lu_x = _implicit_operator(nx, 0.5 * dt * ax, dx)
    lu_y = _implicit_operator(ny, 0.5 * dt * ay, dy)

    def src_row(t: float) -> np.ndarray:
        """Boundary-flux contribution to ay*Dyy at row yhat=0: ay*(-2*phi/dy), phi=u_y(0)=source_fn."""
        if source_fn is None:
            return np.zeros(nx)
        b = np.zeros((ny, nx))
        b[0, :] = ay * (-2.0 / dy) * source_fn(t)
        return b

    def vol(t: float) -> np.ndarray:
        if forcing_field is None:
            return np.zeros((ny, nx))
        # nearest-time slice of the (T,ny,nx) forcing
        j = int(np.clip(round(t * (len(t_grid_forcing) - 1)), 0, len(t_grid_forcing) - 1))
        return forcing_field[j]

    U = np.zeros((ny, nx))
    snaps = np.empty((len(t_snapshots), ny, nx))
    snap_set = sorted(set(int(round(ts * nt)) for ts in t_snapshots))
    snap_map = {int(round(ts * nt)): i for i, ts in enumerate(t_snapshots)}

    if 0 in snap_map:
        snaps[snap_map[0]] = U
    for step in range(1, nt + 1):
        tn = (step - 1) * dt
        tnp = step * dt
        gn = src_row(tn) + ay * 0  # boundary source already scaled; volumetric added below
        # PR step 1: implicit x, explicit y (+ half source at t^n)
        rhs1 = U + 0.5 * dt * ay * _neumann_d2(U, axis=0, d=dy) + 0.5 * dt * (gn + vol(tn))
        Ustar = lu_x.solve(rhs1.T).T                  # solve along x for each y-row
        # PR step 2: implicit y, explicit x (+ half source at t^{n+1})
        gnp = src_row(tnp)
        rhs2 = Ustar + 0.5 * dt * ax * _neumann_d2(Ustar, axis=1, d=dx) + 0.5 * dt * (gnp + vol(tnp))
        U = lu_y.solve(rhs2)                           # solve along y for each x-column
        if step in snap_map:
            snaps[snap_map[step]] = U
    return snaps


def crank_nicolson_2d(
    traj_fn: Callable[[np.ndarray], np.ndarray],
    pi: PiGroups,
    cfg: dict,
    t_snapshots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent reference field u[t,y,x] for the moving boundary flux. Returns (x, y, u)."""
    sigma_star = pi.sigma_star(cfg["physics"]["sigma_factor"])
    nx = int(cfg["fdm"]["nx"])
    x = np.linspace(0.0, 1.0, nx)

    def source_fn(t: float) -> np.ndarray:
        # phi = u_yhat(yhat=0) = -Q_star * ghat(x* - xb*(t))
        xb = float(traj_fn(np.array([t]))[0])
        ghat = np.exp(-0.5 * ((x - xb) / sigma_star) ** 2) / (sigma_star * np.sqrt(2.0 * np.pi))
        return -pi.Q_star * ghat

    u = _solve_pde(source_fn, pi, cfg, np.asarray(t_snapshots, float))
    y = np.linspace(0.0, 1.0, int(cfg["fdm"]["ny"]))
    return x, y, u


def solve_defect(
    residual_field: np.ndarray,
    t_grid: np.ndarray,
    pi: PiGroups,
    cfg: dict,
    t_snapshots: np.ndarray,
) -> np.ndarray:
    """Tier-2B: solve D[e] = -R with homogeneous IC/BC -> error-map estimate e_hat[t,y,x]."""
    return _solve_pde(
        source_fn=None,
        pi=pi,
        cfg=cfg,
        t_snapshots=np.asarray(t_snapshots, float),
        forcing_field=-np.asarray(residual_field, float),
        t_grid_forcing=np.asarray(t_grid, float),
    )
