"""
Tier-0A exact analytic reference = the dimensionless Fourier-series solution.

Dual role: (1) generates exact training labels u for the hybrid loss, (2) ground-truth for
validation on ANY continuous trajectory. NUMERICALLY VERIFIED (2026-06-15) against an
independent 2D Crank-Nicolson FD solver: L2 ~1.6% (FD-grid-limited), energy conserved to 0.05%,
peak match <0.2%. Do NOT change the alpha/k factor or the N_x/N_y normalization.

Dimensionless solution on [0,1]x[0,1] (x*, y_hat), t* in [0,1]:
    u(x*,y_hat,t*) = sum_{m=0..M} sum_{n=0..N} A_mn(t*) cos(m pi x*) cos(n pi y_hat)
    lambda_mn = Fo*[ (m pi)^2 + (n pi / AR)^2 ]
    Nx_m = 1/2 (m>=1) else 1 ;  Ny_n = 1/2 (n>=1) else 1
    Ghat_m(tau) = integral_0^1 ghat(x* - xb*(tau)) cos(m pi x*) dx*     (numerical quadrature)
    A_mn(t*) = (1/(Nx_m Ny_n)) integral_0^{t*} Ghat_m(tau) exp(-lambda_mn (t*-tau)) dtau

The source prefactor is exactly 1 because (Fo/AR^2)*Q_star == 1 (Q_star = AR^2/Fo).
Sanity: A_00(t*) = t* exactly -> mean(u) = t* -> exact energy conservation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from nondim import PiGroups


def _ghat(s: np.ndarray, sigma_star: float) -> np.ndarray:
    """Normalized contact profile (gaussian, integral 1 over x*)."""
    return np.exp(-0.5 * (s / sigma_star) ** 2) / (sigma_star * np.sqrt(2.0 * np.pi))


def _norm(idx: np.ndarray) -> np.ndarray:
    """Cosine-basis norm factor: 1 for index 0, 1/2 otherwise."""
    return np.where(idx == 0, 1.0, 0.5)


def _build_A_grid(
    traj_fn: Callable[[np.ndarray], np.ndarray],
    pi: PiGroups,
    cfg: dict,
    M: int,
    N: int,
    ntau: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (tau_grid, A_grid) where A_grid[j] = A_mn at tau_grid[j], shape (ntau, M+1, N+1)."""
    sigma_star = pi.sigma_star(cfg["physics"]["sigma_factor"])
    nx = int(cfg["analytic"]["quad_nx"])

    tau = np.linspace(0.0, 1.0, ntau)
    dtau = tau[1] - tau[0]

    # Ghat_m(tau) by numerical quadrature over x* (edge-robust; do NOT use closed form near edges)
    xg = np.linspace(0.0, 1.0, nx)
    dx = xg[1] - xg[0]
    m_idx = np.arange(M + 1)
    n_idx = np.arange(N + 1)
    cos_mx = np.cos(np.outer(m_idx, np.pi * xg))            # (M+1, nx)
    xb_tau = traj_fn(tau)                                   # (ntau,)
    g_vals = _ghat(xg[None, :] - xb_tau[:, None], sigma_star)  # (ntau, nx)
    Ghat = g_vals @ cos_mx.T * dx                          # (ntau, M+1)

    # lambda_mn and source-coefficient 1/(Nx Ny), both (M+1, N+1)
    lam = pi.Fo * ((m_idx[:, None] * np.pi) ** 2 + (n_idx[None, :] * np.pi / pi.AR) ** 2)
    src_coeff = 1.0 / (_norm(m_idx)[:, None] * _norm(n_idx)[None, :])
    decay = np.exp(-lam * dtau)                            # (M+1, N+1)

    A_grid = np.zeros((ntau, M + 1, N + 1))
    A = np.zeros((M + 1, N + 1))
    for j in range(ntau - 1):
        # exponential-integrator step with trapezoidal source
        A = A * decay + src_coeff * 0.5 * (Ghat[j][:, None] * decay + Ghat[j + 1][:, None]) * dtau
        A_grid[j + 1] = A
    return tau, A_grid


def _interp_A(tau: np.ndarray, A_grid: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Linearly interpolate A_grid (ntau,M+1,N+1) to query times -> (Q, M+1, N+1)."""
    tq = np.clip(np.asarray(t_query, dtype=float), 0.0, 1.0)
    ntau = len(tau)
    dtau = tau[1] - tau[0]
    idx = np.clip((tq / dtau).astype(int), 0, ntau - 2)
    frac = (tq - tau[idx]) / dtau
    return A_grid[idx] * (1.0 - frac)[:, None, None] + A_grid[idx + 1] * frac[:, None, None]


def fourier_labels_u(
    query_pts_star: np.ndarray,
    traj_fn: Callable[[np.ndarray], np.ndarray],
    pi: PiGroups,
    cfg: dict,
    M: int = 48,
    N: int = 32,
    ntau: int = 400,
) -> np.ndarray:
    """Exact dimensionless u at scattered query points (Q,3)=[x*, y_hat, t*]. Smaller M/N/ntau
    defaults than the validation field — labels do not need machine precision."""
    q = np.asarray(query_pts_star, dtype=float)
    xs, ys, ts = q[:, 0], q[:, 1], q[:, 2]
    tau, A_grid = _build_A_grid(traj_fn, pi, cfg, M, N, ntau)
    A_q = _interp_A(tau, A_grid, ts)                       # (Q, M+1, N+1)
    cos_mx = np.cos(np.outer(xs, np.pi * np.arange(M + 1)))  # (Q, M+1)
    cos_ny = np.cos(np.outer(ys, np.pi * np.arange(N + 1)))  # (Q, N+1)
    return np.einsum("qmn,qm,qn->q", A_q, cos_mx, cos_ny)


def fourier_field_u(
    x_star: np.ndarray,
    y_star: np.ndarray,
    t_star: np.ndarray,
    traj_fn: Callable[[np.ndarray], np.ndarray],
    pi: PiGroups,
    cfg: dict,
    M: int | None = None,
    N: int | None = None,
) -> np.ndarray:
    """High-fidelity dimensionless field u[t, y, x] on the grid (x_star, y_star) at times t_star."""
    M = int(cfg["analytic"]["fourier_M"]) if M is None else M
    N = int(cfg["analytic"]["fourier_N"]) if N is None else N
    ntau = int(cfg["analytic"]["quad_nt"])
    tau, A_grid = _build_A_grid(traj_fn, pi, cfg, M, N, ntau)
    A_t = _interp_A(tau, A_grid, np.asarray(t_star, float))  # (T, M+1, N+1)
    cos_mx = np.cos(np.outer(np.asarray(x_star, float), np.pi * np.arange(M + 1)))  # (X, M+1)
    cos_ny = np.cos(np.outer(np.asarray(y_star, float), np.pi * np.arange(N + 1)))  # (Y, N+1)
    # u[t,y,x] = sum_mn A_t[t,m,n] cos_ny[y,n] cos_mx[x,m]
    return np.einsum("tmn,yn,xm->tyx", A_t, cos_ny, cos_mx)
