"""
Tier-2 intrinsic checks (NO ground truth needed — usable at deployment):
  2A residual_map      — PDE + flux-BC residual statistics
  2B (defect)          — see fdm.solve_defect; error_map_from_defect() wraps it
  2C energy_balance    — mean(u) over the domain must equal t* (exact energy conservation)
  2D check_invariants  — non-negativity (up to Gibbs), monotone energy, IC

predict_field() evaluates the trained model on a grid for one trajectory.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from core.nondim import PiGroups, normalize
from core.physics import bc_flux_residual, pde_residual
from core.trajectory import sample_at_nodes


def _branch_vec(traj_fn, pi: PiGroups, cfg: dict) -> np.ndarray:
    k = int(cfg["trajectory"]["k_sensors"])
    return np.concatenate([sample_at_nodes(traj_fn, k), np.array(normalize(pi, cfg))]).astype(np.float32)


def predict_field(model, traj_fn, pi: PiGroups, cfg: dict,
                  x: np.ndarray, y: np.ndarray, t: np.ndarray, device) -> np.ndarray:
    """Model field u[t, y, x] for one trajectory."""
    bvec = torch.as_tensor(_branch_vec(traj_fn, pi, cfg), device=device)
    X, Y = np.meshgrid(x, y)                       # (ny, nx)
    out = np.empty((len(t), len(y), len(x)), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i, ti in enumerate(t):
            coords = np.stack([X.ravel(), Y.ravel(), np.full(X.size, ti)], axis=1)
            ct = torch.as_tensor(coords, dtype=torch.float32, device=device)
            br = bvec.unsqueeze(0).expand(ct.shape[0], -1)
            u = model(br, ct).cpu().numpy().reshape(len(y), len(x))
            out[i] = u
    return out


def residual_map(model, traj_fn, pi: PiGroups, cfg: dict, device,
                 nx: int = 60, ny: int = 40, nt: int = 20) -> dict:
    """PDE residual stats over the interior + flux-BC residual at yhat=0 (Tier 2A)."""
    bvec = torch.as_tensor(_branch_vec(traj_fn, pi, cfg), device=device)
    sf = float(cfg["physics"]["sigma_factor"])

    xs = np.linspace(0, 1, nx); ys = np.linspace(0.02, 1, ny); ts = np.linspace(0.05, 1, nt)
    X, Y, T = np.meshgrid(xs, ys, ts, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)
    ct = torch.as_tensor(coords, dtype=torch.float32, device=device).requires_grad_(True)
    br = bvec.unsqueeze(0).expand(ct.shape[0], -1)
    Fo = torch.full((ct.shape[0], 1), pi.Fo, device=device)
    AR = torch.full((ct.shape[0], 1), pi.AR, device=device)
    R = pde_residual(model, br, ct, Fo, AR).detach().abs()

    # flux BC at yhat=0
    xb0 = np.linspace(0, 1, nx * 4)
    tb0 = np.linspace(0.05, 1, nt * 2)
    Xb, Tb = np.meshgrid(xb0, tb0)
    cb = np.stack([Xb.ravel(), np.zeros(Xb.size), Tb.ravel()], axis=1)
    cbt = torch.as_tensor(cb, dtype=torch.float32, device=device).requires_grad_(True)
    brb = bvec.unsqueeze(0).expand(cbt.shape[0], -1)
    Q = torch.full((cbt.shape[0], 1), pi.Q_star, device=device)
    xb_at = torch.as_tensor(traj_fn(cb[:, 2]).reshape(-1, 1), dtype=torch.float32, device=device)
    w = torch.full((cbt.shape[0], 1), pi.w, device=device)
    R_bc = bc_flux_residual(model, brb, cbt, Q, xb_at, w, sf).detach().abs()

    return {"mean": float(R.mean()), "max": float(R.max()), "max_bc": float(R_bc.max())}


def energy_balance(u_field: np.ndarray, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Tier 2C: |mean_domain(u) - t*| / t*  per snapshot (domain area = 1)."""
    res = np.empty(len(t))
    for i, ti in enumerate(t):
        mean_u = np.trapezoid(np.trapezoid(u_field[i], x, axis=1), y)
        res[i] = abs(mean_u - ti) / max(ti, 1e-6)
    return res


def check_invariants(u_field: np.ndarray, t: np.ndarray) -> dict:
    """Tier 2D: non-negativity (up to Gibbs), monotone stored energy, near-zero IC."""
    peak = max(abs(u_field).max(), 1e-9)
    nonneg = bool(u_field.min() > -0.06 * peak)
    energies = np.array([u_field[i].mean() for i in range(len(t))])
    mono = bool(np.all(np.diff(energies) > -1e-3 * peak))
    ic_ok = True
    if t[0] < 1e-6:
        ic_ok = bool(abs(u_field[0]).max() < 1e-3 * peak)
    return {"nonneg": nonneg, "mono_energy": mono, "ic": ic_ok}


def error_map_from_defect(model, traj_fn, pi: PiGroups, cfg: dict, device,
                          t_snapshots: np.ndarray) -> np.ndarray:
    """Tier 2B: estimate the error field e_hat by solving D[e] = -R via FDM (no ground truth)."""
    from validation.fdm import solve_defect

    nx, ny = int(cfg["fdm"]["nx"]), int(cfg["fdm"]["ny"])
    nt = int(cfg["fdm"]["nt"])
    x = np.linspace(0, 1, nx); y = np.linspace(0, 1, ny)
    t_grid = np.linspace(0, 1, nt + 1)

    # residual field R(t,y,x) from the model on the FDM grid (interior strong-form residual)
    bvec = torch.as_tensor(_branch_vec(traj_fn, pi, cfg), device=device)
    Rfield = np.zeros((len(t_grid), ny, nx), dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    for j, tj in enumerate(t_grid):
        coords = np.stack([X.ravel(), Y.ravel(), np.full(X.size, tj)], axis=1)
        ct = torch.as_tensor(coords, dtype=torch.float32, device=device).requires_grad_(True)
        br = bvec.unsqueeze(0).expand(ct.shape[0], -1)
        Fo = torch.full((ct.shape[0], 1), pi.Fo, device=device)
        AR = torch.full((ct.shape[0], 1), pi.AR, device=device)
        Rfield[j] = pde_residual(model, br, ct, Fo, AR).detach().cpu().numpy().reshape(ny, nx)
    return solve_defect(Rfield, t_grid, pi, cfg, t_snapshots)
