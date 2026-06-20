"""
Physics residuals for the dimensionless PDE (autograd, create_graph=True).

    PDE:        R     = u_t*  -  Fo*u_x*x*  -  (Fo/AR^2)*u_yhat_yhat
    flux BC:    R_bc0 = u_yhat|_{yhat=0}  +  Q_star * g_hat(x* - xb*(t*))
    insulation: R_ins = u_n   (normal derivative on x*=0, x*=1, yhat=1)  -> 0

Per-point pi-group tensors (Fo, AR, Q_star, w) are passed in because each collocation point
belongs to a trajectory with its own pi-groups. xb_at_t (the source center at each point's t*)
is precomputed from the trajectory spline and passed as data (not differentiated).
"""

from __future__ import annotations

import math

import torch


def g_hat(s: torch.Tensor, sigma_star: torch.Tensor) -> torch.Tensor:
    """Normalized gaussian contact profile (integral 1 over x*)."""
    return torch.exp(-0.5 * (s / sigma_star) ** 2) / (sigma_star * math.sqrt(2.0 * math.pi))


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """d(outputs.sum())/d(inputs), keeping the graph for higher-order derivatives."""
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True,
    )[0]


def pde_residual(model, branch_in, coords, Fo, AR) -> torch.Tensor:
    """Interior PDE residual. coords: (N,3) leaf w/ requires_grad. Fo, AR: (N,1)."""
    u = model(branch_in, coords)               # (N,1)
    g1 = _grad(u, coords)                       # (N,3): u_x, u_y, u_t
    u_x, u_y, u_t = g1[:, 0:1], g1[:, 1:2], g1[:, 2:3]
    u_xx = _grad(u_x, coords)[:, 0:1]
    u_yy = _grad(u_y, coords)[:, 1:2]
    return u_t - Fo * u_xx - (Fo / AR**2) * u_yy


def bc_flux_residual(model, branch_in, coords_y0, Q_star, xb_at_t, w, sigma_factor: float):
    """Moving-flux BC at yhat=0:  u_yhat + Q_star*g_hat(x* - xb*(t*)).  All per-point (N,1)."""
    u = model(branch_in, coords_y0)
    u_y = _grad(u, coords_y0)[:, 1:2]
    sigma_star = w / sigma_factor
    g = g_hat(coords_y0[:, 0:1] - xb_at_t, sigma_star)
    return u_y + Q_star * g


def bc_insulation_residual(model, branch_in, coords, axis: int) -> torch.Tensor:
    """Zero-flux residual: normal derivative along `axis` (0=x for x-edges, 1=y for yhat=1 edge)."""
    u = model(branch_in, coords)
    return _grad(u, coords)[:, axis:axis + 1]
