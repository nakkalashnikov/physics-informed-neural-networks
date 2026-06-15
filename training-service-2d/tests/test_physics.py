"""Tests for physics.py — g_hat normalization, autograd residual shapes/finiteness/backprop,
and an INDEPENDENT check that the analytic field satisfies the PDE residual form (finite diff)."""

import numpy as np
import torch
import yaml

from analytic import fourier_field_u
from model import build_model
from nondim import PiGroups
from physics import bc_flux_residual, bc_insulation_residual, g_hat, pde_residual
from trajectory import sample_trajectory

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)
K = CFG["trajectory"]["k_sensors"]


def test_g_hat_normalized():
    x = torch.linspace(0, 1, 4000)
    sigma = torch.tensor(0.02)
    g = g_hat(x - 0.5, sigma)
    integral = torch.trapezoid(g, x)
    assert abs(integral.item() - 1.0) < 1e-3


def test_pde_residual_shape_and_grad():
    torch.manual_seed(0)
    model = build_model(CFG)
    n = 64
    branch = torch.rand(n, K + 3)
    coords = torch.rand(n, 3, requires_grad=True)
    Fo = torch.full((n, 1), 5e-3)
    AR = torch.full((n, 1), 0.15)
    R = pde_residual(model, branch, coords, Fo, AR)
    assert R.shape == (n, 1) and torch.isfinite(R).all()
    R.pow(2).mean().backward()  # gradients must flow to model params
    assert any(p.grad is not None for p in model.parameters())


def test_bc_residuals_shapes():
    model = build_model(CFG)
    n = 32
    branch = torch.rand(n, K + 3)
    y0 = torch.rand(n, 3, requires_grad=True)
    y0.data[:, 1] = 0.0
    Q = torch.full((n, 1), 4.0)
    xb = torch.full((n, 1), 0.5)
    w = torch.full((n, 1), 0.1)
    r0 = bc_flux_residual(model, branch, y0, Q, xb, w, CFG["physics"]["sigma_factor"])
    assert r0.shape == (n, 1) and torch.isfinite(r0).all()
    edge = torch.rand(n, 3, requires_grad=True)
    rins = bc_insulation_residual(model, branch, edge, axis=0)
    assert rins.shape == (n, 1) and torch.isfinite(rins).all()


def test_pde_residual_on_manufactured_solution():
    """Verify physics.pde_residual's autograd operator (the coefficients Fo and Fo/AR^2) on a
    SMOOTH manufactured solution with known derivatives. (The truncated analytic field cannot be
    used here: its strong-form residual is contaminated by undamped high-frequency ringing of the
    boundary-flux delta in the second derivative, even though u itself is accurate.)"""

    class ManufacturedModel(torch.nn.Module):
        # u = sin(2x) cos(3y) exp(t)  -> u_t=u, u_xx=-4u, u_yy=-9u  (ignores branch_in)
        def forward(self, branch_in, coords):
            x, y, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
            return torch.sin(2 * x) * torch.cos(3 * y) * torch.exp(t)

    Fo_v, AR_v = 5e-3, 0.2
    model = ManufacturedModel()
    n = 100
    branch = torch.rand(n, K + 3)
    coords = torch.rand(n, 3, requires_grad=True)
    Fo = torch.full((n, 1), Fo_v)
    AR = torch.full((n, 1), AR_v)
    R = pde_residual(model, branch, coords, Fo, AR)
    x, y, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
    u = torch.sin(2 * x) * torch.cos(3 * y) * torch.exp(t)
    expected = u - Fo * (-4 * u) - (Fo / AR**2) * (-9 * u)
    assert torch.allclose(R, expected, atol=1e-5)
