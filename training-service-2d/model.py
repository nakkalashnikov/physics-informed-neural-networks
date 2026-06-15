"""
PI-DeepONet for the 2D moving-heat-source operator  G: (trajectory, pi-groups) -> u(x*,y_hat,t*).

BRANCH:  [xb*(t1*)..xb*(tk*)] ++ [logFo_n, AR_n, w_n]  (104) -> MLP -> b in R^p
TRUNK:   (x*, y_hat, t*)  -> RFF(sigma curriculum) -> 3x PirateNet block -> tau in R^p
MERGE:   u = ( sum_i b_i tau_i + b0 ) * t*          (dot product, then *t* = hard IC)

Reuses the proven 1D building blocks (FourierFeatures + full PirateNet block, Wang et al. JMLR
2024). Output u is dimensionless and O(1); physical dT = T_c * u is reconstructed by the caller.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Random Fourier features R^d -> R^{2m} with mutable bandwidth sigma (curriculum)."""

    def __init__(self, d_in: int, m: int, sigma: float):
        super().__init__()
        self.register_buffer("B", torch.randn(m, d_in))
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        self.out_dim = 2 * m

    def set_sigma(self, sigma: float) -> None:
        self.sigma.fill_(sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * self.sigma * (x @ self.B.T)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class PirateBlock(nn.Module):
    """Full PirateNet residual block with shared U/V gating and learnable scalar skip (init 0)."""

    def __init__(self, size: int):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)
        self.linear3 = nn.Linear(size, size)
        self.alpha = nn.Parameter(torch.zeros(1))
        for layer in (self.linear1, self.linear2, self.linear3):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, u, v):
        h = torch.tanh(self.linear1(x))
        h = h * u + (1.0 - h) * v
        h = torch.tanh(self.linear2(h))
        h = h * u + (1.0 - h) * v
        h = torch.tanh(self.linear3(h))
        return self.alpha * h + (1.0 - self.alpha) * x


class BranchNet(nn.Module):
    """Plain tanh MLP encoding [trajectory samples ++ pi-groups] -> latent b in R^p."""

    def __init__(self, in_dim: int, hidden: list[int], out_dim: int):
        super().__init__()
        layers = []
        d = in_dim
        for w in hidden:
            lin = nn.Linear(d, w)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            d = w
        out = nn.Linear(d, out_dim)
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TrunkNet(nn.Module):
    """RFF + PirateNet trunk encoding (x*, y_hat, t*) -> latent tau in R^p."""

    def __init__(self, rff_m: int, sigma_start: float, n_blocks: int, width: int, out_dim: int):
        super().__init__()
        self.fourier = FourierFeatures(d_in=3, m=rff_m, sigma=sigma_start)
        d_in = self.fourier.out_dim
        self.encoder_u = nn.Linear(d_in, width)
        self.encoder_v = nn.Linear(d_in, width)
        self.input_layer = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList([PirateBlock(width) for _ in range(n_blocks)])
        self.output_layer = nn.Linear(width, out_dim)
        for layer in (self.encoder_u, self.encoder_v, self.input_layer, self.output_layer):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def set_sigma(self, sigma: float) -> None:
        self.fourier.set_sigma(sigma)

    def forward(self, coords):
        feats = self.fourier(coords)
        u = torch.tanh(self.encoder_u(feats))
        v = torch.tanh(self.encoder_v(feats))
        h = torch.tanh(self.input_layer(feats))
        for block in self.blocks:
            h = block(h, u, v)
        return self.output_layer(h)


class DeepONet(nn.Module):
    """u = (sum_i b_i tau_i + b0) * t*  with hard IC (u=0 at t*=0)."""

    def __init__(self, cfg: dict):
        super().__init__()
        k = int(cfg["trajectory"]["k_sensors"])
        p = int(cfg["branch"]["out_dim"])
        assert p == int(cfg["trunk"]["out_dim"]), "branch.out_dim must equal trunk.out_dim"
        self.branch = BranchNet(in_dim=k + 3, hidden=list(cfg["branch"]["hidden"]), out_dim=p)
        self.trunk = TrunkNet(
            rff_m=int(cfg["trunk"]["rff_num_features"]),
            sigma_start=float(cfg["trunk"]["rff_sigma_start"]),
            n_blocks=int(cfg["trunk"]["n_pirate_blocks"]),
            width=int(cfg["trunk"]["width"]),
            out_dim=p,
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def set_sigma(self, sigma: float) -> None:
        self.trunk.set_sigma(sigma)

    def forward(self, branch_in: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """branch_in: (N, k+3), coords: (N, 3)=[x*, y_hat, t*]. Returns u: (N, 1).

        branch_in is given per evaluation point (repeat a trajectory's vector across its points).
        """
        b = self.branch(branch_in)                       # (N, p)
        tau = self.trunk(coords)                          # (N, p)
        dot = (b * tau).sum(dim=-1, keepdim=True) + self.bias  # (N, 1)
        t_star = coords[:, 2:3]                            # (N, 1)
        return t_star * dot                               # hard IC


def build_model(cfg: dict) -> DeepONet:
    return DeepONet(cfg)
