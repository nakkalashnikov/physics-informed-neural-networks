"""Inverse PI-DeepONet:  branch([thermocouples ; Fo,AR,w]) ⊗ trunk(t)  ->  xb(t).

Outputs the source POSITION vs time (a function -> genuine DeepONet). For the linear family
xb(t)=x0+v·t, so (x0, v) = xb(0), xb(1)-xb(0). The branch encodes the K×M measurement vector plus the
known normalized material/geometry; the trunk encodes the query time. Smooth low-DOF output -> accurate.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _mlp(d_in: int, hidden: list[int], d_out: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = d_in
    for w in hidden:
        lin = nn.Linear(d, w)
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        d = w
    out = nn.Linear(d, d_out)
    nn.init.xavier_uniform_(out.weight)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class _TrunkT(nn.Module):
    """t -> R^p with Fourier features so the trunk can represent a wiggly xb(t).

    n_freq=0 -> just [t] (affine-capable, enough for linear trajectories); n_freq>0 adds
    sin/cos(kπt) up to n_freq, letting xb(t) carry up to ~n_freq oscillations (arbitrary paths).
    """

    def __init__(self, n_freq: int, hidden: list[int], p: int):
        super().__init__()
        self.register_buffer("freqs", math.pi * torch.arange(1, n_freq + 1, dtype=torch.float32))
        self.net = _mlp(1 + 2 * n_freq, hidden, p)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        ang = t * self.freqs                                  # (T, n_freq)
        feats = torch.cat([t, torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.net(feats)


class InverseDeepONet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        iv = cfg["inverse"]
        self.K, self.M = int(iv["n_sensors"]), int(iv["n_times"])
        p = int(iv["latent"])
        self.branch = _mlp(self.K * self.M + 3, list(iv["branch_hidden"]), p)
        self.trunk = _TrunkT(int(iv.get("trunk_n_freq", 0)), list(iv["trunk_hidden"]), p)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_in: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """branch_in: (N, K*M+3) = [measurements ; Fo_n, AR_n, w_n].  t: (T, 1) shared query times.
        Returns xb: (N, T) — source position at each query time."""
        meas, mat = branch_in[:, : self.K * self.M], branch_in[:, self.K * self.M:]
        # per-sample RMS normalization: the field amplitude spans ~7 orders (Q* range); the trajectory
        # info is in the SHAPE (peak timing), and the amplitude/material is given separately in `mat`.
        meas = meas / (meas.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
        b = self.branch(torch.cat([meas, mat], dim=-1))     # (N, p)
        tau = self.trunk(t)                                  # (T, p)
        return b @ tau.T + self.bias                         # (N, T)

    def recover(self, branch_in: torch.Tensor) -> torch.Tensor:
        """Read off (x0, v) = xb(0), xb(1)-xb(0).  Returns (N, 2)."""
        t = torch.tensor([[0.0], [1.0]], device=branch_in.device, dtype=branch_in.dtype)
        xb = self.forward(branch_in, t)                      # (N, 2)
        return torch.cat([xb[:, 0:1], xb[:, 1:2] - xb[:, 0:1]], dim=-1)


def build_inverse_model(cfg: dict) -> InverseDeepONet:
    return InverseDeepONet(cfg)
