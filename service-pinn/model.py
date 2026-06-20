"""Single-instance PINN: u(x*,ŷ,t*) = TrunkNet(x*,ŷ,t*) · t*  (hard IC u(·,·,0)=0).

A plain trunk (no branch) for ONE fixed problem instance — the classic PINN. Reuses core's
multi-scale Fourier trunk + PirateNet blocks with out_dim=1. forward() takes (branch_in, coords) and
IGNORES branch_in so the shared core.physics residuals (which call model(branch_in, coords)) work
unchanged — pass None for branch_in.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from core.model import TrunkNet


class TrunkPINN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        t = cfg["trunk"]
        self.trunk = TrunkNet(
            rff_m=int(t["rff_num_features"]),
            bands=[float(b) for b in t.get("rff_sigma_bands", [float(t["rff_sigma_end"])])],
            n_blocks=int(t["n_pirate_blocks"]),
            width=int(t["width"]),
            out_dim=1,
        )

    def set_sigma(self, sigma: float) -> None:
        self.trunk.set_sigma(sigma)

    def forward(self, branch_in, coords: torch.Tensor) -> torch.Tensor:
        """branch_in ignored (single instance). coords: (N,3)=[x*,ŷ,t*] leaf w/ grad. Returns u:(N,1)."""
        return coords[:, 2:3] * self.trunk(coords)        # ×t* hard IC


def build_pinn(cfg: dict) -> TrunkPINN:
    return TrunkPINN(cfg)
