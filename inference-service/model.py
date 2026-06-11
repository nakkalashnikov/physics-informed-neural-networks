"""
Fourier Feature PINN for 1D heat equation with a moving heat source.

Network: (x_norm, t_norm, α_n, l_n, i_eff_n, x0_n, v_n) → ΔT
         where ΔT = T − T_amb  (temperature rise above ambient).

Coordinate inputs (x_norm, t_norm) pass through a random Fourier feature
encoding to overcome spectral bias in standard MLPs.  Physics parameters
are normalised to [0, 1] and appended directly after the encoding.

IC is hard-enforced by multiplying the network output by t_norm, which
guarantees ΔT(x, 0) = 0 for all inputs without a soft penalty term.

Fourier sigma is annealed during training via set_sigma(): the network
starts learning smooth global structure (low σ) and gradually gains access
to higher frequencies needed to capture the narrow moving heat source.
"""

import math
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """
    Random Fourier feature map with adjustable bandwidth: R^d → R^{2m}.

    Maps v → [cos(2π σ B v), sin(2π σ B v)]
    where B ∈ R^{m×d} is sampled once from N(0, 1) and frozen.
    σ (sigma) is a mutable scalar buffer — call set_sigma() to update it
    during training without re-sampling the frequency directions.
    """

    def __init__(self, d_in: int, m: int, sigma: float):
        super().__init__()
        B = torch.randn(m, d_in)                      # unit Gaussian directions, frozen
        self.register_buffer("B", B)
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        self.out_dim = 2 * m

    def set_sigma(self, sigma: float) -> None:
        self.sigma.fill_(sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_in)  →  (..., 2m)
        proj = 2.0 * math.pi * self.sigma * (x @ self.B.T)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class PINN(nn.Module):
    """
    Parameterised Physics-Informed Neural Network.

    Input layout
    ─────────────
    coords_norm : (N, 2)  –  [x_norm, t_norm] ∈ [0, 1]
    params_norm : (N, 5)  –  [α_n, l_n, i_eff_n, x0_n, v_n] ∈ [0, 1]

    Output
    ──────
    delta_T : (N, 1)  –  ΔT = T − T_amb  [K]

    Hard IC: output = t_norm * MLP(features), so ΔT(x, 0) ≡ 0.
    """

    def __init__(
        self,
        fourier_m: int = 64,
        fourier_sigma: float = 1.0,
        hidden_layers: int = 4,
        hidden_size: int = 256,
    ):
        super().__init__()

        self.fourier = FourierFeatures(d_in=2, m=fourier_m, sigma=fourier_sigma)

        # 128 Fourier features + 5 normalised physics parameters
        d_in_mlp = self.fourier.out_dim + 5

        layers: list[nn.Module] = [nn.Linear(d_in_mlp, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 1))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def set_sigma(self, sigma: float) -> None:
        """Update Fourier bandwidth. Call once per training step."""
        self.fourier.set_sigma(sigma)

    def forward(
        self,
        coords_norm: torch.Tensor,
        params_norm: torch.Tensor,
    ) -> torch.Tensor:
        encoded  = self.fourier(coords_norm)                        # (N, 128)
        features = torch.cat([encoded, params_norm], dim=-1)        # (N, 133)
        t_norm   = coords_norm[:, 1:2]                              # (N,   1)
        return t_norm * self.mlp(features)                          # hard IC


def build_model(cfg: dict) -> PINN:
    m = cfg["model"]
    sigma_start = m.get("fourier_sigma_start", m.get("fourier_sigma", 1.0))
    return PINN(
        fourier_m=m["fourier_m"],
        fourier_sigma=sigma_start,
        hidden_layers=m["hidden_layers"],
        hidden_size=m["hidden_size"],
    )
