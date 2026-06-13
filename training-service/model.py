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

Hidden layers use the full PirateNet architecture (Wang et al., JMLR 2024):

  1. Two shared encoders U, V computed once from input features:
       U = tanh(W_u · features),   V = tanh(W_v · features)

  2. Each PirateBlock:
       h   = tanh(W · h_prev + b)       standard transform
       h   = h ⊙ U + (1 − h) ⊙ V       element-wise gating with shared U, V
       out = α · h + (1 − α) · h_prev   residual with learnable scalar gate

  α is initialised to 0 (identity map at start) and unconstrained —
  the paper reports it stabilises around O(10⁻²) during training.
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
        B = torch.randn(m, d_in)
        self.register_buffer("B", B)
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        self.out_dim = 2 * m

    def set_sigma(self, sigma: float) -> None:
        self.sigma.fill_(sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * self.sigma * (x @ self.B.T)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class PirateBlock(nn.Module):
    """
    Full PirateNet residual block (Wang et al., JMLR 2024).

    Forward:
        h   = tanh(W · x + b)
        h   = h ⊙ u + (1 − h) ⊙ v      shared encoders gate the transform
        out = α · h + (1 − α) · x        learnable residual skip

    u and v are shared across all blocks and passed in from PINN.forward().
    α is a per-block scalar initialised to 0 (no clamp — paper is unconstrained).
    """

    def __init__(self, size: int):
        super().__init__()
        self.linear = nn.Linear(size, size)
        self.alpha  = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.tanh(self.linear(x))
        h = h * u + (1.0 - h) * v
        return self.alpha * h + (1.0 - self.alpha) * x


class PINN(nn.Module):
    """
    Parameterised Physics-Informed Neural Network with full PirateNet layers.

    Input layout
    ─────────────
    coords_norm : (N, 2)  –  [x_norm, t_norm] ∈ [0, 1]
    params_norm : (N, 5)  –  [α_n, l_n, i_eff_n, x0_n, v_n] ∈ [0, 1]

    Output
    ──────
    delta_T : (N, 1)  –  ΔT = T − T_amb  [K]

    Hard IC: output = t_norm * net(features), so ΔT(x, 0) ≡ 0.
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
        d_in_mlp = self.fourier.out_dim + 5   # 128 + 5 = 133

        # Shared encoders U and V — computed once per forward pass,
        # shared across all PirateBlocks
        self.encoder_u = nn.Linear(d_in_mlp, hidden_size)
        self.encoder_v = nn.Linear(d_in_mlp, hidden_size)

        # Input projection
        self.input_layer = nn.Linear(d_in_mlp, hidden_size)

        # PirateNet residual blocks
        self.blocks = nn.ModuleList([
            PirateBlock(hidden_size) for _ in range(hidden_layers - 1)
        ])

        # Output projection
        self.output_layer = nn.Linear(hidden_size, 1)

        for layer in [self.encoder_u, self.encoder_v,
                      self.input_layer, self.output_layer]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def set_sigma(self, sigma: float) -> None:
        self.fourier.set_sigma(sigma)

    def forward(
        self,
        coords_norm: torch.Tensor,
        params_norm: torch.Tensor,
    ) -> torch.Tensor:
        encoded  = self.fourier(coords_norm)                     # (N, 128)
        features = torch.cat([encoded, params_norm], dim=-1)     # (N, 133)
        t_norm   = coords_norm[:, 1:2]                           # (N,   1)

        # Shared encoders — computed once, reused in every block
        u = torch.tanh(self.encoder_u(features))                 # (N, 256)
        v = torch.tanh(self.encoder_v(features))                 # (N, 256)

        h = torch.tanh(self.input_layer(features))               # (N, 256)
        for block in self.blocks:
            h = block(h, u, v)                                   # (N, 256)

        return t_norm * self.output_layer(h)                     # hard IC


def build_model(cfg: dict) -> PINN:
    m = cfg["model"]
    sigma_start = m.get("fourier_sigma_start", m.get("fourier_sigma", 1.0))
    return PINN(
        fourier_m=m["fourier_m"],
        fourier_sigma=sigma_start,
        hidden_layers=m["hidden_layers"],
        hidden_size=m["hidden_size"],
    )
