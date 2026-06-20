"""
Tier-0B cross-check: thin-plate (2D) quasi-stationary Rosenthal solution for a CONSTANT-velocity
moving point source. Independent of the Fourier reference and of a different physical regime
(semi-infinite, quasi-steady), so agreement on a constant-v sub-case is a strong sanity check.
Valid only mid-domain, after quasi-steady is reached, away from the insulated boundaries.

    dT(xi, y) = P/(2 pi k h) * exp(-v xi / 2 alpha) * K0( v r / 2 alpha ),  r = sqrt(xi^2 + y^2)
    xi = x - v t  (moving coordinate),  K0 = modified Bessel function of the second kind.
"""

from __future__ import annotations

import numpy as np
from scipy.special import kn


def thin_plate_rosenthal(x: np.ndarray, y: np.ndarray, t: float,
                         P: float, k: float, h: float, alpha: float, v: float,
                         x0: float = 0.0) -> np.ndarray:
    """Quasi-steady temperature rise dT[y, x] for a source at x_b(t) = x0 + v t. Dimensional."""
    xb = x0 + v * t
    X, Y = np.meshgrid(x, y)
    xi = X - xb
    r = np.sqrt(xi**2 + Y**2)
    r = np.maximum(r, 1e-9)                       # regularize the log singularity at the source
    return P / (2 * np.pi * k * h) * np.exp(-v * xi / (2 * alpha)) * kn(0, v * r / (2 * alpha))
