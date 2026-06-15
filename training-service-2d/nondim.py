"""
Non-dimensionalization (Buckingham-pi) for the 2D moving-heat-source problem.

Physical PDE (domain [0,l]x[0,h], t in (0,t_end]):
    dT/dt = alpha (T_xx + T_yy),   alpha = k/rho_c
    -k dT/dy|_{y=0} = q(x,t) = P * g(x - x_b(t))     (moving boundary flux)
    insulated on x=0, x=l, y=h ;  T(.,.,0)=0

Dimensionless vars: x*=x/l, y_hat=y/h in [0,1], t*=t/t_end, u=dT/T_c.
    T_c = P * t_end / (rho_c * l * h)
    Fo  = alpha * t_end / l^2
    AR  = h / l
    w   = S / l          (contact width; gaussian std sigma_star = w/sigma_factor)
    Q_star = AR^2 / Fo   (DERIVED — falls out of the T_c choice; verified numerically)

Dimensionless system:
    u_t* = Fo*u_x*x*  +  (Fo/AR^2)*u_yhat_yhat
    u_yhat|_{yhat=0} = -Q_star * g_hat(x* - x_b*(t*))
    u_n = 0 on other edges ;  u(.,.,0)=0  (hard-enforced via *t*)
    dT = T_c * u
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PiGroups:
    """The three independent dimensionless groups for one problem instance."""

    Fo: float
    AR: float
    w: float

    def __post_init__(self) -> None:
        # Domain invariants — fail loud if a caller builds a non-physical instance.
        if not self.Fo > 0.0:
            raise ValueError(f"Fo must be > 0, got {self.Fo}")
        if not 0.0 < self.AR:
            raise ValueError(f"AR must be > 0, got {self.AR}")
        if not 0.0 < self.w < 1.0:
            raise ValueError(f"w (=S/l) must be in (0,1), got {self.w}")

    @property
    def Q_star(self) -> float:
        """Dimensionless flux coefficient — derived, not free."""
        return self.AR**2 / self.Fo

    def sigma_star(self, sigma_factor: float) -> float:
        """Gaussian std of the contact profile in x* units."""
        return self.w / sigma_factor


def sample_pi_groups(cfg: dict, rng: np.random.Generator) -> PiGroups:
    """Sample one instance: Fo log-uniform, AR and w uniform."""
    p = cfg["physics"]
    log_fo = rng.uniform(math.log(p["Fo_min"]), math.log(p["Fo_max"]))
    Fo = math.exp(log_fo)
    AR = rng.uniform(p["AR_min"], p["AR_max"])
    w = rng.uniform(p["w_min"], p["w_max"])
    return PiGroups(Fo=Fo, AR=AR, w=w)


def normalize(pi: PiGroups, cfg: dict) -> tuple[float, float, float]:
    """Map pi-groups to [0,1]^3 for the branch network (Fo in log space)."""
    p = cfg["physics"]
    log_fo_n = (math.log(pi.Fo) - math.log(p["Fo_min"])) / (
        math.log(p["Fo_max"]) - math.log(p["Fo_min"])
    )
    ar_n = (pi.AR - p["AR_min"]) / (p["AR_max"] - p["AR_min"])
    w_n = (pi.w - p["w_min"]) / (p["w_max"] - p["w_min"])
    return (log_fo_n, ar_n, w_n)


def reconstruct_T_c(P: float, t_end: float, rho_c: float, l: float, h: float) -> float:
    """Characteristic temperature: T_c = P*t_end / (rho_c*l*h)."""
    return P * t_end / (rho_c * l * h)


def to_physical(u: np.ndarray, T_c: float) -> np.ndarray:
    """dT = T_c * u."""
    return T_c * u
