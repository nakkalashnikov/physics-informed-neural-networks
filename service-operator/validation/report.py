"""
Validation reporting: a pi-space L2 sweep vs the exact Fourier reference (offline) and an
acceptance card (green/yellow/red) combining the offline L2 with the intrinsic checks.
"""

from __future__ import annotations

import numpy as np

from core.analytic import fourier_field_u
from core.nondim import sample_pi_groups
from core.trajectory import sample_trajectory
from validation.intrinsic import energy_balance, predict_field


def pi_table_sweep(model, cfg: dict, device, n_cases: int | None = None,
                   nx: int = 40, ny: int = 30, seed: int = 12345) -> dict:
    """Relative-L2 of the model vs exact Fourier across random (pi, trajectory) cases.
    Reports median / p90 / max (worst-case matters — never report mean only)."""
    n = int(cfg["validation"]["pi_table_n_cases"]) if n_cases is None else n_cases
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, nx); y = np.linspace(0, 1, ny); t = np.array([0.25, 0.5, 0.75, 1.0])
    errs = []
    for _ in range(n):
        pi = sample_pi_groups(cfg, rng)
        traj = sample_trajectory(cfg, rng)
        u_pred = predict_field(model, traj, pi, cfg, x, y, t, device)
        u_true = fourier_field_u(x, y, t, traj, pi, cfg)
        errs.append(np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-12))
    errs = np.array(errs)
    return {"median": float(np.median(errs)), "p90": float(np.percentile(errs, 90)),
            "max": float(errs.max()), "errors": errs}


def _tier(value: float, green: float, yellow: float) -> str:
    return "green" if value < green else ("yellow" if value < yellow else "red")


def acceptance_card(model, cfg: dict, device, n_cases: int | None = None) -> dict:
    """Combine offline L2 sweep + intrinsic energy balance into a green/yellow/red card."""
    th = cfg["validation"]["thresholds"]
    sweep = pi_table_sweep(model, cfg, device, n_cases=n_cases)

    # intrinsic energy balance on a few held-out cases
    rng = np.random.default_rng(999)
    x = np.linspace(0, 1, 40); y = np.linspace(0, 1, 30); t = np.array([0.5, 1.0])
    en = []
    for _ in range(4):
        pi = sample_pi_groups(cfg, rng)
        traj = sample_trajectory(cfg, rng)
        u = predict_field(model, traj, pi, cfg, x, y, t, device)
        en.append(energy_balance(u, x, y, t).max())
    energy_max = float(np.max(en))

    card = {
        "l2_p90": sweep["p90"],
        "l2_max": sweep["max"],
        "l2_tier": _tier(sweep["p90"], th["l2_green"], th["l2_yellow"]),
        "energy_max": energy_max,
        "energy_tier": _tier(energy_max, th["energy_green"], th["energy_yellow"]),
    }
    tiers = [card["l2_tier"], card["energy_tier"]]
    card["verdict"] = "red" if "red" in tiers else ("yellow" if "yellow" in tiers else "green")
    return card
