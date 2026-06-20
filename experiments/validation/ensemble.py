"""Tier-3 uncertainty: deep ensemble -> mean +/- std (the deployment confidence signal)."""

from __future__ import annotations

import numpy as np
import torch

from validation.intrinsic import predict_field


def predict_ensemble(models: list, traj_fn, pi, cfg: dict,
                     x: np.ndarray, y: np.ndarray, t: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) fields u[t,y,x] over an ensemble of independently-seeded models."""
    fields = np.stack([predict_field(m, traj_fn, pi, cfg, x, y, t, device) for m in models], axis=0)
    return fields.mean(axis=0), fields.std(axis=0)
