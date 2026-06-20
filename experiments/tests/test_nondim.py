"""Tests for nondim.py — pi-group invariants, Q_star identity, T_c round-trip."""

import math

import numpy as np
import pytest

from nondim import (
    PiGroups,
    normalize,
    reconstruct_T_c,
    sample_pi_groups,
    to_physical,
)

CFG = {
    "physics": {
        "Fo_min": 3.0e-6, "Fo_max": 3.0e-2,
        "AR_min": 0.02, "AR_max": 0.30,
        "w_min": 0.02, "w_max": 0.20,
    }
}


def test_q_star_identity():
    pi = PiGroups(Fo=1e-3, AR=0.1, w=0.05)
    assert pi.Q_star == pytest.approx(0.1**2 / 1e-3)


def test_sigma_star():
    pi = PiGroups(Fo=1e-3, AR=0.1, w=0.06)
    assert pi.sigma_star(6.0) == pytest.approx(0.01)


def test_invariants_reject_bad():
    with pytest.raises(ValueError):
        PiGroups(Fo=-1.0, AR=0.1, w=0.05)
    with pytest.raises(ValueError):
        PiGroups(Fo=1e-3, AR=0.1, w=1.5)
    with pytest.raises(ValueError):
        PiGroups(Fo=1e-3, AR=-0.1, w=0.05)


def test_sample_in_range():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        pi = sample_pi_groups(CFG, rng)
        assert CFG["physics"]["Fo_min"] <= pi.Fo <= CFG["physics"]["Fo_max"]
        assert CFG["physics"]["AR_min"] <= pi.AR <= CFG["physics"]["AR_max"]
        assert CFG["physics"]["w_min"] <= pi.w <= CFG["physics"]["w_max"]


def test_normalize_endpoints():
    lo = PiGroups(Fo=3.0e-6, AR=0.02, w=0.02)
    hi = PiGroups(Fo=3.0e-2, AR=0.30, w=0.20)
    assert normalize(lo, CFG) == pytest.approx((0.0, 0.0, 0.0))
    assert normalize(hi, CFG) == pytest.approx((1.0, 1.0, 1.0))


def test_normalize_fo_is_log():
    mid_fo = math.sqrt(3.0e-6 * 3.0e-2)  # geometric mean -> 0.5 in log space
    pi = PiGroups(Fo=mid_fo, AR=0.16, w=0.11)
    log_fo_n, _, _ = normalize(pi, CFG)
    assert log_fo_n == pytest.approx(0.5)


def test_T_c_roundtrip():
    T_c = reconstruct_T_c(P=2.0, t_end=3.0, rho_c=4.0, l=0.5, h=0.1)
    assert T_c == pytest.approx(2.0 * 3.0 / (4.0 * 0.5 * 0.1))
    u = np.array([0.0, 0.5, 1.0])
    assert to_physical(u, T_c) == pytest.approx(T_c * u)
