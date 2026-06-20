"""Generate (measurements, material, true source params) tuples for the inverse PI-DeepONet.

Per case: sample pi-groups (Fo, AR, w) + a linear source (x0, v); evaluate the exact analytic
temperature at K fixed thermocouples over M times -> the measurement vector. Inputs to the inverse
net are [measurements ; normalized (Fo, AR, w)]; the target is the source trajectory, which for the
linear family is fully given by (x0, v). Saves train/val splits to npz (cached) for training.

    python inverse_data.py --config config_inverse.yaml --workers 8
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import yaml

from analytic import fourier_labels_u
from nondim import normalize, sample_pi_groups
from trajectory import sample_trajectory


def sensor_xy(cfg: dict) -> np.ndarray:
    """Fixed thermocouple positions (x_k, ŷ_k): a row spread along x near the bottom."""
    iv = cfg["inverse"]
    K = int(iv["n_sensors"])
    xs = np.linspace(float(iv["sensor_x_lo"]), float(iv["sensor_x_hi"]), K)
    ys = np.full(K, float(iv["sensor_y"]))
    return np.stack([xs, ys], axis=1)                       # (K, 2)


def _gen_chunk(args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg, seed, n = args
    rng = np.random.default_rng(seed)
    iv = cfg["inverse"]
    K, M, Q = int(iv["n_sensors"]), int(iv["n_times"]), int(iv["n_query"])
    margin = float(cfg["trajectory"]["x_margin"])
    arbitrary = bool(cfg["trajectory"].get("arbitrary", False))
    A, B = int(cfg["analytic"]["fourier_M"]), int(cfg["analytic"]["fourier_N"])
    ntau = int(cfg["analytic"]["quad_nt"])

    sensors = sensor_xy(cfg)
    t_s = np.linspace(0.0, 1.0, M)
    t_q = np.linspace(0.0, 1.0, Q)                          # trunk query times (supervision grid)
    xy = np.repeat(sensors, M, axis=0)                      # (K*M, 2)
    tt = np.tile(t_s, K)[:, None]                           # (K*M, 1)
    q = np.concatenate([xy, tt], axis=1)                    # (K*M, 3) = [x, ŷ, t]

    meas = np.empty((n, K * M), np.float32)
    mat = np.empty((n, 3), np.float32)
    traj_q = np.empty((n, Q), np.float32)                  # source position xb(t) at the Q query times
    for i in range(n):
        pi = sample_pi_groups(cfg, rng)
        if arbitrary:
            traj = sample_trajectory(cfg, rng)             # random smooth Fourier trajectory
        else:
            x0, x1 = rng.uniform(margin, 1.0 - margin, size=2)
            traj = lambda t, a=x0, b=x1 - x0: a + b * np.atleast_1d(np.asarray(t, float))
        meas[i] = fourier_labels_u(q, traj, pi, cfg, M=A, N=B, ntau=ntau)
        mat[i] = normalize(pi, cfg)
        traj_q[i] = np.asarray(traj(t_q)).reshape(-1)
    return meas, mat, traj_q


def build(cfg: dict, n: int, seed: int, workers: int):
    base = n // workers
    sizes = [base + (1 if i < n % workers else 0) for i in range(workers)]
    tasks = [(cfg, 7000 + seed * 97 + i, s) for i, s in enumerate(sizes) if s > 0]
    with Pool(workers) as pool:
        parts = pool.map(_gen_chunk, tasks)
    return tuple(np.concatenate([p[j] for p in parts], axis=0) for j in range(3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_inverse.yaml")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    sensors = sensor_xy(cfg)
    t_q = np.linspace(0.0, 1.0, int(cfg["inverse"]["n_query"]))
    for split, n, seed in [("train", cfg["data"]["n_train"], 0), ("val", cfg["data"]["n_val"], 999)]:
        t0 = time.time()
        meas, mat, traj_q = build(cfg, int(n), seed, max(1, args.workers))
        out = cfg["data"]["cache"].replace(".npz", f"_{split}.npz")
        np.savez_compressed(out, meas=meas, mat=mat, traj_q=traj_q, sensors=sensors, t_q=t_q)
        print(f"{split}: {meas.shape[0]} cases, K*M={meas.shape[1]}, Q={traj_q.shape[1]} -> {out}  "
              f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
