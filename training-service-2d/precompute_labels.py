"""
Precompute the analytic Fourier labels (the only CPU-heavy part) into a cache file.

Run this once on the Mac (uses all CPU cores); GPU training then reads the cache and generates
the cheap PDE/BC collocation points on the fly — no Fourier evaluation during training.

    python precompute_labels.py                 # uses config cache.n_cases
    python precompute_labels.py --n 20000 --out labels.npz --workers 8

Each cached case stores: pi-groups [Fo,AR,w], branch vector [101 traj samples ++ 3 pi-norms],
data point coords (x*,y_hat,t*), exact analytic labels u, and the per-trajectory loss scales.
The trajectory itself is recoverable from the 101 branch samples (cubic spline) for the BC source.
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import yaml

from analytic import fourier_labels_u
from nondim import normalize, sample_pi_groups
from trajectory import sample_at_nodes, sample_trajectory


def _gen_chunk(args) -> dict:
    """Worker: generate `n` cases with an independent RNG seed. Spawn-safe (top-level)."""
    cfg, seed, n = args
    rng = np.random.default_rng(seed)
    k = int(cfg["trajectory"]["k_sensors"])
    nd = int(cfg["cache"]["n_data_per_case"])
    M = int(cfg["cache"]["label_M"]); N = int(cfg["cache"]["label_N"])
    ntau = int(cfg["cache"]["label_ntau"])
    sf = float(cfg["physics"]["sigma_factor"])

    pi3 = np.empty((n, 3), np.float32)
    branch = np.empty((n, k + 3), np.float32)
    dcoords = np.empty((n, nd, 3), np.float32)
    du = np.empty((n, nd), np.float32)
    scale = np.empty(n, np.float32)
    fluxscale = np.empty(n, np.float32)

    for i in range(n):
        pi = sample_pi_groups(cfg, rng)
        traj = sample_trajectory(cfg, rng)
        samples = sample_at_nodes(traj, k)
        branch[i] = np.concatenate([samples, np.array(normalize(pi, cfg))])
        dc = rng.random((nd, 3))
        u = fourier_labels_u(dc, traj, pi, cfg, M=M, N=N, ntau=ntau)
        dcoords[i] = dc
        du[i] = u
        pi3[i] = (pi.Fo, pi.AR, pi.w)
        scale[i] = max(float(np.sqrt(np.mean(u**2))), 1e-3)
        sigma_star = pi.w / sf
        fluxscale[i] = max(pi.Q_star / (sigma_star * np.sqrt(2.0 * np.pi)), 1e-3)

    return {"pi3": pi3, "branch": branch, "dcoords": dcoords,
            "du": du, "scale": scale, "fluxscale": fluxscale}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=None, help="override cache.n_cases")
    ap.add_argument("--out", default=None, help="override cache.path")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--linear", action="store_true",
                    help="restrict source to straight-line constant-velocity trajectories")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.linear:
        cfg["trajectory"]["linear"] = True
    n_total = int(args.n if args.n is not None else cfg["cache"]["n_cases"])
    out = args.out or cfg["cache"]["path"]
    workers = max(1, args.workers)

    # split into one chunk per worker, each with a distinct seed
    base = n_total // workers
    sizes = [base + (1 if i < n_total % workers else 0) for i in range(workers)]
    tasks = [(cfg, 1000 + i, s) for i, s in enumerate(sizes) if s > 0]

    print(f"precomputing {n_total} cases on {workers} workers -> {out}")
    t0 = time.time()
    with Pool(workers) as pool:
        parts = pool.map(_gen_chunk, tasks)

    merged = {key: np.concatenate([p[key] for p in parts], axis=0) for key in parts[0]}
    np.savez_compressed(out, **merged)
    dt = time.time() - t0
    mb = sum(v.nbytes for v in merged.values()) / 1e6
    print(f"done: {merged['branch'].shape[0]} cases, {mb:.1f} MB raw, {dt:.1f}s "
          f"({n_total / dt:.0f} cases/s)")


if __name__ == "__main__":
    main()
