"""
Training batch construction for the hybrid loss.

Each step samples n_traj_per_batch trajectories. For every trajectory we draw:
  - interior collocation points (PDE residual),
  - boundary points at yhat=0 (moving-flux residual),
  - insulated-edge points (x*=0, x*=1, yhat=1),
  - data points with exact analytic labels u (data loss).
All groups carry per-point branch vectors and per-point pi-group tensors, then are concatenated
across trajectories. A background prefetcher hides the analytic-label cost on CUDA.

Returns numpy arrays; the trainer converts to torch (coords need requires_grad for autograd).
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np

from analytic import fourier_labels_u
from nondim import PiGroups, normalize, sample_pi_groups
from trajectory import sample_at_nodes, sample_trajectory, spline_interp


@dataclass
class Batch:
    # interior
    int_coords: np.ndarray      # (Ni,3)
    int_branch: np.ndarray      # (Ni,k+3)
    int_Fo: np.ndarray          # (Ni,1)
    int_AR: np.ndarray          # (Ni,1)
    # flux boundary yhat=0
    bc0_coords: np.ndarray
    bc0_branch: np.ndarray
    bc0_Q: np.ndarray
    bc0_xb: np.ndarray          # source center xb*(t*) at each bc0 point
    bc0_w: np.ndarray
    # insulated edges, grouped by normal axis (0 = x-edges, 1 = yhat=1 edge)
    insx_coords: np.ndarray
    insx_branch: np.ndarray
    insy_coords: np.ndarray
    insy_branch: np.ndarray
    # data (analytic labels)
    data_coords: np.ndarray
    data_branch: np.ndarray
    data_u: np.ndarray          # (Nd,1)
    # per-point relative-loss scale (per-trajectory RMS of labels); Q*=AR^2/Fo spans ~7 orders,
    # so without this the loudest samples dominate the loss (the 1D 68%-ceiling pathology)
    int_scale: np.ndarray
    insx_scale: np.ndarray
    insy_scale: np.ndarray
    data_scale: np.ndarray
    # flux-BC residual u_y + Q*ghat has natural magnitude Q**peak(ghat), NOT the solution scale
    bc0_fluxscale: np.ndarray


def _branch_vector(traj_fn, pi: PiGroups, cfg: dict) -> np.ndarray:
    k = int(cfg["trajectory"]["k_sensors"])
    samples = sample_at_nodes(traj_fn, k)              # (k,)
    pi_n = np.array(normalize(pi, cfg))                # (3,)
    return np.concatenate([samples, pi_n]).astype(np.float64)


def build_batch(cfg: dict, rng: np.random.Generator) -> Batch:
    n_traj = int(cfg["batch"]["n_traj_per_batch"])
    n_int = int(cfg["batch"]["n_interior"])
    n_bc0 = int(cfg["batch"]["n_bc0"])
    n_ins = int(cfg["batch"]["n_bc_ins"])
    n_data = int(cfg["batch"]["n_data"])
    sf = float(cfg["physics"]["sigma_factor"])

    acc: dict[str, list] = {k: [] for k in (
        "int_coords", "int_branch", "int_Fo", "int_AR",
        "bc0_coords", "bc0_branch", "bc0_Q", "bc0_xb", "bc0_w",
        "insx_coords", "insx_branch", "insy_coords", "insy_branch",
        "data_coords", "data_branch", "data_u",
        "int_scale", "insx_scale", "insy_scale", "data_scale", "bc0_fluxscale",
    )}

    n_insx = n_ins // 2          # split between x*=0 and x*=1
    n_insy = n_ins - n_insx      # yhat=1 edge

    for _ in range(n_traj):
        pi = sample_pi_groups(cfg, rng)
        traj = sample_trajectory(cfg, rng)
        bvec = _branch_vector(traj, pi, cfg)

        def tile(n):  # repeat the branch vector across n points
            return np.tile(bvec, (n, 1))

        # interior
        ic = rng.random((n_int, 3))
        acc["int_coords"].append(ic)
        acc["int_branch"].append(tile(n_int))
        acc["int_Fo"].append(np.full((n_int, 1), pi.Fo))
        acc["int_AR"].append(np.full((n_int, 1), pi.AR))

        # flux boundary yhat=0
        bc0 = rng.random((n_bc0, 3)); bc0[:, 1] = 0.0
        acc["bc0_coords"].append(bc0)
        acc["bc0_branch"].append(tile(n_bc0))
        acc["bc0_Q"].append(np.full((n_bc0, 1), pi.Q_star))
        acc["bc0_xb"].append(traj(bc0[:, 2]).reshape(-1, 1))
        acc["bc0_w"].append(np.full((n_bc0, 1), pi.w))

        # insulated x-edges (half at x*=0, half at x*=1)
        ix = rng.random((n_insx, 3))
        ix[: n_insx // 2, 0] = 0.0
        ix[n_insx // 2:, 0] = 1.0
        acc["insx_coords"].append(ix)
        acc["insx_branch"].append(tile(n_insx))
        # insulated yhat=1 edge
        iy = rng.random((n_insy, 3)); iy[:, 1] = 1.0
        acc["insy_coords"].append(iy)
        acc["insy_branch"].append(tile(n_insy))

        # data labels (exact analytic u)
        dc = rng.random((n_data, 3))
        du = fourier_labels_u(dc, traj, pi, cfg).reshape(-1, 1)
        acc["data_coords"].append(dc)
        acc["data_branch"].append(tile(n_data))
        acc["data_u"].append(du)

        # per-trajectory relative-loss scale (RMS of labels, floored)
        scale = max(float(np.sqrt(np.mean(du**2))), 1e-3)
        acc["int_scale"].append(np.full((n_int, 1), scale))
        acc["insx_scale"].append(np.full((n_insx, 1), scale))
        acc["insy_scale"].append(np.full((n_insy, 1), scale))
        acc["data_scale"].append(np.full((n_data, 1), scale))
        # flux-BC normalizer = peak of Q*ghat = Q* / (sigma_star*sqrt(2pi)), sigma_star = w/sf
        sigma_star = pi.w / sf
        fluxscale = max(pi.Q_star / (sigma_star * np.sqrt(2.0 * np.pi)), 1e-3)
        acc["bc0_fluxscale"].append(np.full((n_bc0, 1), fluxscale))

    cat = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in acc.items()}
    return Batch(**cat)


class CachedBatcher:
    """Build training batches from precomputed analytic labels (precompute_labels.py).

    Pulls data labels from the cache; generates PDE/BC collocation points fresh each call
    (cheap — no Fourier). The trajectory is rebuilt from the cached 101 branch samples via a
    cubic spline, used for the moving-flux source position xb*(t*). Drop-in for the trainer:
    exposes .get() like BatchPrefetcher.
    """

    def __init__(self, cfg: dict, seed: int, path: str | None = None):
        self._cfg = cfg
        self._rng = np.random.default_rng(seed)
        npz = np.load(path or cfg["cache"]["path"])
        self._pi3 = npz["pi3"]
        self._branch = npz["branch"]
        self._dcoords = npz["dcoords"]
        self._du = npz["du"]
        self._scale = npz["scale"]
        self._fluxscale = npz["fluxscale"]
        self._k = int(cfg["trajectory"]["k_sensors"])
        self._t_nodes = np.linspace(0.0, 1.0, self._k)

    def get(self) -> Batch:
        cfg, rng = self._cfg, self._rng
        n_traj = int(cfg["batch"]["n_traj_per_batch"])
        n_int = int(cfg["batch"]["n_interior"])
        n_bc0 = int(cfg["batch"]["n_bc0"])
        n_ins = int(cfg["batch"]["n_bc_ins"])
        n_data = int(cfg["batch"]["n_data"])
        n_insx = n_ins // 2
        n_insy = n_ins - n_insx
        nd_cache = self._du.shape[1]

        idxs = rng.integers(0, self._branch.shape[0], size=n_traj)
        acc: dict[str, list] = {k: [] for k in (
            "int_coords", "int_branch", "int_Fo", "int_AR",
            "bc0_coords", "bc0_branch", "bc0_Q", "bc0_xb", "bc0_w",
            "insx_coords", "insx_branch", "insy_coords", "insy_branch",
            "data_coords", "data_branch", "data_u",
            "int_scale", "insx_scale", "insy_scale", "data_scale", "bc0_fluxscale",
        )}
        for idx in idxs:
            bvec = self._branch[idx]
            Fo, AR, w = (float(v) for v in self._pi3[idx])
            Q = AR**2 / Fo
            scale = float(self._scale[idx])
            flux = float(self._fluxscale[idx])
            spline = spline_interp(self._t_nodes, bvec[: self._k])

            def tile(n):
                return np.tile(bvec, (n, 1))

            ic = rng.random((n_int, 3))
            acc["int_coords"].append(ic); acc["int_branch"].append(tile(n_int))
            acc["int_Fo"].append(np.full((n_int, 1), Fo)); acc["int_AR"].append(np.full((n_int, 1), AR))
            acc["int_scale"].append(np.full((n_int, 1), scale))

            bc0 = rng.random((n_bc0, 3)); bc0[:, 1] = 0.0
            acc["bc0_coords"].append(bc0); acc["bc0_branch"].append(tile(n_bc0))
            acc["bc0_Q"].append(np.full((n_bc0, 1), Q))
            acc["bc0_xb"].append(spline(bc0[:, 2]).reshape(-1, 1))
            acc["bc0_w"].append(np.full((n_bc0, 1), w))
            acc["bc0_fluxscale"].append(np.full((n_bc0, 1), flux))

            ix = rng.random((n_insx, 3)); ix[: n_insx // 2, 0] = 0.0; ix[n_insx // 2:, 0] = 1.0
            acc["insx_coords"].append(ix); acc["insx_branch"].append(tile(n_insx))
            acc["insx_scale"].append(np.full((n_insx, 1), scale))
            iy = rng.random((n_insy, 3)); iy[:, 1] = 1.0
            acc["insy_coords"].append(iy); acc["insy_branch"].append(tile(n_insy))
            acc["insy_scale"].append(np.full((n_insy, 1), scale))

            sel = rng.choice(nd_cache, size=min(n_data, nd_cache), replace=False)
            acc["data_coords"].append(self._dcoords[idx][sel])
            acc["data_branch"].append(tile(len(sel)))
            acc["data_u"].append(self._du[idx][sel].reshape(-1, 1))
            acc["data_scale"].append(np.full((len(sel), 1), scale))

        cat = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in acc.items()}
        return Batch(**cat)


class BatchPrefetcher:
    """Background thread that builds batches into a queue (numpy only; safe off-main-thread)."""

    def __init__(self, cfg: dict, seed: int, queue_size: int = 4):
        self._cfg = cfg
        self._rng = np.random.default_rng(seed)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(build_batch(self._cfg, self._rng), timeout=1.0)
            except queue.Full:
                continue

    def get(self) -> Batch:
        return self._queue.get()

    def close(self) -> None:
        self._stop.set()
