# Spec: 2D PI-DeepONet Training Throughput

**Status:** ready to implement
**Affects:** training-service-2d — trainer.py, data.py, model.py, config.yaml
**Goal:** raise effective epochs/sec; the 2D run is GPU-starved (~7 it/s, GPU ~98% idle)

---

## Problem

The 200k-step production run did ~7 it/s on an RTX 5070 Ti, yet the model is only ~915k params over ~16k points/step — a few ms of GPU work per step. The step takes ~143 ms, so the GPU sits ~98% idle: the loop is **CPU/pipeline-bound**, not compute-bound. Root cause: when `cache_path` is set the trainer bypasses the prefetcher and builds each batch serially on the main thread (numpy tile/full/concatenate + per-trajectory CubicSpline). Combined with undertraining (only ~32 epochs over 100k cases), slow throughput is the primary blocker to a converged model. A faster GPU would not help until the pipeline is fixed.

## Out of scope

- Any model architecture change (decoder, latent width, trunk) — diagnosed as fine
- Changing the loss formulation, sigma curriculum, or #cases (separate retraining spec)
- Distributed / multi-GPU training
- Inference-service changes

## Solution

- **#1 Prefetch the cache** — wrap `CachedBatcher` in the same background-thread pattern `BatchPrefetcher` already uses, so CPU batch assembly overlaps GPU compute. Highest ROI, trivial.
- **#2 Bigger batch** — make `n_traj_per_batch` / `n_interior` larger and fully config-driven; fills the idle GPU and raises epochs/sec. **Not free for convergence:** a 4× batch lowers gradient noise and changes the effective LR/sigma schedule — scale `lr` (≈√k or linear, tune) and re-check the curriculum; treat as a hyperparameter change, not pure speed.
- **#3 Stop tiling the branch** — pass one branch vector per trajectory + a segment index; `BranchNet` runs once per trajectory, result is gathered to points. The real win is **CPU tile + host→device transfer**; the GPU branch saving is minor because the trunk dominates compute and still runs per-point.
- **#4 BF16 autocast — forward / data-loss ONLY (CUDA only)** — scaler-free `torch.autocast(bfloat16)` around the data-loss forward. **The PDE residual (u_xx, u_yy double-backward) stays in FP32, outside autocast:** BF16 has only 7 mantissa bits (~2 decimal digits), and second derivatives of a sharp Gaussian source in BF16 lose all meaningful precision. This is a precision limit, not just the FP16 death-spiral.
- **#5 torch.compile the forward (CUDA only)** — compile `model.forward` (not the grad graph) and/or set `donated_buffer=False`, guarded so it can't break `create_graph` double-backward.
- **Ordering is load-bearing:** #4/#5 yield ~nothing until #1–#3 remove GPU starvation. Land and measure 1→5 in order.

## Step 0 — Measure first (MANDATORY, before any change)

The "GPU ~98% idle" claim is an inference, never confirmed (GPU was decommissioned). Do not optimise blind — that is the exact failure that cost an 8-hour run. Add instrumentation, then gate every change on it.

- Per-step timers around: (a) `source.get()` (CPU batch build/transfer), (b) forward+loss, (c) backward, (d) `opt.step()`. Log a rolling mean every N steps.
- `nvidia-smi dmon -s u` (or `torch.cuda.utilization()`) sampled during a 200-step warm run → record baseline GPU-util + it/s.
- Record the **baseline numbers in this spec / NOTES** before touching code.
- **STOP-gate between optimisations:** after each landed change, re-measure. If GPU-util > ~80% (compute-bound) → stop pipeline work; skip remaining #3 and go straight to #4/#5. If still starved → continue. Never implement an optimisation whose bottleneck the timers say is already gone.

## Priority / expected gain

| # | Change | Expected gain | Where the win is | Effort | Risk |
|---|---|---|---|---|---|
| 1 | Prefetch CachedBatcher | ×2–5 | overlap CPU build w/ GPU | trivial | low |
| 2 | Bigger batch (configurable) | ×2–4 useful work | fill idle GPU; +epochs/sec | trivial | low (but see LR note) |
| 3 | Branch-once + broadcast | mostly **CPU tile + host→device transfer**; GPU saving is minor (trunk dominates compute, still per-point) | medium | medium (indexing correctness) |
| 4 | BF16 autocast — **forward/data only** (CUDA) | ×1.5–2 *(after 1–3)* | tensor-core matmuls | medium | medium (precision — see below) |
| 5 | torch.compile forward (CUDA) | ×1.3–1.7 *(after 1–3; small model → modest)* | kernel fusion | high | high (create_graph conflict) |

## Files changed

| File | Change |
|---|---|
| `trainer.py` | Always route batches through a prefetcher (incl. cache path); add `bf16`/`compile` toggles around the step; keep non-finite-grad guard; CUDA-gate bf16/compile |
| `data.py` | Generalize `BatchPrefetcher` to wrap any `.get()` source (incl. `CachedBatcher`); add per-trajectory branch + segment index to `Batch` (no full tile); keep numpy-only off-main-thread |
| `model.py` | `forward` accepts either tiled branch (back-comp) or `(branch_unique, seg_idx)`; compute `b = branch(branch_unique)[seg_idx]`; optional `@compile`-friendly split |
| `config.yaml` | `batch.n_traj_per_batch`, `batch.n_interior` (raise + document); add `training.use_bf16`, confirm `training.use_compile` honored |

## Branch-broadcast contract (#3)

```python
# Batch (new fields; replaces per-group *_branch tile)
branch_unique : (n_traj, k+3)     # one vector per trajectory
int_seg       : (Ni,)  int64      # which trajectory each interior point belongs to
bc0_seg, insx_seg, insy_seg, data_seg : likewise

# model.forward(branch_unique, coords, seg_idx):
b   = self.branch(branch_unique)      # (n_traj, p)
b   = b[seg_idx]                       # (N, p) gathered, no recompute
tau = self.trunk(coords)               # (N, p)
u   = (b * tau).sum(-1, keepdim=True) + bias) * t_star
```
- Back-compat: if `seg_idx is None`, treat `branch_unique` as already per-point (old path) so `validation/intrinsic.predict_field` keeps working unchanged.

## BF16 precision contract (#4)

```python
# data-loss forward: BF16 OK (matmul-heavy, no derivatives)
with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
    u_pred = model(branch, data_coords)
    L_data = (((u_pred - du) / ds) ** 2).mean()

# PDE / BC residuals: FP32, autocast DISABLED — 2nd derivatives need precision
with torch.autocast("cuda", enabled=False):
    R   = pde_residual(model, ib, ic_fp32, Fo, AR)   # u_xx, u_yy in fp32
    L_pde = ((R / isc) ** 2).mean()
    ...bc terms likewise...
```
- Rationale: BF16 = 7 mantissa bits. `u_xx ≈ ∂²u/∂x²` of a Gaussian with σ*≈w/6 (down to ~0.003) has huge curvature; BF16 rounding destroys it. Range was never the issue — precision is.
- `use_bf16` is CUDA-only; on CPU/MPS the autocast context is a no-op (`enabled=False`).

## Failure modes

| Failure | Detected by | Handling |
|---|---|---|
| Prefetch thread dies (exception in batch build) | empty queue / join timeout | log + fall back to synchronous build for that step |
| BF16 leaks into the residual (autocast not disabled there) → garbage 2nd derivatives | timers/loss diverge; unit test residual==fp32 | residual forward MUST be under `autocast(enabled=False)`; assert dtype fp32 in test |
| torch.compile errors on `create_graph` (donated buffers) | exception at first step | catch, log, run eager (no compile) |
| Bigger batch → OOM on smaller GPU | CUDA OOM | config-driven; document VRAM per setting; keep 16/1024 default safe |
| seg-index gather mismatch (wrong branch per point) | unit test vs tiled reference | assert equality in test before trusting |
| MPS/CPU run hits a CUDA-only path | device check | bf16/compile no-op off CUDA; prefetch + batch + branch-broadcast device-agnostic |

## Test plan

- [ ] **Equivalence (#3):** branch-broadcast forward == old tiled forward, same inputs, `allclose` (rtol 1e-5)
- [ ] **Prefetch (#1):** cached path yields identical batches with/without prefetch (seeded); queue stays non-empty under load
- [ ] **Throughput:** it/s before vs after #1–#3 on CUDA (and MPS sanity); assert ≥2× on #1 alone
- [ ] **BF16 (#4):** residual forward runs in **fp32** (assert dtype); data-loss in bf16; loss curve tracks full-FP32 within noise; non-finite-skip rate ≈ 0
- [ ] **Batch/LR (#2):** after raising batch, a short run with scaled vs unscaled `lr` — confirm scaled LR converges no worse than the 16/1024 baseline (guards against silent convergence regression)
- [ ] **Compile (#5):** first-step compile succeeds OR cleanly falls back to eager; numerics match eager
- [ ] **Back-compat:** `validation/intrinsic.predict_field` and `validate_checkpoint.py` run unchanged (old per-point branch path)
- [ ] **Device matrix:** CPU + MPS runs still train (bf16/compile auto-off); CUDA uses all paths
- [ ] **Guard intact:** inject a NaN grad → step skipped, training continues

---

**Anything missing or wrong before I start implementing?**
