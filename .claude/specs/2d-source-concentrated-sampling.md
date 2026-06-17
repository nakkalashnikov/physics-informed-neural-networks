# Spec: Source-Concentrated Collocation Sampling (2D PI-DeepONet)

**Status:** ready to implement
**Affects:** training-service-2d — data.py, config.yaml
**Goal:** break the ~75–80% relative-L2 plateau by making the physics residual actually "see" the sharp moving Gaussian source.

---

## Problem

The flux BC at `yhat=0` carries a Gaussian source `Q* · g_hat(x* − xb*(t*))` with `sigma_star = w/6 ∈ [0.0033, 0.033]`. Interior/BC collocation points are drawn **uniformly** in `x*` (`data.py` `build_batch` and `CachedBatcher.get`). With 256 bc0 / 1024 interior points across `x*∈[0,1]`, essentially none land within a few `sigma_star` of the moving center `xb*(t*)`, so the residual gradient about the hottest region is ~0 and training plateaus. Literature (arXiv [2508.19847](https://arxiv.org/pdf/2508.19847), [2310.19590](https://arxiv.org/pdf/2310.19590)) names exactly this — uniform sampling vs sharp sources — and prescribes source-region concentration.

## Out of scope

- Data-loss points: come from the fixed 512-point cache per case (`du`/`dcoords`); **not** resampled here.
- Architecture, sigma curriculum, LR, case count — separate, already in flight.
- Source-frame (advected) coordinates — the earlier attempt "hurt slightly" (memory `project_2d_deeponet_diagnosis`). **Open question, NOT this spec:** was that a flawed implementation rather than a dead end? Revisit only after this lands.
- `physics.py` (create_graph double-backward) — untouched.

## Solution

- **Mixture sampling** of the **physics** collocation points (interior + bc0): a fraction `source_frac` drawn **near the source**, the rest uniform (cold domain stays covered).
- For a source point: draw `t* ~ U(0,1)`, compute per-point center `c = xb*(t*)` (build_batch: `traj(t*)`; cache: `spline(t*)` — already reconstructed for `bc0_xb`), then `x* = clip(c + N(0, (source_band_sigmas·sigma_star)²), 0, 1)`. `yhat`: bc0 fixed at 0; interior stays uniform (x* is the razor-sharp axis; yhat decays over the broader diffusion length — concentration there is a later extension).
- **No importance reweighting.** The mixture deliberately makes the loss a source-weighted residual norm — that is the goal; the uniform component prevents the cold field from being ignored. Per-trajectory `scale` (RMS of labels) is a per-trajectory constant, independent of point distribution → **unchanged and still valid**.
- **Vectorized & per-point center** (source moves): `t*` is an array, `c = spline(t*)` is an array — one extra spline eval per group (cache path already does this for `bc0_xb`), so throughput is unaffected.
- `source_frac = 0.0` ⇒ exact current behavior (guarded path) for clean A/B.

## Sampling contract

| Param (config `batch.`) | Meaning | Default |
|---|---|---|
| `source_frac` | fraction of interior & bc0 points concentrated near source | `0.5` |
| `source_band_sigmas` | Gaussian std in units of `sigma_star` (`std = k·w/6`) | `3.0` |

```
n_src = round(source_frac · n)         # n = n_int  or  n_bc0
# --- source points ---
t  = rng.random(n_src)
c  = traj(t)            # build_batch        (per-point center, source moves)
c  = spline(t)          # CachedBatcher
x  = np.clip(c + rng.normal(0, source_band_sigmas*sigma_star, n_src), 0, 1)
# interior: yhat = rng.random(n_src); bc0: yhat = 0 ; t* column = t
# --- uniform points (n - n_src) ---  unchanged
# concatenate, shuffle order irrelevant (per-point branch is identical within a trajectory)
sigma_star = w / sigma_factor          # w = pi.w  (build_batch) / pi3[idx] (cache)
```

## Files changed

| File | Change |
|---|---|
| `data.py` | Add `_mix_sample_x(n, traj_or_spline, sigma_star, cfg, rng)` returning `(x*, yhat, t*)` cols for interior and the `x*,t*` for bc0; call it in `build_batch` and `CachedBatcher.get` for `int_coords` and `bc0_coords`. `bc0_xb` must use the **same** per-point `t*` so center & sample agree. |
| `config.yaml` | `batch.source_frac: 0.5`, `batch.source_band_sigmas: 3.0` (documented). |

## Failure modes

| Failure | Detected by | Handling |
|---|---|---|
| Center near edge → concentrated points pile at clip boundary | coverage test | acceptable (still valid points at the truncated peak); no special-case |
| Source points dominate mean, cold residual silently grows | log pde on uniform vs source subsets (debug) | intended emphasis; uniform fraction guards coverage; tune `source_frac` |
| `bc0_xb` center ≠ x*-sample center (t* mismatch) | equality test `c == traj(t*)` for source rows | single `t*` array feeds both center and `bc0_xb` |
| RNG draw-order change breaks seeded repro of other groups | seeded distribution test | new draws are local; `source_frac=0` keeps the old path exactly |
| `sigma_star` tiny (0.003) → degenerate spike | finite-loss / NaN-guard (already in trainer) | clip handles range; scale normalization unchanged |
| Cache path: `w`/spline unavailable | KeyError at load | `w = pi3[idx][2]`, spline already built — assert present |

## Test plan

- [ ] **Coverage:** `source_frac=0.5` ⇒ ~50% of x* within `±3·source_band_sigmas·sigma_star` of `c`; rest ~uniform (KS test vs U(0,1) on the uniform half).
- [ ] **Center agreement:** for source bc0 rows, `bc0_xb == traj(t*)` (build_batch) / `spline(t*)` (cache), `allclose`.
- [ ] **Back-compat:** `source_frac=0.0` reproduces current uniform distribution (seeded, distributional).
- [ ] **Consistency:** `build_batch` vs `CachedBatcher.get` give statistically matching x* distributions for the same pi/trajectory.
- [ ] **Scaling intact:** per-trajectory `scale`/`fluxscale` unchanged; loss finite; no new NaN-skip.
- [ ] **Throughput:** it/s within noise of current (one extra vectorized spline eval).
- [ ] **Convergence (the real test):** short 2k-case run, `source_frac=0.5` vs `0.0`, compare VAL median at 25k/50k — concentrated should break below the uniform plateau.

---

**Anything missing or wrong before I start implementing?**
