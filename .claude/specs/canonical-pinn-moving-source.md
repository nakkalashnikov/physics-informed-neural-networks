# Spec: Canonical single-instance PINN — 2D moving heat source

**Status:** ready to implement
**New service dir:** `training-service-pinn/`
**Goal:** a guaranteed-working forward PINN for ONE fixed instance of the 2D moving-source heat problem (constant-velocity "soldering iron"), validated against the exact analytic field. Deadline-safe deliverable; reuses the verified analytic/physics stack from `training-service-2d`.

---

## Problem

The PI-DeepONet (operator over arbitrary trajectories) plateaus and its multi-case transfer is unproven (session finding: on 64 cases multi-scale only matches the single-σ baseline — the operator's bottleneck is generalisation across trajectories, not single-field representation). For the final project we need a reliable result. A canonical PINN drops the operator: it solves a SINGLE instance (one trajectory + one pi-group set) with a pure trunk network `u(x*,ŷ,t*)` — exactly the regime where the multi-scale fix DOES work (1-case fits a sharp-source field to ~12–23% rel-L2). This service makes that a clean, plotted deliverable via a controlled source-sharpness sweep.

## Out of scope

- Operator learning / generalisation to unseen trajectories (that is the DeepONet, `training-service-2d`).
- Inverse problems, 3D, temperature-dependent material properties.
- New physics — PDE/BC/analytic are reused unchanged.
- Trajectory shapes other than constant velocity (the framework accepts any `t→xb`, but the deliverable fixes a straight line).

## Solution

- **Pure-trunk PINN:** `u = trunk(x*,ŷ,t*) · t*` (scalar out, hard IC via ×t*). Drop the DeepONet branch + dot product entirely.
- **Multi-scale Fourier trunk** (session finding): RFF bands `[2,6,12]` + 3 PirateNet blocks. Single-σ kept as an ablation baseline for the report.
- **Controlled study, not scattered cases:** straight trajectory `xb*(t*) = x0 + v·t*`; hold `(Fo, AR, x0, v)` fixed and **sweep only source width `w`** (wide→razor) × `{single-σ, multi-scale}` (table below). One difficulty axis, one figure.
- **Two clean, non-conflicting loss recipes** (NEVER mix data-labels with the flux BC — they disagree at ŷ=0, see [[2d-source-concentrated-sampling]] / check_consistency):
  - **Hybrid-robust (primary, proven):** `λ_data·L_data + λ_pde·L_pde + λ_bc·L_ins` — source carried by sparse analytic anchors; NO flux BC.
  - **Pure-physics (ablation/headline-PINN):** `λ_pde·L_pde + λ_flux·L_flux + λ_bc·L_ins` — source carried by the flux BC; NO data.
- **Validate** each instance against the exact analytic field (`analytic.fourier_field_u`): relative L2, energy, peak-match; field + error heatmaps; loss curves.

## Controlled source-sharpness sweep (the headline study)

Hold block, material, and motion FIXED; vary ONLY the source width `w` (contact patch) wide→razor. `Q*=AR²/Fo` is w-independent and `ĝ` is normalised to `∫=1`, so **total heat is constant across the sweep** — it isolates SHARPNESS (spectral content) from energy, directly probing spectral bias. Cleaner than 4 scattered instances: one controlled axis, one figure.

**Fixed across the sweep:** `Fo=3.0e-3, AR=0.15, x0=0.10, v=0.8` → `Q*=7.5`, source travels 0.10→0.90, all within `[0.05,0.95]`, `|v|≤speed_max_star(=4)`.

| w | σ*=w/6 | peak flux `Q*/(σ*√2π)` | character |
|---|---|---|---|
| 0.18 | 0.030 | ~100 | broad — easy |
| 0.12 | 0.020 | ~150 | mild |
| 0.08 | 0.013 | ~224 | moderate |
| 0.05 | 0.0083 | ~360 | sharp |
| 0.03 | 0.0050 | ~599 | razor — single-σ expected to fail |

**Each w trained TWICE** — `single` (one band `[3.0]`) and `multi` (`[2,6,12]`) → **5×2 = 10 runs**. Headline figure: rel-L2 vs w for both archs. Expected **crossover** (single-σ climbs as w→0, multi-scale stays flat) = the project's central result, the physical-parameter analogue of the session's σ-sweep. (w-list and point-count adjustable; 3-5 enough for the curve.)

## Module layout

| file | purpose | reuse from training-service-2d |
|---|---|---|
| `model_pinn.py` | `TrunkPINN`: MultiScaleFourierFeatures + PirateBlock + Linear(width,1), ×t* | copy `MultiScaleFourierFeatures`, `PirateBlock`, `FourierFeatures` |
| `physics.py` | PDE/flux/insulation residuals | **import/copy unchanged** |
| `analytic.py` | exact field (validation + optional anchors) | **import/copy unchanged** |
| `instance.py` | `linear_traj(x0,v,margin)`, `Instance` dataclass (Fo,AR,w,x0,v) | trajectory clip logic from `trajectory.py` |
| `data_pinn.py` | per-step collocation sampling for ONE instance (interior/bc0/ins + optional anchors) | adapt `_mix_x` source-concentrated sampler |
| `train_pinn.py` | single-run loop (one w, one arch): sample → loss → Adam(warmup+cosine) → grad-clip + nan-guard | adapt `trainer.py` (drop case batching) |
| `sweep_pinn.py` | driver: loop w × {single,multi}, train each, collect rel-L2, plot the headline curve | new |
| `validate_pinn.py` | rel-L2 / energy / peak vs analytic; field+error PNGs | adapt `validate_checkpoint.py` |
| `config_pinn.yaml` | sweep def (fixed params + w list + archs) + loss + trunk + training | new |

## Network

```
coords (N,3)=[x*,ŷ,t*]
  -> MultiScaleFourierFeatures(d_in=3, m=64, bands=[2,6,12])   # 384 feats
  -> encoder_u, encoder_v, input_layer (Linear -> width=256, tanh)
  -> 3 × PirateBlock(256)
  -> output_layer Linear(256, 1)
  -> u_raw (N,1)
u = u_raw * t*            # hard IC: u(·,·,0)=0 exactly
```
- σ curriculum = global multiplier on the comb (`rff_sigma_start→end`, ramp_frac); default `0.3→1.0` (smooth start, then full multi-scale).

## Loss (dimensionless, per-instance)

| term | residual | scale (relative) | recipe |
|---|---|---|---|
| `L_data` | `u_pred − u_analytic` at anchor pts | RMS of analytic field | hybrid only |
| `L_pde` | `u_t − Fo·u_xx − (Fo/AR²)·u_yy` (interior) | RMS(u_t) or field RMS | both |
| `L_flux` | `u_ŷ + Q*·ĝ(x*−xb*(t*))` at ŷ=0 | `Q*/(σ*·√2π)` | pure-physics only |
| `L_ins` | `u_n` on x*=0, x*=1, ŷ=1 | field RMS | both |

- Default weights (hybrid): `data=10, pde=1, flux=0, bc=10`. Pure-physics: `data=0, pde=1, flux=10, bc=10`.
- create_graph=True double-backward (u_xx,u_yy) unchanged. grad-clip + non-finite-grad guard kept.

## Collocation sampling (per step, one instance)

- interior `n_int`, bc0 `n_bc0`, insulation `n_ins`, anchors `n_data` (hybrid).
- **Source-concentrated**: `source_frac` of interior + bc0 drawn `x* ~ N(xb*(t*), (band·σ*)²)` clipped to [0,1]; rest uniform (covers the razor source — same fix as the operator service).
- anchors: sample analytic `u` at random (x*,ŷ,t*) each step (cheap Fourier, single trajectory) OR precompute a fixed anchor set.

## CLI / config

```
python train_pinn.py    --w 0.03 --arch multi|single [--recipe hybrid|physics]
                        [--steps N] [--device cuda] [--out ckpt.pt]
python sweep_pinn.py     # runs the full w × arch matrix, saves ckpts, plots rel-L2 vs w
python validate_pinn.py --checkpoint ckpt.pt --w 0.03 --plot
```
`config_pinn.yaml`:
```yaml
sweep:
  fixed:    {Fo: 3.0e-3, AR: 0.15, x0: 0.10, v: 0.8}   # everything but w
  w_values: [0.18, 0.12, 0.08, 0.05, 0.03]             # wide -> razor
  archs:    {single: [3.0], multi: [2.0, 6.0, 12.0]}   # rff_sigma_bands per arch
loss:    {lambda_data: 10, lambda_pde: 1, lambda_flux: 0, lambda_bc: 10}  # hybrid (proven)
trunk:   {rff_num_features: 64, rff_sigma_start: 0.3, rff_sigma_end: 1.0,
          rff_sigma_ramp_frac: 0.5, n_pirate_blocks: 3, width: 256}   # bands from sweep.archs
batch:   {n_interior: 2048, n_bc0: 512, n_bc_ins: 512, n_data: 1024,
          source_frac: 0.5, source_band_sigmas: 3.0}
training:{lr: 1.0e-3, lr_min: 1.0e-5, warmup_steps: 2000, total_steps: 40000,
          grad_clip_norm: 1.0, seed: 0}
physics: {sigma_factor: 6.0}
```
(visits is a non-issue — one instance per run, so 40k steps = 40k visits. 10 runs × ~25 min on a 3060 ≈ 4 h.)

## Outputs / deliverables

- `ckpt_w<w>_<arch>.pt` (model + cfg + final metrics), 10 files.
- per-run report line: `rel-L2 median/p90/max`, energy error, peak-match.
- **HEADLINE figure: `rel-L2 vs w`, two curves (single vs multi)** — the expected crossover is the central result.
- supporting: `pred|true|error` heatmaps at t*∈{0.25,0.5,0.75,1.0} for the sharpest w (both archs, to *see* where single-σ smears the peak); loss curves.
- σ-sweep tradeoff figure (reuse session data) — the mechanism behind the crossover.

## Failure modes

| failure | detected by | handling |
|---|---|---|
| sharp source under-resolved (high rel-L2 near peak) | error heatmap hot at ŷ=0 | multi-scale bands + source-concentrated sampling (already in) |
| pure-physics stalls (data/curvature tradeoff) | rel-L2 plateau, pde floor | switch to hybrid recipe (proven), or raise top band |
| high-σ band instability (pde spikes) | huge pde in log | σ curriculum (0.3→1.0) + grad-clip + nan-guard |
| data⊕flux mixed by mistake → boundary conflict | rel-L2 stuck ~60% | enforce: hybrid has flux=0, physics has data=0 (config presets) |
| trajectory leaves domain | assert in `linear_traj` | clip / validate x0+v ≤ 1−margin at load |

## Test plan

- [ ] Smoke: build TrunkPINN, forward + double-backward (pde_residual) finite, loss.backward() ok.
- [ ] Hard IC: `u(x,y,0)==0` exactly for random x,y.
- [ ] Overfit a mild w (0.12, multi, 40k): rel-L2 median < 15% vs analytic.
- [ ] **Crossover claim:** at the sharpest w (0.03) multi-scale beats single-σ on rel-L2 by a clear margin; at broad w (0.18) they are comparable. (The central result.)
- [ ] Energy: `mean(u)` tracks `t*` (A_00=t* identity) within a few %.
- [ ] Recipe guard: hybrid run has flux weight 0; physics run has data weight 0 (no boundary conflict).
- [ ] Full matrix runs to completion; `rel-L2 vs w` headline figure produced.

## Reuse map (fast path)

1. `cp training-service-2d/{physics,analytic}.py training-service-pinn/` — unchanged.
2. Lift `MultiScaleFourierFeatures`, `FourierFeatures`, `PirateBlock` into `model_pinn.py`; new `TrunkPINN` (out_dim=1, ×t*).
3. `train_pinn.py` = `trainer.py` minus CachedBatcher/branch/case-loop; one fixed instance, anchors from `analytic.fourier_labels_u`.
4. The existing `labels_one.npz`-style 1-case runs are the proof this converges — port their working config (bands [2,6,12], flux off, data on).

---

**Anything missing or wrong before I start implementing?**
