# Spec: Refactor — split the working code into two clean PINN microservices

**Status:** ready to implement
**Goal:** lift the two on-topic, physics-informed deliverables out of the research junkyard
(`training-service-2d`) into two clean, atomic microservices — **canonical single-instance PINN** and
**PI-DeepONet forward operator** — each self-contained (own config + train + `checkpoints/`), sharing a
small `core/`. Both have the PDE in the loss → both are genuinely PINN (on-topic). Then retrain both
cleanly to confirm the services work and reach final numbers.

---

## Problem

Everything works but lives in `training-service-2d`, a research junkyard: the forward operator, the
inverse, dozens of diagnostic scripts and configs, dead-ends. For the final project we want two clean
services, one per genuinely-PINN model, that a reader/grader can run atomically. The shared physics/
analytic building blocks should sit in one `core/`, not be duplicated.

## Out of scope

- The **inverse operator** — drifted to data-only DeepONet (off the PINN topic); kept as committed
  bonus / "future work" inside `experiments/`, not a service.
- Nonlinear physics; deleting `training-service` (old 1D) — left as legacy.
- New modelling — we MOVE and CLEAN working code, we do not rewrite the logic.

## Solution

- **`core/`** — shared building blocks copied verbatim from `training-service-2d`: `physics.py`,
  `analytic.py`, `nondim.py`, `trajectory.py`, `model.py` (multi-scale RFF + PirateNet + DeepONet).
  One copy; both services import it.
- **`service-pinn/`** — canonical single-instance PINN (see [[canonical-pinn-moving-source]]): trunk-only
  `u(x,y,t)·t*`, loss = PDE residual + insulation BC + sparse analytic anchors (physics IS in the loss),
  trains ONE instance → `checkpoints/`. Validated vs the exact analytic field.
- **`service-operator/`** — PI-DeepONet forward operator over linear sources **with VARIED material/
  geometry** (Fo, AR, w sampled, not pinned): branch(traj + pi) ⊗ trunk → field, loss = data + PDE + BC.
  More general (one model for all materials + linear paths), so expect a HIGHER floor than the
  fixed-material 40% — the 7-order Q* spread is a hard generalisation axis (~45–55% likely). A
  `lambda_pde` knob gives the **DeepONet (data-only) vs PI-DeepONet (+physics)** ablation from one service.
- **Rename** `training-service-2d` → `experiments/` (kept: all diagnostics + inverse code + configs).
- Each service **atomic**: `python train.py --config config.yaml` → model in its `checkpoints/`.

## Target structure

```
core/
  __init__.py
  physics.py  analytic.py  nondim.py  trajectory.py  model.py   (copies from training-service-2d)
service-pinn/
  config.yaml            # one instance: Fo,AR,w,x0,v + loss weights + trunk
  model.py               # TrunkPINN: core multi-scale RFF + PirateNet, out_dim=1, ×t* hard IC
  train.py  validate.py
  checkpoints/
service-operator/
  config.yaml            # PI-DeepONet (data + pde + bc)
  config_deeponet.yaml   # ablation: lambda_pde=0 (data-only DeepONet)
  data.py  precompute_labels.py  train.py  validate.py
  checkpoints/
experiments/             # ← renamed training-service-2d (research junkyard + inverse code, kept as-is)
training-service/        # old 1D — legacy, untouched
inference-service/  ui/  # untouched
```

**Core sharing (no path hacks):** `core/` is a proper installable package (`core/pyproject.toml`),
installed once with `pip install -e ./core`. Both services then `import core.physics` etc. like any
library — runs from anywhere, clean, no sys.path shims. `requirements.txt`: one per service (each lists
`-e ../core` + torch/numpy/scipy/pyyaml) so each service is independently installable.

**service-pinn model is NEW (not a copy):** `TrunkPINN` reuses core's `MultiScaleFourierFeatures` +
`PirateBlock` but with `out_dim=1` and the `×t*` hard-IC head — a thin new wrapper, ~30 lines.

## service-pinn contract (canonical PINN)

| item | value |
|---|---|
| network | trunk-only: `core` MultiScaleFourierFeatures + PirateNet → Linear(.,1); `u = raw·t*` (hard IC) |
| loss | `λ_pde·R_pde + λ_bc·(insulation) + λ_data·anchors` (PDE in loss = genuine PINN) |
| instance | fixed `(Fo, AR, w, x0, v)` from config; linear constant-velocity source |
| output | `checkpoints/ckpt_pinn.pt` + rel-L2 / energy vs analytic + field PNGs |
| target | ~12–23% rel-L2 (or better with clean full training) |

## service-operator contract (PI-DeepONet forward)

| item | value |
|---|---|
| network | `core` DeepONet: branch(traj+pi) ⊗ trunk(x,y,t), multi-scale RFF + PirateNet |
| loss | `λ_data·data + λ_pde·pde + λ_bc·bc` (PI-DeepONet); `λ_pde=0` → data-only DeepONet (ablation) |
| data | linear sources, **VARIED material/geometry** (Fo, AR, w sampled); source-concentrated sampling |
| output | `checkpoints/ckpt_operator.pt` + VAL rel-L2 + the DeepONet-vs-PI-DeepONet comparison |
| target | ~45–55% rel-L2 (varied material is harder than the 40% fixed-material run — Q* spans ~7 orders) |

## Migration map

| from `training-service-2d/` | to |
|---|---|
| physics.py, analytic.py, nondim.py, trajectory.py, model.py | `core/` (copy) |
| trainer.py, data.py, precompute_labels.py, config_linear*.yaml | `service-operator/` (cleaned) |
| overfit_one.py + the 1-case / canonical pieces | `service-pinn/` (formalised per its spec) |
| everything else (inverse_*, check_consistency, configs, logs) | stays in `experiments/` (renamed dir) |

## Failure modes

| failure | detected by | handling |
|---|---|---|
| service can't import core | ImportError on run | path shim / PYTHONPATH=core documented in each README |
| stale cross-imports after move | run each train.py | grep imports; point to `core.` |
| checkpoint cfg incompatibility | load test | each service self-describes cfg in its checkpoint |
| forward retrain OOM (big model) | CUDA OOM | n_traj cap + expandable_segments (known) |
| rename breaks paths in experiments/ | run a diagnostic | experiments is legacy; not required to run |

## Retraining plan (after refactor)

```
1. service-pinn:      python train.py --config config.yaml   → confirm service + ~12-23%, field plots
2. service-operator:  precompute_labels.py → train.py        → confirm service + ~40% (full run, capacity floor)
   + train config_deeponet.yaml (λ_pde=0) → the DeepONet-vs-PI-DeepONet ablation number
3. collect: rel-L2 numbers, field/error plots, capacity curve, ablation → report assets
```

## Test plan

- [ ] `core/` imports standalone; both services import from it (no cross-service imports).
- [ ] `service-pinn/train.py` runs end-to-end → checkpoint + rel-L2 vs analytic printed.
- [ ] `service-operator/train.py` runs end-to-end → checkpoint + VAL rel-L2 printed.
- [ ] Ablation: `config_deeponet.yaml` (λ_pde=0) trains and reports a comparable number.
- [ ] `experiments/` (renamed) still present; nothing deleted; inverse code intact.
- [ ] Each service self-contained: a fresh clone + `pip install -r requirements.txt` runs both.

---

**Anything missing or wrong before I start implementing?**
