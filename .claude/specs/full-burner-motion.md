# Spec: Full 1D Burner Motion

**Status:** ready to implement
**Affects:** sampler, physics, model, trainer, inspect_model, predictor, config

---

## Problem

Current model covers one physical scenario: constant-speed, left-to-right burner
starting in the first 30% of the pipe. Three restrictions are hard-coded:

| Restriction | Current | Reality |
|---|---|---|
| `x0_fraction_range` | [0.0, 0.3] | burner can start anywhere |
| `velocity_range` | [0.005, 0.3] | can move left, stop, reverse |
| acceleration | absent | real burners decelerate / reverse |

The PDE source `S = δ_gauss(x*; x0_norm + β·t*, σ*)` is linear in t* — no curvature.

---

## Out of scope

- Multiple heating passes / on-off source cycle
- 2D/3D geometry or variable intensity
- Any change to network depth / width / PirateNet architecture

---

## Math change

### New burner position

```
x_b*(t*) = x0_norm + β·t* + ½γ·t*²
```

| Symbol | Definition | Notes |
|---|---|---|
| β | v·t_total / l | signed travel fraction (was always > 0) |
| γ | a·t_total² / l | dimensionless acceleration (new 4th π-group) |

### Normalizer bounds — constraint-derived, hardcoded

**Why not from config ranges:** γ = a·t_total²/l with physical ranges spans ±720 in
theory, but the in-pipe constraint `x_b*(t*) ∈ [0,1] ∀ t* ∈ [0,1]` limits what is
actually achievable. Analysis:

- Endpoint: `x0_norm + β + ½γ ∈ [0,1]` → `|½γ| ≤ 1` when others are at extremes → **|γ| ≤ 2**
- Turning point (if `t*_turn = −β/γ ∈ (0,1)`): `x0_norm − β²/(2γ) ∈ [0,1]` — tighter, but covered by |γ| ≤ 2 given |β| ≤ 1

| Key | Old bounds | New bounds | Normalization |
|---|---|---|---|
| `beta` | (0.0, 1.0) | **(-1.0, 1.0)** | linear |
| `gamma` | absent | **(-2.0, 2.0)** | linear |

These are hardcoded constants in `Normalizer.__init__`, not computed from config —
identical to the existing pattern for `beta: (0.0, 1.0)`.

### Network input dim

```
d_in_mlp = fourier.out_dim + 4   # 128 + 4 π-groups = 132  (was 131)
```

---

## Config changes (`config.yaml`)

```yaml
# Before → After
x0_fraction_range: [0.0, 0.3]   →   [0.0, 1.0]
velocity_range:    [0.005, 0.3]  →   [-0.3, 0.3]   # signed; 0 allowed (stationary)
# New key:
acceleration_range: [-0.05, 0.05]  # m/s²; γ is sampled via γ=a·t²/l then constrained
```

**v = 0 is valid.** Stationary burner is a legitimate physical case (point source fixed in
space). The model should cover it. It was previously excluded by `v_lo = 0.005`.

**x0 anywhere.** Starting at the right end (x0_norm ≈ 1) and moving left is now valid.
The constraint loop (see below) handles it automatically.

---

## Sampling constraint

After sampling `(x0, v, a, l, t_total)`, enforce `x_b*(t*) ∈ [0, 1]` for all
`t* ∈ [0, 1]` by checking three critical points:

| Point | Expression | Condition |
|---|---|---|
| Start (t*=0) | `x0_norm` | always satisfied by construction |
| End (t*=1) | `x0_norm + β + ½γ` | must be in [0, 1] |
| Turning point (only if `t*_turn = −β/γ ∈ (0,1)`) | `x0_norm − β²/(2γ)` | must be in [0, 1] |

**Strategy:** vectorised rejection loop — resample entire rows that fail any check,
cap at 20 iterations, log a warning if still failing after 10. Expected acceptance rate
> 85% per iteration given the chosen ranges.

The source in `physics.py` also hard-clamps `x_b*.clamp(0, 1)` as a numerical
safety net (protects against the rare floating-point edge case post-sampling).

---

## Files changed

### `config.yaml`
- `x0_fraction_range`, `velocity_range`, add `acceleration_range`

### `training-service/sampler.py`

| Function | Change |
|---|---|
| `Normalizer.__init__` | `beta: (-1.0, 1.0)`, add `gamma: (-2.0, 2.0)` |
| `sample_params` | add `a` from `acceleration_range`; enforce signed `v` (no `v_lo` floor); add rejection loop for trajectory constraint; return `a` in dict |
| `compute_pi_groups` | add `gamma = raw["a"] * raw["t_total"]**2 / raw["l"]` |
| `_pi_norm_for_set` | 4-column output `[Fo_n, x0_n, β_n, γ_n]` |
| `rad_resample_pde` | pass `gamma` in `raw_keys`; pass to `pde_residuals_fd` |
| `build_batch` | pass `gamma` in `raw_pde` |

### `training-service/physics.py`

| Function | Change |
|---|---|
| `pde_loss` | `x_b = x0_norm + beta*t + 0.5*gamma*t**2`; unpack `gamma` from `raw`; update docstring |
| `pde_residuals_fd` | same x_b update; unpack `gamma` from `raw` |
| `analytical_delta_T` | add `a=0.0` param; closed-form path when `a==0` (unchanged); scipy-quad path when `a!=0` (see below) |

### `training-service/model.py`
- `d_in_mlp = self.fourier.out_dim + 4`
- Update docstring input layout

### `training-service/trainer.py`
- `validate()`: pass `gamma` from `compute_pi_groups` into `_build_pi_norm`

### `training-service/inspect_model.py`
- Add `a` / `gamma` to table printout and `_build_pi_norm` call

### `inference-service/predictor.py`
- `_pi_groups()`: add `a` param, compute and return `gamma`
- `_build_pi_norm()`: 4-column output
- `predict_point` / `predict_heatmap`: add `acceleration: float = 0.0` arg
- `analytical_delta_T`: same scipy-quad path

---

## `analytical_delta_T` with acceleration

Current closed-form evaluates `Iₙ(t) = ∫₀ᵗ exp(μₙτ) cos(nπ(x₀+vτ)/l) dτ` analytically.
With acceleration the integrand becomes `cos(nπ(x₀+vτ+½aτ²)/l)` — no closed form.

```python
# a == 0: existing closed-form (unchanged, fast)
# a != 0: numerical quadrature per-n
from scipy.integrate import quad

def _I_n(mu_n, n, x0, v, a, l, t):
    def integrand(tau):
        return math.exp(mu_n * tau) * math.cos(
            n * math.pi * (x0 + v * tau + 0.5 * a * tau**2) / l
        )
    val, _ = quad(integrand, 0.0, t, limit=200)
    return val
```

**Performance:** 150 terms × 8 validation sets × 1 time point ≈ 1 200 quad calls
≈ 0.1 s per validation checkpoint. Acceptable.
Heatmap (60×60 grid): ~15 s. Warn in docstring; inference is not latency-critical.

---

## Failure modes

| Failure | Detected by | Handling |
|---|---|---|
| Rejection loop doesn't converge | iteration counter ≥ 20 | log warning, use last valid batch |
| γ slightly outside [-2,2] due to floating point | Normalizer clamps nothing | network gets slightly out-of-range input; acceptable |
| Old 3-group checkpoint loaded with 4-group code | `load_state_dict` strict=True → RuntimeError | checkpoint stores `model_cfg`; check `d_in_mlp` at load, raise clear error |
| scipy absent in training env | ImportError on first a≠0 validation | guard with `try/except`; fall back to `a=0` analytical with warning |

---

## Test plan

- [ ] **γ=0, β>0**: output identical to pre-change model (same network input, same weights)
- [ ] **β=0, γ=0** (stationary burner): source stays at x0 for all t; PINN output matches analytical
- [ ] **β<0** (right-to-left): burner position decreases over time; residual peak tracks correctly
- [ ] **γ>0, β<0** (reverse then forward): turning point inside pipe; source moves left then right
- [ ] **x0_norm=0.9** (right-end start, β<0): no rejection loop issue; trajectory valid
- [ ] **Constraint check**: 10k sampled batches — assert `x_b*.min() ≥ 0`, `x_b*.max() ≤ 1` at t*=0, 0.5, 1.0 and at analytical turning point
- [ ] **analytical_delta_T**: `a≠0` scipy result matches RK4 finite-difference PDE solve for 3 random sets
- [ ] **Checkpoint guard**: loading old checkpoint raises `RuntimeError` with message mentioning π-group count mismatch
