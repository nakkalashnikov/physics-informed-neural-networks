# Spec: Inverse PI-DeepONet — recover source trajectory from measurements + material/geometry

**Status:** ready to implement
**Files:** `training-service-2d/` (reuses analytic.py); new `inverse_*.py`, `analytic_torch.py`
**Goal:** an amortized inverse **PI-DeepONet** — given thermocouple readings plus the KNOWN material
and geometry, output the heat-source **trajectory** `xb(t)`; for the linear family this is `(x0, v)`.
Genuine DeepONet (function output via trunk), physics-informed (recovered trajectory must reproduce
the measurements through the PDE), and accurate (**<20%, likely <10%** — it outputs a smooth low-DOF
trajectory, not the sharp field).

---

## Problem

The forward operator (trajectory→field) is fixed at ~40% (sharp-field representational floor). The
inverse flips the difficulty: the field is hard, but the source MOTION is a smooth low-DOF function,
strongly encoded in the hot-streak timing across sensors. One amortized network recovers it for ANY
measurement set + material/geometry — instant, no re-solve. Industrial framing: "from thermocouples,
recover where the weld started and how fast it moved."

## Out of scope

- Nonlinear physics (k(T), phase change) — future work, the real deal-breaker regime.
- Reconstructing the full sharp field (that IS the 40% forward operator).
- Single-instance inverse PINN (re-solve per query) — we want the amortized operator.

## Solution

- **Inverse PI-DeepONet:**
  ```
  branch( [thermocouples K×M ; Fo ; AR ; w] )  ⊗  trunk( t )  →  xb(t)
  ```
  Output is the source **position vs time** — a function → genuine DeepONet. For the linear family
  `xb(t)=x0+v·t`, so `(x0, v) = xb(0), slope`. Generalizes over material/geometry (Fo, AR, w vary) AND
  extends to arbitrary trajectories (the output is a function, not 2 scalars).
- **Two training modes (architecture identical):**
  - **(A) Supervised (primary, fast, accurate):** `L = ‖xb_pred(t) − xb_true(t)‖`. Labels = the known
    trajectory per case. No forward model needed → trains in minutes, even on Mac. The <20% deliverable.
  - **(B) Physics-informed (the genuine PI):** `L += ‖forward(xb_pred, material; sensors) − measurements‖`
    via a **differentiable torch forward** (`analytic_torch.py`). Physics closes the loop: "your recovered
    trajectory, run through the PDE, must reproduce the sensors." No dependence on the 40% forward op.
- **Data from analytic.py** (exact): sample material + `(x0,x1)` → linear traj → field → sensor readings.
- **Validate:** trajectory / `(x0,v)` recovery error on held-out (unseen measurements + material), noise
  sweep, sensor-count ablation, torch-forward-vs-numpy correctness.

## Architecture

```
branch: [meas (K*M); Fo_n; AR_n; w_n]  -> MLP [512,512,256] tanh -> b ∈ R^p
trunk:  t*  -> RFF + MLP                                       -> τ(t) ∈ R^p
xb(t) = clip( Σ_i b_i τ_i(t) + b0 , margin, 1-margin )
readout (linear family): x0 = xb(0), v = xb(1) − xb(0)
```
`p` ~ 64. Smooth output ⇒ small, fast, accurate. (Material/geometry enter the branch as normalized scalars
appended to the measurement vector.)

## Differentiable forward (`analytic_torch.py`) — the PI engine

Port `analytic._build_A_grid` + field eval to **torch**, evaluated only at the K sensor `(x,ŷ)` over M
times (small): given `xb(t)` (network output, sampled at τ-nodes) + `(Fo,AR,w)` → `u` at sensors,
differentiable. Mode (B) backprops `‖u_torch − measurements‖`. Coarser modes/quadrature OK (consistency
needs no machine precision). Verify against numpy `analytic` (<2%).

## Inputs / data generation

| knob | meaning | default |
|---|---|---|
| `K` | thermocouples (row along x near bottom; x-spread > y-spread) | **9** (start), ablate 2/3/4/6/9 |
| sensor x | e.g. evenly in [0.15, 0.9] at ŷ≈0.15 | fixed grid |
| `M` | time samples per sensor | 30 |
| material/geom | Fo, AR, w — VARIED (the operator generalizes over them) | full ranges |
| `N_train` | cases | 8000–20000 (cheap, low-dim) |
| noise | optional Gaussian on measurements | 0 (sweep later) |

```
per case: Fo,AR,w ~ ranges ; x0,x1 ~ U(margin,1-margin) -> linear traj
          -> analytic field -> T at K sensors × M times = measurement vector
          inputs: [measurements ; Fo_n ; AR_n ; w_n]   target: xb_true(t) (the line)
```

## Module layout

| file | purpose | reuse |
|---|---|---|
| `inverse_data.py` | generate (measurements, material, xb_true) pairs; fixed sensor grid | analytic.fourier_field_u |
| `analytic_torch.py` | differentiable torch forward at sensor points (PI engine) | port of analytic.py |
| `model_inverse.py` | branch(meas+material) ⊗ trunk(t) → xb(t) | trunk bits from model.py |
| `train_inverse.py` | mode A (supervised) + mode B (physics-consistency) | analytic_torch for B |
| `validate_inverse.py` | xb/(x0,v) error, noise & sensor ablation, consistency plot | analytic |
| `config_inverse.yaml` | sensors (K, x-grid, M), material ranges, model, training | — |

## Failure modes

| failure | detected by | handling |
|---|---|---|
| sensors miss the path (uninformative) | high error for some cases | x-spread covers domain; ablate K |
| ill-posed | error floor | linear x0/v identifiable from streak timing+slope; report |
| torch forward ≠ numpy analytic | correctness test | verify <2%; match modes/quadrature |
| measurement-scale spread (Q* range) | unstable train | per-sample normalization |
| noise degrades recovery | noise sweep | report curve; more sensors / denoise |
| mode B slow (torch convolution loop) | wall time | sensors-only eval + coarse modes; A doesn't need it |

## Test plan

- [ ] `analytic_torch` matches numpy `analytic` at sensor points (<2%).
- [ ] Mode A recovers `(x0,v)` to **<20% (target <10%)** on held-out (unseen measurements + material).
- [ ] Per-param: x0 error and v error separately; also full `xb(t)` rel-L2.
- [ ] Mode B trains via physics-consistency (recovers without trajectory labels, or A+B hybrid).
- [ ] Noise robustness: 1/5/10% measurement noise → degradation curve.
- [ ] Sensor-count ablation: K = 2/3/4/6/9 → accuracy vs #sensors ("how few thermocouples").

## Report framing (the 4th point)

> "Amortized inverse **PI-DeepONet**: from K thermocouples + the known material/geometry it instantly
> recovers the heat-source trajectory `xb(t)` (start x0, speed v) to <X%, for any scenario, no re-solve.
> Physics-informed — the recovered trajectory, run through the PDE, reproduces the measurements. Forward
> (trajectory→field) and inverse (measurements→trajectory) close the loop."

The **<20% respect number** (smooth low-DOF output) + the regime where the neural method beats classical
(native inverse, no adjoint). Caps the arc: PINN (single-instance) → DeepONet (forward op) → PI-DeepONet
(physics-informed + dimensionality collapse + capacity curve) → **inverse PI-DeepONet**.

---

**Anything missing or wrong before I start implementing?**
