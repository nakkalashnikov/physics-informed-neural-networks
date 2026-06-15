# Spec: 2D Moving Heat Source — PI-DeepONet Forward Operator + Validation

**Status:** ready to implement. **Implementer:** Sonnet. **Design rationale:**
`memory/problems/forward-2d-pideeponet.md` (read first; all physics derived there).
This spec is the implementation contract — every number, signature, and order is fixed.
Where a value is given, use it verbatim; make no independent modeling decisions.

---

## Problem
Learn an operator `G: (burner trajectory x_b(t), material/geometry params) → ΔT(x,y,t)` for a
thin bar `ℓ×h` heated by a burner moving arbitrarily along x (boundary flux on `y=0`), so any
new recorded trajectory yields the full temperature field in one forward pass, no retraining.
Forward problem + validation only; inverse is out of scope.

## Out of scope
- Inverse problem (recovering α or trajectory from sensors).
- Real thermocouple data (synthetic only: Fourier + FDM).
- 3D geometry; temperature-dependent k/ρc; phase change; radiation/convection losses.
- Inference service / UI (separate later spec). This spec stops at a trained, validated model + checkpoint.
- Reuse of old 1D `model.py/physics.py/sampler.py` — write fresh; do not import them.

## Solution
- New `training-service-2d/` package. Python 3.11, PyTorch (reuse existing `.venv`).
- PI-DeepONet (branch=trajectory+π-groups, trunk=(x*,ŷ,t*)), non-dimensional, hard-IC, RFF trunk.
- Hybrid loss: exact Fourier labels (cheap) + PDE residual + flux/insulation BC residuals.
- Validation package implementing Tiers 0A/0B/1/2/3 with a fixed acceptance card.
- All physics in non-dimensional form; physical ΔT reconstructed as `T_c·u` outside the net.

---

## Canonical equations (USE VERBATIM — do not re-derive)

```
# Non-dimensional vars: x*=x/ℓ, ŷ=y/h ∈[0,1], t*=t/t_end, u=ΔT/T_c
T_c = P * t_end / (rho_c * ℓ * h)
Fo  = alpha * t_end / ℓ**2          # alpha = k/rho_c
AR  = h / ℓ
w   = S / ℓ                          # contact width; gaussian std sigma_star = w/6
Q_star = AR**2 / Fo                  # DERIVED

# PDE residual (interior):
R = u_t  -  Fo*u_xx  -  (Fo/AR**2)*u_yy

# Flux BC residual at ŷ=0:
R_bc0 = u_y(ŷ=0)  +  Q_star * g_hat(x* - xb_star(t*))     # g_hat: gaussian, ∫=1, std=w/6
# Insulation BC residual at x*=0, x*=1, ŷ=1:
R_ins = normal_derivative(u)         # must be 0

# IC: hard-enforced by u = (...)·t*  → no IC loss term
```

Analytic Fourier solution (Tier 0A), dimensional — see design doc §4 for derivation:
```
ΔT(x,y,t) = Σ_{m=0}^{M} Σ_{n=0}^{N} A_mn(t)·cos(mπx/ℓ)·cos(nπy/h)
λ_mn  = alpha*((mπ/ℓ)² + (nπ/h)²)
Nx_m  = ℓ/2 if m>=1 else ℓ ;  Ny_n = h/2 if n>=1 else h
Q_m(τ)= ∫₀^ℓ P*g(x−xb(τ))*cos(mπx/ℓ) dx                   # NUMERICAL quadrature over x (≥400 pts)
        # do NOT use closed-form P*cos(mπxb/ℓ)*exp(-½(mπσ_s/ℓ)²): that assumes infinite domain
        # and is wrong within ~3σ_s of an edge (trajectory reaches 0.05ℓ). σ_s = S/6.
A_mn(t)= (alpha/(k*Nx_m*Ny_n)) * ∫₀ᵗ Q_m(τ)*exp(-λ_mn*(t-τ))dτ
# evaluate the integral by exponential-integrator recursion over the time grid:
# A_mn[j+1] = A_mn[j]*exp(-λ_mn*Δt) + (alpha/(k*Nx_m*Ny_n)) * 0.5*(Q_m[j]*exp(-λ_mn*Δt)+Q_m[j+1])*Δt
# NOTE: this whole formula is NUMERICALLY VERIFIED (2026-06-15) vs an independent 2D Crank–Nicolson
# FD solver: L2=1.6% (grid-limited), energy conserved to 0.05%, peak match <0.2%, α/k==1/ρc exact.
# The α/k factor and N_m^x/N_n^y normalization are CONFIRMED — do not change them.
# CAUTION: the analytic field is accurate in VALUE (use for data labels / value comparison) but its
# strong-form PDE residual is contaminated by undamped high-freq ringing of the δ(ŷ) boundary source
# in u_yy (damped in u, ×λ_mn in ∇²u). Do NOT compute strong-form residuals from the analytic field.
# Verify physics.py's residual operator on a SMOOTH manufactured solution instead.
```

---

## File structure
```
training-service-2d/
  config.yaml              # ALL hyperparameters & ranges (table below)
  nondim.py                # PiGroups dataclass, to_nondim(), to_physical(), normalize/denormalize
  trajectory.py            # random trajectory generator + Spline interpolant
  analytic.py              # Tier 0A Fourier series (also = label generator)
  model.py                 # BranchNet, TrunkNet (RFF+PirateNet), DeepONet, hard IC
  physics.py               # pde_residual, bc_flux_residual, bc_insulation_residual (autograd)
  data.py                  # Sample dataclass, batch builder, async prefetcher
  trainer.py               # hybrid-loss training loop, Adam+cosine, checkpoint, val hook
  train.py                 # CLI entry: parse args, device select, call trainer
  validation/
    rosenthal.py           # Tier 0B thin-plate K0 (constant-v sanity only)
    fdm.py                 # Tier 1 Crank–Nicolson 2D reference
    intrinsic.py           # Tier 2: residual map, defect-FDM, energy, invariants
    ensemble.py            # Tier 3: deep ensemble mean±std
    report.py              # π-table sweep + acceptance card
  tests/
    test_analytic.py  test_fdm.py  test_trajectory.py
    test_nondim.py  test_model.py  test_physics.py  test_intrinsic.py
```

---

## config.yaml (fixed values)

| Group | Key | Value |
|---|---|---|
| physics ranges | Fo_min / Fo_max | 3.0e-6 / 3.0e-2 (log-uniform sample) |
| | AR_min / AR_max | 0.02 / 0.30 |
| | w_min / w_max | 0.02 / 0.20 |
| | P, rho_c, k, ℓ, h, S, t_end | sampled per-case from Fo/AR/w + fixed P=1.0, t_end=1.0 ref (T_c handles scale) |
| trajectory | k_sensors | 101 |
| | n_fourier_modes (J) | 5 |
| | spectrum_decay (p) | 2.0 |
| | sigma0 | 1.0 |
| | x_margin | 0.05 (range [0.05, 0.95]) |
| | speed_max_star | 4.0 (|dx_b*/dt*| cap) |
| branch | input_dim | 101 + 3 |
| | hidden | [256, 256, 256] |
| | activation | tanh |
| | out_dim p | 128 |
| trunk | input_dim | 3 (x*, ŷ, t*) |
| | rff_num_features | 64 (→128 after sin/cos) |
| | rff_sigma_start / end | 1.0 / 3.5 (linear over training) |
| | n_pirate_blocks | 3 |
| | width | 256 |
| | out_dim p | 128 |
| loss | lambda_data / pde / bc / ic | 10 / 1 / 10 / 0 |
| collocation/step | n_traj_per_batch | 16 |
| | n_interior | 1024 |
| | n_bc0 (flux y=0) | 256 |
| | n_bc_ins (other edges) | 256 |
| | n_data (Fourier label pts) | 512 |
| training | optimizer | Adam, lr 1e-3 |
| | scheduler | CosineAnnealingLR → 1e-5 |
| | total_steps | 200000 |
| | grad_clip_norm | 1.0 |
| | use_amp / use_compile / use_fp64 | false / false / false (1D notes: incompatibilities) |
| | use_prefetch | true on CUDA, false on CPU/MPS |
| | seed | 0 (ensemble: 0..N-1) |
| analytic/fdm | fourier_M / fourier_N | 64 / 48 |
| | fdm_nx / fdm_ny / fdm_nt | 201 / 101 / 400 (Crank–Nicolson) |
| validation | pi_table_n_cases | 64 |
| | ensemble_size | 5 |
| | thresholds | L2 p90 .02/.05; energy .02/.05; defect .03/.08 |

---

## Module contracts (signatures Sonnet implements exactly)

### nondim.py
```python
@dataclass
class PiGroups:
    Fo: float; AR: float; w: float
    @property
    def Q_star(self) -> float: return self.AR**2 / self.Fo
def sample_pi_groups(cfg, rng) -> PiGroups           # log-uniform Fo, uniform AR,w
def normalize(pi: PiGroups, cfg) -> tuple[float,float,float]   # → (logFo_n, AR_n, w_n) ∈[0,1]³
def reconstruct_T_c(P, t_end, rho_c, l, h) -> float
def to_physical(u, T_c) -> "ΔT"                       # ΔT = T_c * u
```

### trajectory.py
```python
def sample_trajectory(cfg, rng) -> Callable[[np.ndarray], np.ndarray]   # xb_star(t_star)∈[0.05,0.95]
    # random Fourier series: xb = c0 + Σ_{j=1..J} a_j sin(2πj t*)+b_j cos(2πj t*),
    # a_j,b_j ~ N(0, (sigma0/j**p)²); affine-map to [margin,1-margin]; rescale to satisfy speed cap.
def sample_at_nodes(traj_fn, k) -> np.ndarray         # k uniform t* nodes → branch input
def spline_interp(nodes_t, nodes_x) -> Callable        # C¹ cubic; used in physics source term
# property tests: range bound, continuity, speed cap.
```

### analytic.py
```python
def fourier_field(x_grid, y_grid, t_grid, traj_fn, pi: PiGroups, cfg,
                  M=64, N=48) -> np.ndarray            # returns ΔT[t,y,x] dimensional via T_c
    # uses exponential-integrator recursion (canonical eqns above). Vectorize over modes.
def fourier_labels(query_pts_star, traj_fn, pi, cfg) -> np.ndarray   # u at (x*,ŷ,t*) for L_data
```

### model.py
```python
class BranchNet(nn.Module):  # MLP [104→256→256→256→128], tanh
class TrunkNet(nn.Module):   # RFF(B fixed N(0,1) 64×3, sigma curriculum) → 3×PirateBlock(256) → 128
    def set_sigma(self, sigma: float): ...            # called each step from curriculum
class PirateBlock(nn.Module): # U/V shared encoders + 2 gates + learnable α skip (init 0); see 1D notes
class DeepONet(nn.Module):
    def forward(self, branch_in, coords_star) -> u    # u = (Σ b_i τ_i + b0) * coords_star[...,t*]
```

### physics.py  (all operate on non-dim u; autograd create_graph=True)
```python
def pde_residual(model, branch_in, coords, pi) -> Tensor      # R interior
def bc_flux_residual(model, branch_in, coords_y0, pi, traj_spline) -> Tensor   # R_bc0
def bc_insulation_residual(model, branch_in, coords_edges) -> Tensor           # R_ins
def g_hat(s, w) -> Tensor                                     # gaussian ∫=1, std=w/6
```

### data.py
```python
@dataclass
class Sample: branch_in; interior; bc0; bc_ins; data_pts; data_u; pi; traj_spline
def build_batch(cfg, rng, device) -> list[Sample]            # n_traj_per_batch samples
class BatchPrefetcher: ...                                   # bg thread, queue=4 (1D notes pattern)
```

### trainer.py
```python
def train(cfg, device, resume=None) -> checkpoint_path
    # per step: sigma curriculum; for each sample sum L_data+λ·L_pde+λ·L_bc; backward; clip; step.
    # cosine LR; checkpoint every 5k; validate every 5k via validation.report.quick_card.
```

### validation/
```python
# rosenthal.py
def thin_plate_rosenthal(x, y, P, k, h, alpha, v) -> ΔT       # quasi-steady, ξ=x−vt, K0
# fdm.py
def crank_nicolson_2d(traj_fn, pi, cfg) -> ΔT[t,y,x]          # validate vs fourier first
def solve_defect(residual_field, pi, cfg) -> e_hat[t,y,x]     # 𝒟[e]=−R, same CN solver
# intrinsic.py
def residual_map(model, sample) -> dict(mean,max,max_bc)
def energy_balance(field, P, t_end, rho_c, l, h) -> float     # |E_in−E_stored|/E_in
def check_invariants(field) -> dict(nonneg, mono_energy, causality) -> bool flags
# ensemble.py
def predict_ensemble(models, branch_in, coords) -> (mean, std)
# report.py
def pi_table_sweep(model, cfg, n_cases=64) -> DataFrame       # L2 vs fourier; median/p90/max
def acceptance_card(...) -> dict                              # green/yellow/red per thresholds
```

---

## Failure modes

| Failure | Detected by | Handling |
|---|---|---|
| Trunk mode collapse (DeepONet instability) | val L2 plateaus high; τ variance ~0 | log τ std per step; if collapse, raise p, lower lr |
| Sharp source overflows residual | NaN loss step 1 | AMP stays OFF (1D notes); clamp σ*; if NaN, skip step + log |
| Fourier label inaccurate near y=0 | label vs FDM mismatch >1% | increase N (y-modes) until energy conserves <0.5% |
| FDM ref untrustworthy | FDM vs Fourier >0.5% on canonical | refine grid; gate: FDM unusable until it passes 0A |
| Trajectory speed-cap distorts shape | continuity/speed test fails | rescale time amplitude, not clip position |
| Off-distribution trajectory poor | Tier-0A L2 high on OOD test set | widen trajectory distribution (J, sigma0); retrain |
| Energy not conserved by NN | Tier-2C residual >5% | raise λ_data; check T_c reconstruction |

---

## Test plan
- [ ] nondim: round-trip `to_physical(to_nondim(x))==x`; `Q_star==AR²/Fo`.
- [ ] analytic: energy conservation `ρc∫∫ΔT == ∫P dt` within 0.5%; m=n=0 mode == mean rise; static source vs separable 1D limit.
- [ ] fdm: matches analytic on constant-v and on a random trajectory within 0.5% (grid-refined).
- [ ] trajectory: output ∈[0.05,0.95]; C¹ continuous; speed ≤ cap; reproducible per seed.
- [ ] model: `u(t*=0)==0` exactly (hard IC); output O(1); shapes correct for batched coords.
- [ ] physics: pde_residual→0 when fed the analytic field (finite-diff sanity); g_hat integrates to 1.
- [ ] intrinsic: defect-FDM error map matches true error (vs Fourier) within 20% relative; energy_balance sign/scale.
- [ ] integration (smoke): 500-step train run decreases loss; checkpoint loads; quick_card runs.
- [ ] acceptance: after full train, pi_table_sweep p90 L2 < 5% (🟡 gate to ship).

---

## Label cache workflow (CPU/GPU decoupling — implemented)
- `precompute_labels.py` (multiprocessing) generates a bank of `cache.n_cases` (trajectory, pi)
  cases with exact analytic labels -> `labels.npz`. Only the Fourier labels are precomputed;
  PDE/BC collocation points are cheap and generated fresh each training step.
- `data.py::CachedBatcher` reads the npz and builds batches with fresh collocation + cached labels;
  rebuilds the trajectory spline from the cached 101 branch samples for the flux-BC source.
- `train.py --cache labels.npz` uses it; GPU training then performs NO Fourier evaluation.
- Status: 100k-case cache generated on Mac (752 MB file, ~860 MB RAM, 2.2 s load). git-ignored.

## Build order (linear — each step testable before the next)
1. `config.yaml` + `nondim.py`  → test_nondim.
2. `trajectory.py` → test_trajectory.
3. `analytic.py` → test_analytic (energy conservation gates correctness).
4. `validation/fdm.py` → test_fdm (must match analytic before trusted).
5. `model.py` → test_model (hard IC, shapes).
6. `physics.py` → test_physics (residual→0 on analytic field).
7. `data.py` (batch + prefetcher).
8. `trainer.py` + `train.py` → smoke 500-step run.
9. `validation/intrinsic.py` → test_intrinsic (defect map, energy, invariants).
10. `validation/ensemble.py` + `validation/report.py` → pi-table + acceptance card.
11. `validation/rosenthal.py` → cross-check on one constant-v case.
12. Full 200k train → acceptance card → ship checkpoint.
