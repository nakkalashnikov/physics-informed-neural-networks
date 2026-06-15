# Forward Problem (v2): 2D Moving Heat Source via PI-DeepONet

> Supersedes the 1D parametric-PINN forward formulation. The 1D code (training-service/
> model.py, physics.py, sampler.py) describes the OLD problem and may be discarded.
> This document is the physics + design rationale. The implementation contract is
> `.claude/specs/pinn-2d-pideeponet-forward.md`.

## 1. Problem (from the hand-drawn statement)

A thin rectangular bar (length `ℓ`, thickness `h`) lies along the x-axis. A burner moves
*underneath* it along x and heats it through a contact of fixed area `S`. The burner's
position `x_b(t)` is an **arbitrary continuous function** given as a recording sampled at
intervals ("запис руху з інтервалами") — NOT `x₀ + v·t`. Constraints on motion:
- moves only along x,
- `S = const`,
- `x_b(t)` continuous (no teleport); otherwise free (back-and-forth, dwell, accelerate).

**Wanted:** temperature field `ΔT(x, y, t)` in Kelvin, at any point `(x,y)` and time `t`,
for *any* new recorded trajectory, **without retraining** → operator learning (PI-DeepONet).

## 2. Governing physics (dimensional)

Domain Ω = [0,ℓ]×[0,h], t ∈ (0, t_end]. ΔT = T − T_ambient.

```
∂ΔT/∂t = α (∂²ΔT/∂x² + ∂²ΔT/∂y²)              α = k/(ρc)        [m²/s]

−k ∂ΔT/∂y |_{y=0} = q(x,t) = P · g(x − x_b(t))     ← moving boundary flux (burner)
∂ΔT/∂n = 0   on x=0, x=ℓ, y=h                      ← insulated
ΔT(x,y,0) = 0
```
- `P` — burner power per unit out-of-plane depth [W/m].
- `g(·)` — contact profile, normalized `∫g ds = 1`, width set by `S` (Gaussian, σ_s = S/6).
- Source is a **boundary flux on y=0**, NOT a volumetric term (decision: burner touches from below).

## 3. Non-dimensionalization (Buckingham-π) — the conditioning lever

Variables: `x* = x/ℓ`, `ŷ = y/h ∈[0,1]`, `t* = t/t_end`, `u = ΔT/T_c`.

Characteristic temperature (total injected energy spread over the bar):
```
T_c = P · t_end / (ρc · ℓ · h)
```

Non-dimensional system (derive once; Sonnet must NOT re-derive — use as-is):
```
∂u/∂t* = Fo · ∂²u/∂x*²  +  (Fo / AR²) · ∂²u/∂ŷ²            (PDE)

∂u/∂ŷ |_{ŷ=0} = −Q* · ĝ(x* − x_b*(t*); w)                  (flux BC)
∂u/∂n = 0  on x*=0, x*=1, ŷ=1                              (insulated)
u(x*,ŷ,0) = 0                                              (IC, hard-enforced)
ΔT = T_c · u                                               (reconstruct outside net)
```
where
```
Fo = α·t_end/ℓ²        (Fourier number; ~4 orders → log-scaled)
AR = h/ℓ               (aspect ratio)
w  = S/ℓ               (relative contact width; sets ĝ std σ* = w/6)
Q* = AR² / Fo          (DERIVED, not a free input — falls out of the T_c choice)
ĝ  — Gaussian in x*, ∫ĝ dx* = 1, std σ* = w/6
```

**Independent π-groups: {Fo, AR, w}** plus the trajectory function `x_b*(t*)`.
Because `T_c` absorbs the power scale, `u = O(1)` everywhere and `Q*` is fixed by {Fo,AR}.
This is the same lever that broke the 68% ceiling in 1D: residual is O(1) across the whole
parameter space, so the loss weights every sample equally. See [[notes-nondimensionalization]].

## 4. Exact analytic reference (Tier 0A) — Fourier series, ANY trajectory

Rigorous solution on the all-Neumann rectangle with inhomogeneous flux on y=0. Derived by
eigenfunction projection onto `cos(mπx/ℓ)cos(nπy/h)`; the boundary flux enters each mode as
a source term (Green's-identity boundary term). Final closed form (dimensional):

```
ΔT(x,y,t) = Σ_{m=0}^{M} Σ_{n=0}^{N} A_mn(t) · cos(mπx/ℓ) · cos(nπy/h)

λ_mn   = α·[ (mπ/ℓ)² + (nπ/h)² ]
N_m^x  = ℓ/2 (m≥1), ℓ (m=0);   N_n^y = h/2 (n≥1), h (n=0)
Q_m(τ) = ∫₀^ℓ q(x,τ)·cos(mπx/ℓ) dx     ← compute by NUMERICAL quadrature over x
A_mn(t)= (α / (k·N_m^x·N_n^y)) · ∫₀^t Q_m(τ)·exp(−λ_mn·(t−τ)) dτ
```
Closed form `Q_m = P·cos(mπx_b/ℓ)·exp(−½(mπσ_s/ℓ)²)` holds only in the interior (infinite-domain
approx); it errors within ~3σ_s of an edge, so use numerical quadrature.

**VERIFIED numerically (2026-06-15):** full 2D moving-source formula matches an independent 2D
Crank–Nicolson FD solver — L2=1.6% (FD-grid-limited), energy conserved to 0.05%, peak match <0.2%,
and `α/k == 1/ρc` exact. The `α/k` factor and `N_m^x/N_n^y` normalization are confirmed correct.

- Works for **any continuous x_b(τ)**: the trajectory enters only pointwise inside `Q_m(τ)`;
  the convolution is numerical quadrature over the recorded/interpolated trajectory.
- **Efficient evaluation:** exponential-integrator recursion over the time grid
  `A_mn(t_{j+1}) = A_mn(t_j)·e^{−λ_mn·Δt} + (source increment)` (exact for piecewise-const Q).
- `n`-modes resolve the y-profile (sharp near y=0 where flux enters → need high n there).
- This is the 2D analog of the old 1D `analytical_delta_T` Fourier series.
- **Dual use:** (a) generates unlimited exact training labels for the hybrid loss;
  (b) is the ground-truth for validation on any trajectory (even out-of-distribution).

Sanity sub-limits: with `M,N` large enough, energy must conserve (Tier 2C); the n=0,m=0 mode
equals mean temperature rise = injected energy / (ρc·ℓ·h).

## 5. Architecture: PI-DeepONet

Operator `G: (x_b*(·), {Fo,AR,w}) ⟼ ΔT(x,y,t)`.

```
BRANCH:  [x_b*(t₁*)..x_b*(t_k*)] (k=101) ++ [logFo_n, AR_n, w_n]  →  MLP → b ∈ ℝ^p (p=128)
TRUNK:   (x*, ŷ, t*) → RFF(σ curriculum 1→3.5) → 3×PirateNet(width 256) → τ ∈ ℝ^p
MERGE:   u = ( Σ_{i=1}^{p} b_i·τ_i + b₀ ) · t*        ← dot product, then ×t* = hard IC
ΔT = T_c · u
```
Carried over from the working 1D build (all proven): Random Fourier Features (anti spectral
bias for the sharp moving contact), adaptive σ curriculum, PirateNet residual-gate blocks,
hard IC via ×t* (so λ_ic=0). Separable-DeepONet trunk factorization is a documented FUTURE
optimization (config flag, off in v1).

## 6. Training: HYBRID data + physics loss

Because Tier-0A Fourier gives exact labels cheaply for ANY trajectory, train hybrid (matches
[[feedback-hybrid-pinn]] — hybrid beat pure-physics in 1D):
```
L = λ_data·‖u_NN − u_Fourier‖²
  + λ_pde ·‖u_t* − Fo·u_x*x* − (Fo/AR²)·u_ŷŷ‖²            (interior collocation)
  + λ_bc  ·‖u_ŷ|_{ŷ=0} + Q*·ĝ(x*−x_b*(t*))‖²  + ‖∂u/∂n‖²_other_edges
```
- Data labels make training stable/fast; physics term generalizes off-distribution.
- Source term in residual uses a **continuous spline interpolant** of the k trajectory
  samples (branch encodes it; physics loss evaluates ĝ(x*−x_b*(t*)) from the spline).
- Weights: λ_data=10, λ_pde=1, λ_bc=10, λ_ic=0.

## 7. Trajectory distribution (defines operator competence)

Sample smooth random trajectories (random Fourier series, smoothness-decaying spectrum),
mapped into [0.05, 0.95], speed-limited. Covers reversals, dwells, accel. See spec §6 for the
exact generator. Operator quality = richness of this distribution; mix in real recordings if
their character is known.

## 8. Validation (full hierarchy — synthetic only, no real sensors)

| Tier | Tool | Needs truth? | Role |
|---|---|---|---|
| 0A | Fourier series (§4) | exact truth | PRIMARY ref, any trajectory |
| 0B | thin-plate Rosenthal K₀ | exact (quasi-steady) | sanity on constant-v sub-case only |
| 1 | Crank–Nicolson FDM | reference | general motion; validate vs 0A first |
| 2A | PDE residual map | NO | a posteriori indicator |
| 2B | defect equation 𝒟[e]=−R (FDM) | NO | error map without truth (arXiv 2603.15526) |
| 2C | energy balance ∫P dt = ρc∫∫ΔT | NO | global amplitude check |
| 2D | invariants (ΔT≥0, mono energy, causality) | NO | fast flags |
| 3 | deep ensemble (5–10 seeds) | NO | ΔT ± σ confidence |

Report worst-case over π-space (median/p90/**max**), never just mean — the 1D 68% ceiling was
a worst-case masking problem. Acceptance: L2_rel p90 <2%🟢/<5%🟡; energy_residual <2%🟢/<5%🟡;
defect ‖ê‖/‖ΔT‖ <3%🟢/<8%🟡; any invariant fail → 🔴.

## 9. Key references
- Wang et al., *Physics-informed DeepONets*, Science Advances 2021 — PI-DeepONet foundation.
- Koric & Abueidda, IJHMT 2023 — PI-DeepONet for heat conduction w/ parametric source (precedent).
- Separable PI-DeepONet, ScienceDirect 2024 — curse-of-dimensionality for 3D (x,y,t).
- "Building Trust in PINNs", arXiv 2603.15526 — defect-equation error map (Tier 2B).
- Rosenthal / Wilson-Rosenthal moving heat source — Tier 0B.

## 10. Scope
- IN: forward operator (trajectory → field) + full validation. Synthetic data only.
- OUT (future): inverse problem (recover α / trajectory from sensors) — see [[project-direction]];
  real thermocouple data; 3D geometry; temperature-dependent material properties.
