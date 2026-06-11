# Inverse PINN: Identify Material & Burner Parameters from Sparse Sensors

## 1. Problem Statement

**Goal**: Given temperature readings from 2–3 fixed sensors on a pipe at known positions,
infer the unknown physical parameters of the system in real time.

**Why this is hard for classical methods**:
- The system is described by an integral over the source history (moving Gaussian convolved with the Green's function)
- Classical optimization over analytical solution requires evaluating 150+ Fourier terms per iteration of Levenberg–Marquardt / gradient descent — slow
- The parameter space is multi-modal: multiple (α, v) combinations can produce similar sensor traces at a single time instant
- Classical methods need a good initial guess; PINN-based gradient flow can escape poor initializations because physics constraints regularize the landscape

**Why PINN is the right tool**:
- Physics constraints prevent non-physical solutions that would fit the sensor data but violate the PDE
- Automatic differentiation gives exact gradients of predictions w.r.t. physical parameters
- Single unified computational graph: sensor loss + PDE residual + BC/IC are all differentiable w.r.t. both network weights AND unknown physics parameters simultaneously
- Literature (Raissi et al. 2019, Emergent Mind 2024) shows <3% parameter error with only 5 sensors; single-point monitoring can achieve <5% for some parameters

---

## 2. Full Mathematical Formulation

### 2.1 Governing PDE (forward problem)

```
∂ΔT/∂t  =  α · ∂²ΔT/∂x²  +  (i/ρc) · δ(x − x_b(t))     (1)
```

| Symbol   | Meaning                                      | Unit      |
|----------|----------------------------------------------|-----------|
| ΔT(x,t)  | Temperature rise above ambient               | K         |
| α = k/ρc | Thermal diffusivity                          | m²/s      |
| ρc       | Volumetric heat capacity                     | J/(m³·K)  |
| i        | Linear heat intensity (burner power)         | W/m       |
| i_eff    | Effective source = i/ρc                      | K·m/s     |
| x_b(t)   | Burner position = x₀ + v·t                  | m         |
| v        | Burner velocity                              | m/s       |
| x₀       | Initial burner position                      | m         |
| l        | Pipe length                                  | m         |

**Boundary conditions (insulated ends)**:
```
∂ΔT/∂x = 0   at  x = 0  and  x = l       (Neumann / zero-flux)        (2)
```

**Initial condition**:
```
ΔT(x, 0) = 0   ∀x ∈ [0, l]                                            (3)
```

### 2.2 Exact analytical solution (Fourier series)

For validation and data generation:

```
ΔT(x,t) = (i_eff · t / l)
         + Σ_{n=1}^{N} aₙ(t) · cos(nπx/l)
```

where:
```
μₙ  = α · (nπ/l)²
bₙ  = nπv/l
cₙ  = nπx₀/l

aₙ(t) = (2·i_eff / l) · [f(t) − exp(−μₙt)·f(0)] / (μₙ² + bₙ²)

f(τ) = μₙ·cos(bₙτ + cₙ) + bₙ·sin(bₙτ + cₙ)
```

The n=0 mode represents total energy accumulation; n≥1 modes capture spatial redistribution.
The series converges absolutely for t > 0 (N=150 terms is sufficient for practical ranges).

---

## 3. Inverse Problem Formulation

### 3.1 Unknowns to identify

**Set A — Material** (per pipe installation, changes slowly):
```
θ_phys = {α, ρc}      →  i_eff = i / ρc  (i is measurable separately)
```
Or equivalently treat `i_eff` as a single unknown if intensity `i` is also unknown.

**Set B — Process** (changes per heating cycle):
```
θ_proc = {x₀, v}
```

**Combined unknown vector**:
```
Λ = [α, i_eff, x₀, v]     (4 unknowns, or 3 if α known from material spec)
```

Pipe length `l` is assumed measurable geometrically.

### 3.2 Sensor model

Sensors at known fixed positions `{s₁, s₂, ..., sₖ}` (k = 2 or 3),
recording temperature at times `{t₁, t₂, ..., tₘ}`:

```
y_j = ΔT(sⱼ, tⱼ) + ε_j       ε_j ~ N(0, σ_noise²)               (5)
```

Total observations: N_data = k × m  (e.g., 3 sensors × 20 timestamps = 60 points).

### 3.3 Identifiability — what can be recovered

Theoretical analysis (from "On the identification of source term in heat equation", arxiv 1908.02015):

| Parameter | Identifiable from sensors? | Notes                                     |
|-----------|---------------------------|-------------------------------------------|
| α         | ✅ Yes                    | Controls diffusion rate — visible in peak width over time |
| v         | ✅ Yes                    | Controls peak displacement speed          |
| x₀        | ✅ Yes                    | Controls initial peak position            |
| i_eff     | ✅ Yes                    | Controls peak amplitude                   |
| l         | ⚠️ Partially              | Detectable from BC reflection; better measured directly |

**Minimum sensor requirements** (from literature):
- 1 sensor: can identify i_eff and x₀ if α and v partially known
- 2 sensors: can jointly identify all 4 parameters with sufficient temporal samples
- 3 sensors: robust identification even with 10–30% measurement noise

**Pathological cases** (ill-posed):
- Sensors all at same x → cannot distinguish x₀ from v  
- Very short measurement window (t << l²/α) → α and v are correlated
- Low signal-to-noise → i_eff and α are correlated (both affect peak height)

---

## 4. Inverse PINN Loss Function

### 4.1 Full loss (Raissi 2019 formulation, extended)

```
L_total(θ_NN, Λ) = λ_data · L_data  +  λ_pde · L_pde  +  λ_bc · L_bc

                                                                        (6)
```

**Data loss** (sensor fidelity):
```
L_data = (1/N_data) · Σ_{j=1}^{N_data} [ΔT_NN(sⱼ, tⱼ; Λ) − yⱼ]²    (7)
```

**PDE residual loss** (causal, Wang et al.):
```
L_pde = (1/M) · Σ_{k=1}^{M} w_k · (1/|Bₖ|) · Σ_{(x,t)∈Bₖ} R(x,t)²

R = ∂ΔT/∂t − α·∂²ΔT/∂x² − i_eff·δ_σ(x − x₀ − v·t)

w_k = exp(−ε · Σ_{j<k} L_j^{pde})   [causal weights]                (8)
```

**BC loss** (Neumann, finite-difference):
```
L_bc = (1/N_bc) · Σ_t [(ΔT(ε,t) − ΔT(0,t))²  +  (ΔT(l,t) − ΔT(l−ε,t))²] / ε²  (9)
```

**IC** is hard-enforced via `ΔT_NN = t_norm · MLP(x,t,Λ)` — no IC loss term.

### 4.2 Key insight: Λ is TRAINABLE

```python
# Λ treated as nn.Parameter alongside θ_NN
log_alpha = nn.Parameter(torch.tensor(log(alpha_init)))   # log-space for positivity
log_i_eff = nn.Parameter(torch.tensor(log(i_eff_init)))
x0        = nn.Parameter(torch.tensor(x0_init))           # ∈ [0, l]
v         = nn.Parameter(torch.tensor(v_init))             # ∈ [v_min, v_max]

# Optimizer sees both NN weights and Λ
optimizer = Adam([*model.parameters(), log_alpha, log_i_eff, x0, v], lr=1e-3)
```

Gradient flow: `∂L/∂Λ` is computed automatically by autograd through:
```
L_data → ΔT_NN(s,t; Λ) → MLP(FourierFeatures([s,t]), norm(Λ)) → α, i_eff (via exp)
L_pde  → R(x,t; Λ) → α appears in α·∂²ΔT/∂x², i_eff in source term Q
```

### 4.3 Gradient flow diagram

```
                     ┌─────────────────────────────────┐
                     │        θ_NN  (MLP weights)       │
                     └───────────────┬─────────────────-┘
                                     │
     sensor data y_j          ┌──────▼──────┐        collocation points
          ↓                   │   ΔT_NN     │              ↓
     L_data ──────────────────►  (x, t; Λ) ◄─────────── L_pde
                               └──────┬──────┘
                                      │   ∂/∂Λ via autograd
                     ┌────────────────▼───────────────────┐
                     │  Λ = {log_α, log_i_eff, x₀, v}    │
                     │      (trainable parameters)         │
                     └────────────────────────────────────┘
```

### 4.4 Why log-space for α and i_eff

Both α and i_eff span orders of magnitude:
- α: 2.9×10⁻⁶ to 1.2×10⁻⁴  (41× range)
- i_eff: 1.25×10⁻⁴ to 5.0×10⁻³ (40× range)

Using `α = exp(log_α)` keeps values positive and makes the optimization landscape
more symmetric — a unit step in log-space corresponds to multiplication by e,
not an additive shift. This significantly improves gradient conditioning.

---

## 5. Architecture Design

### 5.1 Option A — Joint forward + inverse (recommended)

Reuse the trained forward PINN (from training-service) as the backbone.
Freeze or fine-tune network weights. Only optimize Λ.

```
Input:  (x_norm, t_norm)  +  Λ_norm(Λ)   →   ΔT_pred
```

**Advantage**: The network already "knows" the PDE structure.
**Phase 1**: Train forward PINN fully (current training-service).
**Phase 2**: For each inference query (new sensor data), run ~1000 gradient steps
            on Λ only (or jointly with a small LR for the NN).

### 5.2 Option B — Dedicated inverse network

A separate small network maps `(sensor_readings, sensor_positions, timestamps) → Λ_hat`.
Trained on synthetic data from analytical solution + noise.

```
Input:  [y₁, y₂, ..., yₙ, s₁, ..., sₖ, t₁, ..., tₘ]  →  [α̂, i_eff_hat, x̂₀, v̂]
```

**Advantage**: Inference is a single forward pass — milliseconds.
**Disadvantage**: Requires many labeled training examples; doesn't generalize outside training distribution.

### 5.3 Recommended approach: Option A with warm start

```python
# Inference time:
# 1. Load trained forward PINN (frozen weights)
# 2. Initialize Λ from physical prior or uniform random in valid range
# 3. Run Adam(Λ, lr=5e-3) for 500–2000 steps on:
#       L = L_data(Λ) + λ_pde * L_pde(Λ)
# 4. Return Λ_final + ΔT(x, t; Λ_final) for the full field
```

Typical convergence: 200–500 steps to <5% parameter error (literature benchmark).

---

## 6. Training Data Strategy

Since we have the exact analytical solution, synthetic sensor data is free:

```python
def generate_sensor_data(params, sensor_positions, t_grid, noise_std=0.001):
    """
    params = {alpha, rho_c, l, intensity, x0, v, t_total}
    Returns: {t: array, measurements: dict[sensor_pos -> array]}
    """
    for s in sensor_positions:
        dT_true = analytical_delta_T(np.array([s]), t_grid, ...)
        dT_noisy = dT_true + np.random.normal(0, noise_std, len(t_grid))
    ...
```

Noise model: σ_noise = 0.001 K (realistic for PT100 / thermocouple with good ADC).

---

## 7. Implementation Plan

```
Phase 0 (done):   Forward PINN working with <50% error
                  → training-service/trainer.py + model.py + physics.py

Phase 1 (next):   Inverse solver module
                  → training-service/inverse_solver.py
                  - class InverseSolver(forward_model, sensor_data)
                  - trainable Λ as nn.ParameterDict
                  - .solve(n_steps=1000) → Λ_hat
                  - .reconstruct_field(x_grid, t_grid) → ΔT(x,t)

Phase 2:          Inference service endpoint
                  → inference-service/main.py  POST /inverse
                  - Input: {sensor_positions, measurements, timestamps}
                  - Output: {alpha, rho_c, v, x0, temperature_field}

Phase 3:          UI panel for inverse problem
                  → ui/components/InversePanel.tsx
                  - Sensor placement widget (drag 2–3 sensors on pipe diagram)
                  - Simulated or uploaded measurement data
                  - Real-time parameter estimation display
                  - Reconstructed temperature field heatmap
```

---

## 8. Loss Weight Recommendations (from literature)

| Regime                    | λ_data | λ_pde | λ_bc |
|---------------------------|--------|-------|------|
| Many sensors (k ≥ 5)      | 100    | 1     | 10   |
| Sparse sensors (k = 2–3)  | 10     | 1     | 10   |
| Single sensor             | 1      | 1     | 10   |

With sparse sensors the PDE residual carries more regularization weight relative to data.
Higher λ_data → faster convergence but higher risk of getting stuck in local minima
if sensor coverage is poor.

**Adaptive weights** (from IAW-PINN, ScienceDirect 2025):
After every 100 steps, rebalance:
```
λ_data ← λ_data * (L_pde / L_data)^0.5    (keep contributions balanced)
```

---

## 9. Expected Performance (from literature)

| Noise level | # sensors | Parameter error (typical) | Reference                 |
|-------------|-----------|--------------------------|---------------------------|
| 0%          | 3         | < 0.1%                   | Raissi 2019               |
| 1%          | 3         | < 2%                     | AIAA SciTech 2023         |
| 5%          | 3         | < 5%                     | Emergent Mind survey 2024 |
| 10%         | 3         | < 10%                    | ASME Heat Transfer 2021   |
| 30%         | 5         | < 15% (PINNverse 2025)   | arxiv 2511.15543          |

For our problem (high-quality industrial sensors, σ_noise ~ 0.001–0.005 K, 3 sensors):
**Expected: <3% error on α and v, <5% on i_eff**.

---

## 10. Key Papers & Sources

| Paper | Key contribution | URL |
|-------|-----------------|-----|
| Raissi et al. 2019 | Original PINN forward + inverse, trainable λ₁,λ₂ in NS | JMLR |
| Wang et al. 2022/2024 | Causal PDE loss, temporal weighting | [JMLR 2024](https://www.jmlr.org/papers/volume25/24-0313/24-0313.pdf) |
| AIAA SciTech 2023 | Inverse heat conduction with PINN, sparse sensors | [arc.aiaa.org](https://arc.aiaa.org/doi/10.2514/6.2023-0537) |
| ScienceDirect 2024 | Adaptive Fourier sigma, diminishing spectral bias | [link](https://www.sciencedirect.com/science/article/abs/pii/S0893608024008153) |
| arxiv 2511.15543 | Optimal sensor placement + parameter estimation PINN | [link](https://arxiv.org/pdf/2511.15543) |
| arxiv 1908.02015 | Identifiability of source term in heat equation | [link](https://arxiv.org/pdf/1908.02015) |
| ScienceDirect 2026 | Hierarchical neural operator, unsteady heat conduction inverse | [link](https://www.sciencedirect.com/science/article/abs/pii/S0017931026000116) |
| ScienceDirect 2023 | Moving heat source detection from opposite-surface sensors | [link](https://www.sciencedirect.com/science/article/abs/pii/S0017931023009857) |
| Hard constraints 2024 | Architecture-level BC/IC enforcement | [arxiv](https://arxiv.org/html/2404.16189v2) |

---

## 11. Open Questions / Risks

1. **Local minima in Λ space**: The joint loss landscape has saddle points, especially
   when sensors are poorly placed. Mitigation: multiple random restarts for Λ, keep best.

2. **α–i_eff correlation**: Both control peak amplitude at a single time. Mitigation:
   use multiple time snapshots per sensor (not just one), or use a prior on i_eff from
   known burner power rating.

3. **v–x₀ correlation**: If measurement window is short, moving from (x₀=0.1, v=0.1)
   vs (x₀=0.15, v=0.05) produces similar sensor traces at early times.
   Mitigation: 3+ sensors spread along the pipe, or longer measurement window.

4. **Latency**: For real-time use (< 1s response), need to limit inverse solver to
   ~500 Adam steps. For offline analysis, 2000+ steps give better accuracy.

5. **Noise robustness**: σ_noise >> 0.01 K makes α estimation unreliable with only 2 sensors.
   Consider ensemble / Bayesian PINN (E-PINN) for uncertainty quantification.
