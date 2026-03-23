# Research Synthesis: Addressing Forward Model Mismatch in INR-based SoS Reconstruction

**Compiled 2026-03-22 | Deep Academic Research**

---

## Executive Summary

Six parallel research directions were investigated to address the central challenge: the straight-ray L-matrix introduces a tiny (~0.11%) but catastrophically amplified mismatch against full-wave (k-wave) data, and this mismatch is low-frequency (85-87%), making it inseparable from the signal via spectral filtering.

**The three most promising approaches, ranked by feasibility and expected impact:**

1. **Kaipio-Somersalo Approximation Error Method** — statistically characterize the model error and incorporate it into the inversion (Tier 1, implement first)
2. **SVD-Constrained INR with Mode-Dependent Regularization** — restrict the INR to the stable subspace of L and use mismatch-aware spectral weighting (Tier 1, implement alongside)
3. **Eikonal-Based Iterative L-Matrix Update** — recompute bent-ray paths using fast marching, updating L during INR optimization (Tier 2, implement after Tier 1 results)

---

## 1. The Problem Restated

| Property | Value |
|----------|-------|
| Forward model | d = L @ s, L is (131072 × 4096), straight-ray |
| Mismatch energy | 0.11% (kwave_geom), 0.44% (kwave_blob) |
| Mismatch frequency | 85-87% low-frequency — same band as signal |
| Mismatch structure | 66% systematic (shared template), 34% sample-specific |
| L condition number | 2.69 × 10^13 |
| Amplification ratio | 4.53 × 10^11 (last/first SVD mode) |
| Effective rank | 4050/4096; 90% energy in top 86 modes, 99% in top 221 |
| Available paired data | 32 samples (kwave_geom), 70 samples (kwave_blob) |
| Current baselines | L1: MAE 7.00, L2: MAE 9.33, Best INR: MAE ~5-10, Oracle INR: MAE ~1.8 |

**Root cause**: The mismatch looks like signal (can't remove it) AND the inversion amplifies it (can't survive it). The gap between Oracle (~1.8) and Best INR (~5-10) is entirely due to the physics mismatch.

---

## 2. Research Directions: Detailed Findings

### 2.1 Bent-Ray Tomography & Ray Path Correction (Reviewer Suggestion #1)

**Core insight**: Straight-ray travel times are correct to **first order** in the velocity perturbation (by Fermat's principle). The bending correction is second-order. For your tissue contrasts (5-7%), second-order effects are non-negligible (ray deflection ~2-4 pixels over a 64-pixel path).

**Recommended approach: Eikonal + Backtracing**

Rather than tracing 131,072 individual rays, solve the eikonal equation `|∇T(x)|² = s(x)²` once per source (8 solves on 64×64 grid) using the Fast Marching Method (FMM), then backtrace rays from receivers along `-∇T`.

- **Cost**: ~200ms per L-update (8 eikonal solves + 131K backtraces + sparse matrix build). Negligible vs. INR training.
- **Integration**: Update L every N=50-200 epochs during INR optimization.
- **Fermat's principle shortcut**: The Jacobian dt/ds = L_bent (evaluated on current ray paths) is mathematically exact — no need to differentiate through the ray tracer. Your existing INR backprop works unchanged with L_bent as a constant.

**Algorithm**:
```
1. Initialize s⁰ from straight-ray reconstruction
2. For each outer iteration k:
   a. Extract current SoS: c = 1/INR(grid_coords), reshape to 64×64
   b. For each source i (i = 1..8):
      - Solve eikonal: |∇T_i|² = 1/c² with T_i(x_source) = 0
      - Backtrace rays from all receivers along -∇T_i
   c. Rebuild L_bent as sparse matrix from traced paths
   d. Continue INR training with L_bent for N epochs
```

**When bent-ray is NOT enough**: If residual mismatch remains, it's from finite-frequency effects (Fresnel zone width ~8 pixels for your setup), diffraction at sub-wavelength structures, or multiple scattering. These require wave-theoretic corrections (Section 2.6).

**Key references**:
- Andersen & Kak (1982) — bent-ray ultrasound tomography
- Sethian (1996) — Fast Marching Method
- Li et al. (2009) — FMM for bent-ray breast USCT
- `scikit-fmm` Python package for eikonal solving

---

### 2.2 Ray Selection & Weighting (Reviewer Suggestion #2)

**Critical finding: Standard ray selection/weighting WILL NOT solve your problem.** The mismatch is:
- **Uniform** across rays (1.2-1.3× max/min ratio across firing pairs)
- **Systematic** (66% shared template, not random outliers)
- **Low-frequency** (no spectral signature to identify "bad" rays)

There is no subset of rays that is mismatch-free. Robust loss functions (Huber, Tukey) address outlier noise, not systematic model error.

**What DOES work: SVD-domain mode weighting** (asking "which inversion modes amplify the mismatch most?" instead of "which rays are bad?"):

```
Loss = Σᵢ wᵢ² (uᵢᵀd - σᵢvᵢᵀs)²

where wᵢ = σᵢ² / (σᵢ² + λ / min(1, SMRᵢ))
and SMRᵢ = |uᵢᵀd| / (|uᵢᵀε_template| + η)
```

This selectively suppresses SVD modes where mismatch dominates signal while preserving well-constrained modes.

**Ancillary value of ray weighting**: Template subtraction (removing the 66% systematic component) followed by Huber IRLS on the residual can provide modest additional improvement on top of template correction.

**Key references**:
- Hansen (1998) — SVD filtering for ill-posed problems
- Rawlinson et al. (2010) — iterative reweighting in travel-time tomography (confirms that reweighting cannot fix systematic model error)

---

### 2.3 SVD-Constrained Reconstruction (Your Phase 2)

**Implementation recommendation: INR + Hard Projection (Option B)**

```python
s_raw = INR(coords)                     # (4096, 1)
alpha = V_K.T @ s_raw                   # project to K-dim subspace
s_proj = V_K @ alpha                    # back to pixel space
d_pred = (U_K * S_K) @ alpha            # reduced forward model (fast!)
```

**Choosing K**:
- Because 85% of mismatch is low-frequency and concentrated in modes 1-50, using too small a K is counterproductive — you don't escape the mismatch, you only lose signal.
- **Recommended sweep**: K ∈ {50, 86, 100, 150, 200, 220, 300, 500, 1000, 2000, 4096}
- **Expected optimal**: K ~ 200-400, balancing signal recovery vs. mismatch amplification
- Use the **discrepancy principle**: choose K such that ||L·s_TSVD - d|| ≈ ||ε||

| K | Energy | Resolution | Cond(reduced) | Behavior |
|---|--------|-----------|---------------|----------|
| 86 | 90% | ~4-8 px | ~10² | Smooth, gross anatomy, ultra-stable |
| 221 | 99% | ~2-3 px | ~10³-10⁴ | Good boundaries, recommended start |
| 500 | 99.9%+ | ~1-2 px | ~10⁵-10⁷ | Near-full resolution, mild artifacts |
| 4096 | 100% | full | 10¹³ | Catastrophic amplification |

**Advanced techniques**:
- **Progressive K** (borrowed from multi-scale seismic FWI): Start K=50, increase by 50 every 200 iterations. INR learns coarse structure first, then adds detail.
- **Soft projection**: Gaussian taper `wᵢ = exp(-(i-K)²/2τ²)` for i > K instead of hard cutoff.
- **Wiener filter** (oracle baseline): Mode-dependent filter estimated from ground truth `σ²_signal_i / (σ²_signal_i + σ²_mismatch_i)`. Gives the theoretically optimal linear estimator to beat.

**Interaction with INR spectral bias**: INR spectral bias (low-frequency-first in spatial domain) and SVD subspace (top modes of L) are approximately aligned but not identical. Double regularization should be more stable than either alone. Monitor for over-regularization by comparing against pure TSVD baseline.

**Key references**:
- Hansen (1998, 2010) — TSVD theory and parameter choice
- Schwab, Antholzer & Haltmeier (2019) — deep null-space learning
- Dittmer et al. (2020) — regularization by architecture
- Bunks et al. (1995) — progressive frequency inclusion in seismic FWI

---

### 2.4 Data-Domain Correction (Kaipio-Somersalo Approximation Error)

**This is the most theoretically principled approach for your specific problem.**

**Core idea**: Don't try to remove the model error — characterize it statistically and incorporate those statistics into the inversion.

**Mathematical framework** (Kaipio & Somersalo, 2006):

True model: `d = A(s) + e`
Simplified model: `d = L·s + ε(s) + e` where `ε(s) = A(s) - L·s`

Characterize ε by its first two moments from your 32 paired samples:
```
μ_ε = (1/32) Σᵢ εᵢ             (mean model error = your template)
Γ_ε = (1/31) Σᵢ (εᵢ - μ_ε)(εᵢ - μ_ε)ᵀ   (covariance)
```

The MAP estimate becomes:
```
ŝ = argmin_s  (d - μ_ε - L·s)ᵀ (Γ_ε + Γ_e)⁻¹ (d - μ_ε - L·s)  +  λ·R(s)
```

This is your existing weighted least-squares problem with two modifications:
1. **Mean-corrected data**: subtract μ_ε from d_meas
2. **Structured weighting**: use (Γ_ε + Γ_e)⁻¹ instead of identity

**Practical computation** (handles the 131072 × 131072 covariance):
1. Compute εᵢ for all 32 samples, subtract mean
2. PCA on residuals: Γ_ε ≈ U_K Λ_K U_Kᵀ + σ²_floor · I (at most 31 nonzero components)
3. Woodbury identity for inversion: (UΛUᵀ + σ²I)⁻¹ = (1/σ²)I - (1/σ⁴)U(Λ⁻¹ + (1/σ²)I)⁻¹Uᵀ
4. Leave-one-out evaluation: for sample j, compute μ_ε and Γ_ε from the other 31 samples

**Why this is ideal**:
- **Small training set is sufficient** — only needs first two moments (mean + top 10-20 PCA modes)
- **Theoretically rigorous** — Bayesian foundations with known convergence properties
- **Compatible with all solvers** — L1/L2/INR work with modified inner product
- **Addresses amplification directly** — downweights measurement directions with high model error variance
- **No neural network training required**

**Integration with INR**: Modify the training loss:
```python
# Instead of: loss = ||d_meas - L @ s||²
# Use:        loss = ||d_meas - μ_ε - L @ s||²_W
residual = d_meas - mu_epsilon - L @ s_pred
loss = residual @ W @ residual  # W = (Γ_ε + σ²I)⁻¹
```

**Key references**:
- Kaipio & Somersalo (2006) — "Statistical and Computational Inverse Problems," Ch. 7
- Kaipio & Somersalo (2007) — "Statistical inverse problems: Discretization, model reduction and inverse crimes"
- Calvetti, Dunlop & Somersalo (2020) — iterative model error updating
- Tick, Pulkkinen & Tarvainen (2020) — approximation error for SoS in photoacoustic tomography
- Lunz et al. (2021) — "On Learned Operator Correction in Inverse Problems"

---

### 2.5 Joint Correction + Reconstruction (Fixing Phase 1 Failure)

**Why Phase 1 (Bias-Absorber) failed**: Gradient descent follows the path of least resistance. The bias INR has a direct, well-conditioned path to the loss (`d INR_bias / d θ_bias`), while the anatomy INR must pass through the ill-conditioned L (`L^T · residual`). The optimizer naturally favors updating the bias network — this is a structural problem, not a hyperparameter tuning issue.

**The literature converges on three strategies to prevent shortcut learning:**

#### Strategy A: Architectural Capacity Control (Most Important — Hard Constraint)

Make it **physically impossible** for the correction to represent the signal:
- **INR_anatomy**: 8 layers, 256 units, L_encoding = 10 (high-frequency Fourier features)
- **INR_bias**: 2 layers, 16 units, L_encoding = 2 (only very smooth functions)
- Or use **low-rank parameterization**: r = U_basis @ α (K-dimensional, K ~ 10-50)

This is a hard constraint that cannot be defeated by optimization dynamics.

#### Strategy B: Staged Training (Second Most Important — Temporal Separation)

```
Phase 1 (Burn-in):       r = 0, train INR_anatomy only for N1 iterations
Phase 2 (Correction):    Freeze anatomy, train INR_bias on residual d - L@s
Phase 3 (Joint tuning):  Unfreeze both, asymmetric LRs (lr_anatomy = 10 × lr_bias)
```

The anatomy network must explain as much data as possible before the correction is introduced.

#### Strategy C: Penalty Calibration (Supporting Role Only)

Set τ in `τ·||r||²` based on the expected mismatch energy (~0.11% of ||d||²). But **soft penalties alone are insufficient** — your Phase 1 failure confirms this. Must combine with Strategy A or B.

**Alternating minimization is more stable than joint optimization** for this architecture:
```
Step 1: Fix r, solve min_s ||L@s + r - d||² + λ·reg(s)
Step 2: Fix s, solve min_r ||L@s + r - d||² + τ·||r||²  →  r* = (d - L@s) / (1 + τ)
```

**Key references**:
- Gilton, Ongie & Willett (2021) — model adaptation for inverse problems
- arXiv:2403.04847 — untrained NNs for model mismatch
- Tancik et al. (2020) — frequency control via Fourier features

---

### 2.6 Full-Waveform & Hybrid Physics Corrections

**Two approaches that can work within the pre-evaluated L-matrix constraint:**

#### A. Finite-Frequency (Banana-Doughnut) L-Matrix Replacement

**Perhaps the single most impactful correction.** Replace the infinitely-thin ray kernel with the wave-theoretic finite-frequency sensitivity kernel:

```
L_FF[i,j] = ∫_{pixel_j} K(x; s_i, r_i) dx
```

where K is the banana-doughnut kernel (from seismology). Key properties:
- **Zero sensitivity ON the geometric ray** (the "doughnut hole")
- Maximum sensitivity in the first Fresnel zone, slightly off-ray
- Width scales as √(λ·D) where λ ~ 0.5mm (at 3MHz), D ~ 32mm → width ~ 4mm ~ 8 pixels

**For a homogeneous background in 2D:**
```python
K(x; x_s, x_r, ω, c₀) = -2k²/c₀ · Im{ G(x,x_s) · G(x_r,x) / G(x_r,x_s) }
```
where G are 2D Green's functions (Hankel functions).

- **Same matrix dimensions as L_ray** — drop-in replacement
- **One-time precomputation** (seconds to minutes)
- **Captures Fresnel zone physics**: diffraction, finite aperture, frequency-dependent resolution

#### B. Rytov Approximation L-Matrix

The Rytov approximation works with the complex phase of the wave field, giving a linear relationship between SoS perturbations and scattered field data that naturally captures diffraction:

```
L_Rytov[i,j] = Rytov kernel integrated over pixel j
```

More rigorous than banana-doughnut but complex-valued (requires handling phase/amplitude separately).

**Key references**:
- Dahlen et al. (2000) — Fréchet kernels for finite-frequency travel times
- Woodward (1992) — wave-equation tomography
- Marquering et al. (1999) — banana-doughnut kernels
- Devaney (1981) — Rytov approximation for inverse scattering

---

## 3. Prioritized Research Roadmap

### Tier 1: Implement First (1-2 weeks, highest expected impact)

| # | Approach | What to Do | Expected Impact |
|---|----------|-----------|----------------|
| **1A** | **Kaipio-Somersalo Approximation Error** | Compute μ_ε and Γ_ε from 32 samples. Solve weighted least-squares with mean-corrected data. LOO evaluation. | Directly addresses amplification; rigorous; no NN training |
| **1B** | **SVD-Constrained INR** | Implement INR + V_K projection. Sweep K ∈ {86,150,200,300,500}. Compare against TSVD/LSQR baselines. | Prevents mismatch amplification in tail modes |
| **1C** | **SVD-Domain Mismatch-Aware Loss** | Weight loss by per-mode signal-to-mismatch ratio (SMRᵢ). Requires ε_template projected to SVD space. | Best spectral control of what's preserved vs. suppressed |

**These three can be combined**: Use Kaipio-Somersalo mean correction + SVD subspace projection + mode-weighted loss together.

### Tier 2: Implement Second (2-4 weeks)

| # | Approach | What to Do | Expected Impact |
|---|----------|-----------|----------------|
| **2A** | **Eikonal-Based L-Matrix Update** | Install scikit-fmm. Implement FMM solve + backtrace + L rebuild. Update L every N epochs during INR training. | Corrects the physics directly; ~10-30% improvement at boundaries |
| **2B** | **Finite-Frequency L-Matrix** | Compute banana-doughnut kernels for all source-receiver pairs. Build L_FF as drop-in replacement for L_ray. | Captures wave physics the ray model misses; one-time precomputation |
| **2C** | **Joint Correction + Reconstruction (Fixed Phase 1)** | Implement with architectural capacity control (tiny INR_bias) + staged training + alternating minimization | Addresses sample-specific mismatch component |

### Tier 3: Exploratory (4+ weeks)

| # | Approach | What to Do | Expected Impact |
|---|----------|-----------|----------------|
| **3A** | **Generate more k-wave simulations** | Run 200-500 additional k-wave simulations to train a correction U-Net (Lozenski approach) | Enables supervised learning of the correction |
| **3B** | **Neural operator for forward model correction** | Train FNO/DeepONet to predict d_wave - L·s from simulation data | Fast differentiable correction; requires training data |
| **3C** | **Learned spectral regularization** | Learn optimal per-mode filter g(σᵢ) from paired data (Alberti et al., 2021) | Subsumes TSVD, Tikhonov, Wiener — data-adaptive |

---

## 4. Concrete Implementation Plan for Tier 1

### Step 1: Precompute SVD and Mismatch Statistics

```python
# One-time computation (save to disk)
L_valid = L[mask]  # (61843, 4096)
U, S, Vt = torch.linalg.svd(L_valid, full_matrices=False)
V = Vt.T  # (4096, 4096)

# Mismatch statistics from 32 kwave_geom samples
epsilons = [d_meas[i] - L @ s_true[i] for i in range(32)]
mu_epsilon = np.mean(epsilons, axis=0)
residuals = [eps - mu_epsilon for eps in epsilons]
# PCA of residuals for covariance
U_eps, S_eps, _ = np.linalg.svd(np.stack(residuals), full_matrices=False)
# Keep top K_eps components (K_eps ≤ 31)
```

### Step 2: Kaipio-Somersalo Corrected Inversion

```python
# Mean-corrected data
d_corrected = d_meas - mu_epsilon

# Weighted reconstruction using (Γ_ε + σ²I)⁻¹ via Woodbury
# ... integrate into existing L1/L2 solver with modified norm

# Leave-one-out evaluation across 32 samples
```

### Step 3: SVD-Constrained INR

```python
# Modify engines.py: add subspace projection after INR output
V_K = V[:, :K].to(device)  # (4096, K)
s_raw = model(sample.coords)
alpha = V_K.T @ s_raw
s_proj = V_K @ alpha
d_pred = (U[:, :K] * S[:K]) @ alpha  # fast reduced forward model
```

### Step 4: Combine

```python
# Combined loss: KS correction + SVD weighting + subspace projection
d_corrected = d_meas - mu_epsilon
s_raw = model(sample.coords)
alpha = V_K.T @ s_raw
d_pred_svd = S[:K] * alpha  # in SVD coordinates
d_target_svd = U[:, :K].T @ d_corrected  # target in SVD coordinates
mode_weights = compute_smr_weights(d_target_svd, eps_mode_template[:K])
loss = (mode_weights * (d_pred_svd - d_target_svd)**2).sum()
```

---

## 5. Key Insight: Why These Approaches Work Together

The three Tier 1 approaches attack the problem at different levels:

```
                          ┌─────────────────────────────────┐
                          │  MEASUREMENT SPACE               │
                          │  Kaipio-Somersalo corrects       │
   d_meas ──────────────► │  mean bias + covariance weighting │
                          └───────────┬─────────────────────┘
                                      │ d_corrected
                          ┌───────────▼─────────────────────┐
                          │  SVD SPACE                        │
                          │  Mode-weighted loss suppresses    │
                          │  modes where mismatch dominates   │
                          └───────────┬─────────────────────┘
                                      │ weighted residual
                          ┌───────────▼─────────────────────┐
                          │  IMAGE SPACE                      │
                          │  SVD subspace projection keeps    │
                          │  INR in stable reconstruction     │
                          │  manifold                         │
                          └─────────────────────────────────┘
```

1. **Kaipio-Somersalo** removes the known systematic mismatch before inversion
2. **SVD mode weighting** handles the residual mismatch by suppressing its amplification
3. **Subspace projection** prevents the INR from generating patterns in the amplification-prone tail

---

## 6. Comprehensive Reference List

### Foundational Theory
- Hansen (1998) "Rank-Deficient and Discrete Ill-Posed Problems." SIAM.
- Kaipio & Somersalo (2006) "Statistical and Computational Inverse Problems." Springer, Ch. 7.
- Engl, Hanke & Neubauer (1996) "Regularization of Inverse Problems." Kluwer.
- Cerveny (2001) "Seismic Ray Theory." Cambridge University Press.

### Approximation Error Method
- Kaipio & Somersalo (2007) "Statistical inverse problems: Discretization, model reduction and inverse crimes." J. Comp. Applied Math.
- Lehikoinen et al. (2007) "Approximation errors and model reduction." Inverse Problems.
- Calvetti, Dunlop & Somersalo (2020) "Iterative Updating of Model Error for Bayesian Inversion." Inverse Problems.
- Tick, Pulkkinen & Tarvainen (2020) "Modelling of errors due to SoS variations in PAT." Biomed. Phys. Eng. Express.

### SVD / Subspace Methods
- Halko, Martinsson & Tropp (2011) "Finding Structure with Randomness." SIAM Review.
- Schwab, Antholzer & Haltmeier (2019) "Deep Null Space Learning." Inverse Problems.
- Dittmer et al. (2020) "Regularization by Architecture." J. Math. Imaging Vision.
- Alberti et al. (2021) "Learning the Optimal Tikhonov Regularizer." NeurIPS.
- Lieberman, Willcox & Ghattas (2010) "Parameter and State Model Reduction." SIAM Review.

### Bent-Ray / Eikonal
- Andersen & Kak (1982) "Digital ray tracing in 2D refractive fields." JASA.
- Sethian (1996) "A fast marching level set method." PNAS.
- Zhao (2005) "A fast sweeping method for eikonal equations." Math. Comp.
- Li et al. (2009) "Refraction corrected transmission USCT." Med. Phys.
- Rau et al. (2021) "SoS imaging using first-arrival USCT." IEEE TUFFC.

### Finite-Frequency / Wave-Theoretic Corrections
- Woodward (1992) "Wave-equation tomography." Geophysics.
- Dahlen et al. (2000) "Fréchet kernels for finite-frequency traveltimes." GJI.
- Marquering et al. (1999) "Banana-doughnut kernels." GJI.
- Devaney (1981) "Inverse-scattering theory within the Rytov approximation."
- Perez-Liva et al. (2017) "Rytov-based USCT." JASA.

### Joint Correction / Model Mismatch
- Gilton, Ongie & Willett (2021) "Model Adaptation for Inverse Problems." IEEE TCI.
- arXiv:2403.04847 — "Solving Inverse Problems with Model Mismatch using Untrained NNs."
- Lunz et al. (2021) "On Learned Operator Correction." SIAM J. Imaging Sci.
- Lozenski et al. (2025) "Learned Correction Methods for USCT." arXiv:2502.09546.
- Darestani & Heckel (2022) "Test-Time Training for Compressed Sensing." ICML.

### Robust / Weighted Estimation
- Rawlinson et al. (2010) "Seismic tomography." Physics of Earth and Planetary Interiors.
- Aster, Borchers & Thurber (2018) "Parameter Estimation and Inverse Problems." 3rd ed.

### USCT-Specific
- Sanabria et al. (2018) "Spatial domain SoS reconstruction." PMB.
- Hormati et al. (2010) "Robust ultrasound travel-time tomography." IEEE TMI.
- Bernhardt et al. (2023) "DNN with Tikhonov pseudo-inverse prior."
- Stahli et al. (2020) "Improved Forward Model." Ultrasonics.

### Neural Network / INR Methods
- Tancik et al. (2020) "Fourier Features Let Networks Learn High Frequency Functions."
- Sitzmann et al. (2020) "SIREN." NeurIPS.
- Bunks et al. (1995) "Multiscale seismic FWI." Geophysics.
- Adler & Oktem (2017) "Iterative deep NNs for inverse problems." Inverse Problems.
- Aggarwal et al. (2019) "MoDL." IEEE TMI.

---

## 7. What Won't Work (and Why)

| Approach | Why It Fails |
|----------|-------------|
| Standard denoising (DIP, spectral bias) | Mismatch is same frequency as signal; Architecture 1 already failed (MAE=40.53) |
| Huber/robust loss alone | Mismatch is systematic, not outlier-type; all rays similarly affected |
| Ray selection by geometry | Mismatch is uniform across rays (1.2-1.3× variation); no "clean" subset |
| Template subtraction alone | Only captures 66% of mismatch; remaining 34% + amplification still dominates |
| Very small TSVD (K < 100) | Mismatch is concentrated in modes 1-50; small K loses signal without escaping mismatch |
| Unconstrained dual-INR (Phase 1) | Shortcut learning — correction absorbs signal due to gradient path imbalance |
