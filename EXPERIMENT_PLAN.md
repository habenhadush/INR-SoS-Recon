# Experiment Plan: Forward Model Mismatch Correction

**Created**: 2026-03-23 | **Base branch**: `k-wave-validation`
**Based on**: `research-synthesis.md`

**Goal**: Close the gap between Oracle INR and current best, on both datasets.

### Datasets

| | kwave_geom | kwave_blob |
|---|---|---|
| **Samples** | 32 | 70 |
| **Shapes** | Simple geometric (circles, rectangles) | Realistic blobs, intricate boundaries |
| **SoS range (m/s)** | 1400–1550 | 1415–1618 |
| **Contrast (m/s)** | 45.0 ± 17.2 | 62.6 ± 38.0 |
| **Valid rays** | 54.3% ± 0.7% | 47.2% ± 6.3% |
| **A-matrix** | Embedded in .mat | External A.mat |
| **Mismatch energy** | 0.11% | 0.44% |
| **Difficulty** | Easier (development) | Harder (validation) |

**Strategy**: Develop and tune on kwave_geom (simpler, faster iteration). Validate on kwave_blob (realistic, the real test). A method that works on geom but fails on blob is insufficient.

### Baselines

**kwave_geom** (32 samples):

| Method | CNR | SSIM | RMSE (m/s) | MAE (m/s) |
|--------|-----|------|------------|-----------|
| L1 (LASSO) | TBD | 0.649 | 10.35 ± 2.96 | 7.00 ± 2.58 |
| L2 (Tikhonov) | TBD | 0.726 | 12.01 ± 2.03 | 9.33 ± 1.64 |
| Best INR sweep | TBD | TBD | TBD | ~5-10 |
| Oracle INR (direct GT) | TBD | TBD | TBD | ~1.8 |

**kwave_blob** (70 samples):

| Method | CNR | SSIM | RMSE (m/s) | MAE (m/s) |
|--------|-----|------|------------|-----------|
| L1 (LASSO) | TBD | 0.612 | 28.8 ± 14.0 | 24.0 ± 13.7 |
| L2 (Tikhonov) | TBD | 0.673 | 24.1 ± 10.5 | 19.1 ± 10.8 |
| Best INR sweep | TBD | TBD | TBD | TBD |
| Oracle INR (direct GT) | TBD | TBD | TBD | TBD |

**First task**: Re-evaluate all baselines with CNR on both datasets to establish ground truth for comparison.

**Evaluation protocol**: Leave-one-out on 32 kwave_geom samples. Report all four metrics. Compare every experiment against L1 baseline and Oracle.

**Primary metrics (in order of importance)**:
1. **CNR** (Contrast-to-Noise Ratio) — Can the inclusion be distinguished from the background? A reconstruction that blurs the inclusion into the background is a failure regardless of MAE. Uses Otsu segmentation on GT for ROI/background masks.
2. **SSIM** — Structural similarity captures perceptual quality (edges, contrast, luminance). More meaningful than pixel-wise metrics for image reconstruction.
3. **RMSE** — Penalizes large errors more than MAE. A few badly reconstructed pixels matter.
4. **MAE** — Pixel-wise average error. Useful for comparison with prior work but can be misleadingly low if the reconstruction is over-smoothed to the mean background SoS.

**Failure mode to watch**: Low MAE + low CNR = reconstruction collapsed to background. This means the method "gave up" on the inclusion and just predicted ~1500 m/s everywhere. Always check CNR alongside MAE.

---

## Experiment 1: Kaipio-Somersalo Approximation Error Method

**Branch**: `experiment/kaipio-somersalo`
**Priority**: Tier 1 — implement first
**Status**: [ ] NOT STARTED

### Hypothesis

Statistically characterizing the model error (mean + covariance) and incorporating it into the inversion will reduce the mismatch amplification without requiring any neural network training.

### Method

1. Compute ε_i = d_meas_i − L @ s_true_i for all 32 samples
2. Compute μ_ε = mean(ε_i) — the systematic mismatch template
3. Compute Γ_ε via PCA of (ε_i − μ_ε) — low-rank covariance (≤ 31 components)
4. Solve modified problem: argmin_s (d − μ_ε − L·s)ᵀ W (d − μ_ε − L·s) + λ·‖s‖₁
   where W = (Γ_ε + σ²I)⁻¹ via Woodbury identity
5. Leave-one-out: for sample j, compute statistics from the other 31

### Sub-experiments

- [ ] **1a**: Template subtraction only (d_corrected = d − μ_ε), then standard L1/L2 solve
- [ ] **1b**: Template + covariance weighting (full Kaipio-Somersalo)
- [ ] **1c**: Template + covariance + INR reconstruction (combine with existing INR pipeline)
- [ ] **1d**: Vary number of PCA components K_ε ∈ {5, 10, 15, 20, 31}

### Success criteria

- CNR improves over L1 baseline (inclusion is more visible, not blurred away)
- SSIM > 0.649 (geom) / 0.612 (blob)
- MAE < 7.00 (geom) / < 24.0 (blob)

### Results — kwave_geom

| Sub-exp | CNR | SSIM | RMSE | MAE | Notes |
|---------|-----|------|------|-----|-------|
| 1a | — | — | — | — | |
| 1b | — | — | — | — | |
| 1c | — | — | — | — | |
| 1d | — | — | — | — | |

### Results — kwave_blob

| Sub-exp | CNR | SSIM | RMSE | MAE | Notes |
|---------|-----|------|------|-----|-------|
| 1a | — | — | — | — | |
| 1b | — | — | — | — | |
| 1c | — | — | — | — | |
| 1d | — | — | — | — | |

---

## Experiment 2: SVD-Constrained INR

**Branch**: `experiment/svd-constrained-inr`
**Priority**: Tier 1 — implement alongside Exp 1
**Status**: [ ] NOT STARTED

### Hypothesis

Projecting the INR output onto the top-K right singular vectors of L prevents the network from generating patterns in the catastrophically ill-conditioned tail modes (modes 3800-4096), reducing mismatch amplification while preserving spatial resolution.

### Method

1. Precompute SVD: L = U Σ Vᵀ (one-time, save to disk)
2. Modify INR forward pass: s_proj = V_K @ (V_Kᵀ @ s_raw)
3. Use reduced forward model: d_pred = (U_K · Σ_K) @ α where α = V_Kᵀ @ s_raw
4. Sweep truncation level K

### Sub-experiments

- [ ] **2a**: Classical TSVD baseline (no INR) — sweep K ∈ {86, 150, 200, 300, 500, 1000}
- [ ] **2b**: LSQR with early stopping baseline — sweep iterations ∈ {50, 100, 200, 500}
- [ ] **2c**: INR + hard projection — sweep K ∈ {86, 150, 200, 300, 500}
- [ ] **2d**: INR + progressive K (start K=50, increase by 50 every 200 iterations)
- [ ] **2e**: INR + soft projection (Gaussian taper beyond K)

### Success criteria

- INR + projection beats pure TSVD (proving INR adds value beyond linear)
- CNR improves (inclusion preserved, not smoothed away by truncation)
- Identify optimal K range where CNR and SSIM peak (not just MAE)

### Results — kwave_geom

| Sub-exp | K | CNR | SSIM | RMSE | MAE | Notes |
|---------|---|-----|------|------|-----|-------|
| 2a | — | — | — | — | — | |
| 2b | — | — | — | — | — | |
| 2c | — | — | — | — | — | |
| 2d | — | — | — | — | — | |
| 2e | — | — | — | — | — | |

### Results — kwave_blob

| Sub-exp | K | CNR | SSIM | RMSE | MAE | Notes |
|---------|---|-----|------|------|-----|-------|
| 2a | — | — | — | — | — | |
| 2b | — | — | — | — | — | |
| 2c | — | — | — | — | — | |
| 2d | — | — | — | — | — | |
| 2e | — | — | — | — | — | |

---

## Experiment 3: SVD-Domain Mismatch-Aware Loss

**Branch**: `experiment/svd-mismatch-loss`
**Priority**: Tier 1 — implement alongside Exp 1 & 2
**Status**: [ ] NOT STARTED

### Hypothesis

Weighting the loss by per-mode signal-to-mismatch ratio (SMR) selectively suppresses SVD modes where the mismatch dominates, providing finer control than uniform truncation.

### Method

1. Project mismatch template into SVD space: ε_mode_i = uᵢᵀ @ ε_template
2. Compute per-mode SMR: SMR_i = |uᵢᵀd| / (|ε_mode_i| + η)
3. Mode-weighted loss: Loss = Σᵢ wᵢ · (uᵢᵀd − σᵢ vᵢᵀs)²
   where wᵢ = σᵢ² / (σᵢ² + λ / min(1, SMR_i))

### Sub-experiments

- [ ] **3a**: SVD-weighted loss with standard INR (no subspace projection)
- [ ] **3b**: SVD-weighted loss + subspace projection (combine with Exp 2)
- [ ] **3c**: Wiener filter baseline (oracle — uses ground truth signal power per mode)
- [ ] **3d**: Learned spectral filter (fit g(σᵢ) to minimize LOO error on 32 samples)

### Success criteria

- Beats uniform TSVD truncation on CNR (inclusion visibility) and SSIM
- Mode weighting + INR approaches Oracle performance

### Results — kwave_geom

| Sub-exp | CNR | SSIM | RMSE | MAE | Notes |
|---------|-----|------|------|-----|-------|
| 3a | — | — | — | — | |
| 3b | — | — | — | — | |
| 3c | — | — | — | — | |
| 3d | — | — | — | — | |

### Results — kwave_blob

| Sub-exp | CNR | SSIM | RMSE | MAE | Notes |
|---------|-----|------|------|-----|-------|
| 3a | — | — | — | — | |
| 3b | — | — | — | — | |
| 3c | — | — | — | — | |
| 3d | — | — | — | — | |

---

## Experiment 4: Combined Tier 1 (Best of Exp 1-3)

**Branch**: `experiment/combined-tier1`
**Priority**: Tier 1 — after Exp 1-3 results
**Status**: [ ] NOT STARTED

### Hypothesis

The three Tier 1 approaches attack different levels of the problem and should stack:
- Kaipio-Somersalo corrects known systematic mismatch in measurement space
- SVD mode weighting handles residual mismatch by suppressing amplification
- Subspace projection keeps INR in the stable reconstruction manifold

### Method

```
d_corrected = d_meas − μ_ε                     (mean correction)
Loss = Σᵢ wᵢ · (uᵢᵀ d_corrected − σᵢ αᵢ)²    (mode-weighted)
s = V_K @ α                                     (subspace projected)
```

### Sub-experiments

- [ ] **4a**: Best config from each of Exp 1-3 combined
- [ ] **4b**: Joint hyperparameter sweep on combined approach
- [ ] **4c**: Evaluate on kwave_blob (70 samples) for generalization

### Success criteria

- CNR significantly higher than individual Tier 1 experiments
- Approach Oracle performance on kwave_geom
- Meaningful improvement on kwave_blob (the real test — intricate shapes must be resolved)

### Results

| Sub-exp | Dataset | CNR | SSIM | RMSE | MAE | Notes |
|---------|---------|-----|------|------|-----|-------|
| 4a | kwave_geom | — | — | — | — | |
| 4a | kwave_blob | — | — | — | — | |
| 4b | kwave_geom | — | — | — | — | |
| 4b | kwave_blob | — | — | — | — | |

---

## Experiment 5: Eikonal-Based L-Matrix Update (Bent-Ray)

**Branch**: `experiment/eikonal-bent-ray`
**Priority**: Tier 2 — after Tier 1 results
**Status**: [x] COMPLETE — FAILED (fundamental mismatch between ray-traced and beamformed L)

### Hypothesis

Recomputing the L-matrix using bent-ray paths (via Fast Marching eikonal solver) corrects the physics directly, reducing the mismatch at its source rather than mitigating it statistically.

### Method (as implemented)

Sub-experiments were restructured during implementation to diagnose the failure incrementally:

- **5a**: Baseline — train INR with original (real) L-matrix (control)
- **5b**: Straight-ray Siddon L, per-row normalized to match real L row norms
- **5c**: Eikonal bent-ray L (scikit-fmm + gradient backtrace), per-row normalized
- **5d (planned)**: Gaussian-broadened Siddon L + per-row normalized — abandoned after broadening analysis showed correlation ceiling of 0.097

### Geometry Exploration (Phase 0)

Comprehensive notebook analysis (`notebooks/experiment5_geometry_exploration.ipynb`) identified:
- DT pixel indexing convention: `row_idx = ix * 128 + iz`
- 8 element pairs with δ_channel=17: (27,44), (37,54), (45,62), (54,71), (55,72), (63,80), (71,88), (80,97)
- Sign convention: L = ray(right_elem → pixel) - ray(left_elem → pixel)
- Grid: SoS 64×64 (0.6mm), DT 128×128 (0.3mm), elements at z=0

### Why It Failed — Root Cause Analysis

**The real L-matrix is NOT a ray-tracing object.** It encodes beamforming sensitivity:

1. **Rasterization mismatch**: Siddon ray tracing produces thin 1-pixel lines (~1.3% nnz per row). The real L has broad sensitivity regions (~13.9% nnz per row). The spatial support is 10× too narrow.

2. **Per-row normalization cannot fix support mismatch**: Matching row magnitudes preserves direction but the direction itself is wrong — the sensitivity pattern from beamforming (coherent summation across many elements in a sub-aperture) is fundamentally different from a single element-to-pixel ray.

3. **Gaussian broadening fails**: Sweeping isotropic Gaussian blur σ from 0.5 to 8.0 pixels, the best row-wise correlation between blurred Siddon and real L was only **0.097** (essentially zero). The real L is not a blurred version of a ray — it encodes multi-element beamforming physics that cannot be replicated by any spatial filter on a geometric ray.

4. **The real L encodes**: coherent sub-aperture summation, steering delays, apodization, the beamformer PSF, and wave-physics effects (diffraction, interference). A Siddon ray from one element to one pixel is the wrong abstraction entirely.

**Conclusion**: Building our own L from ray tracing (straight or bent) is a dead end for this measurement geometry. The iterative bent-ray approach from USCT literature works for ring-array setups where the L IS ray-based, but not for CUTE where the L comes from beamforming.

### Results — kwave_geom (2 debug samples)

| Sub-exp | Description | CNR | SSIM | RMSE | MAE | Notes |
|---------|-------------|-----|------|------|-----|-------|
| 5a | Baseline (real L) | 2.36 | 0.77 | 7.86 | 3.01 | Control — expected performance |
| 5b | Siddon straight + norm | 0.13 | 0.87 | 23.43 | 9.28 | No inclusion localization, fan artifacts |
| 5c | Eikonal bent + norm | 0.10 | 0.71 | 22.18 | 9.86 | Noisy, speckled, no inclusion |
| 5d | Broadened Siddon | — | — | — | — | Abandoned: broadening corr=0.097 |

### What Would Make This Work — Questions for Deniz/Orcun

The approach fails because we cannot replicate the beamformed L from ray tracing. To unblock this direction, we need:

1. **The code that generates the L-matrix (A-matrix).** Is it analytically derived from the beamforming model or numerically computed (finite-difference Jacobian)? With the generation code, we could modify it to use eikonal-based travel times while preserving the beamforming structure.

2. **Which elements contribute to each DT pixel?** Our experiment showed the sensitivity is much broader than just the two firing-pair elements. Understanding the sub-aperture, steering delays, and apodization would let us build a correct forward model.

3. **Can L be recomputed for a non-homogeneous SoS background?** Currently L is linearized around homogeneous 1510 m/s. If recomputable for an arbitrary SoS map, that would be the beamforming-native equivalent of "bent-ray L" — exactly what we need.

4. **Is the L computation differentiable (or could it be)?** If so, end-to-end optimization through the beamforming model would bypass the need for a fixed L entirely.

Without the code, a mathematical description of how L is formed (the exact delay-and-sum formula, sub-aperture definition, apodization weights) would let us reimplement it.

### Key Files

- `notebooks/experiment5_geometry_exploration.ipynb` — geometry exploration + broadening analysis (Sections 1-16)
- `scripts/run_eikonal_bent_ray.py` — experiment script with sub-experiments 5a-5d
- `scripts/data/experiment5_eikonal/kwave_geom/20260326_184929/` — results and plots

---

## Experiment 6: Finite-Frequency L-Matrix (Banana-Doughnut)

**Branch**: `experiment/finite-frequency-L`
**Priority**: Tier 2 — after Tier 1 results
**Status**: [ ] NOT STARTED

### Hypothesis

Replacing the infinitely-thin ray kernel with the wave-theoretic finite-frequency sensitivity kernel (banana-doughnut) captures Fresnel zone physics that straight rays miss, particularly for structures near or below the Fresnel zone width (~8 pixels).

### Method

1. Compute banana-doughnut kernels for all source-receiver pairs:
   K(x; x_s, x_r, ω, c₀) = −2k²/c₀ · Im{ G(x,x_s)·G(x_r,x) / G(x_r,x_s) }
2. Build L_FF by integrating K over each pixel
3. Average over transducer frequency band for broadband
4. Use L_FF as drop-in replacement for L_ray

### Sub-experiments

- [ ] **6a**: L_FF with standard L1/L2 solvers
- [ ] **6b**: L_FF with best INR configuration
- [ ] **6c**: L_FF combined with Tier 1 corrections
- [ ] **6d**: Compare mismatch energy: ε_FF = d_meas − L_FF @ s_true vs ε_ray

### Success criteria

- Mismatch energy (ε_FF) significantly lower than ε_ray (0.11% geom, 0.44% blob)
- CNR improves — Fresnel zone physics should help resolve inclusions near the zone width (~8 pixels)
- kwave_blob especially should benefit (wider Fresnel zone + higher contrast = more diffraction)

### Results — kwave_geom

| Sub-exp | ε energy | CNR | SSIM | RMSE | MAE | Notes |
|---------|----------|-----|------|------|-----|-------|
| 6a | — | — | — | — | — | |
| 6b | — | — | — | — | — | |
| 6c | — | — | — | — | — | |
| 6d | — | — | — | — | — | |

### Results — kwave_blob

| Sub-exp | ε energy | CNR | SSIM | RMSE | MAE | Notes |
|---------|----------|-----|------|------|-----|-------|
| 6a | — | — | — | — | — | |
| 6b | — | — | — | — | — | |
| 6c | — | — | — | — | — | |
| 6d | — | — | — | — | — | |

---

## Experiment 7: Joint Correction + Reconstruction (Phase 1 v2)

**Branch**: `experiment/joint-correction-v2`
**Priority**: Tier 2 — after Tier 1 results
**Status**: [ ] NOT STARTED

### Hypothesis

The Phase 1 dual-INR failure was due to shortcut learning. With architectural capacity control + staged training + alternating minimization, the correction INR can be constrained to capture only the model mismatch.

### Method

Architecture:
- INR_anatomy: 8 layers, 256 units, L_encoding=10
- INR_bias: 2 layers, 16 units, L_encoding=2 (cannot represent high-freq anatomy)

Training protocol:
1. Stage 1 (burn-in): r=0, train INR_anatomy only for N1 steps
2. Stage 2 (correction): freeze anatomy, train INR_bias on residual
3. Stage 3 (joint): asymmetric LRs, lr_anatomy = 10 × lr_bias

### Sub-experiments

- [ ] **7a**: Capacity-controlled dual-INR with staged training
- [ ] **7b**: Alternating minimization instead of joint optimization
- [ ] **7c**: Low-rank residual parameterization: r = U_basis @ α
- [ ] **7d**: Combine with Kaipio-Somersalo mean correction

### Success criteria

- Dual-INR produces sensible anatomy (no shortcut learning) — check CNR > 0
- CNR comparable to or better than single-INR (inclusion is preserved, not absorbed by bias)
- ||r|| / ||d|| converges to ~0.001-0.005 (expected mismatch magnitude)
- If ||r||/||d|| >> 0.01 or CNR ≈ 0 → shortcut learning detected, experiment failed

### Results — kwave_geom

| Sub-exp | ||r||/||d|| | CNR | SSIM | RMSE | MAE | Shortcut? | Notes |
|---------|-----------|-----|------|------|-----|-----------|-------|
| 7a | — | — | — | — | — | — | |
| 7b | — | — | — | — | — | — | |
| 7c | — | — | — | — | — | — | |
| 7d | — | — | — | — | — | — | |

### Results — kwave_blob

| Sub-exp | ||r||/||d|| | CNR | SSIM | RMSE | MAE | Shortcut? | Notes |
|---------|-----------|-----|------|------|-----|-----------|-------|
| 7a | — | — | — | — | — | — | |
| 7b | — | — | — | — | — | — | |
| 7c | — | — | — | — | — | — | |
| 7d | — | — | — | — | — | — | |

---

## Experiment 8: DeepONet Forward Operator Learning

**Branch**: `experiment/deeponet-forward`
**Priority**: Tier 2 — HIGH (mid-term presentation)
**Status**: [ ] NOT STARTED

### Hypothesis

The L-matrix is a linear approximation of the true nonlinear forward operator A: s → d. A DeepONet (Deep Operator Network) can learn this operator from paired (s, d) data, capturing beamforming physics, diffraction, and refraction effects that the linear L misses. If the learned operator generalizes, it can replace L entirely as a differentiable forward model for INR-based reconstruction.

### Motivation

- The L-matrix encodes a first-order linearization around homogeneous 1510 m/s background
- Real wave propagation is nonlinear: diffraction, refraction, scattering create structured mismatch
- DeepONet (Lu et al., 2021) provides a universal approximation framework for nonlinear operators
- We have abundant inverse crime data where d = L @ s is exact — train on this, test transfer

### Method

**Architecture** (DeepONet):
- **Branch net**: Takes flattened SoS field s (4096,) as input → encodes the function
- **Trunk net**: Takes ray index/coordinates as input → encodes the output location
- Output: predicted displacement d_pred for each ray given SoS field s
- Both branch and trunk are standard MLPs with configurable depth/width

**Training strategy** (staged):

1. **Stage 1: Learn operator on inverse crime data**
   - Train DeepONet to map s → d using inverse crime paired data where d = L @ s is exact
   - Validation: DeepONet should reproduce L @ s with very low error
   - This proves the architecture can represent the forward mapping

2. **Stage 2: Test zero-shot transfer to k-wave data**
   - Apply the inverse-crime-trained DeepONet to k-wave SoS fields
   - Compare DeepONet(s_true) vs d_kwave_measured
   - If residual is smaller than L @ s_true vs d_kwave_measured → learned operator captures some wave physics

3. **Stage 3 (if Stage 2 fails): Fine-tune on k-wave data**
   - Fine-tune DeepONet on 32 k-wave paired samples (leave-one-out)
   - If insufficient data: request more k-wave simulations from supervisors

4. **Stage 4: Use learned operator for reconstruction**
   - Replace `d_pred = L @ s` with `d_pred = DeepONet(s)` in INR training loop
   - INR optimizes: min_θ ||DeepONet(INR(coords)) - d_meas||²
   - The learned operator is differentiable → standard backprop through both networks

### Sub-experiments

- [ ] **8a**: Train DeepONet on inverse crime data, validate reconstruction (should match Oracle)
- [ ] **8b**: Zero-shot transfer — apply to k-wave data, measure residual vs L-matrix residual
- [ ] **8c**: Fine-tune on k-wave (LOO), reconstruct with learned operator
- [ ] **8d**: Compare reconstruction quality: DeepONet vs L-matrix vs Oracle

### Success criteria

- 8a: DeepONet reproduces inverse crime forward model with <1% relative error
- 8b: DeepONet residual on k-wave data is smaller than L-matrix residual (0.11% energy)
- 8c/8d: Reconstruction MAE closer to Oracle (1.8) than L-matrix INR (3.5)
- CNR improvement: inclusion more visible than with L-matrix reconstruction

### Key references

- Lu et al. (2021) — "Learning nonlinear operators via DeepONet" (Nature Machine Intelligence)
- Lu et al. (2022) — "A comprehensive and fair comparison of two neural operators" (arXiv:2111.05512)
- Lunz et al. (2021) — "On Learned Operator Correction in Inverse Problems"

### Results — kwave_geom

| Sub-exp | Operator error | CNR | SSIM | RMSE | MAE | Notes |
|---------|---------------|-----|------|------|-----|-------|
| 8a | — | — | — | — | — | |
| 8b | — | — | — | — | — | |
| 8c | — | — | — | — | — | |
| 8d | — | — | — | — | — | |

---

## Experiment 9: CNR Improvement via Regularization Priors

**Branch**: `experiment/cnr-improvement`
**Priority**: Tier 2 — HIGH (mid-term presentation)
**Status**: [ ] NOT STARTED

### Hypothesis

The INR achieves low MAE (3.5) by accurately predicting the ~95% homogeneous background but fails to resolve inclusions (low CNR). This is partly because MSE loss treats all pixels equally, favoring the majority background. Regularization priors that explicitly encourage spatial contrast — Total Variation (TV), edge-preserving penalties, or inclusion-aware loss weighting — can improve CNR while maintaining or improving MAE relative to L1/L2 baselines.

### Motivation

- Current INR: low MAE but poor inclusion visibility (smeared into vertical stripes)
- L1 (LASSO): MAE 7.0 — worse MAE but may have better edge structure
- L2 (Tikhonov): MAE 9.3 — smooth, poor edges
- The limited-angle geometry (8 firing pairs) inherently limits angular resolution, but within those constraints, better priors can improve contrast
- TV regularization is well-established for preserving edges in ill-posed inverse problems

### Method

**Approach 1: Total Variation Regularization**
- Add TV penalty to INR training loss: `Loss = ||L @ s - d||² + λ_TV · TV(s)`
- TV(s) = Σ|∇s| computed on the 64×64 grid (anisotropic or isotropic)
- Encourages piecewise-constant reconstructions → sharp inclusion boundaries
- Sweep λ_TV to find optimal trade-off between data fit and edge preservation

**Approach 2: Weighted Loss (Gradient-Aware)**
- Upweight pixels with high spatial gradient in the reconstruction
- Two-pass: (1) reconstruct with standard loss, (2) compute gradient magnitude, (3) retrain with gradient-weighted loss
- Focuses the optimization on inclusion boundaries rather than the flat background

**Approach 3: Multi-Scale / Coarse-to-Fine**
- Stage 1: Train INR at reduced resolution (32×32) to capture gross contrast
- Stage 2: Upsample and refine at 64×64 with TV regularization
- Coarse scale has better angular coverage → captures inclusion location

**Approach 4: Post-Processing Enhancement**
- Apply contrast-limited adaptive histogram equalization (CLAHE) or similar
- Not a reconstruction improvement per se, but may make inclusions visible for presentation

### Sub-experiments

- [ ] **9a**: TV regularization — sweep λ_TV ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1.0}
- [ ] **9b**: Gradient-aware weighted loss (two-pass)
- [ ] **9c**: Coarse-to-fine multi-scale
- [ ] **9d**: Comparison panel: L1 vs L2 vs INR vs INR+TV vs Oracle (all 4 metrics)

### Success criteria

- CNR significantly higher than baseline INR (inclusion clearly distinguishable from background)
- MAE remains competitive (< 7.0, i.e., still beats L1)
- SSIM improves over L1 baseline (0.649)
- Visual comparison shows clear inclusion boundary vs the vertical smearing artifact

### Results — kwave_geom

| Sub-exp | λ_TV | CNR | SSIM | RMSE | MAE | Notes |
|---------|------|-----|------|------|-----|-------|
| 9a | — | — | — | — | — | |
| 9b | — | — | — | — | — | |
| 9c | — | — | — | — | — | |
| 9d | — | — | — | — | — | |

### Results — kwave_blob

| Sub-exp | λ_TV | CNR | SSIM | RMSE | MAE | Notes |
|---------|------|-----|------|------|-----|-------|
| 9a | — | — | — | — | — | |
| 9b | — | — | — | — | — | |
| 9c | — | — | — | — | — | |
| 9d | — | — | — | — | — | |

---

## Experiment 10: Measurement-Domain INR Denoiser (Blind)

**Branch**: `experiment/measurement-denoiser`
**Priority**: Tier 2 — COMPLETE
**Status**: [x] COMPLETE — modest improvement, ceiling reached

### Hypothesis

The INR's spectral bias can act as a measurement-domain regularizer: fitting a small INR to each firing pair's 128×128 DT displacement grid smooths high-frequency model mismatch and fills NaN holes within the cone, producing cleaner measurements for downstream reconstruction.

### Method

Two-stage pipeline (no changes to reconstruction engine):
1. **Stage 1 (Denoise)**: For each of 8 firing pairs, train a FourierMLP INR on valid pixels of the 128×128 DT grid. The INR learns (x_dt, z_dt) → displacement. Spectral bias smooths valid pixels; spatial interpolation fills NaN holes.
2. **Stage 2 (Reconstruct)**: Replace `d_meas` with denoised measurements, keep original mask, run standard `optimize_full_forward_operator`.

### Key insights

1. **What the denoiser actually does**: It low-pass filters the displacement field, removing high-frequency components indiscriminately (both real signal and mismatch). It is measurement-domain regularization, not denoising in the traditional sense.
2. **Why it helps**: The ill-conditioned inverse (κ = 2.69×10¹³) amplifies high frequencies disproportionately. Removing them prevents amplification into artifacts.
3. **Why it can't close the gap**: The blind INR cannot distinguish mismatch from signal. 85-87% of mismatch energy is low-frequency (passes through the low-pass filter). The fundamental problem is the L-matrix, not the measurements.
4. **Oracle paradox**: Denoised measurements are actually farther from `L @ s_true` (-20 to -27% worse), yet reconstruction improves. This confirms the smoothing removes real signal too, but the net effect is positive because the removed components were being amplified by the ill-conditioning.

### Sweep results — kwave_geom (10 samples, 75 configs)

| Method | MAE | CNR | SSIM |
|--------|-----|-----|------|
| **L1 Baseline** | **6.02±0.81** | **2.64±1.25** | **0.730** |
| L2 Baseline | 9.49±1.32 | 2.73±1.03 | 0.802 |
| Best denoiser (s5.0_st500_h128) | 12.67±3.47 | 1.88±0.91 | 0.529 |
| Raw INR (no denoising) | 13.95±2.60 | 1.69±0.87 | 0.525 |

- Best denoiser: ~9% MAE improvement over raw INR (13.95 → 12.67)
- Still 2× worse than L1 baseline (6.02)
- Sweet spot: scale=5.0 (moderate bandwidth), steps=300-500, hidden=128+
- High scale (10.0) catastrophically fails (MAE 40-260): INR overfits mismatch
- Low scale (0.5) ≈ raw INR: over-smoothing destroys real signal

### Takeaway

The blind measurement-domain denoiser provides modest improvement (~9%) but hits a hard ceiling far from L1 baseline. The fundamental limitation is that it cannot selectively suppress mismatch vs signal. **Next step: joint optimization where the denoiser has access to the forward model**, allowing it to learn which measurement components to suppress based on their effect on reconstruction quality.

### Key files

- `src/inr_sos/training/denoise_engine.py` — per-pair INR denoiser
- `scripts/run_denoised_reconstruction.py` — two-stage experiment script
- `scripts/run_denoiser_sweep.py` — hyperparameter sweep script
- `scripts/data/denoiser_sweep/kwave_geom/20260403_152216/` — sweep results

---

## Experiment 11: Joint Denoiser + Reconstructor Optimization

**Branch**: `experiment/joint-denoiser-recon`
**Priority**: Tier 1 — HIGH (natural evolution of Exp 10)
**Status**: [ ] NOT STARTED

### Hypothesis

The blind denoiser (Exp 10) hits a ceiling because it has no knowledge of the forward model — it smooths indiscriminately. By jointly optimizing the denoiser INR and the reconstruction INR end-to-end, the denoiser can learn to produce measurements that are specifically compatible with the L-matrix reconstruction, selectively suppressing components that the ill-conditioned inverse would amplify into artifacts.

### Motivation

- Exp 10 showed: blind smoothing gives ~9% improvement but plateaus at MAE 12.67 (vs L1 baseline 6.02)
- The denoiser's limitation: can't distinguish mismatch from signal without forward model knowledge
- Joint training gives the denoiser a reconstruction-quality gradient signal — "this change to d improved/hurt the reconstruction"
- This is the measurement-domain analog of learned regularization

### Method

**Architecture**: Two INRs, one pipeline:
- **INR_denoise**: Per-pair FourierMLP (same as Exp 10). Maps (x_dt, z_dt) → denoised displacement. Trained on valid pixels with MSE to raw measurements + regularized by reconstruction loss.
- **INR_recon**: Standard ReluMLP (same as current pipeline). Maps (x_sos, z_sos) → slowness.

**Forward pass**:
```
d_denoised = INR_denoise(coords_dt)          # per-pair, concat 8 pairs
s_pred = INR_recon(coords_sos)               # (4096, 1) slowness field
d_pred = L @ s_pred                           # forward model prediction
loss = ||mask * (d_pred - d_denoised)||²      # data fidelity on valid rays
      + λ_fit · ||mask * (d_denoised - d_raw)||²  # denoiser stays close to data
```

**Training strategies** (sub-experiments):

1. **End-to-end**: Both INRs optimized simultaneously. Gradient flows from reconstruction loss through d_denoised back to INR_denoise.
2. **Alternating**: Fix denoiser → optimize reconstructor for N steps → fix reconstructor → optimize denoiser for M steps → repeat.
3. **Staged**: Stage 1: pretrain denoiser (blind, as Exp 10). Stage 2: pretrain reconstructor on denoised data. Stage 3: joint fine-tuning with both gradients.

### Sub-experiments

- [ ] **11a**: End-to-end joint optimization (both INRs, single loss)
- [ ] **11b**: Alternating minimization (fix one, optimize other)
- [ ] **11c**: Staged training (pretrain → joint fine-tune)
- [ ] **11d**: Ablation: vary λ_fit (denoiser data-fidelity weight) to control how far denoiser can deviate from raw measurements
- [ ] **11e**: Best config on kwave_blob (generalization test)

### Success criteria

- MAE significantly below blind denoiser ceiling (12.67) — target: < 8.0
- Approach or beat L1 baseline (6.02)
- CNR > L1 baseline (2.64) — joint optimization should preserve inclusion structure
- Denoiser learns structured corrections (not just uniform smoothing) — visualize d_denoised vs d_raw difference maps, should show anatomically meaningful patterns
- No collapse: denoiser shouldn't ignore raw data entirely (monitor ||d_denoised - d_raw|| / ||d_raw||)

### Risks

- **Denoiser shortcut**: INR_denoise learns to output d = L @ s_pred directly (bypassing raw measurements). Mitigate with λ_fit penalty.
- **Mode collapse**: Both INRs converge to trivial solution (uniform background). Monitor CNR throughout training.
- **Gradient competition**: Denoiser and reconstructor fight each other. Alternating minimization (11b) may be more stable.

### Results — kwave_geom

| Sub-exp | λ_fit | MAE | CNR | SSIM | RMSE | Notes |
|---------|-------|-----|-----|------|------|-------|
| 11a | — | — | — | — | — | |
| 11b | — | — | — | — | — | |
| 11c | — | — | — | — | — | |
| 11d | — | — | — | — | — | |
| 11e | — | — | — | — | — | |

### Results — kwave_blob

| Sub-exp | λ_fit | MAE | CNR | SSIM | RMSE | Notes |
|---------|-------|-----|-----|------|------|-------|
| 11e | — | — | — | — | — | |

---

## Experiment Dependency Graph

```
Exp 1 (KS) ──────┐
                  ├──► Exp 4 (Combined Tier 1) ──► Exp 5 (Eikonal) [BLOCKED: needs L-gen code]
Exp 2 (SVD-INR) ──┤                              ──► Exp 6 (Finite-Freq) [BLOCKED: needs L-gen code]
                  │                               ──► Exp 7 (Joint v2)
Exp 3 (SVD-Loss) ─┘

Exp 8 (DeepONet) ──────── independent, uses inverse crime data first
Exp 9 (CNR Improvement) ── independent, works with existing INR pipeline
Exp 10 (Blind Denoiser) ──► Exp 11 (Joint Denoiser+Recon) ← ACTIVE
```

Experiments 1, 2, 3 are independent (COMPLETE/SKIPPED).
Experiment 4 combines the best of 1-3 (SKIPPED).
Experiments 5, 6 blocked pending L-generation code from Deniz/Orcun.
**Exp 10 complete** — blind denoiser ceiling reached (~9% improvement).
**Exp 11 is the active priority** — joint optimization to break through the ceiling.

---

## Branching Strategy

```
k-wave-validation (base)
├── experiment/kaipio-somersalo       (Exp 1) — COMPLETE
├── experiment/svd-constrained-inr    (Exp 2) — COMPLETE
├── experiment/svd-mismatch-loss      (Exp 3) — SKIPPED
├── experiment/combined-tier1         (Exp 4) — SKIPPED
├── experiment/eikonal-bent-ray       (Exp 5) — COMPLETE (FAILED)
├── experiment/finite-frequency-L     (Exp 6) — BLOCKED
├── experiment/joint-correction-v2    (Exp 7) — TODO
├── experiment/deeponet-forward       (Exp 8) — TODO ← mid-term
├── experiment/cnr-improvement        (Exp 9) — TODO ← mid-term
├── experiment/measurement-denoiser   (Exp 10) — COMPLETE (ceiling reached)
└── experiment/joint-denoiser-recon   (Exp 11) — TODO ← ACTIVE PRIORITY
```

Each experiment branch starts fresh from `k-wave-validation`. Results are recorded in this file on the base branch after each experiment concludes.

---

## Key Takeaways from Completed Experiments

### What has been tried and what we learned

**Tier 1 — Measurement/Solution-Side Corrections (Exp 1-4):**
- **Exp 1 (Kaipio-Somersalo):** Subtracting the mean mismatch template from measurements does nothing for the INR (MAE 3.49 vs 3.58 baseline). The mismatch energy (0.11%) is too small relative to the INR's implicit regularization. Classical linear solvers (LSQR) catastrophically fail with or without correction (MAE ~283).
- **Exp 2 (SVD-Constrained INR):** Truncating the INR output to the top-K singular vectors of L hurts at every K tested. The INR's spectral bias already provides optimal implicit regularization — explicit truncation removes information the INR needs. Only unconstrained (K=4096) works.
- **Exp 3-4 (SVD Loss, Combined):** Skipped. Experiments 1-2 conclusively showed that solution-side constraints cannot close the gap.

**Tier 2 — Forward Model Replacement (Exp 5):**
- **Exp 5 (Eikonal Bent-Ray):** Attempted to build our own L-matrix from ray tracing (Siddon's algorithm) with eikonal-based bent rays. Failed fundamentally: the real L-matrix encodes beamforming sensitivity (coherent multi-element summation), not geometric ray paths. Row-wise correlation between Siddon L and real L was only 0.097. Gaussian broadening cannot bridge this gap. To make this work, we need the code that generates the real L (the beamforming Jacobian).

### The real problem (reframed 2026-03-28)

The MAE gap (3.5 vs Oracle 1.8) was the initial focus, but the actual clinical problem is **CNR — inclusion visibility**. The INR achieves MAE 3.5 (beating L1=7.0, L2=9.3) by accurately predicting the ~95% homogeneous background, while the inclusion is barely visible (smeared into vertical stripes). This is fundamentally a **limited-angle tomography** problem: 8 firing pairs from a linear array provide too few angular views to resolve a compact inclusion.

### Remaining directions

| Experiment | Status | Priority | Prospects |
|---|---|---|---|
| Exp 6 (Banana-Doughnut) | Not started | Blocked | Same obstacle as Exp 5 — needs beamforming model, not ray kernels |
| Exp 7 (Joint Correction v2) | Not started | Medium | Superseded by Exp 11 (same idea, better framing from Exp 10 insights) |
| Exp 8 (DeepONet Forward Operator) | Not started | **HIGH — mid-term** | Learn operator s→d on inverse crime data, test transfer to k-wave |
| Exp 9 (CNR Improvement) | Not started | **HIGH — mid-term** | TV regularization + inclusion-aware priors to beat L1/L2 visually |
| Exp 10 (Blind Denoiser) | **COMPLETE** | — | ~9% improvement, ceiling reached. Blind INR can't separate mismatch from signal. |
| **Exp 11 (Joint Denoiser+Recon)** | Not started | **HIGH — ACTIVE** | Break blind denoiser ceiling by giving denoiser forward model knowledge. Target: approach L1 baseline. |
| Ask Deniz/Orcun for L-generation code | Pending | HIGH | Would unblock Exp 5/6 and enable FMM with correct beamforming model |

### Action item: L-generation code
Ask Deniz/Orcun for the MATLAB code that generates the L-matrix (A-matrix). With it, we could recompute L for non-homogeneous SoS backgrounds — the beamforming-native equivalent of bent-ray iteration. This would unblock Exp 5/6 entirely.

---

## Change Log

| Date | Experiment | Change | Result |
|------|-----------|--------|--------|
| 2026-03-23 | — | Plan created | — |
| 2026-03-25 | Exp 1 | Kaipio-Somersalo complete | 1a/1b/1d: MAE~283 (catastrophic). 1c: MAE 3.49 (marginal) |
| 2026-03-25 | Exp 2 | SVD-Constrained INR complete | Truncation hurts at all K. Unconstrained best (MAE 3.58) |
| 2026-03-25 | Exp 3,4 | Skipped | Solution-side constraints don't help |
| 2026-03-26 | Exp 5 | Geometry exploration complete | DT indexing, element pairs, rasterization mismatch identified |
| 2026-03-27 | Exp 5 | Eikonal bent-ray experiments complete | 5a: MAE 3.01, 5b: MAE 9.28, 5c: MAE 9.86. Root cause: beamformed L ≠ ray-traced L |
| 2026-03-28 | Exp 5 | Broadening analysis complete | Gaussian blur correlation ceiling 0.097. Approach abandoned |
| 2026-03-29 | Exp 8 | Plan added: DeepONet Forward Operator | Train on inverse crime, test transfer to k-wave |
| 2026-03-29 | Exp 9 | Plan added: CNR Improvement | TV regularization + priors to improve inclusion visibility |
| 2026-03-29 | — | Mid-term pivot | Focus on Exp 7, 8, 9 for presentation. Exp 5/6 blocked pending L-gen code |
| 2026-04-02 | Exp 10 | Blind measurement denoiser implemented | Per-pair FourierMLP denoiser, two-stage pipeline |
| 2026-04-03 | Exp 10 | Denoiser sweep complete (75 configs, 10 samples) | Best MAE 12.67 (s5.0_st500_h128) vs raw 13.95 — ~9% improvement. L1 baseline still 6.02. Ceiling reached. |
| 2026-04-04 | Exp 11 | Joint denoiser+reconstructor plan added | End-to-end, alternating, and staged training strategies. Target: break blind denoiser ceiling, approach L1 baseline. |
