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
**Status**: [x] COMPLETE (2026-03-24)

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

- [x] **1a**: Template subtraction only (d_corrected = d − μ_ε), then standard L1/L2 solve
- [x] **1b**: Template + covariance weighting (full Kaipio-Somersalo)
- [x] **1c**: Template + covariance + INR reconstruction (combine with existing INR pipeline)
- [x] **1d**: Vary number of PCA components K_ε ∈ {5, 10, 15, 20, 31}

### Success criteria

- CNR improves over L1 baseline (inclusion is more visible, not blurred away)
- SSIM > 0.649 (geom) / 0.612 (blob)
- MAE < 7.00 (geom) / < 24.0 (blob)

### Results — kwave_geom

| Sub-exp | CNR | SSIM | RMSE | MAE | Notes |
|---------|-----|------|------|-----|-------|
| 1a | 0.28 | 0.60 | 287 | 281 | Template correction negligible; LSQR catastrophic |
| 1b | 0.27 | 0.65 | 288 | 283 | Covariance weighting no help; 31-sample Γ too low-rank |
| 1c | **1.22** | **0.83** | 9.0 | **3.49** | KS-INR marginal over raw INR (3.58); INR does the work |
| 1d | 0.28 | 0.65 | 288 | 283 | K sweep (5-31): no effect on linear solver |

### Results — kwave_blob

| Sub-exp | CNR | SSIM | RMSE | MAE | Notes |
|---------|-----|------|------|-----|-------|
| 1c | **0.49** | **0.77** | 10.5 | **5.09** | KS-INR marginal over raw INR (5.14) |

### Key Findings

1. **Linear solvers (LSQR) are catastrophic** (MAE ~280) regardless of KS correction — L-matrix ill-conditioning dominates.
2. **KS mean correction is negligible** — mismatch energy (0.11%) too small relative to regularization-induced error.
3. **KS covariance weighting fails** — 31-sample covariance too low-rank.
4. **INR provides the real regularization** — spectral bias acts as implicit prior, making explicit mismatch correction redundant.
5. **Conclusion**: Measurement-domain correction alone is insufficient. Need to control the *inversion amplification* (→ Experiment 2).

---

## Experiment 2: SVD-Constrained INR

**Branch**: `experiment/svd-constrained-inr`
**Priority**: Tier 1 — implement alongside Exp 1
**Status**: [~] IN PROGRESS (2026-03-24)

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
**Status**: [ ] NOT STARTED

### Hypothesis

Recomputing the L-matrix using bent-ray paths (via Fast Marching eikonal solver) corrects the physics directly, reducing the mismatch at its source rather than mitigating it statistically.

### Method

1. Install scikit-fmm (or implement fast sweeping)
2. After every N epochs of INR training:
   - Extract SoS map: c = 1/INR(coords), reshape to 64×64
   - For each source (8 total): solve |∇T|² = 1/c² via FMM
   - Backtrace rays from receivers along −∇T
   - Rebuild L_bent as sparse matrix
3. Continue INR training with L_bent

### Sub-experiments

- [ ] **5a**: Single L-update after straight-ray INR converges
- [ ] **5b**: Iterative L-update every N=100 epochs
- [ ] **5c**: Iterative L-update every N=50 epochs
- [ ] **5d**: Combine with best Tier 1 approach (KS + SVD + bent-ray)

### Success criteria

- CNR and SSIM improve over Tier 1 (bent rays should sharpen inclusion boundaries)
- Verify that L-update cost is negligible (<1% of total training time)
- kwave_blob is the real test — higher contrast (62.6 m/s) means more refraction

### Results — kwave_geom

| Sub-exp | N_update | CNR | SSIM | RMSE | MAE | L-update time | Notes |
|---------|----------|-----|------|------|-----|--------------|-------|
| 5a | once | — | — | — | — | — | |
| 5b | 100 | — | — | — | — | — | |
| 5c | 50 | — | — | — | — | — | |
| 5d | — | — | — | — | — | — | |

### Results — kwave_blob

| Sub-exp | N_update | CNR | SSIM | RMSE | MAE | L-update time | Notes |
|---------|----------|-----|------|------|-----|--------------|-------|
| 5a | once | — | — | — | — | — | |
| 5b | 100 | — | — | — | — | — | |
| 5c | 50 | — | — | — | — | — | |
| 5d | — | — | — | — | — | — | |

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

## Experiment Dependency Graph

```
Exp 1 (KS) ──────┐
                  ├──► Exp 4 (Combined Tier 1) ──► Exp 5 (Eikonal)
Exp 2 (SVD-INR) ──┤                              ──► Exp 6 (Finite-Freq)
                  │                               ──► Exp 7 (Joint v2)
Exp 3 (SVD-Loss) ─┘
```

Experiments 1, 2, 3 are independent — run in parallel.
Experiment 4 combines the best of 1-3.
Experiments 5, 6, 7 build on Tier 1 results.

---

## Branching Strategy

```
k-wave-validation (base)
├── experiment/kaipio-somersalo       (Exp 1)
├── experiment/svd-constrained-inr    (Exp 2)
├── experiment/svd-mismatch-loss      (Exp 3)
├── experiment/combined-tier1         (Exp 4, merges best of 1-3)
├── experiment/eikonal-bent-ray       (Exp 5)
├── experiment/finite-frequency-L     (Exp 6)
└── experiment/joint-correction-v2    (Exp 7)
```

Each experiment branch starts fresh from `k-wave-validation`. Results are recorded in this file on the base branch after each experiment concludes.

---

## Change Log

| Date | Experiment | Change | Result |
|------|-----------|--------|--------|
| 2026-03-23 | — | Plan created | — |
| 2026-03-24 | Exp 1 (KS) | All sub-experiments complete | KS negligible; INR does the heavy lifting (MAE 3.49 vs 3.58) |
| 2026-03-24 | Exp 2 (SVD) | Implementation started | Script: `scripts/run_svd_constrained.py` |
