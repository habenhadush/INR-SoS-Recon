# Tackling the Physics Mismatch: Approaches & Literature Review

## 1. Problem Recap

We reconstruct speed-of-sound (SoS) from ultrasound time-of-flight data using:

```
d = L · s
```

where **L** ∈ ℝ^{131072 × 4096} is a precomputed ray-tracing matrix and **s** is the slowness field. The true measurement satisfies:

```
d_meas = L · s_true + ε
```

where **ε** is the **structured, low-frequency, systematic model error** arising from the physics gap between ray acoustics (L) and full-wave propagation (reality).

### What We Know About ε (Phase 1 Diagnostics)

| Property | Finding |
|----------|---------|
| Energy | 0.11% of ‖d_meas‖² (kwave_geom) |
| Structure | Systematic, geometry-dependent (LOO correlation = 0.66) |
| Spectrum | 85–87% low-frequency — same band as signal |
| Per-pair | Uniformly distributed across 8 firing pairs |
| Amplification | L⁺ε amplified by up to 4.53 × 10¹¹ through ill-conditioned tail |
| Effect | ≈3% structured error in image space → MAE ≈ 7–17 m/s |

### Why It's Hard

The optimizer solves `min_θ ‖L·f_θ(X) − d_meas‖²` and finds `s* = s_true + L⁺ε`, absorbing the mismatch as structured image artifacts. Three compounding factors:
1. **Same spectral band**: ε overlaps signal → no frequency-based separation
2. **Column-space absorption**: ε ∈ col(L) → optimizer fits it as a wrong slowness
3. **Ill-conditioning**: κ(L) ≈ 10¹³ → small ε in measurement space becomes large Δs in image space

### Architecture 1 (Denoiser) Failed — Why

We tried Deep Image Prior (DIP) style denoising: train an INR to fit d_meas, hoping spectral bias captures signal before mismatch. **MAE = 40.53** (worse than uncorrected).

**Root cause**: DIP spectral bias separates high-frequency random noise from low-frequency signal [Chakrabarty & Maji, 2019; Jo et al., IJCV 2022]. But our ε is **low-frequency and structured** — it sits in the exact band DIP learns first. The denoiser captures ε along with signal, offering no separation.

This is consistent with theory: "structured high-frequency image details with self-similarity are fitted better and faster" [Jo et al., 2022], and by extension, structured low-frequency corruption is inseparable from structured low-frequency signal.

---

## 2. Taxonomy of Approaches

Based on the literature, approaches to forward model mismatch fall into five categories:

### Category A: Measurement-Domain Correction (Pre-processing d)
*Learn/estimate a mapping that transforms measured data to be consistent with the simplified model*

### Category B: Joint Reconstruction + Correction (Dual-variable optimization)
*Simultaneously estimate the image and a correction term during reconstruction*

### Category C: Learned Operator Correction (Fix the forward model) — OUT OF SCOPE
*Learn a corrected forward-adjoint operator pair that better approximates the true physics.*
**Excluded**: The L-matrix is fixed and precomputed — this is a project constraint. Included here for theoretical context only (Lunz et al.'s adjoint insight informs why some approaches may fail).

### Category D: Bayesian Approximation Error (Statistical treatment)
*Model the mismatch statistically and incorporate its covariance into the inversion*

### Category E: Robust Loss / Regularization (Suppress mismatch effects)
*Use loss functions and regularizers that are less sensitive to structured model error*

---

## 3. Detailed Approach Analysis

### 3A. Measurement-Domain Correction

#### Theory
Instead of correcting after inversion, correct the measurements before:

```
d_corrected = Ψ(d_meas) ≈ L · s_true
```

Then reconstruct from d_corrected using the standard pipeline. The key insight: correction in measurement space avoids the ill-conditioned amplification that makes image-space correction fragile.

#### Literature Support

**[Lozenski et al., 2024; 2025]** — "Learned Correction Methods for USCT Imaging Using Simplified Physics Models"
- Systematically compared data-domain vs image-domain correction for USCT with Born approximation (simplified) vs wave equation (accurate)
- **Data correction** (CNN mapping d_wave → d_born): minor visual artifacts, AUC=0.951
- **Image correction** (CNN mapping ŝ_born → s_true): "heavy bias on training data, resulting in hallucinations", AUC=0.925
- **Out-of-distribution**: Data correction maintained AUC≈0.945 on unseen tissue types; image correction dropped to 0.885
- **Key quote**: "Learning a correction in the data domain led to better task performance and robust out-of-distribution generalization compared to correction in the image domain"
- **Why**: "The number and size of measurements are often much larger than reconstructed images, providing richer training sets" and the mapping is "easier to learn and more generalizable"

**[Bezek & Goksel, IEEE TMI 2024]** — "Learning the Imaging Model via Convolutional Formulation"
- Our supervisors' paper. Learns a convolutional kernel that replaces the hand-crafted L-matrix
- Reduces contrast error by 38% vs conventional model
- On cancerous breast: learned model achieves SoS contrast of 34.6 m/s vs conventional 3.4 m/s (10× improvement)
- Demonstrates that the forward model itself is the bottleneck

#### Applicability to Our Problem

**Challenge**: Lozenski et al. use paired training data {d_wave, d_born} from simulation. We have d_meas (k-wave) and L (ray-tracing), but need *supervised pairs* to learn the mapping.

**Possible path**: Use our training set (N=30 kwave_geom samples) where we know s_true:
- Compute d_model = L · s_true (what L predicts)
- Observe d_meas (what k-wave produces)
- Learn Ψ: d_meas → d_model (remove the mismatch)

**Risk**: N=30 samples may be too few for a CNN. But since ε is systematic (geometry-dependent, not sample-dependent), a low-capacity model might suffice.

---

### 3B. Joint Reconstruction + Correction (Architecture 3)

#### Theory
Decompose the measurement into a model-consistent part and a residual:

```
min_{θ,φ}  ‖L · f_θ(X) + g_φ(r) − d_meas‖² + λ·R(f_θ) + τ·‖g_φ‖²
```

where:
- f_θ: INR for slowness (the reconstruction)
- g_φ: INR for residual correction (absorbs ε)
- τ·‖g_φ‖²: penalty preventing trivial solution (g_φ explains everything)

#### Literature Support

**[Gilton et al., 2024 (ICLR 2025)]** — "Solving Inverse Problems with Model Mismatch using Untrained Neural Networks"
- Core formulation: `y = A₀(x) + f_θ(x, A₀) + ε`
- Untrained residual block f_θ optimized per-instance
- Full loss: `½‖y − A₀(z) − f_θ(z,A₀)‖² + γr(x) + τ‖f_θ‖² + λ‖x−z‖²`
- **Convergence proven** under mild conditions (Proposition 4.1): monotonic descent with bounded gradients
- τ penalty: critical for preventing trivial solutions; τ=0.1 works across experiments
- **Results**: +2–3 dB PSNR over robust baselines across deblurring, deconvolution, defogging
- **Key advantage**: untrained (no data needed), per-instance, works with any approximate forward model
- Random initialization performs identically to saved weights (<0.01 dB difference)

**[Ring artifact removal via INR decomposition, 2024]**
- Decomposes sinogram: `P(x) = IS(x) + SA(x)`
- IS parameterized by multi-resolution hash grid + MLP; SA by simple learnable matrix
- Directional regularization: smooth IS in detector direction, sparse SA in angular direction
- Loss: `|IS + SA − P| + λ_IS·smooth(IS) + λ_SA·sparse(SA)`
- **PSNR = 49.0 dB** (vs 39.5 for next best)
- Works because artifacts have directional structure different from signal

#### Applicability to Our Problem

**Strong fit**: This is essentially our Architecture 3. The Gilton et al. framework provides:
1. A principled loss function with τ-penalized residual
2. Convergence guarantees
3. No training data requirement (self-supervised, per-instance)

**Key question**: What features should g_φ take as input? Options:
- Ray features (transmitter/receiver positions, angles) — captures geometry-dependent ε
- Same coordinates as f_θ but in measurement space — models ε as a function of ray index
- Learned embeddings per ray

**The ring artifact paper** suggests that if ε has directional/structural regularities (ours does — it's geometry-dependent), then directional regularization can separate it. Our ε is strongest at fan edges of each firing pair, suggesting angular/spatial structure we can exploit.

---

### 3C. Learned Operator Correction

#### Theory
Instead of correcting measurements or adding a residual, learn corrections to the operator itself:

```
A_Θ = F_Θ ∘ L̃    (corrected forward)
A_Φ* = G_Φ ∘ L̃*   (corrected adjoint)
```

Then solve: `min_x ½‖A_Θ(x) − d‖² + λR(x)`

#### Literature Support

**[Lunz, Hauptmann, Arridge et al., SIAM 2021]** — "On Learned Operator Correction in Inverse Problems"
- **Critical theorem**: Correcting only the forward operator is *insufficient*. Gradient descent iterates are confined to range(L̃*), and if s_true ∉ closure(range(L̃*)), no forward-only correction can recover it.
- **Solution**: Forward-adjoint correction — learn both F_Θ (measurement space) and G_Φ (image space)
- Convergence to δ-neighborhood of true solution when:
  - `‖L‖ · ‖(A − A_Θ)(x_n)‖ < δ/4`
  - `‖(A* − A_Φ*)(A_Θ(x_n) − y)‖ < δ/4`
- **Recursive training**: Must include later optimization iterates in training set (training only on initial backprojections degrades over iterations)
- Results: L2 error ≈ 0.15 (learned) vs 0.19 (Bayesian approx error) vs 0.55 (uncorrected) on PAT
- **Key insight**: "the range of the corrected fidelity term's gradient is limited by the range of the approximate adjoint" — both must be corrected

**[Gilton et al., 2023]** — "Inverse Problems with Learned Forward Operators"
- Two paradigms: operator-agnostic learning (learns on training data subspace) and physics-informed learning (corrects simplified model)
- Both require learning the adjoint alongside the forward model
- Convergence to neighborhood of accurate solution under controlled subdifferential conditions

#### Applicability to Our Problem

**Out of scope**: The L-matrix is fixed/precomputed as a project constraint. However, the theoretical insights are valuable:
- **Key takeaway** (Lunz et al.): If we correct only in measurement space without also correcting the adjoint, gradient-based reconstruction may still fail — because iterates are confined to range(L*). For our L (effective rank 4050/4096), range(L*) is nearly full-rank, so this is less of a concern.
- **Key takeaway** (Bezek & Goksel): The forward model itself is the bottleneck — their learned convolutional kernel improved contrast 10×. This validates that our mismatch diagnosis is correct and motivates the in-scope approaches that work around the fixed L.

---

### 3D. Bayesian Approximation Error (BAE)

#### Theory
Model the mismatch statistically:

```
d_meas = L · s + ε + e
```

where ε ~ N(η_ε, Γ_ε) is model error and e ~ N(0, Γ_e) is measurement noise. The variational problem becomes:

```
min_s  ½‖L_ε(L·s − d + η_ε)‖² + λR(s)
```

where L_ε = Γ_ε^{-1/2} is a whitening transform that down-weights directions with high model error.

#### Literature Support

**[Kaipio & Somersalo, 2007]** — Statistical Inverse Problems
- Original framework for approximation error modeling
- Estimate mean η_ε and covariance Γ_ε from training samples
- Incorporate into the data-fidelity term as a modified noise model
- Successfully applied in electrical impedance tomography, diffuse optical tomography

**[Lunz et al., SIAM 2021]** — comparison with learned correction
- BAE is more stable but less expressive (Gaussian assumption, linear correction)
- In PAT experiments: BAE achieves L2 error ≈ 0.19 vs learned correction ≈ 0.15
- BAE converges slowly but stably; learned methods converge faster when well-trained

#### Applicability to Our Problem

**Good fit for our data characteristics**:
- ε is systematic and geometry-dependent → η_ε ≠ 0 (can estimate from N=30 samples)
- Covariance Γ_ε captures per-ray error structure
- No neural network needed — purely statistical

**Implementation**:
1. Compute ε_i = d_meas_i − L · s_true_i for all training samples
2. Estimate η_ε = mean(ε_i) and Γ_ε = cov(ε_i)
3. Modify the loss: `‖Γ_ε^{-1/2}(L·f_θ(X) − d_meas + η_ε)‖²`

**Challenges**:
- Γ_ε ∈ ℝ^{131072 × 131072} — too large to compute/invert directly
- Need low-rank or diagonal approximation
- Gaussian assumption may not capture structured spatial patterns
- With N=30 samples, covariance estimation is rank-deficient (rank ≤ 29)

**Practical variant**: Diagonal BAE — just reweight each ray by 1/σ²_i (per-ray error variance). Simple, no matrix inversion, directly implementable.

---

### 3E. Robust Loss & Regularization

#### Theory
Instead of correcting ε, make the reconstruction robust to it:
- **Robust losses** (Huber, L1, Cauchy): down-weight large residuals that may be mismatch-dominated
- **Total Variation**: penalize high-frequency artifacts from amplified mismatch
- **Truncated SVD regularization**: discard the ill-conditioned tail modes where ε amplification is worst

#### Literature Support

**L1 loss / Huber loss**:
- L1 minimization via IRLS is equivalent to iteratively reweighting residuals, giving less weight to outliers
- Already observed in our sweeps: Huber loss improves over MSE but doesn't close the gap to baselines

**Total Variation Regularization** [Rudin-Osher-Fatemi, 1992]:
- Penalizes image gradient magnitude → promotes piecewise-constant reconstructions
- More robust to model mismatch than L2 regularization because it doesn't penalize sharp features
- "TV regularization for full-waveform inversion" [Aghamiry et al., 2016] shows TV improves robustness to model errors in seismic imaging

**Truncated SVD / spectral filtering**:
- Our SVD analysis shows 90% of signal energy in top 86 modes, but mismatch amplification explodes in the tail (modes 3800–4096)
- A spectral filter that reduces sensitivity to tail modes could suppress the worst amplification
- Can be implemented as a weighted loss in the SVD basis

#### Applicability to Our Problem

**Already partially explored**: Huber loss helps but doesn't close the gap. These are complementary techniques that can be combined with any of the above approaches.

**Key opportunity — SVD-informed regularization**:
Our detailed SVD analysis (Section 9–10 of Phase 1) gives us a map of exactly where the mismatch amplification occurs. We can design a custom spectral regularizer:

```
Loss = Σ_k w_k · (u_k^T · (L·f_θ − d_meas))²
```

where w_k decreases for high-k (ill-conditioned) singular modes. This directly targets the amplification mechanism.

---

## 4. Recommended Strategy: Layered Approach

Based on the literature and our specific problem characteristics, I recommend a **layered approach** that combines multiple strategies, ordered by implementation complexity:

### Layer 1: SVD-Informed Spectral Regularization (Simplest, immediate)

**Rationale**: We already have the SVD decomposition. The mismatch amplification is worst in modes 3800–4096. By downweighting these modes in the loss, we suppress the worst artifacts without any training data.

**Implementation**:
```python
# Project residual onto SVD basis
r = L @ s_pred - d_meas
r_svd = U.T @ r  # project onto singular vectors
# Weight by inverse amplification factor
weights = 1.0 / (1.0 + alpha * amplification_ratio)
loss = (weights * r_svd**2).sum()
```

**Literature basis**: Truncated SVD regularization [Hansen, 1987]; spectral filtering for ill-conditioned problems

**Expected impact**: Should reduce artifacts from the amplified tail, but won't correct the low-frequency mismatch in the well-conditioned modes.

### Layer 2: Diagonal BAE — Per-Ray Error Reweighting (Low complexity, uses training data)

**Rationale**: With N=30 samples, we can compute per-ray mismatch variance and mean. Subtracting the mean ε and reweighting by 1/σ²_ε is a simple, principled correction.

**Implementation**:
```python
# From training data
epsilon = d_meas - L @ s_true  # shape: (N, 131072)
eta = epsilon.mean(axis=0)      # systematic bias
sigma2 = epsilon.var(axis=0)    # per-ray variance

# Modified loss
residual = L @ s_pred - (d_meas - eta)  # bias-corrected
weights = 1.0 / (sigma2 + sigma2_noise)  # reweighting
loss = (weights * residual**2).sum()
```

**Literature basis**: [Kaipio & Somersalo, 2007]; BAE framework

**Expected impact**: Should correct the mean bias and downweight unreliable rays. Simple and interpretable.

### Layer 3: Joint Reconstruction + Residual Correction (Medium complexity)

**Rationale**: The Gilton et al. (ICLR 2025) framework provides a principled way to jointly estimate slowness and mismatch. No training data needed (self-supervised, per-instance). Proven convergence.

**Implementation**:
```python
# Two networks
inr_s = SirenMLP(2, 1)         # (x,z) → slowness
inr_r = MLP(ray_features, 1)   # ray_idx → residual correction

# Loss (per Gilton et al.)
d_pred = L @ inr_s(coords) + inr_r(ray_idx)
loss = ||d_pred - d_meas||² + lambda * TV(inr_s) + tau * ||inr_r||²
```

The τ penalty is critical: too small → inr_r explains everything (trivial solution); too large → no correction.

**Literature basis**: [Gilton et al., ICLR 2025]; [Ring artifact INR decomposition, 2024]

**Expected impact**: Should capture geometry-dependent mismatch in inr_r while keeping inr_s clean. The key question is whether τ can be tuned to achieve good separation.

### Layer 4: Learned Measurement Correction (Higher complexity, strongest evidence)

**Rationale**: Lozenski et al. (2025) showed data-domain correction outperforms image-domain correction on USCT with simplified physics. This is the most directly relevant prior work.

**Challenge for us**: Requires supervised pairs. With N=30, need a low-capacity model.

**Implementation options**:
- **Linear correction**: Learn a matrix C such that d_corrected = C · d_meas + b. Low-rank C reduces parameters.
- **Per-ray affine**: d_corrected_i = a_i · d_meas_i + b_i (2 parameters per ray). Very simple, may capture scaling/offset errors.
- **Small CNN on ray features**: If rays have spatial structure, a small 1D-CNN could learn local corrections.
- **SVD-domain correction**: Learn corrections in the SVD basis where the structure is clearest.

**Literature basis**: [Lozenski et al., SPIE 2024; arXiv 2025]; [Bezek & Goksel, IEEE TMI 2024]

**Expected impact**: Strongest theoretical support, but limited by our small training set. The per-ray affine correction is the minimum viable version.

---

## 5. Key Literature Summary

### Directly Relevant Papers (New)

| # | Paper | Key Contribution | Relevance |
|---|-------|-----------------|-----------|
| 16 | Gilton et al. (ICLR 2025) | Untrained NN residual block for model mismatch; convergence proof; τ-penalized correction | Direct framework for Architecture 3 |
| 17 | Lozenski et al. (arXiv 2025) | Data correction > image correction for USCT; OOD generalization | Strongest evidence for measurement-domain approach |
| 18 | Lunz et al. (SIAM 2021) | Forward-adjoint correction; proves forward-only insufficient; convergence theorem | Theory: must correct adjoint too |
| 19 | Gilton et al. (2023) | Inverse problems with learned forward operators; two paradigms | Theoretical framework for operator correction |
| 20 | Jo et al. (IJCV 2022) | Measuring & controlling spectral bias of DIP | Explains why Arch 1 failed |
| 21 | Kaipio & Somersalo (2007) | BAE framework; model error as Gaussian; covariance reweighting | Framework for Layer 2 (diagonal BAE) |
| 22 | Bezek & Goksel (IEEE TMI 2024) | Convolutional formulation for SoS imaging model | Supervisors' paper; demonstrates learned model improves contrast 10× |
| 23 | Lozenski et al. (SPIE 2024) | CNN corrects pressure data for Born approximation in USCT | Precursor to arXiv 2025 paper |

### Previously Collected (Papers 1–15)
See `references_kwave_gap.txt` for the original 15 references collected during Phase 1.

---

## 6. Experimental Plan

### Experiment 1: SVD-Weighted Loss
- Modify loss to downweight ill-conditioned singular modes
- Sweep over truncation threshold / weighting schemes
- Compare: uniform weights vs SVD-informed weights
- **Baseline**: Current best INR MAE ≈ 17 (standard), L1 baseline = 7.0

### Experiment 2: Diagonal BAE (Mean-Subtraction + Reweighting)
- Compute per-ray ε statistics from training data
- Subtract mean mismatch, reweight by 1/σ²
- Test with and without SVD weighting
- **Evaluation**: LOO cross-validation on N=30 samples

### Experiment 3: Joint Reconstruction + Residual (τ sweep)
- Implement dual-INR engine (f_θ for slowness, g_φ for residual)
- Sweep τ ∈ [0.001, 0.01, 0.1, 1.0, 10.0]
- Monitor separation: ‖g_φ‖ / ‖L·f_θ‖ ratio
- **Evaluation**: MAE, SSIM, visual inspection of g_φ for structure

### Experiment 4: Per-Ray Affine Correction
- Learn a_i, b_i per ray from training data (2 × 131072 parameters, N=30 equations per ray)
- Apply to test samples
- Combine with INR reconstruction
- **Evaluation**: Compare corrected d to L·s_true; then reconstruct and compare MAE

### Experiment 5: Combined Pipeline
- Best of Layers 1–4 stacked
- E.g., mean-subtract + SVD-weight + joint residual
- **Target**: MAE < 7.0 (beat L1 baseline)

---

## 7. Open Questions

1. **Is N=30 enough for any supervised correction?** The mismatch is systematic (LOO=0.66), so a low-rank correction might generalize. But we need LOO validation to be sure.

2. **What should g_φ's input be?** Ray index? Transmitter-receiver geometry? Spatial coordinates? The mismatch structure (strongest at fan edges) suggests geometric features.

3. **Can we combine BAE with the joint residual?** Use BAE for the mean/reweighting and the residual INR for sample-specific deviations from the mean mismatch.

4. **How does the optimal τ relate to ‖ε‖/‖d‖?** Gilton et al. used τ=0.1 universally. Our ε/d ratio is ~0.03 — does this change the optimal τ?

5. **Lunz et al.'s warning**: Forward-only correction may be insufficient if s_true ∉ closure(range(L*)). Is this the case for our L? Our L-matrix has effective rank ≈ 4050/4096, so range(L*) is nearly full-rank — this may not be an issue for us.

---

## References (Full List)

1. Byra et al. (2024) — INR for SoS, differentiable beamformer. [arXiv:2409.14035](https://arxiv.org/abs/2409.14035)
2. Lozenski et al. (2025) — Learned correction for USCT with simplified models. [arXiv:2502.09546](https://arxiv.org/abs/2502.09546)
3. P2INR-FWI (MICCAI 2025) — Polar INR for FWI. [MICCAI](https://papers.miccai.org/miccai-2025/paper/2163_paper.pdf)
4. Plug-and-Play untrained NN for FWI (2024). [arXiv:2406.08523](https://arxiv.org/abs/2406.08523)
5. Physics and Deep Learning in Computational Wave Imaging (2024). [arXiv:2410.08329](https://arxiv.org/abs/2410.08329)
6. Robust DL for Pulse-echo SoS via Time-shift Maps (2024). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12629671/)
7. 3D Forward Model with Transducer Modeling (IEEE TUFFC 2023). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10775680/)
8. DNN-based SoS with Tikhonov Pseudo-Inverse Prior (2023). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10448397/)
9. Ring Artifact Removal via INR Sinogram Decomposition (2024). [arXiv:2409.15731](https://arxiv.org/abs/2409.15731)
10. Diff-INR: Generative Regularization for EIT (2024). [arXiv:2409.04494](https://arxiv.org/abs/2409.04494)
11. Gilton et al. — Untrained NN for Inverse Problems with Model Mismatch (ICLR 2025). [arXiv:2403.04847](https://arxiv.org/abs/2403.04847)
12. FINER: Spectral-Bias Tuning in INR (CVPR 2024). [CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_FINER_Flexible_Spectral-bias_Tuning_in_Implicit_NEural_Representation_by_Variable-periodic_CVPR_2024_paper.pdf)
13. Deep Image Prior Survey (2025). [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S105120042500257X)
14. NN-Based Regularization for Inverse Problems (2024). [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/gamm.202470004)
15. Accelerated INR for CT Reconstruction (2025). [arXiv:2504.13390](https://arxiv.org/abs/2504.13390)
16. Lunz, Hauptmann, Arridge et al. — Learned Operator Correction (SIAM 2021). [SIAM](https://epubs.siam.org/doi/10.1137/20M1338460)
17. Gilton et al. — Inverse Problems with Learned Forward Operators (2023). [arXiv:2311.12528](https://arxiv.org/abs/2311.12528)
18. Bezek & Goksel — Learning the Imaging Model via Convolutional Formulation (IEEE TMI 2024). [PubMed](https://pubmed.ncbi.nlm.nih.gov/39401112/)
19. Jo et al. — Measuring and Controlling Spectral Bias of DIP (IJCV 2022). [Springer](https://link.springer.com/article/10.1007/s11263-021-01572-7)
20. Kaipio & Somersalo — Statistical and Computational Inverse Problems (2005, 2007). [Google Books](https://books.google.com/books/about/Statistical_and_Computational_Inverse_Pr.html?id=h0i-Gi4rCZIC)
21. Noise2Inverse — Self-supervised denoising for tomography (2020). [arXiv:2001.11801](https://arxiv.org/abs/2001.11801)
22. Hendriksen et al. — Noise2Inverse (IEEE TCI 2020). [IEEE](https://ieeexplore.ieee.org/document/9178467/)
23. Bezek et al. — Uncertainty estimation for SoS reconstruction (IJCARS 2025). [Springer](https://link.springer.com/article/10.1007/s11548-025-03402-4)
24. IGA-INR — Inductive Gradient Adjustment for spectral bias (2024). [arXiv:2410.13271](https://arxiv.org/abs/2410.13271)
