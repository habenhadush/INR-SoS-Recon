# Adaptive Lambda Strategies for Joint Denoiser+Reconstructor

## Joint Loss Formulation

The staged joint pipeline optimizes two INRs: a per-pair denoiser $f_\theta$ and a reconstruction INR $g_\phi$. In Stage 3 (joint fine-tuning), the combined loss is:

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda(t) \cdot \mathcal{L}_{\text{fit}}$$

where:

$$\mathcal{L}_{\text{recon}} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \left( \tau \cdot \left[ (\mathbf{L} \hat{\mathbf{s}})_i - \hat{d}_i \right] \right)^2$$

$$\mathcal{L}_{\text{fit}} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \left( \tau \cdot \left[ \hat{d}_i - d_i^{\text{raw}} \right] \right)^2$$

- $\hat{\mathbf{s}} = g_\phi(\mathbf{x})$: reconstructed slowness field (clamped to $[1/1800, 1/1200]$ s/m)
- $\hat{\mathbf{d}} = f_\theta(\mathbf{x}_{\text{DT}})$: denoised displacement field (8 per-pair INRs)
- $\mathbf{d}^{\text{raw}}$: raw measured displacements
- $\mathcal{V}$: set of valid (non-NaN) ray indices
- $\tau$: time_scale ($1/\text{pix2time}$) for numerical stability
- $\mathbf{L}$: precomputed forward model matrix $(131072 \times 4096)$

## Staged Training Protocol

1. **Stage 1** ($N_1$ steps): Pretrain denoiser blind — MSE on raw valid pixels only
2. **Stage 2** ($N_2$ steps): Pretrain reconstructor on denoised data, denoiser frozen
3. **Stage 3** ($N_3$ steps): Joint fine-tuning, both networks, $\text{lr} \times \alpha_{\text{lr}}$

## Lambda Strategies (Stage 3 only)

### Strategy 1: Fixed

$$\lambda(t) = \lambda_0 \quad \forall t$$

Single hyperparameter $\lambda_0$. Best MAE at $\lambda_0 = 0.07$ on kwave_geom (MAE 5.6), but sample-dependent — one sample had MAE 12.7 while others were 3-6.

### Strategy 2: Cosine Decay

$$\lambda(t) = \lambda_{\min} + \frac{1}{2}(\lambda_{\max} - \lambda_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

Starts conservative (high $\lambda$, denoiser stays near raw data), decays to allow more correction. Best CNR on kwave_geom (3.50) but MAE degraded (10.3) due to late-stage drift.

**Hyperparameters**: $\lambda_{\max}$, $\lambda_{\min}$

### Strategy 3: Loss-Ratio Balanced

Maintains a target ratio between the two loss terms by adjusting $\lambda$ each step:

$$r(t) = \frac{\mathcal{L}_{\text{recon}}(t)}{\mathcal{L}_{\text{fit}}(t) + \epsilon}$$

$$\lambda(t+1) = \begin{cases} \lambda(t) \cdot (1 - \eta) & \text{if } r(t) > r_{\text{target}} \quad \text{(denoiser too conservative)} \\ \lambda(t) \cdot (1 + \eta) & \text{if } r(t) < r_{\text{target}} \quad \text{(denoiser drifting)} \end{cases}$$

$$\lambda(t+1) = \text{clamp}(\lambda(t+1), \lambda_{\min}, \lambda_{\max})$$

Fully adaptive per-sample — the loss ratio naturally reflects mismatch severity.

**Hyperparameters**: $r_{\text{target}}$, $\eta$ (update rate), $\lambda_{\min}$, $\lambda_{\max}$

### Strategy 4: Residual-Normalized

Sets $\lambda$ per-sample from the post-Stage-2 residual ratio:

$$r_0 = \mathcal{L}_{\text{recon}}^{(S2)}, \quad f_0 = \mathcal{L}_{\text{fit}}^{(S2)}$$

$$\lambda_{\text{sample}} = \alpha \cdot \frac{r_0}{f_0}$$

If $r_0 \gg f_0$: large mismatch remains, low $\lambda$ grants more denoising freedom.
If $r_0 \approx f_0$: moderate $\lambda$.

Fixed throughout Stage 3 but automatically adapts per sample.

**Hyperparameters**: $\alpha$ (global scaling)

## Results Summary (kwave_geom, 8 samples)

| Strategy | Config | MAE (mean) | CNR (mean) | Notes |
|----------|--------|-----------|-----------|-------|
| L1 baseline | — | 5.8 | 2.88 | Reference |
| L2 baseline | — | 9.2 | 2.99 | Reference |
| Raw INR | — | 13.6–14.5 | 2.12–2.45 | No denoising |
| Fixed | $\lambda_0 = 0.07$ | **5.6** | 2.84 | Best MAE, but 1 outlier sample (12.7) |
| Cosine | $5.0 \to 0.001$ | 10.3 | **3.50** | Best CNR, MAE degraded |
| Residual | $\alpha = 0.1$ | 5.7 | 3.22 | Best MAE-CNR tradeoff, no outliers |
| Residual | $\alpha = 0.05$ | 6.7 | 3.15 | Slightly looser |
| Residual | $\alpha = 0.01$ | 20.3 | 1.80 | Too much freedom, collapsed |

## Key Takeaway

The residual strategy ($\alpha = 0.1$) achieves the best balance: MAE comparable to the L1 baseline while improving CNR by ~15%. It automatically adapts $\lambda$ per sample based on the actual mismatch level, eliminating the sample-dependent outlier problem seen with fixed $\lambda$.

However, none of these strategies transfer to kwave_blob, where the joint pipeline underperforms L1/L2 baselines regardless of $\lambda$ configuration. The blob dataset has higher mismatch energy (0.44% vs 0.11%) and differently-structured inclusions (amorphous vs geometric), suggesting the current pipeline architecture needs further work for more challenging data.
