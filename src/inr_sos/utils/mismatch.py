"""Mismatch correction utilities for Layer 1 (SVD weighting) and Layer 2 (diagonal BAE).

Layer 1 — SVD-weighted loss:
    Downweight ill-conditioned singular modes where mismatch amplification is worst.
    Uses the SVD of the masked L-matrix to construct per-ray weights in the SVD basis.

Layer 2 — Diagonal Bayesian Approximation Error (BAE):
    Compute per-ray mismatch statistics (mean, variance) from training data,
    then subtract the mean bias and reweight by 1/sigma^2.
"""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_svd_weights(L_matrix, mask, config):
    """Compute SVD-based loss weights for the masked L-matrix.

    Constructs a projection into the SVD basis and returns weights that
    downweight the ill-conditioned tail modes (k > svd_top_k).

    Args:
        L_matrix: (M, N) tensor — full L-matrix.
        mask: (M, 1) tensor — binary ray mask.
        config: ExperimentConfig with svd_top_k and svd_tail_damping.

    Returns:
        dict with:
            'U': (M_valid, K) left singular vectors (on CPU)
            'S': (K,) singular values
            'Vt': (K, N) right singular vectors
            'weights': (M, 1) per-ray weights (on same device as L_matrix)
    """
    device = L_matrix.device
    mask_flat = mask.flatten()
    valid_idx = torch.where(mask_flat > 0.5)[0]

    # Extract masked L for SVD (move to CPU for large SVD)
    L_masked = L_matrix[valid_idx].cpu().float()
    logger.info(f"Computing SVD of masked L-matrix: {L_masked.shape}")

    # Economy SVD
    U, S, Vt = torch.linalg.svd(L_masked, full_matrices=False)
    K = S.shape[0]
    logger.info(f"SVD complete: {K} singular values, condition number: {S[0]/S[-1]:.2e}")

    # Build per-mode weights: full weight for top_k, damped for tail
    top_k = min(config.svd_top_k, K)
    mode_weights = torch.ones(K, dtype=torch.float32)
    mode_weights[top_k:] = config.svd_tail_damping
    logger.info(
        f"SVD weighting: top_k={top_k}, tail_damping={config.svd_tail_damping}, "
        f"tail modes={K - top_k}"
    )

    # Project mode weights back to ray space:
    # For each ray i, its effective weight = sum_k (U[i,k]^2 * w_k)
    # This gives higher weight to rays that are well-explained by the
    # well-conditioned modes, and lower weight to rays dominated by the tail.
    U_sq = U ** 2  # (M_valid, K)
    ray_weights_valid = U_sq @ mode_weights  # (M_valid,)

    # Normalize to mean 1.0 for stable loss scaling
    ray_weights_valid = ray_weights_valid / (ray_weights_valid.mean() + 1e-8)

    # Map back to full ray space
    ray_weights = torch.zeros(L_matrix.shape[0], 1, dtype=torch.float32, device=device)
    ray_weights[valid_idx, 0] = ray_weights_valid.to(device)

    return {
        'U': U,
        'S': S,
        'Vt': Vt,
        'mode_weights': mode_weights,
        'weights': ray_weights,
    }


def compute_bae_stats(dataset, L_matrix, mask_intersection=None, max_samples=None):
    """Compute per-ray mismatch statistics from training data.

    For each sample i, computes epsilon_i = d_meas_i - L @ s_true_i,
    then returns mean(epsilon) and var(epsilon) across samples.

    Args:
        dataset: USDataset instance.
        L_matrix: (M, N) tensor — L-matrix.
        mask_intersection: (M, 1) optional — common valid mask across samples.
            If None, uses per-sample masks intersected.
        max_samples: int — limit samples for faster computation.

    Returns:
        dict with:
            'eta': (M, 1) mean mismatch per ray
            'sigma2': (M, 1) variance per ray
            'n_samples': int — number of samples used
            'mask_common': (M, 1) common valid mask
    """
    N = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    M = L_matrix.shape[0]

    logger.info(f"Computing BAE statistics from {N} samples...")

    # Collect mismatch vectors
    epsilons = []
    masks = []

    for i in range(N):
        sample = dataset[i]
        d_meas = sample['d_meas']        # (M, 1)
        s_gt = sample['s_gt_raw']         # (4096, 1)
        m = sample['mask']                # (M, 1)

        # d_model = L @ s_true
        d_model = L_matrix @ s_gt         # (M, 1)

        # epsilon = d_meas - L @ s_true
        eps = d_meas - d_model            # (M, 1)

        epsilons.append(eps)
        masks.append(m)

    epsilons = torch.stack(epsilons, dim=0)  # (N, M, 1)
    masks = torch.stack(masks, dim=0)        # (N, M, 1)

    # Common valid mask (intersection across all samples)
    if mask_intersection is not None:
        mask_common = mask_intersection
    else:
        mask_common = (masks.prod(dim=0) > 0.5).float()  # (M, 1)

    n_valid = mask_common.sum().item()
    logger.info(f"Common valid rays: {int(n_valid)} / {M} ({100*n_valid/M:.1f}%)")

    # Masked statistics
    epsilons_masked = epsilons * mask_common.unsqueeze(0)  # (N, M, 1)

    eta = epsilons_masked.mean(dim=0)        # (M, 1) — mean mismatch
    sigma2 = epsilons_masked.var(dim=0)      # (M, 1) — per-ray variance

    # For invalid rays, set sigma2 to large value (won't be used, but safe)
    sigma2 = sigma2 * mask_common + (1.0 - mask_common) * 1e10

    # Summary statistics
    valid_mask_bool = mask_common.flatten() > 0.5
    eta_valid = eta.flatten()[valid_mask_bool]
    sigma2_valid = sigma2.flatten()[valid_mask_bool]

    logger.info(
        f"BAE stats: mean|eta|={eta_valid.abs().mean():.2e}, "
        f"mean(sigma)={sigma2_valid.sqrt().mean():.2e}, "
        f"max|eta|={eta_valid.abs().max():.2e}"
    )

    return {
        'eta': eta,
        'sigma2': sigma2,
        'n_samples': N,
        'mask_common': mask_common,
    }


def compute_bae_weights(bae_stats, config, noise_floor=1e-16):
    """Convert BAE statistics to per-ray loss weights.

    Args:
        bae_stats: dict from compute_bae_stats.
        config: ExperimentConfig.
        noise_floor: minimum variance to prevent division by zero.

    Returns:
        (M, 1) tensor of per-ray weights, normalized to mean 1.0 over valid rays.
    """
    sigma2 = bae_stats['sigma2']
    mask = bae_stats['mask_common']

    # Inverse-variance weighting: w_i = 1 / (sigma2_i + noise_floor)
    weights = 1.0 / (sigma2 + noise_floor)

    # Zero out invalid rays
    weights = weights * mask

    # Normalize to mean 1.0 over valid rays
    valid_sum = (weights * mask).sum()
    n_valid = mask.sum() + 1e-8
    weights = weights * (n_valid / (valid_sum + 1e-8))

    return weights


def correct_measurements(d_meas, bae_stats, config):
    """Apply BAE mean-subtraction to measurements.

    Args:
        d_meas: (M, 1) measured displacement.
        bae_stats: dict from compute_bae_stats.
        config: ExperimentConfig.

    Returns:
        (M, 1) corrected measurements (d_meas - eta if enabled).
    """
    if config.bae_subtract_mean:
        eta = bae_stats['eta'].to(d_meas.device)
        return d_meas - eta
    return d_meas
