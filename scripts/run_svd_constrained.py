"""
Experiment 2: SVD-Constrained INR

Addresses forward model mismatch by restricting the INR output to the top-K
right singular vectors of L, preventing catastrophic amplification in the
ill-conditioned tail modes.

Sub-experiments:
  2a) Classical TSVD baseline (no INR) — sweep K
  2b) LSQR with early stopping baseline — sweep iterations
  2c) INR + hard SVD projection — sweep K
  2d) INR + progressive K (coarse-to-fine)
  2e) INR + soft projection (Gaussian taper beyond K)

Usage:
    source .venv/bin/activate
    uv run python scripts/run_svd_constrained.py --dataset kwave_geom --sub_exp 2a
    uv run python scripts/run_svd_constrained.py --dataset kwave_geom --sub_exp all
"""

import argparse
import copy
import logging
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse.linalg import lsqr

# -- Project imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inr_sos.utils.data import USDataset
from inr_sos.evaluation.metrics import calculate_metrics, calculate_cnr
from inr_sos.io.paths import DATA_DIR
from inr_sos.utils.config import ExperimentConfig
from inr_sos.models.mlp import FourierMLP
from inr_sos.models.siren import SirenMLP
from inr_sos.training.engines import (
    _DEVICE, _SLOWNESS_MIN, _SLOWNESS_MAX,
    _compute_data_loss, _maybe_clamp_slowness, _EarlyStopper,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# -- Paths ---------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_OUT_DIR = SCRIPTS_DIR / "data" / "experiment2_svd_constrained"
PLOTS_DIR = SCRIPTS_DIR / "plots" / "experiment2_svd_constrained"
SVD_CACHE_DIR = SCRIPTS_DIR / "data" / "svd_cache"

# -- SoS constants -------------------------------------------------------------
SOS_BG = 1540.0
SOS_MIN = 1380.0
SOS_MAX = 1620.0


# =============================================================================
#  Dataset loading (shared with Exp 1)
# =============================================================================

def load_dataset_config(key: str) -> dict:
    cfg_path = SCRIPTS_DIR / "datasets.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ds = cfg["datasets"][key]
    ds["key"] = key
    ds["data_path"] = DATA_DIR + ds["data_file"]
    return ds


def load_dataset(ds_cfg: dict) -> USDataset:
    data_file = ds_cfg["data_path"]
    grid_path = DATA_DIR + ds_cfg.get("grid_path",
                                       "/DL-based-SoS/forward_model_lr/grid_parameters.mat")
    ds_kwargs = {}
    if not ds_cfg.get("has_A_matrix", True):
        matrix_file = ds_cfg.get("matrix_file")
        if matrix_file:
            ds_kwargs["matrix_path"] = DATA_DIR + matrix_file
            ds_kwargs["use_external_L_matrix"] = True
    if ds_cfg.get("h5_keys"):
        ds_kwargs["h5_keys"] = ds_cfg["h5_keys"]
    return USDataset(data_file, grid_path, **ds_kwargs)


# =============================================================================
#  SVD Precomputation & Caching
# =============================================================================

def compute_or_load_svd(L_matrix, dataset_name):
    """Compute full SVD of L and cache to disk.

    Returns:
        U: (M, N) left singular vectors
        S: (N,) singular values
        Vt: (N, N) right singular vectors (transposed)

    where M = num rays (131072), N = num pixels (4096).
    """
    SVD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = SVD_CACHE_DIR / f"svd_{dataset_name}.npz"

    if cache_path.exists():
        logger.info(f"Loading cached SVD from {cache_path}")
        data = np.load(cache_path)
        return data["U"], data["S"], data["Vt"]

    logger.info("Computing full SVD of L-matrix (this may take a minute)...")
    L_np = L_matrix.numpy() if isinstance(L_matrix, torch.Tensor) else np.asarray(L_matrix)

    # Economy SVD: L is (M, N) with M >> N, so U is (M, N), S is (N,), Vt is (N, N)
    U, S, Vt = np.linalg.svd(L_np, full_matrices=False)

    logger.info(
        f"SVD complete: U {U.shape}, S {S.shape}, Vt {Vt.shape}\n"
        f"  Condition number: {S[0]/S[-1]:.2e}\n"
        f"  90% energy at K={np.searchsorted(np.cumsum(S**2)/np.sum(S**2), 0.9) + 1}\n"
        f"  99% energy at K={np.searchsorted(np.cumsum(S**2)/np.sum(S**2), 0.99) + 1}"
    )

    np.savez_compressed(cache_path, U=U, S=S, Vt=Vt)
    logger.info(f"SVD cached to {cache_path}")
    return U, S, Vt


# =============================================================================
#  Plotting utilities
# =============================================================================

def _slowness_to_sos(s, grid_shape=(64, 64)):
    s_flat = np.asarray(s, dtype=np.float32).flatten()
    v = np.clip(1.0 / (s_flat + 1e-8), SOS_MIN, SOS_MAX)
    return v.reshape(grid_shape)


def _get_plot_indices(n_samples, n_plots=6):
    if n_samples <= n_plots:
        return list(range(n_samples))
    step = n_samples // n_plots
    return list(range(0, n_samples, step))[:n_plots]


def plot_sample_comparison(sample_idx, s_gt, methods, dataset_name, sub_exp,
                           grid_shape=(64, 64)):
    n_methods = len(methods)
    n_cols = n_methods + 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    fig.patch.set_facecolor("white")

    v_gt = _slowness_to_sos(s_gt, grid_shape)
    v_min = max(SOS_MIN, float(v_gt.min()) - 10)
    v_max = min(SOS_MAX, float(v_gt.max()) + 10)
    bg_sos = float(np.median(v_gt))
    norm = mcolors.TwoSlopeNorm(vmin=v_min, vcenter=bg_sos, vmax=v_max)

    im = axes[0].imshow(v_gt, cmap="RdBu_r", norm=norm,
                         interpolation="nearest", origin="upper")
    axes[0].set_title("Ground Truth", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    best_err_map = None
    best_mae = float("inf")
    for i, (name, s_rec) in enumerate(methods.items()):
        v_rec = _slowness_to_sos(s_rec, grid_shape)
        err_map = np.abs(v_gt - v_rec)
        mae = float(np.mean(err_map))
        metrics = calculate_metrics(s_rec, s_gt, grid_shape)

        im = axes[i + 1].imshow(v_rec, cmap="RdBu_r", norm=norm,
                                 interpolation="nearest", origin="upper")
        axes[i + 1].set_title(
            f"{name}\nMAE={metrics['MAE']:.1f}  CNR={metrics['CNR']:.2f}",
            fontsize=10)
        axes[i + 1].axis("off")
        plt.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)

        if mae < best_mae:
            best_mae = mae
            best_err_map = err_map

    im = axes[-1].imshow(best_err_map, cmap="hot", vmin=0, vmax=50,
                          interpolation="nearest", origin="upper")
    axes[-1].set_title(f"Best Abs Error\nMAE={best_mae:.1f} m/s", fontsize=10)
    axes[-1].axis("off")
    cb = plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    cb.set_label("|GT - Rec| (m/s)", fontsize=9)

    fig.suptitle(f"Exp2{sub_exp} | {dataset_name} | Sample {sample_idx}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_dir = PLOTS_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"sub_{sub_exp}_sample_{sample_idx:03d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Plot saved: {save_path}")


def plot_metrics_summary(all_results, dataset_name, sub_exp):
    methods = list(all_results.keys())
    metric_names = ["CNR", "SSIM", "RMSE", "MAE"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("white")
    x = np.arange(len(methods))

    for ax, metric in zip(axes, metric_names):
        means = [np.mean([m[metric] for m in all_results[meth]])
                 for meth in methods]
        stds = [np.std([m[metric] for m in all_results[meth]])
                for meth in methods]
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="gray")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Exp2{sub_exp} Metrics Summary | {dataset_name}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_dir = PLOTS_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"sub_{sub_exp}_metrics_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Summary plot saved: {save_path}")


def plot_k_sweep(sweep_results, dataset_name, sub_exp):
    """Line plot of metrics vs truncation level K."""
    k_labels = list(sweep_results.keys())
    k_values = [int(k.split("=")[1]) for k in k_labels]
    metric_names = ["CNR", "SSIM", "RMSE", "MAE"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("white")

    for ax, metric in zip(axes, metric_names):
        means = [np.mean([m[metric] for m in sweep_results[k]])
                 for k in k_labels]
        stds = [np.std([m[metric] for m in sweep_results[k]])
                for k in k_labels]
        ax.errorbar(k_values, means, yerr=stds, marker="o", capsize=4,
                    linewidth=2, markersize=6)
        ax.set_xlabel("K (truncation level)", fontsize=10)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Exp2{sub_exp} K Sweep | {dataset_name}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_dir = PLOTS_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"sub_{sub_exp}_k_sweep.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  K sweep plot saved: {save_path}")


# =============================================================================
#  Sub-experiment 2a: Classical TSVD (no INR)
# =============================================================================

def run_2a_tsvd(dataset, dataset_name, U, S, Vt,
                k_values=None):
    """Classical Truncated SVD reconstruction — sweep K."""
    if k_values is None:
        k_values = [50, 86, 150, 200, 300, 500, 1000, 2000, 4096]

    logger.info("=" * 60)
    logger.info(f"SUB-EXPERIMENT 2a: Classical TSVD — K in {k_values}")
    logger.info("=" * 60)

    n_samples = len(dataset)
    plot_indices = set(_get_plot_indices(n_samples))
    results = {}
    # Store reconstructions per sample for cross-K comparison plots
    per_sample_recons = {idx: {} for idx in plot_indices}
    per_sample_gt = {}

    for K in k_values:
        logger.info(f"\n--- K = {K} ---")
        k_label = f"K={K}"
        results[k_label] = []

        # Precompute truncated components
        U_K = U[:, :K]      # (M, K)
        S_K = S[:K]          # (K,)
        V_K = Vt[:K, :].T   # (4096, K)

        for test_idx in range(n_samples):
            sample = dataset[test_idx]
            s_gt = sample['s_gt_raw'].numpy().flatten()
            d_meas = sample['d_meas'].numpy().flatten()
            mask = sample['mask'].numpy().flatten()

            # Project data into SVD space: alpha = diag(1/sigma_k) @ U_K^T @ d
            d_valid = d_meas * mask
            alpha = (U_K.T @ d_valid) / (S_K + 1e-15)  # (K,)

            # Reconstruct: s = V_K @ alpha
            s_tsvd = V_K @ alpha  # (4096,)

            metrics = calculate_metrics(s_tsvd, s_gt, grid_shape=(64, 64))
            results[k_label].append(metrics)

            if test_idx in plot_indices:
                per_sample_recons[test_idx][f"TSVD K={K}"] = s_tsvd
                per_sample_gt[test_idx] = s_gt

            if test_idx % 10 == 0:
                logger.info(
                    f"  Sample {test_idx}: MAE={metrics['MAE']:.2f}, "
                    f"CNR={metrics['CNR']:.3f}, SSIM={metrics['SSIM']:.4f}"
                )

        # Summary for this K
        mae_mean = np.mean([m['MAE'] for m in results[k_label]])
        cnr_mean = np.mean([m['CNR'] for m in results[k_label]])
        logger.info(f"  K={K}: mean MAE={mae_mean:.2f}, mean CNR={cnr_mean:.3f}")

    # Plot representative samples with a selection of K values
    plot_k_selection = [k for k in k_values if k in [86, 200, 500, 2000, 4096]]
    if not plot_k_selection:
        plot_k_selection = k_values[:min(5, len(k_values))]
    for idx in plot_indices:
        methods_to_plot = {k: v for k, v in per_sample_recons[idx].items()
                          if any(f"K={pk}" in k for pk in [str(x) for x in plot_k_selection])}
        if methods_to_plot:
            plot_sample_comparison(idx, per_sample_gt[idx], methods_to_plot,
                                   dataset_name, "2a")

    _print_results_table("2a", results, dataset_name)
    plot_k_sweep(results, dataset_name, "2a")
    plot_metrics_summary(results, dataset_name, "2a")
    return results


# =============================================================================
#  Sub-experiment 2b: LSQR with early stopping
# =============================================================================

def run_2b_lsqr_early_stop(dataset, dataset_name,
                            iter_values=None):
    """LSQR baseline with varying iteration limits (implicit regularization)."""
    if iter_values is None:
        iter_values = [10, 25, 50, 100, 200, 500, 1000]

    logger.info("=" * 60)
    logger.info(f"SUB-EXPERIMENT 2b: LSQR Early Stopping — iters in {iter_values}")
    logger.info("=" * 60)

    L_np = dataset.L_matrix.numpy()
    n_samples = len(dataset)
    plot_indices = set(_get_plot_indices(n_samples))
    results = {}
    per_sample_recons = {idx: {} for idx in plot_indices}
    per_sample_gt = {}

    for n_iter in iter_values:
        logger.info(f"\n--- LSQR iter_lim = {n_iter} ---")
        k_label = f"iter={n_iter}"
        results[k_label] = []

        for test_idx in range(n_samples):
            sample = dataset[test_idx]
            s_gt = sample['s_gt_raw'].numpy().flatten()
            d_meas = sample['d_meas'].numpy().flatten()
            mask = sample['mask'].numpy().flatten()
            valid = mask > 0.5

            L_valid = L_np[valid]
            d_valid = d_meas[valid]

            s_rec, *_ = lsqr(L_valid, d_valid, damp=0.0, iter_lim=n_iter)

            metrics = calculate_metrics(s_rec, s_gt, grid_shape=(64, 64))
            results[k_label].append(metrics)

            if test_idx in plot_indices:
                per_sample_recons[test_idx][f"LSQR i={n_iter}"] = s_rec
                per_sample_gt[test_idx] = s_gt

            if test_idx % 10 == 0:
                logger.info(
                    f"  Sample {test_idx}: MAE={metrics['MAE']:.2f}, "
                    f"CNR={metrics['CNR']:.3f}"
                )

        mae_mean = np.mean([m['MAE'] for m in results[k_label]])
        cnr_mean = np.mean([m['CNR'] for m in results[k_label]])
        logger.info(f"  iter={n_iter}: mean MAE={mae_mean:.2f}, mean CNR={cnr_mean:.3f}")

    # Plot representative samples with a selection of iteration counts
    plot_iter_selection = [n for n in iter_values if n in [25, 100, 500, 1000]]
    if not plot_iter_selection:
        plot_iter_selection = iter_values[:min(4, len(iter_values))]
    for idx in plot_indices:
        methods_to_plot = {k: v for k, v in per_sample_recons[idx].items()
                          if any(f"i={pi}" in k for pi in [str(x) for x in plot_iter_selection])}
        if methods_to_plot:
            plot_sample_comparison(idx, per_sample_gt[idx], methods_to_plot,
                                   dataset_name, "2b")

    _print_results_table("2b", results, dataset_name)
    plot_metrics_summary(results, dataset_name, "2b")
    return results


# =============================================================================
#  SVD-Constrained INR Engine
# =============================================================================

def optimize_svd_constrained(sample, L_matrix, model, config,
                              U_K, S_K, V_K, label="SVD-INR"):
    """INR training with hard SVD subspace projection.

    Instead of d_pred = L @ s, we use the reduced forward model:
        s_raw  = INR(coords) * s_std + s_mean
        alpha  = V_K^T @ s_raw          (K,)
        d_pred = (U_K * S_K) @ alpha    (M,)   [reduced, well-conditioned]
        s_proj = V_K @ alpha            for final reconstruction

    Args:
        U_K: (M, K) left singular vectors (torch, on device)
        S_K: (K,) singular values (torch, on device)
        V_K: (4096, K) right singular vectors (torch, on device)
    """
    model = model.to(_DEVICE)
    coords = sample['coords'].to(_DEVICE)
    d_meas = sample['d_meas'].to(_DEVICE)
    mask = sample['mask'].to(_DEVICE)
    s_mean = sample['s_stats'][0].item()
    s_std = sample['s_stats'][1].item()

    # Precompute U_K_scaled = U_K * S_K for fast forward model
    U_K_scaled = U_K * S_K.unsqueeze(0)  # (M, K)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)

    stopper = _EarlyStopper(mask, config, model)
    train_mask = stopper.get_train_mask(mask)

    loss_history = []

    for step in range(config.steps):
        model.train()
        optimizer.zero_grad()

        # INR output -> physical slowness
        s_norm = model(coords)  # (4096, 1)
        s_phys = _maybe_clamp_slowness(s_norm * s_std + s_mean, config)

        # SVD projection: project slowness into K-dim subspace
        alpha = V_K.T @ s_phys  # (K, 1)

        # Reduced forward model
        d_pred = U_K_scaled @ alpha  # (M, 1)

        residual_seconds = d_pred - d_meas
        loss = _compute_data_loss(residual_seconds, train_mask, config)

        # Regularization (on projected slowness)
        reg_loss = 0
        if config.reg_weight > 0:
            reg_loss += config.reg_weight * (s_norm ** 2).mean()
        if config.tv_weight > 0:
            s_proj = V_K @ alpha  # (4096, 1)
            s_img = s_proj.reshape(64, 64)
            tv_x = ((s_img[:, 1:] - s_img[:, :-1]) ** 2).mean()
            tv_z = ((s_img[1:, :] - s_img[:-1, :]) ** 2).mean()
            reg_loss += config.tv_weight * (tv_x + tv_z)

        total_loss = loss + reg_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(total_loss.item())

        if step % 50 == 0:
            logging.info(f"  [{label}] step {step}: loss={loss.item():.4f}")

        # Early stopping
        if stopper.enabled and step % config.log_interval == 0:
            with torch.no_grad():
                val_loss, should_stop = stopper.evaluate(
                    residual_seconds.detach(), config, model
                )
            if should_stop:
                logging.info(f"  [{label}] Early stopping at step {step}")
                break

    stopper.restore_best(model)

    # Final reconstruction: project to subspace
    model.eval()
    with torch.no_grad():
        s_norm = model(coords)
        s_phys = _maybe_clamp_slowness(s_norm * s_std + s_mean, config)
        alpha = V_K.T @ s_phys
        s_proj = V_K @ alpha  # (4096, 1) — lives in span(V_K)

    return {
        's_phys': s_proj.detach().cpu(),
        's_norm': s_norm.detach().cpu(),
        'loss_history': loss_history,
        'model_state': model.state_dict(),
    }


def optimize_svd_progressive(sample, L_matrix, model, config,
                               U, S, Vt, k_schedule, label="Prog-SVD-INR"):
    """INR training with progressive K (coarse-to-fine).

    k_schedule: list of (step_threshold, K) tuples, e.g.:
        [(0, 50), (200, 100), (400, 200), (600, 400)]
    K increases at specified steps.
    """
    model = model.to(_DEVICE)
    coords = sample['coords'].to(_DEVICE)
    d_meas = sample['d_meas'].to(_DEVICE)
    mask = sample['mask'].to(_DEVICE)
    s_mean = sample['s_stats'][0].item()
    s_std = sample['s_stats'][1].item()

    # Convert SVD to torch tensors on device
    U_full = torch.tensor(U, dtype=torch.float32, device=_DEVICE)
    S_full = torch.tensor(S, dtype=torch.float32, device=_DEVICE)
    Vt_full = torch.tensor(Vt, dtype=torch.float32, device=_DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)

    stopper = _EarlyStopper(mask, config, model)
    train_mask = stopper.get_train_mask(mask)

    loss_history = []
    current_K = k_schedule[0][1]
    schedule_idx = 0

    for step in range(config.steps):
        # Update K according to schedule
        while schedule_idx < len(k_schedule) - 1 and step >= k_schedule[schedule_idx + 1][0]:
            schedule_idx += 1
            new_K = k_schedule[schedule_idx][1]
            if new_K != current_K:
                logger.info(f"  [{label}] step {step}: K {current_K} -> {new_K}")
                current_K = new_K

        # Current truncated components
        U_K = U_full[:, :current_K]
        S_K = S_full[:current_K]
        V_K = Vt_full[:current_K, :].T

        model.train()
        optimizer.zero_grad()

        s_norm = model(coords)
        s_phys = _maybe_clamp_slowness(s_norm * s_std + s_mean, config)

        alpha = V_K.T @ s_phys
        U_K_scaled = U_K * S_K.unsqueeze(0)
        d_pred = U_K_scaled @ alpha

        residual_seconds = d_pred - d_meas
        loss = _compute_data_loss(residual_seconds, train_mask, config)

        reg_loss = 0
        if config.reg_weight > 0:
            reg_loss += config.reg_weight * (s_norm ** 2).mean()
        if config.tv_weight > 0:
            s_proj = V_K @ alpha
            s_img = s_proj.reshape(64, 64)
            tv_x = ((s_img[:, 1:] - s_img[:, :-1]) ** 2).mean()
            tv_z = ((s_img[1:, :] - s_img[:-1, :]) ** 2).mean()
            reg_loss += config.tv_weight * (tv_x + tv_z)

        total_loss = loss + reg_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(total_loss.item())

        if step % 50 == 0:
            logging.info(f"  [{label}] step {step} (K={current_K}): loss={loss.item():.4f}")

        if stopper.enabled and step % config.log_interval == 0:
            with torch.no_grad():
                val_loss, should_stop = stopper.evaluate(
                    residual_seconds.detach(), config, model
                )
            if should_stop:
                logging.info(f"  [{label}] Early stopping at step {step}")
                break

    stopper.restore_best(model)

    # Final: use maximum K from schedule for reconstruction
    final_K = k_schedule[-1][1]
    V_K_final = Vt_full[:final_K, :].T
    model.eval()
    with torch.no_grad():
        s_norm = model(coords)
        s_phys = _maybe_clamp_slowness(s_norm * s_std + s_mean, config)
        alpha = V_K_final.T @ s_phys
        s_proj = V_K_final @ alpha

    return {
        's_phys': s_proj.detach().cpu(),
        's_norm': s_norm.detach().cpu(),
        'loss_history': loss_history,
        'model_state': model.state_dict(),
    }


def optimize_svd_soft(sample, L_matrix, model, config,
                       U, S, Vt, K_center, taper_width,
                       label="Soft-SVD-INR"):
    """INR training with soft SVD taper (Gaussian rolloff beyond K_center).

    Weight for mode i: w_i = exp(-max(0, i - K_center)^2 / (2 * taper_width^2))
    Modes below K_center have weight 1, modes above are smoothly suppressed.
    """
    model = model.to(_DEVICE)
    coords = sample['coords'].to(_DEVICE)
    d_meas = sample['d_meas'].to(_DEVICE)
    mask = sample['mask'].to(_DEVICE)
    s_mean = sample['s_stats'][0].item()
    s_std = sample['s_stats'][1].item()

    N = len(S)  # 4096
    U_t = torch.tensor(U, dtype=torch.float32, device=_DEVICE)
    S_t = torch.tensor(S, dtype=torch.float32, device=_DEVICE)
    V = torch.tensor(Vt.T, dtype=torch.float32, device=_DEVICE)  # (4096, 4096)

    # Compute taper weights
    mode_idx = torch.arange(N, dtype=torch.float32, device=_DEVICE)
    taper = torch.exp(-torch.clamp(mode_idx - K_center, min=0) ** 2 / (2 * taper_width ** 2))
    # taper shape: (4096,)

    # Weighted forward model: d_pred = U @ diag(S * taper) @ V^T @ s
    S_tapered = S_t * taper  # (4096,)
    U_scaled = U_t * S_tapered.unsqueeze(0)  # (M, 4096)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)

    stopper = _EarlyStopper(mask, config, model)
    train_mask = stopper.get_train_mask(mask)

    loss_history = []

    for step in range(config.steps):
        model.train()
        optimizer.zero_grad()

        s_norm = model(coords)
        s_phys = _maybe_clamp_slowness(s_norm * s_std + s_mean, config)

        # Tapered forward: project s into SVD space, apply taper, back to measurement
        alpha = V.T @ s_phys  # (4096, 1)
        d_pred = U_scaled @ alpha  # (M, 1)

        residual_seconds = d_pred - d_meas
        loss = _compute_data_loss(residual_seconds, train_mask, config)

        reg_loss = 0
        if config.reg_weight > 0:
            reg_loss += config.reg_weight * (s_norm ** 2).mean()
        if config.tv_weight > 0:
            s_proj = V @ (taper.unsqueeze(1) * alpha)
            s_img = s_proj.reshape(64, 64)
            tv_x = ((s_img[:, 1:] - s_img[:, :-1]) ** 2).mean()
            tv_z = ((s_img[1:, :] - s_img[:-1, :]) ** 2).mean()
            reg_loss += config.tv_weight * (tv_x + tv_z)

        total_loss = loss + reg_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(total_loss.item())

        if step % 50 == 0:
            logging.info(f"  [{label}] step {step}: loss={loss.item():.4f}")

        if stopper.enabled and step % config.log_interval == 0:
            with torch.no_grad():
                val_loss, should_stop = stopper.evaluate(
                    residual_seconds.detach(), config, model
                )
            if should_stop:
                logging.info(f"  [{label}] Early stopping at step {step}")
                break

    stopper.restore_best(model)

    model.eval()
    with torch.no_grad():
        s_norm = model(coords)
        s_phys = _maybe_clamp_slowness(s_norm * s_std + s_mean, config)
        alpha = V.T @ s_phys
        s_proj = V @ (taper.unsqueeze(1) * alpha)

    return {
        's_phys': s_proj.detach().cpu(),
        's_norm': s_norm.detach().cpu(),
        'loss_history': loss_history,
        'model_state': model.state_dict(),
    }


# =============================================================================
#  Sub-experiment 2c: INR + Hard SVD Projection — sweep K
# =============================================================================

def _make_inr_config():
    """Default INR config for SVD experiments (matches best from sweeps)."""
    return ExperimentConfig(
        project_name="Exp2-SVD-INR",
        experiment_group="svd_constrained",
        model_type="FourierMLP",
        hidden_features=256,
        hidden_layers=4,
        mapping_size=64,
        scale=5.0,
        lr=5e-4,
        steps=500,
        time_scale=1e6,
        tv_weight=1e-2,
        reg_weight=1e-3,
        loss_type="mse",
        clamp_slowness=True,
        early_stopping=True,
        patience=100,
    )


def _make_model(config):
    return FourierMLP(
        in_features=config.in_features,
        hidden_features=config.hidden_features,
        hidden_layers=config.hidden_layers,
        mapping_size=config.mapping_size,
        scale=config.scale,
    )


def run_2c_inr_hard_projection(dataset, dataset_name, U, S, Vt,
                                k_values=None, n_eval_samples=12):
    """INR + hard SVD projection — sweep K."""
    if k_values is None:
        k_values = [50, 86, 150, 200, 300, 500, 1000, 2000, 4096]

    logger.info("=" * 60)
    logger.info(f"SUB-EXPERIMENT 2c: INR + Hard SVD Projection — K in {k_values}")
    logger.info("=" * 60)

    n_samples = len(dataset)
    eval_indices = list(range(min(n_eval_samples, n_samples)))
    plot_indices = set(_get_plot_indices(len(eval_indices)))
    config = _make_inr_config()
    results = {}
    # Store reconstructions for cross-K plots: {sample_idx: {method_name: s_rec}}
    per_sample_recons = {eval_indices[i]: {} for i in plot_indices}
    per_sample_gt = {}

    for K in k_values:
        logger.info(f"\n--- K = {K} ---")
        k_label = f"K={K}"
        results[k_label] = []

        # Prepare truncated SVD components on device
        U_K = torch.tensor(U[:, :K], dtype=torch.float32, device=_DEVICE)
        S_K = torch.tensor(S[:K], dtype=torch.float32, device=_DEVICE)
        V_K = torch.tensor(Vt[:K, :].T, dtype=torch.float32, device=_DEVICE)

        for i, test_idx in enumerate(eval_indices):
            sample = dataset[test_idx]
            s_gt_np = sample['s_gt_raw'].numpy().flatten()

            model = _make_model(config)
            result = optimize_svd_constrained(
                sample=sample,
                L_matrix=dataset.L_matrix,
                model=model,
                config=config,
                U_K=U_K, S_K=S_K, V_K=V_K,
                label=f"K={K}",
            )

            s_rec = result['s_phys']
            if hasattr(s_rec, 'numpy'):
                s_rec = s_rec.numpy()
            s_rec = np.asarray(s_rec).flatten()

            metrics = calculate_metrics(s_rec, s_gt_np, grid_shape=(64, 64))
            results[k_label].append(metrics)

            if i in plot_indices:
                per_sample_recons[test_idx][f"INR K={K}"] = s_rec
                per_sample_gt[test_idx] = s_gt_np

            logger.info(
                f"  Sample {test_idx}: MAE={metrics['MAE']:.2f}, "
                f"CNR={metrics['CNR']:.3f}, SSIM={metrics['SSIM']:.4f}"
            )

        mae_mean = np.mean([m['MAE'] for m in results[k_label]])
        cnr_mean = np.mean([m['CNR'] for m in results[k_label]])
        logger.info(f"  K={K}: mean MAE={mae_mean:.2f}, mean CNR={cnr_mean:.3f}")

    # Plot representative samples with a selection of K values
    plot_k_selection = [k for k in k_values if k in [86, 200, 500, 2000, 4096]]
    if not plot_k_selection:
        plot_k_selection = k_values[:min(5, len(k_values))]
    for idx in per_sample_recons:
        methods_to_plot = {k: v for k, v in per_sample_recons[idx].items()
                          if any(f"K={pk}" in k for pk in [str(x) for x in plot_k_selection])}
        if methods_to_plot:
            plot_sample_comparison(idx, per_sample_gt[idx], methods_to_plot,
                                   dataset_name, "2c")

    _print_results_table("2c", results, dataset_name)
    plot_k_sweep(results, dataset_name, "2c")
    plot_metrics_summary(results, dataset_name, "2c")
    return results


# =============================================================================
#  Sub-experiment 2d: INR + Progressive K
# =============================================================================

def run_2d_progressive_k(dataset, dataset_name, U, S, Vt,
                          n_eval_samples=12):
    """INR with progressive K (coarse-to-fine schedule)."""
    logger.info("=" * 60)
    logger.info("SUB-EXPERIMENT 2d: INR + Progressive K")
    logger.info("=" * 60)

    n_samples = len(dataset)
    eval_indices = list(range(min(n_eval_samples, n_samples)))
    config = _make_inr_config()

    # Define progressive schedules to compare
    schedules = {
        "50→200": [(0, 50), (125, 100), (250, 150), (375, 200)],
        "50→500": [(0, 50), (100, 100), (200, 200), (300, 350), (400, 500)],
        "86→300": [(0, 86), (167, 150), (334, 200), (450, 300)],
    }

    results = {}
    plot_indices = set(_get_plot_indices(len(eval_indices)))
    per_sample_recons = {eval_indices[i]: {} for i in plot_indices}
    per_sample_gt = {}

    for sched_name, schedule in schedules.items():
        logger.info(f"\n--- Schedule: {sched_name} ---")
        results[sched_name] = []

        for i, test_idx in enumerate(eval_indices):
            sample = dataset[test_idx]
            s_gt_np = sample['s_gt_raw'].numpy().flatten()

            model = _make_model(config)
            result = optimize_svd_progressive(
                sample=sample,
                L_matrix=dataset.L_matrix,
                model=model,
                config=config,
                U=U, S=S, Vt=Vt,
                k_schedule=schedule,
                label=sched_name,
            )

            s_rec = np.asarray(result['s_phys']).flatten()
            metrics = calculate_metrics(s_rec, s_gt_np, grid_shape=(64, 64))
            results[sched_name].append(metrics)

            if i in plot_indices:
                per_sample_recons[test_idx][f"Prog {sched_name}"] = s_rec
                per_sample_gt[test_idx] = s_gt_np

            logger.info(
                f"  Sample {test_idx}: MAE={metrics['MAE']:.2f}, "
                f"CNR={metrics['CNR']:.3f}"
            )

    # Plot representative samples comparing all schedules
    for idx in per_sample_recons:
        if per_sample_recons[idx]:
            plot_sample_comparison(idx, per_sample_gt[idx], per_sample_recons[idx],
                                   dataset_name, "2d")

    _print_results_table("2d", results, dataset_name)
    plot_metrics_summary(results, dataset_name, "2d")
    return results


# =============================================================================
#  Sub-experiment 2e: INR + Soft Projection (Gaussian taper)
# =============================================================================

def run_2e_soft_projection(dataset, dataset_name, U, S, Vt,
                            n_eval_samples=12):
    """INR with soft SVD taper — sweep K_center and taper_width."""
    logger.info("=" * 60)
    logger.info("SUB-EXPERIMENT 2e: INR + Soft SVD Projection")
    logger.info("=" * 60)

    n_samples = len(dataset)
    eval_indices = list(range(min(n_eval_samples, n_samples)))
    config = _make_inr_config()

    # Sweep configurations: (K_center, taper_width)
    taper_configs = {
        "K=200,τ=50":  (200, 50),
        "K=200,τ=100": (200, 100),
        "K=300,τ=100": (300, 100),
        "K=300,τ=200": (300, 200),
        "K=500,τ=200": (500, 200),
    }

    results = {}
    plot_indices = set(_get_plot_indices(len(eval_indices)))
    per_sample_recons = {eval_indices[i]: {} for i in plot_indices}
    per_sample_gt = {}

    for name, (K_center, taper_width) in taper_configs.items():
        logger.info(f"\n--- {name} ---")
        results[name] = []

        for i, test_idx in enumerate(eval_indices):
            sample = dataset[test_idx]
            s_gt_np = sample['s_gt_raw'].numpy().flatten()

            model = _make_model(config)
            result = optimize_svd_soft(
                sample=sample,
                L_matrix=dataset.L_matrix,
                model=model,
                config=config,
                U=U, S=S, Vt=Vt,
                K_center=K_center,
                taper_width=taper_width,
                label=name,
            )

            s_rec = np.asarray(result['s_phys']).flatten()
            metrics = calculate_metrics(s_rec, s_gt_np, grid_shape=(64, 64))
            results[name].append(metrics)

            if i in plot_indices:
                per_sample_recons[test_idx][name] = s_rec
                per_sample_gt[test_idx] = s_gt_np

            logger.info(
                f"  Sample {test_idx}: MAE={metrics['MAE']:.2f}, "
                f"CNR={metrics['CNR']:.3f}"
            )

    # Plot representative samples comparing all taper configs
    for idx in per_sample_recons:
        if per_sample_recons[idx]:
            plot_sample_comparison(idx, per_sample_gt[idx], per_sample_recons[idx],
                                   dataset_name, "2e")

    _print_results_table("2e", results, dataset_name)
    plot_metrics_summary(results, dataset_name, "2e")
    return results


# =============================================================================
#  Utilities
# =============================================================================

def _print_results_table(sub_exp, results, dataset_name):
    logger.info("\n" + "=" * 70)
    logger.info(f"RESULTS -- Sub-experiment {sub_exp} ({dataset_name})")
    logger.info("=" * 70)
    logger.info(f"{'Method':<25} | {'CNR':>8} | {'SSIM':>8} | {'RMSE':>8} | {'MAE':>8}")
    logger.info("-" * 70)

    for method, metrics_list in results.items():
        cnr = np.mean([m['CNR'] for m in metrics_list])
        cnr_std = np.std([m['CNR'] for m in metrics_list])
        ssim_val = np.mean([m['SSIM'] for m in metrics_list])
        ssim_std = np.std([m['SSIM'] for m in metrics_list])
        rmse = np.mean([m['RMSE'] for m in metrics_list])
        rmse_std = np.std([m['RMSE'] for m in metrics_list])
        mae = np.mean([m['MAE'] for m in metrics_list])
        mae_std = np.std([m['MAE'] for m in metrics_list])
        logger.info(
            f"{method:<25} | {cnr:5.3f}+/-{cnr_std:.3f} | "
            f"{ssim_val:5.4f}+/-{ssim_std:.4f} | "
            f"{rmse:5.2f}+/-{rmse_std:.2f} | "
            f"{mae:5.2f}+/-{mae_std:.2f}"
        )


def save_results(results, sub_exp, dataset_name):
    out_dir = DATA_OUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    filepath = out_dir / f"sub_exp_{sub_exp}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(filepath, 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    logger.info(f"Results saved to {filepath}")


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 2: SVD-Constrained INR")
    parser.add_argument("--dataset", type=str, default="kwave_geom",
                        choices=["kwave_geom", "kwave_blob"])
    parser.add_argument("--sub_exp", type=str, default="2a",
                        choices=["2a", "2b", "2c", "2d", "2e", "all"])
    parser.add_argument("--n_eval", type=int, default=12,
                        help="Number of samples for INR sub-experiments (default: 12)")
    args = parser.parse_args()

    # Load dataset
    ds_cfg = load_dataset_config(args.dataset)
    logger.info(f"Loading dataset: {ds_cfg['name']} ({args.dataset})")
    dataset = load_dataset(ds_cfg)
    logger.info(f"Dataset loaded: {len(dataset)} samples, L shape: {dataset.L_matrix.shape}")

    # Precompute SVD (cached)
    U, S, Vt = compute_or_load_svd(dataset.L_matrix, args.dataset)

    # Run sub-experiments
    sub_exps = ["2a", "2b", "2c", "2d", "2e"] if args.sub_exp == "all" else [args.sub_exp]

    for se in sub_exps:
        if se == "2a":
            results = run_2a_tsvd(dataset, args.dataset, U, S, Vt)
        elif se == "2b":
            results = run_2b_lsqr_early_stop(dataset, args.dataset)
        elif se == "2c":
            results = run_2c_inr_hard_projection(
                dataset, args.dataset, U, S, Vt, n_eval_samples=args.n_eval
            )
        elif se == "2d":
            results = run_2d_progressive_k(
                dataset, args.dataset, U, S, Vt, n_eval_samples=args.n_eval
            )
        elif se == "2e":
            results = run_2e_soft_projection(
                dataset, args.dataset, U, S, Vt, n_eval_samples=args.n_eval
            )

        save_results(results, se, args.dataset)


if __name__ == "__main__":
    main()
