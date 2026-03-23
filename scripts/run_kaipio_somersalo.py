"""
Experiment 1: Kaipio-Somersalo Approximation Error Method

Addresses forward model mismatch by:
  1a) Template subtraction: d_corrected = d_meas - mu_epsilon
  1b) Template + covariance weighting (full KS)
  1c) Template + covariance + INR reconstruction
  1d) Sweep PCA components K_epsilon

Usage:
    source .venv/bin/activate
    uv run python scripts/run_kaipio_somersalo.py --dataset kwave_geom --sub_exp 1a
    uv run python scripts/run_kaipio_somersalo.py --dataset kwave_geom --sub_exp all
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
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse.linalg import lsqr

# -- Project imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inr_sos.utils.data import USDataset
from inr_sos.evaluation.metrics import calculate_metrics, calculate_cnr
from inr_sos.io.paths import DATA_DIR
from inr_sos.utils.config import ExperimentConfig
from inr_sos.models.mlp import FourierMLP, GeluMLP
from inr_sos.models.siren import SirenMLP
from inr_sos.training.engines import optimize_full_forward_operator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# -- Paths ---------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_OUT_DIR = SCRIPTS_DIR / "data" / "experiment1_kaipio_somersalo"
PLOTS_DIR = SCRIPTS_DIR / "plots" / "experiment1_kaipio_somersalo"

# -- SoS constants (matching plot_reconstruction.py) ---------------------------
SOS_BG = 1540.0
SOS_MIN = 1380.0
SOS_MAX = 1620.0


# =============================================================================
#  Dataset loading from datasets.yaml
# =============================================================================

def load_dataset_config(key: str) -> dict:
    """Load dataset config from scripts/datasets.yaml."""
    cfg_path = SCRIPTS_DIR / "datasets.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ds = cfg["datasets"][key]
    ds["key"] = key
    ds["data_path"] = DATA_DIR + ds["data_file"]
    return ds


def load_dataset(ds_cfg: dict) -> USDataset:
    """Instantiate USDataset from a datasets.yaml config dict.

    Follows the same pattern as run_reconstruction.py.
    """
    data_file = ds_cfg["data_path"]
    grid_path = DATA_DIR + ds_cfg.get("grid_path",
                                       "/DL-based-SoS/forward_model_lr/grid_parameters.mat")

    ds_kwargs = {}
    if not ds_cfg.get("has_A_matrix", True):
        matrix_file = ds_cfg.get("matrix_file")
        if matrix_file:
            ds_kwargs["matrix_path"] = DATA_DIR + matrix_file
            ds_kwargs["use_external_L_matrix"] = True
            logger.info(f"External L-matrix: {ds_kwargs['matrix_path']}")
    if ds_cfg.get("h5_keys"):
        ds_kwargs["h5_keys"] = ds_cfg["h5_keys"]

    return USDataset(data_file, grid_path, **ds_kwargs)


# =============================================================================
#  Plotting utilities
# =============================================================================

def _slowness_to_sos(s, grid_shape=(64, 64)):
    """Convert slowness (s/m) to SoS (m/s) image, clamped to physical range."""
    s_flat = np.asarray(s, dtype=np.float32).flatten()
    v = np.clip(1.0 / (s_flat + 1e-8), SOS_MIN, SOS_MAX)
    return v.reshape(grid_shape)


def _get_plot_indices(n_samples, n_plots=6):
    """Select evenly-spaced sample indices for plotting."""
    if n_samples <= n_plots:
        return list(range(n_samples))
    step = n_samples // n_plots
    return list(range(0, n_samples, step))[:n_plots]


def plot_sample_comparison(sample_idx, s_gt, methods, dataset_name, sub_exp,
                           grid_shape=(64, 64)):
    """Plot GT vs reconstruction methods for one sample.

    Args:
        sample_idx: sample index (for filename)
        s_gt:       ground truth slowness (4096,)
        methods:    dict of {method_name: reconstructed_slowness (4096,)}
        dataset_name: e.g. 'kwave_geom'
        sub_exp:    e.g. '1a'
    """
    n_methods = len(methods)
    n_cols = n_methods + 2  # GT + methods + error map
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    fig.patch.set_facecolor("white")

    v_gt = _slowness_to_sos(s_gt, grid_shape)
    v_min = max(SOS_MIN, float(v_gt.min()) - 10)
    v_max = min(SOS_MAX, float(v_gt.max()) + 10)
    bg_sos = float(np.median(v_gt))
    norm = mcolors.TwoSlopeNorm(vmin=v_min, vcenter=bg_sos, vmax=v_max)

    # GT panel
    im = axes[0].imshow(v_gt, cmap="RdBu_r", norm=norm,
                         interpolation="nearest", origin="upper")
    axes[0].set_title("Ground Truth", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Method panels
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

    # Error panel (best method)
    im = axes[-1].imshow(best_err_map, cmap="hot", vmin=0, vmax=50,
                          interpolation="nearest", origin="upper")
    axes[-1].set_title(f"Best Abs Error\nMAE={best_mae:.1f} m/s", fontsize=10)
    axes[-1].axis("off")
    cb = plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    cb.set_label("|GT - Rec| (m/s)", fontsize=9)

    fig.suptitle(f"Exp1{sub_exp} | {dataset_name} | Sample {sample_idx}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_dir = PLOTS_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"sub_{sub_exp}_sample_{sample_idx:03d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Plot saved: {save_path}")


def plot_metrics_summary(all_results, dataset_name, sub_exp):
    """Bar chart summary of metrics across methods."""
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

    fig.suptitle(f"Exp1{sub_exp} Metrics Summary | {dataset_name}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_dir = PLOTS_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"sub_{sub_exp}_metrics_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Summary plot saved: {save_path}")


def plot_pca_sweep(sweep_results, dataset_name):
    """Line plot of metrics vs PCA component count for sub-experiment 1d."""
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
        ax.set_xlabel("K_epsilon (PCA components)", fontsize=10)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Exp1d PCA Sweep | {dataset_name}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_dir = PLOTS_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "sub_1d_pca_sweep.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  PCA sweep plot saved: {save_path}")


# =============================================================================
#  CORE: Mismatch Statistics Computation
# =============================================================================

def compute_mismatch_statistics(dataset, exclude_idx=None):
    """Compute mismatch template mu_epsilon and PCA covariance from paired data.

    Args:
        dataset:     USDataset with ground truth and L-matrix.
        exclude_idx: Index to exclude (for leave-one-out). None = use all.

    Returns:
        mu_epsilon:  (M,) mean mismatch vector
        U_pca:       (M, K) PCA eigenvectors of residual covariance
        S_pca:       (K,) PCA eigenvalues
        epsilons:    list of per-sample mismatch vectors
    """
    L_np = dataset.L_matrix.numpy()
    n_samples = len(dataset)
    indices = [i for i in range(n_samples) if i != exclude_idx]

    epsilons = []
    for i in indices:
        sample = dataset[i]
        s_gt = sample['s_gt_raw'].numpy().flatten()
        d_meas = sample['d_meas'].numpy().flatten()
        mask = sample['mask'].numpy().flatten()

        d_pred = L_np @ s_gt
        epsilon = (d_meas - d_pred) * mask  # zero invalid rays
        epsilons.append(epsilon)

    epsilons = np.stack(epsilons, axis=0)  # (N, M)
    mu_epsilon = np.mean(epsilons, axis=0)  # (M,)

    # PCA of residuals (epsilon_i - mu_epsilon)
    residuals = epsilons - mu_epsilon[np.newaxis, :]  # (N, M)
    # Economy SVD: residuals is (N, M) with N << M
    C_small = residuals @ residuals.T / (len(indices) - 1)  # (N, N)
    eigvals, eigvecs_small = np.linalg.eigh(C_small)

    # Sort descending
    idx_sort = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_sort]
    eigvecs_small = eigvecs_small[:, idx_sort]

    # Convert back to measurement-space eigenvectors
    U_pca = residuals.T @ eigvecs_small  # (M, N)
    norms = np.linalg.norm(U_pca, axis=0, keepdims=True) + 1e-15
    U_pca = U_pca / norms

    # Keep only positive eigenvalues
    pos_mask = eigvals > 1e-15
    U_pca = U_pca[:, pos_mask]
    S_pca = eigvals[pos_mask]

    logger.info(
        f"Mismatch stats: {len(indices)} samples, "
        f"{pos_mask.sum()} PCA components, "
        f"top-5 eigenvalues: {S_pca[:5]}"
    )

    return mu_epsilon, U_pca, S_pca, epsilons


# =============================================================================
#  Sub-experiment 1a: Template Subtraction + L1/L2
# =============================================================================

def run_1a_template_subtraction(dataset, dataset_name):
    """Subtract mean mismatch template, then solve with L1/L2."""
    logger.info("=" * 60)
    logger.info("SUB-EXPERIMENT 1a: Template Subtraction + L1/L2")
    logger.info("=" * 60)

    L_np = dataset.L_matrix.numpy()
    n_samples = len(dataset)
    results = {
        "raw_L2": [], "corrected_L2": [],
        "raw_L1_proxy": [], "corrected_L1_proxy": [],
    }
    # Store reconstructions for plotting
    reconstructions = {k: [] for k in results}
    ground_truths = []
    plot_indices = set(_get_plot_indices(n_samples))

    for test_idx in range(n_samples):
        sample = dataset[test_idx]
        s_gt = sample['s_gt_raw'].numpy().flatten()
        d_meas = sample['d_meas'].numpy().flatten()
        mask = sample['mask'].numpy().flatten()
        valid = mask > 0.5

        ground_truths.append(s_gt)

        # LOO: compute template from all OTHER samples
        mu_epsilon, _, _, _ = compute_mismatch_statistics(dataset, exclude_idx=test_idx)

        # Raw (no correction)
        d_raw = d_meas.copy()
        d_raw[~valid] = 0.0

        # Corrected
        d_corrected = (d_meas - mu_epsilon) * mask
        d_corrected[~valid] = 0.0

        # Solve with LSQR (Tikhonov-like iterative solver)
        L_valid = L_np[valid]
        d_raw_valid = d_raw[valid]
        d_corr_valid = d_corrected[valid]

        # Raw L2 solve
        s_raw, *_ = lsqr(L_valid, d_raw_valid, damp=1e-3, iter_lim=500)
        metrics_raw = calculate_metrics(s_raw, s_gt, grid_shape=(64, 64))
        results["raw_L2"].append(metrics_raw)
        reconstructions["raw_L2"].append(s_raw)

        # Corrected L2 solve
        s_corr, *_ = lsqr(L_valid, d_corr_valid, damp=1e-3, iter_lim=500)
        metrics_corr = calculate_metrics(s_corr, s_gt, grid_shape=(64, 64))
        results["corrected_L2"].append(metrics_corr)
        reconstructions["corrected_L2"].append(s_corr)

        # Raw L1-proxy solve (more iterations, lighter damping)
        s_raw_l1, *_ = lsqr(L_valid, d_raw_valid, damp=5e-4, iter_lim=1000)
        metrics_raw_l1 = calculate_metrics(s_raw_l1, s_gt, grid_shape=(64, 64))
        results["raw_L1_proxy"].append(metrics_raw_l1)
        reconstructions["raw_L1_proxy"].append(s_raw_l1)

        # Corrected L1-proxy solve
        s_corr_l1, *_ = lsqr(L_valid, d_corr_valid, damp=5e-4, iter_lim=1000)
        metrics_corr_l1 = calculate_metrics(s_corr_l1, s_gt, grid_shape=(64, 64))
        results["corrected_L1_proxy"].append(metrics_corr_l1)
        reconstructions["corrected_L1_proxy"].append(s_corr_l1)

        if test_idx % 5 == 0:
            logger.info(
                f"  Sample {test_idx}: raw MAE={metrics_raw['MAE']:.2f}, "
                f"corrected MAE={metrics_corr['MAE']:.2f}, "
                f"raw CNR={metrics_raw['CNR']:.3f}, "
                f"corrected CNR={metrics_corr['CNR']:.3f}"
            )

        # Plot representative samples
        if test_idx in plot_indices:
            plot_sample_comparison(
                test_idx, s_gt,
                {"Raw L2": s_raw, "Corrected L2": s_corr,
                 "Raw L1": s_raw_l1, "Corrected L1": s_corr_l1},
                dataset_name, "1a",
            )

    # Summary table and plot
    _print_results_table("1a", results, dataset_name)
    plot_metrics_summary(results, dataset_name, "1a")

    # Save reconstructions
    save_reconstructions(reconstructions, ground_truths, "1a", dataset_name)

    return results


# =============================================================================
#  Sub-experiment 1b: Template + Covariance Weighting (Full KS)
# =============================================================================

def run_1b_full_kaipio_somersalo(dataset, dataset_name, n_pca=15):
    """Full Kaipio-Somersalo: mean correction + covariance-weighted solve."""
    logger.info("=" * 60)
    logger.info(f"SUB-EXPERIMENT 1b: Full Kaipio-Somersalo (K_pca={n_pca})")
    logger.info("=" * 60)

    L_np = dataset.L_matrix.numpy()
    n_samples = len(dataset)
    results = {"ks_weighted": []}
    reconstructions = {"ks_weighted": []}
    ground_truths = []
    plot_indices = set(_get_plot_indices(n_samples))

    for test_idx in range(n_samples):
        sample = dataset[test_idx]
        s_gt = sample['s_gt_raw'].numpy().flatten()
        d_meas = sample['d_meas'].numpy().flatten()
        mask = sample['mask'].numpy().flatten()
        valid = mask > 0.5

        ground_truths.append(s_gt)

        # LOO mismatch statistics
        mu_epsilon, U_pca, S_pca, _ = compute_mismatch_statistics(
            dataset, exclude_idx=test_idx
        )

        # Mean-corrected data
        d_corrected = (d_meas - mu_epsilon) * mask

        # Covariance-weighted solve via Woodbury identity
        K = min(n_pca, len(S_pca))
        U_K = U_pca[:, :K]    # (M, K)
        lam_k = S_pca[:K]     # (K,)

        # Noise floor estimate
        sigma2 = np.median(S_pca[K:]) if len(S_pca) > K else 1e-18

        L_valid = L_np[valid]
        d_corr_valid = d_corrected[valid]
        U_K_valid = U_K[valid, :]

        # Scaling factors per PCA direction
        scale_factors = np.sqrt(sigma2 / (sigma2 + lam_k))  # (K,)

        # Whiten data and L-matrix
        def apply_whitening(x):
            proj = U_K_valid.T @ x
            correction = U_K_valid @ ((1.0 - scale_factors) * proj)
            return x - correction

        d_whitened = apply_whitening(d_corr_valid)

        proj_L = U_K_valid.T @ L_valid  # (K, 4096)
        L_whitened = L_valid - U_K_valid @ ((1.0 - scale_factors[:, np.newaxis]) * proj_L)

        # Solve whitened system
        s_ks, *_ = lsqr(L_whitened, d_whitened, damp=1e-3, iter_lim=500)
        metrics = calculate_metrics(s_ks, s_gt, grid_shape=(64, 64))
        results["ks_weighted"].append(metrics)
        reconstructions["ks_weighted"].append(s_ks)

        if test_idx % 5 == 0:
            logger.info(
                f"  Sample {test_idx}: MAE={metrics['MAE']:.2f}, "
                f"CNR={metrics['CNR']:.3f}, SSIM={metrics['SSIM']:.4f}"
            )

        if test_idx in plot_indices:
            plot_sample_comparison(
                test_idx, s_gt,
                {"KS Weighted": s_ks},
                dataset_name, "1b",
            )

    _print_results_table("1b", results, dataset_name)
    plot_metrics_summary(results, dataset_name, "1b")
    save_reconstructions(reconstructions, ground_truths, "1b", dataset_name)
    return results


# =============================================================================
#  Sub-experiment 1c: KS Correction + INR Reconstruction
# =============================================================================

def run_1c_ks_plus_inr(dataset, dataset_name, n_pca=15, n_eval_samples=12):
    """Kaipio-Somersalo mean correction fed into INR training."""
    logger.info("=" * 60)
    logger.info("SUB-EXPERIMENT 1c: KS Correction + INR")
    logger.info("=" * 60)

    n_samples = len(dataset)
    eval_indices = list(range(min(n_eval_samples, n_samples)))
    results = {"ks_inr": [], "raw_inr": []}
    reconstructions = {"ks_inr": [], "raw_inr": []}
    ground_truths = []
    plot_indices = set(_get_plot_indices(len(eval_indices)))

    config = ExperimentConfig(
        project_name="Exp1-KS-INR",
        experiment_group="kaipio_somersalo",
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

    for i, test_idx in enumerate(eval_indices):
        sample = dataset[test_idx]
        s_gt = sample['s_gt_raw']
        s_gt_np = s_gt.numpy().flatten() if hasattr(s_gt, 'numpy') else np.asarray(s_gt).flatten()
        ground_truths.append(s_gt_np)

        # --- Raw INR (baseline) ---
        model_raw = FourierMLP(
            in_features=config.in_features,
            hidden_features=config.hidden_features,
            hidden_layers=config.hidden_layers,
            mapping_size=config.mapping_size,
            scale=config.scale,
        )
        cfg_raw = copy.deepcopy(config)
        cfg_raw.experiment_group = "raw_inr"
        result_raw = optimize_full_forward_operator(
            sample=sample,
            L_matrix=dataset.L_matrix,
            model=model_raw,
            label="FourierMLP",
            config=cfg_raw,
            use_wandb=False,
        )
        s_raw_phys = result_raw['s_phys']
        if hasattr(s_raw_phys, 'detach'):
            s_raw_phys = s_raw_phys.detach().cpu().numpy()
        s_raw_phys = np.asarray(s_raw_phys).flatten()

        metrics_raw = calculate_metrics(s_raw_phys, s_gt_np, grid_shape=(64, 64))
        results["raw_inr"].append(metrics_raw)
        reconstructions["raw_inr"].append(s_raw_phys)

        # --- KS-corrected INR ---
        mu_epsilon, _, _, _ = compute_mismatch_statistics(dataset, exclude_idx=test_idx)

        sample_ks = copy.deepcopy(sample)
        mu_eps_tensor = torch.tensor(mu_epsilon, dtype=torch.float32).unsqueeze(1)
        sample_ks['d_meas'] = sample['d_meas'] - mu_eps_tensor * sample['mask']

        model_ks = FourierMLP(
            in_features=config.in_features,
            hidden_features=config.hidden_features,
            hidden_layers=config.hidden_layers,
            mapping_size=config.mapping_size,
            scale=config.scale,
        )
        cfg_ks = copy.deepcopy(config)
        cfg_ks.experiment_group = "ks_inr"
        result_ks = optimize_full_forward_operator(
            sample=sample_ks,
            L_matrix=dataset.L_matrix,
            model=model_ks,
            label="FourierMLP",
            config=cfg_ks,
            use_wandb=False,
        )
        s_ks_phys = result_ks['s_phys']
        if hasattr(s_ks_phys, 'detach'):
            s_ks_phys = s_ks_phys.detach().cpu().numpy()
        s_ks_phys = np.asarray(s_ks_phys).flatten()

        metrics_ks = calculate_metrics(s_ks_phys, s_gt_np, grid_shape=(64, 64))
        results["ks_inr"].append(metrics_ks)
        reconstructions["ks_inr"].append(s_ks_phys)

        logger.info(
            f"  Sample {test_idx}: "
            f"raw MAE={metrics_raw['MAE']:.2f} CNR={metrics_raw['CNR']:.3f} | "
            f"KS  MAE={metrics_ks['MAE']:.2f} CNR={metrics_ks['CNR']:.3f}"
        )

        if i in plot_indices:
            plot_sample_comparison(
                test_idx, s_gt_np,
                {"Raw INR": s_raw_phys, "KS INR": s_ks_phys},
                dataset_name, "1c",
            )

    _print_results_table("1c", results, dataset_name)
    plot_metrics_summary(results, dataset_name, "1c")
    save_reconstructions(reconstructions, ground_truths, "1c", dataset_name)
    return results


# =============================================================================
#  Sub-experiment 1d: Sweep PCA Components
# =============================================================================

def run_1d_sweep_pca(dataset, dataset_name, k_values=None):
    """Sweep number of PCA components K_epsilon in the KS covariance."""
    if k_values is None:
        k_values = [5, 10, 15, 20, 31]
    logger.info("=" * 60)
    logger.info(f"SUB-EXPERIMENT 1d: PCA Component Sweep K_epsilon in {k_values}")
    logger.info("=" * 60)

    results = {}
    for k in k_values:
        logger.info(f"\n--- K_epsilon = {k} ---")
        k_results = run_1b_full_kaipio_somersalo(dataset, dataset_name, n_pca=k)
        results[f"K={k}"] = k_results["ks_weighted"]

    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info(f"PCA SWEEP SUMMARY ({dataset_name})")
    logger.info("=" * 70)
    logger.info(f"{'K_eps':>5} | {'CNR':>8} | {'SSIM':>8} | {'RMSE':>8} | {'MAE':>8}")
    logger.info("-" * 50)
    for k_label, metrics_list in results.items():
        cnr = np.mean([m['CNR'] for m in metrics_list])
        ssim_val = np.mean([m['SSIM'] for m in metrics_list])
        rmse = np.mean([m['RMSE'] for m in metrics_list])
        mae = np.mean([m['MAE'] for m in metrics_list])
        logger.info(f"{k_label:>5} | {cnr:8.3f} | {ssim_val:8.4f} | {rmse:8.2f} | {mae:8.2f}")

    # PCA sweep plot
    plot_pca_sweep(results, dataset_name)
    plot_metrics_summary(results, dataset_name, "1d")

    return results


# =============================================================================
#  Utilities
# =============================================================================

def _print_results_table(sub_exp, results, dataset_name):
    """Print aggregated results for a sub-experiment."""
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
    """Save metrics results to JSON."""
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


def save_reconstructions(reconstructions, ground_truths, sub_exp, dataset_name):
    """Save reconstructed slowness fields to .npz for later analysis."""
    out_dir = DATA_OUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_dict = {"ground_truths": np.array(ground_truths)}
    for method, recons in reconstructions.items():
        save_dict[method] = np.array(recons)

    filepath = out_dir / f"sub_exp_{sub_exp}_reconstructions.npz"
    np.savez_compressed(filepath, **save_dict)
    logger.info(f"Reconstructions saved to {filepath}")


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Kaipio-Somersalo")
    parser.add_argument("--dataset", type=str, default="kwave_geom",
                        choices=["kwave_geom", "kwave_blob"])
    parser.add_argument("--sub_exp", type=str, default="1a",
                        choices=["1a", "1b", "1c", "1d", "all"])
    parser.add_argument("--n_pca", type=int, default=15,
                        help="Number of PCA components for 1b (default: 15)")
    parser.add_argument("--n_eval", type=int, default=12,
                        help="Number of samples to evaluate for 1c INR (default: 12)")
    args = parser.parse_args()

    # Load dataset from datasets.yaml
    ds_cfg = load_dataset_config(args.dataset)
    logger.info(f"Loading dataset: {ds_cfg['name']} ({args.dataset})")
    dataset = load_dataset(ds_cfg)
    logger.info(f"Dataset loaded: {len(dataset)} samples, L shape: {dataset.L_matrix.shape}")

    # Run sub-experiments
    sub_exps = ["1a", "1b", "1c", "1d"] if args.sub_exp == "all" else [args.sub_exp]

    for se in sub_exps:
        if se == "1a":
            results = run_1a_template_subtraction(dataset, args.dataset)
        elif se == "1b":
            results = run_1b_full_kaipio_somersalo(dataset, args.dataset, n_pca=args.n_pca)
        elif se == "1c":
            results = run_1c_ks_plus_inr(dataset, args.dataset, n_pca=args.n_pca,
                                         n_eval_samples=args.n_eval)
        elif se == "1d":
            results = run_1d_sweep_pca(dataset, args.dataset)

        save_results(results, se, args.dataset)


if __name__ == "__main__":
    main()
