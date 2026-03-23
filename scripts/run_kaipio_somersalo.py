"""
Experiment 1: Kaipio-Somersalo Approximation Error Method

Addresses forward model mismatch by:
  1a) Template subtraction: d_corrected = d_meas - μ_ε
  1b) Template + covariance weighting (full KS)
  1c) Template + covariance + INR reconstruction
  1d) Sweep PCA components K_ε

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
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsqr
from scipy.linalg import svd as scipy_svd

# ── Project imports ──────────────────────────────────────────────────────
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

# ── Dataset registry (mirrors datasets.yaml) ────────────────────────────
DATASET_CONFIGS = {
    "kwave_geom": {
        "data_file": f"{DATA_DIR}/DL-based-SoS/test_kWaveGeom_l2rec_l1rec_unifiedvar.mat",
        "grid_file": f"{DATA_DIR}/DL-based-SoS/grid_parameters.mat",
        "matrix_file": None,
        "use_external_L": False,
        "n_samples": 32,
    },
    "kwave_blob": {
        "data_file": f"{DATA_DIR}/DL-based-SoS/test_kWaveBlob_final.mat",
        "grid_file": f"{DATA_DIR}/DL-based-SoS/grid_parameters.mat",
        "matrix_file": f"{DATA_DIR}/DL-based-SoS/A.mat",
        "use_external_L": True,
        "n_samples": 70,
    },
}

# ── Output directory ─────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "experiment1_kaipio_somersalo"


# =========================================================================
#  CORE: Mismatch Statistics Computation
# =========================================================================

def compute_mismatch_statistics(dataset, exclude_idx=None):
    """Compute mismatch template μ_ε and PCA covariance from paired data.

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

    # PCA of residuals (ε_i - μ_ε)
    residuals = epsilons - mu_epsilon[np.newaxis, :]  # (N, M)
    # Economy SVD: residuals is (N, M) with N << M, so we get at most N components
    # Use residuals @ residuals.T which is (N, N) — much smaller
    C_small = residuals @ residuals.T / (len(indices) - 1)  # (N, N)
    eigvals, eigvecs_small = np.linalg.eigh(C_small)

    # Sort descending
    idx_sort = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_sort]
    eigvecs_small = eigvecs_small[:, idx_sort]

    # Convert back to measurement-space eigenvectors
    # U_pca = residuals.T @ eigvecs_small, then normalize
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


# =========================================================================
#  Sub-experiment 1a: Template Subtraction + L1/L2
# =========================================================================

def run_1a_template_subtraction(dataset, dataset_name):
    """Subtract mean mismatch template, then solve with L1/L2."""
    logger.info("=" * 60)
    logger.info("SUB-EXPERIMENT 1a: Template Subtraction + L1/L2")
    logger.info("=" * 60)

    L_np = dataset.L_matrix.numpy()
    n_samples = len(dataset)
    results = {"raw_L2": [], "corrected_L2": [], "raw_L1_proxy": [], "corrected_L1_proxy": []}

    for test_idx in range(n_samples):
        sample = dataset[test_idx]
        s_gt = sample['s_gt_raw'].numpy().flatten()
        d_meas = sample['d_meas'].numpy().flatten()
        mask = sample['mask'].numpy().flatten()
        valid = mask > 0.5

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

        # Corrected L2 solve
        s_corr, *_ = lsqr(L_valid, d_corr_valid, damp=1e-3, iter_lim=500)
        metrics_corr = calculate_metrics(s_corr, s_gt, grid_shape=(64, 64))
        results["corrected_L2"].append(metrics_corr)

        # Raw L1-proxy solve (more iterations, lighter damping)
        s_raw_l1, *_ = lsqr(L_valid, d_raw_valid, damp=5e-4, iter_lim=1000)
        metrics_raw_l1 = calculate_metrics(s_raw_l1, s_gt, grid_shape=(64, 64))
        results["raw_L1_proxy"].append(metrics_raw_l1)

        # Corrected L1-proxy solve
        s_corr_l1, *_ = lsqr(L_valid, d_corr_valid, damp=5e-4, iter_lim=1000)
        metrics_corr_l1 = calculate_metrics(s_corr_l1, s_gt, grid_shape=(64, 64))
        results["corrected_L1_proxy"].append(metrics_corr_l1)

        if test_idx % 5 == 0:
            logger.info(
                f"  Sample {test_idx}: raw MAE={metrics_raw['MAE']:.2f}, "
                f"corrected MAE={metrics_corr['MAE']:.2f}, "
                f"raw CNR={metrics_raw['CNR']:.3f}, "
                f"corrected CNR={metrics_corr['CNR']:.3f}"
            )

    # Aggregate
    _print_results_table("1a", results, dataset_name)
    return results


# =========================================================================
#  Sub-experiment 1b: Template + Covariance Weighting (Full KS)
# =========================================================================

def run_1b_full_kaipio_somersalo(dataset, dataset_name, n_pca=15):
    """Full Kaipio-Somersalo: mean correction + covariance-weighted solve."""
    logger.info("=" * 60)
    logger.info(f"SUB-EXPERIMENT 1b: Full Kaipio-Somersalo (K_pca={n_pca})")
    logger.info("=" * 60)

    L_np = dataset.L_matrix.numpy()
    n_samples = len(dataset)
    results = {"ks_weighted": []}

    for test_idx in range(n_samples):
        sample = dataset[test_idx]
        s_gt = sample['s_gt_raw'].numpy().flatten()
        d_meas = sample['d_meas'].numpy().flatten()
        mask = sample['mask'].numpy().flatten()
        valid = mask > 0.5

        # LOO mismatch statistics
        mu_epsilon, U_pca, S_pca, _ = compute_mismatch_statistics(
            dataset, exclude_idx=test_idx
        )

        # Mean-corrected data
        d_corrected = (d_meas - mu_epsilon) * mask

        # Covariance-weighted solve via Woodbury identity
        # W = (Γ_ε + σ²I)⁻¹ where Γ_ε = U_K Λ_K U_K^T
        # Woodbury: W = (1/σ²)I - (1/σ⁴) U_K (Λ_K⁻¹ + (1/σ²)I)⁻¹ U_K^T
        #
        # Instead of forming W explicitly (M×M), we transform the problem:
        #   min_s ||W^{1/2}(d_corr - L·s)||² + λ||s||²
        # by whitening: d_w = W^{1/2} d_corr, L_w = W^{1/2} L
        #
        # For efficiency with the Woodbury structure, we work in the
        # reduced space. The key insight: W^{1/2} can be applied via
        # the PCA components without forming the full matrix.

        K = min(n_pca, len(S_pca))
        U_K = U_pca[:, :K]    # (M, K)
        lam_k = S_pca[:K]     # (K,)

        # Noise floor estimate (from small eigenvalues or residual)
        sigma2 = np.median(S_pca[K:]) if len(S_pca) > K else 1e-18

        # Apply W^{1/2} via: W^{1/2} x = (1/σ)x - (1/σ²) U_K D U_K^T x
        # where D = diag(σ / sqrt(σ² + λ_k) - 1) (derived from Woodbury sqrt)
        # Simpler: use the preconditioned LSQR approach
        #
        # Transform: for each direction u_k, scale the component by
        # sqrt(σ² / (σ² + λ_k)) — this downweights directions with high
        # model error variance.

        L_valid = L_np[valid]
        d_corr_valid = d_corrected[valid]
        U_K_valid = U_K[valid, :]

        # Scaling factors per PCA direction
        scale_factors = np.sqrt(sigma2 / (sigma2 + lam_k))  # (K,)

        # Transform: x_transformed = x - U_K (I - diag(scale)) U_K^T x
        # This downweights the high-variance directions
        def apply_whitening(x):
            """Apply approximate W^{1/2} to a vector x (valid rays only)."""
            proj = U_K_valid.T @ x  # (K,)
            correction = U_K_valid @ ((1.0 - scale_factors) * proj)  # (M_valid,)
            return x - correction

        d_whitened = apply_whitening(d_corr_valid)

        # Whiten each column of L (expensive but one-time per sample)
        L_whitened = L_valid.copy()
        proj_L = U_K_valid.T @ L_valid  # (K, 4096)
        L_whitened = L_valid - U_K_valid @ ((1.0 - scale_factors[:, np.newaxis]) * proj_L)

        # Solve whitened system
        s_ks, *_ = lsqr(L_whitened, d_whitened, damp=1e-3, iter_lim=500)
        metrics = calculate_metrics(s_ks, s_gt, grid_shape=(64, 64))
        results["ks_weighted"].append(metrics)

        if test_idx % 5 == 0:
            logger.info(
                f"  Sample {test_idx}: MAE={metrics['MAE']:.2f}, "
                f"CNR={metrics['CNR']:.3f}, SSIM={metrics['SSIM']:.4f}"
            )

    _print_results_table("1b", results, dataset_name)
    return results


# =========================================================================
#  Sub-experiment 1c: KS Correction + INR Reconstruction
# =========================================================================

def run_1c_ks_plus_inr(dataset, dataset_name, n_pca=15, n_eval_samples=12):
    """Kaipio-Somersalo mean correction fed into INR training."""
    logger.info("=" * 60)
    logger.info("SUB-EXPERIMENT 1c: KS Correction + INR")
    logger.info("=" * 60)

    n_samples = len(dataset)
    eval_indices = list(range(min(n_eval_samples, n_samples)))
    results = {"ks_inr": [], "raw_inr": []}

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

    for test_idx in eval_indices:
        sample = dataset[test_idx]
        s_gt = sample['s_gt_raw']

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
        metrics_raw = calculate_metrics(result_raw['s_phys'], s_gt, grid_shape=(64, 64))
        results["raw_inr"].append(metrics_raw)

        # --- KS-corrected INR ---
        # Compute LOO mismatch template
        mu_epsilon, _, _, _ = compute_mismatch_statistics(dataset, exclude_idx=test_idx)

        # Create corrected sample (modify d_meas)
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
        metrics_ks = calculate_metrics(result_ks['s_phys'], s_gt, grid_shape=(64, 64))
        results["ks_inr"].append(metrics_ks)

        logger.info(
            f"  Sample {test_idx}: "
            f"raw MAE={metrics_raw['MAE']:.2f} CNR={metrics_raw['CNR']:.3f} | "
            f"KS  MAE={metrics_ks['MAE']:.2f} CNR={metrics_ks['CNR']:.3f}"
        )

    _print_results_table("1c", results, dataset_name)
    return results


# =========================================================================
#  Sub-experiment 1d: Sweep PCA Components
# =========================================================================

def run_1d_sweep_pca(dataset, dataset_name, k_values=None):
    """Sweep number of PCA components K_ε in the KS covariance."""
    if k_values is None:
        k_values = [5, 10, 15, 20, 31]
    logger.info("=" * 60)
    logger.info(f"SUB-EXPERIMENT 1d: PCA Component Sweep K_ε ∈ {k_values}")
    logger.info("=" * 60)

    results = {}
    for k in k_values:
        logger.info(f"\n--- K_ε = {k} ---")
        k_results = run_1b_full_kaipio_somersalo(dataset, dataset_name, n_pca=k)
        results[f"K={k}"] = k_results["ks_weighted"]

    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info(f"PCA SWEEP SUMMARY ({dataset_name})")
    logger.info("=" * 70)
    logger.info(f"{'K_ε':>5} | {'CNR':>8} | {'SSIM':>8} | {'RMSE':>8} | {'MAE':>8}")
    logger.info("-" * 50)
    for k_label, metrics_list in results.items():
        cnr = np.mean([m['CNR'] for m in metrics_list])
        ssim_val = np.mean([m['SSIM'] for m in metrics_list])
        rmse = np.mean([m['RMSE'] for m in metrics_list])
        mae = np.mean([m['MAE'] for m in metrics_list])
        logger.info(f"{k_label:>5} | {cnr:8.3f} | {ssim_val:8.4f} | {rmse:8.2f} | {mae:8.2f}")

    return results


# =========================================================================
#  Utilities
# =========================================================================

def _print_results_table(sub_exp, results, dataset_name):
    """Print aggregated results for a sub-experiment."""
    logger.info("\n" + "=" * 70)
    logger.info(f"RESULTS — Sub-experiment {sub_exp} ({dataset_name})")
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
            f"{method:<25} | {cnr:5.3f}±{cnr_std:.3f} | "
            f"{ssim_val:5.4f}±{ssim_std:.4f} | "
            f"{rmse:5.2f}±{rmse_std:.2f} | "
            f"{mae:5.2f}±{mae_std:.2f}"
        )


def save_results(results, sub_exp, dataset_name):
    """Save results to JSON for later comparison."""
    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
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


# =========================================================================
#  Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Kaipio-Somersalo")
    parser.add_argument("--dataset", type=str, default="kwave_geom",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--sub_exp", type=str, default="1a",
                        choices=["1a", "1b", "1c", "1d", "all"])
    parser.add_argument("--n_pca", type=int, default=15,
                        help="Number of PCA components for 1b (default: 15)")
    parser.add_argument("--n_eval", type=int, default=12,
                        help="Number of samples to evaluate for 1c INR (default: 12)")
    args = parser.parse_args()

    # Load dataset
    ds_cfg = DATASET_CONFIGS[args.dataset]
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = USDataset(
        data_path=ds_cfg["data_file"],
        grid_path=ds_cfg["grid_file"],
        matrix_path=ds_cfg["matrix_file"],
        use_external_L_matrix=ds_cfg["use_external_L"],
    )
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
