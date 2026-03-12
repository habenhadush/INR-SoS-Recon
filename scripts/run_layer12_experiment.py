"""Layer 1 & 2 Experiment: SVD-weighted loss + Diagonal BAE correction.

Runs on kwave_geom dataset. Compares:
  - Baseline: standard MSE loss (no correction)
  - Layer 1 (SVD): downweight ill-conditioned tail modes
  - Layer 2 (BAE): mean-subtract mismatch + inverse-variance reweighting
  - Layer 1+2 (SVD+BAE): combined

Uses leave-one-out: for each test sample, BAE stats are computed from
all OTHER samples. SVD weights are computed once (geometry-dependent).

Usage:
    source .venv/bin/activate
    cd scripts
    uv run python run_layer12_experiment.py [--n_test 5] [--steps 2000]
"""

import argparse
import sys
import time
import logging
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inr_sos.utils.data import USDataset
from inr_sos.utils.config import ExperimentConfig
from inr_sos.utils.mismatch import compute_svd_weights, compute_bae_stats
from inr_sos.training.engines import optimize_full_forward_operator
from inr_sos.evaluation.metrics import calculate_metrics, compute_model_error_floor
from inr_sos.io.paths import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def build_model(config):
    """Create INR model from config."""
    from inr_sos.models.mlp import ReluMLP, FourierMLP
    from inr_sos.models.siren import SirenMLP

    if config.model_type == "SirenMLP":
        return SirenMLP(
            in_features=config.in_features,
            hidden_features=config.hidden_features,
            hidden_layers=config.hidden_layers,
            omega_0=config.omega,
        )
    elif config.model_type == "FourierMLP":
        return FourierMLP(
            in_features=config.in_features,
            hidden_features=config.hidden_features,
            hidden_layers=config.hidden_layers,
            mapping_size=config.mapping_size,
            scale=config.scale,
        )
    else:
        return ReluMLP(
            in_features=config.in_features,
            hidden_features=config.hidden_features,
            hidden_layers=config.hidden_layers,
        )


def run_single_experiment(sample, L_matrix, config, label, bae_stats=None, svd_info=None):
    """Run one reconstruction and return metrics."""
    model = build_model(config)
    t0 = time.time()

    result = optimize_full_forward_operator(
        sample, L_matrix, model, label, config,
        use_wandb=False, bae_stats=bae_stats, svd_info=svd_info,
    )

    elapsed = time.time() - t0
    metrics = calculate_metrics(result['s_phys'], sample['s_gt_raw'])
    metrics['time_s'] = elapsed

    return metrics, result


def compute_loo_bae_stats(dataset, L_matrix, test_idx):
    """Compute BAE stats from all samples EXCEPT test_idx (leave-one-out)."""
    N = len(dataset)
    M = L_matrix.shape[0]

    epsilons = []
    masks = []

    for i in range(N):
        if i == test_idx:
            continue
        sample = dataset[i]
        d_meas = sample['d_meas']
        s_gt = sample['s_gt_raw']
        m = sample['mask']

        d_model = L_matrix @ s_gt
        eps = d_meas - d_model
        epsilons.append(eps)
        masks.append(m)

    epsilons = torch.stack(epsilons, dim=0)
    masks = torch.stack(masks, dim=0)

    mask_common = (masks.prod(dim=0) > 0.5).float()
    epsilons_masked = epsilons * mask_common.unsqueeze(0)

    eta = epsilons_masked.mean(dim=0)
    sigma2 = epsilons_masked.var(dim=0)
    sigma2 = sigma2 * mask_common + (1.0 - mask_common) * 1e10

    return {
        'eta': eta,
        'sigma2': sigma2,
        'n_samples': N - 1,
        'mask_common': mask_common,
    }


def main():
    parser = argparse.ArgumentParser(description="Layer 1+2 mismatch correction experiments")
    parser.add_argument("--n_test", type=int, default=5, help="Number of test samples")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps per run")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_type", type=str, default="SirenMLP", help="Model architecture")
    parser.add_argument("--svd_top_k", type=int, default=3800, help="SVD: keep top-k modes")
    parser.add_argument("--svd_tail_damping", type=float, default=0.01, help="SVD: tail weight")
    parser.add_argument("--output_dir", type=str, default="layer12_results", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # --- Load dataset ---
    data_file = f"{DATA_DIR}/DL-based-SoS/test_kWaveGeom_l2rec_l1rec_unifiedvar.mat"
    grid_file = f"{DATA_DIR}/DL-based-SoS/grid_parameters.mat"

    logger.info(f"Loading kwave_geom from {data_file}")
    dataset = USDataset(data_file, grid_file)
    L_matrix = dataset.L_matrix
    N_total = len(dataset)
    n_test = min(args.n_test, N_total)

    logger.info(f"Dataset: {N_total} samples, L-matrix: {L_matrix.shape}")
    logger.info(f"Testing on {n_test} samples")

    # --- Precompute SVD weights (once, geometry-dependent) ---
    # Use first sample's mask as representative (they're ~84.5% systematic)
    sample0 = dataset[0]
    logger.info("Precomputing SVD weights...")
    svd_config = ExperimentConfig(svd_top_k=args.svd_top_k, svd_tail_damping=args.svd_tail_damping)
    svd_info = compute_svd_weights(L_matrix, sample0['mask'], svd_config)
    logger.info(f"SVD weights computed: condition number={svd_info['S'][0]/svd_info['S'][-1]:.2e}")

    # --- Define experiment configs ---
    base_kwargs = dict(
        model_type=args.model_type,
        steps=args.steps,
        lr=args.lr,
        hidden_features=256,
        hidden_layers=3,
        omega=30.0,
        clamp_slowness=True,
        time_scale=1e6,
    )

    configs = {
        "baseline": ExperimentConfig(**base_kwargs, loss_weighting="none"),
        "svd_only": ExperimentConfig(**base_kwargs, loss_weighting="svd",
                                      svd_top_k=args.svd_top_k,
                                      svd_tail_damping=args.svd_tail_damping),
        "bae_only": ExperimentConfig(**base_kwargs, loss_weighting="bae"),
        "svd+bae":  ExperimentConfig(**base_kwargs, loss_weighting="svd+bae",
                                      svd_top_k=args.svd_top_k,
                                      svd_tail_damping=args.svd_tail_damping),
    }

    # --- Run experiments ---
    all_results = {name: [] for name in configs}

    for test_idx in range(n_test):
        sample = dataset[test_idx]
        logger.info(f"\n{'='*60}")
        logger.info(f"Sample {test_idx}/{n_test}")

        # Error floor for reference
        floor = compute_model_error_floor(sample, L_matrix)
        logger.info(f"Error floor: MSE={floor['floor_mse']:.4f}, MAE={floor['floor_mae']:.4f}")

        # LOO BAE stats for this sample
        logger.info("Computing LOO BAE statistics...")
        bae_stats = compute_loo_bae_stats(dataset, L_matrix, test_idx)
        logger.info(f"BAE stats from {bae_stats['n_samples']} training samples")

        for name, config in configs.items():
            logger.info(f"\n--- Running: {name} ---")

            # Determine which correction to pass
            _svd = svd_info if 'svd' in config.loss_weighting else None
            _bae = bae_stats if 'bae' in config.loss_weighting else None

            metrics, result = run_single_experiment(
                sample, L_matrix, config, f"{name}_s{test_idx}",
                bae_stats=_bae, svd_info=_svd,
            )

            metrics['sample_idx'] = test_idx
            all_results[name].append(metrics)

            logger.info(
                f"  {name}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, "
                f"SSIM={metrics['SSIM']:.4f}, CNR={metrics['CNR']:.2f}, "
                f"time={metrics['time_s']:.1f}s"
            )

    # --- Summarize results ---
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    summary = {}
    for name, results_list in all_results.items():
        maes = [r['MAE'] for r in results_list]
        rmses = [r['RMSE'] for r in results_list]
        ssims = [r['SSIM'] for r in results_list]

        summary[name] = {
            'MAE_mean': float(np.mean(maes)),
            'MAE_std': float(np.std(maes)),
            'RMSE_mean': float(np.mean(rmses)),
            'RMSE_std': float(np.std(rmses)),
            'SSIM_mean': float(np.mean(ssims)),
            'SSIM_std': float(np.std(ssims)),
            'per_sample': results_list,
        }

        logger.info(
            f"  {name:12s}: MAE={np.mean(maes):6.2f} +/- {np.std(maes):5.2f}, "
            f"RMSE={np.mean(rmses):6.2f} +/- {np.std(rmses):5.2f}, "
            f"SSIM={np.mean(ssims):.4f} +/- {np.std(ssims):.4f}"
        )

    logger.info(f"\nBaseline references: L1 MAE=7.0, L2 MAE=9.3")

    # --- Save results ---
    results_file = output_dir / f"layer12_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
