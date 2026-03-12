"""Layer 3 Experiment: Joint reconstruction + residual correction.

Based on Gilton et al. (ICLR 2025) — untrained residual INR absorbs the
forward-model mismatch ε while the reconstruction INR recovers s_true.

Compares:
  - Baseline: standard full-matrix reconstruction
  - Layer 3:  joint f_θ + g_φ with τ-penalty sweep

Reuses existing plotting tools (plot_method_comparison).

Usage:
    source .venv/bin/activate && cd scripts
    uv run python run_layer3_experiment.py [--samples 0 2 4] [--steps 5000]
    uv run python run_layer3_experiment.py --tau_values 0.01 0.1 1.0 --steps 5000
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml
from inr_sos.utils.data import USDataset
from inr_sos.utils.config import ExperimentConfig
from inr_sos.training.engines import (
    optimize_full_forward_operator,
    optimize_joint_residual,
)
from inr_sos.evaluation.metrics import calculate_metrics, compute_model_error_floor
from inr_sos.visualization.plot_reconstruction import plot_method_comparison
from inr_sos.io.paths import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent


def load_dataset_config(key=None):
    with open(SCRIPTS_DIR / "datasets.yaml") as f:
        cfg = yaml.safe_load(f)
    key = key or cfg["active"]
    ds = cfg["datasets"][key]
    ds["key"] = key
    ds["data_path"] = DATA_DIR + ds["data_file"]
    return ds


def load_dataset(ds_cfg):
    data_file = ds_cfg["data_path"]
    grid_file = DATA_DIR + "/DL-based-SoS/forward_model_lr/grid_parameters.mat"
    if not ds_cfg.get("has_A_matrix", True) and "matrix_file" in ds_cfg:
        matrix_path = DATA_DIR + ds_cfg["matrix_file"]
        return USDataset(data_file, grid_file, matrix_path=matrix_path)
    return USDataset(data_file, grid_file)


def build_model(config):
    from inr_sos.models.mlp import ReluMLP, FourierMLP
    from inr_sos.models.siren import SirenMLP
    if config.model_type == "SirenMLP":
        return SirenMLP(in_features=config.in_features,
                        hidden_features=config.hidden_features,
                        hidden_layers=config.hidden_layers, omega_0=config.omega)
    elif config.model_type == "FourierMLP":
        return FourierMLP(in_features=config.in_features,
                          hidden_features=config.hidden_features,
                          hidden_layers=config.hidden_layers,
                          mapping_size=config.mapping_size, scale=config.scale)
    return ReluMLP(in_features=config.in_features,
                   hidden_features=config.hidden_features,
                   hidden_layers=config.hidden_layers)


def main():
    parser = argparse.ArgumentParser(description="Layer 3: Joint residual correction")
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tau_values", type=float, nargs="+",
                        default=[0.01, 0.1, 1.0],
                        help="τ values to sweep (Gilton et al. recommend 0.1)")
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--residual_layers", type=int, default=3)
    parser.add_argument("--residual_omega", type=float, default=10.0)
    parser.add_argument("--residual_lr", type=float, default=1e-3)
    parser.add_argument("--tv_weight", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="layer3_results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # ── Load dataset ───────────────────────────────────────────────────────
    ds_cfg = load_dataset_config(args.dataset)
    logger.info(f"Dataset: {ds_cfg['name']} ({ds_cfg['key']})")
    dataset = load_dataset(ds_cfg)
    L_matrix = dataset.L_matrix

    # ── Shared config for reconstruction INR ───────────────────────────────
    base_kwargs = dict(
        model_type="SirenMLP", steps=args.steps, lr=args.lr,
        hidden_features=256, hidden_layers=3, omega=30.0,
        clamp_slowness=True, time_scale=1e6,
        tv_weight=args.tv_weight,
        residual_hidden=args.residual_hidden,
        residual_layers=args.residual_layers,
        residual_omega=args.residual_omega,
        residual_lr=args.residual_lr,
    )

    # ── Run experiments ────────────────────────────────────────────────────
    all_results = {}  # {method_name: [per_sample_metrics]}

    for sidx in args.samples:
        sample = dataset[sidx]
        logger.info(f"\n{'='*60}\nSample {sidx}\n{'='*60}")

        floor = compute_model_error_floor(sample, L_matrix)
        logger.info(f"Error floor: MSE={floor['floor_mse']:.4f}")

        recon_results = {}  # for plotting

        # ── Baseline ───────────────────────────────────────────────────────
        name = "Baseline"
        logger.info(f"\n--- {name} ---")
        config_base = ExperimentConfig(**base_kwargs)
        model = build_model(config_base)
        t0 = time.time()
        result = optimize_full_forward_operator(
            sample, L_matrix, model, name, config_base, use_wandb=False
        )
        metrics = calculate_metrics(result['s_phys'], sample['s_gt_raw'])
        metrics['time_s'] = time.time() - t0
        metrics['sample_idx'] = sidx
        all_results.setdefault(name, []).append(metrics)
        recon_results[f"{name} (MAE={metrics['MAE']:.1f})"] = result
        logger.info(f"  MAE={metrics['MAE']:.2f}, SSIM={metrics['SSIM']:.4f}")

        # ── Layer 3: τ sweep ───────────────────────────────────────────────
        for tau in args.tau_values:
            name = f"Joint τ={tau}"
            logger.info(f"\n--- {name} ---")
            config_l3 = ExperimentConfig(**base_kwargs, tau=tau)
            model = build_model(config_l3)
            t0 = time.time()
            result = optimize_joint_residual(
                sample, L_matrix, model, name, config_l3, use_wandb=False
            )
            metrics = calculate_metrics(result['s_phys'], sample['s_gt_raw'])
            metrics['time_s'] = time.time() - t0
            metrics['sample_idx'] = sidx
            all_results.setdefault(name, []).append(metrics)

            # Residual diagnostics
            g = result['residual']
            mask = sample['mask']
            g_energy = (g * mask).pow(2).sum() / (mask.sum() + 1e-8)
            d_energy = (sample['d_meas'] * mask).pow(2).sum() / (mask.sum() + 1e-8)

            recon_results[f"{name} (MAE={metrics['MAE']:.1f})"] = result
            logger.info(
                f"  MAE={metrics['MAE']:.2f}, SSIM={metrics['SSIM']:.4f}, "
                f"|g|²/|d|²={g_energy/d_energy:.4f}"
            )

        # ── Comparison plot ────────────────────────────────────────────────
        save_path = output_dir / f"layer3_comparison_sample{sidx}.png"
        plot_method_comparison(
            recon_results, sample,
            title=f"Layer 3 Joint Residual — Sample {sidx} ({ds_cfg['key']})",
            save_path=str(save_path), show=False,
        )
        plt.close("all")
        logger.info(f"Plot saved: {save_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    summary = {}
    for name, results_list in all_results.items():
        maes = [r['MAE'] for r in results_list]
        ssims = [r['SSIM'] for r in results_list]
        summary[name] = {
            'MAE_mean': float(np.mean(maes)),
            'MAE_std': float(np.std(maes)),
            'SSIM_mean': float(np.mean(ssims)),
            'SSIM_std': float(np.std(ssims)),
            'per_sample': results_list,
        }
        logger.info(
            f"  {name:20s}: MAE={np.mean(maes):6.2f} ± {np.std(maes):5.2f}, "
            f"SSIM={np.mean(ssims):.4f}"
        )
    logger.info(f"\nBaseline references: L1 MAE=7.0, L2 MAE=9.3")

    results_file = output_dir / f"layer3_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
