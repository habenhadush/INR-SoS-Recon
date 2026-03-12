"""Generate reconstruction comparison plots for Layer 1+2 experiments.

Re-runs reconstructions for selected samples and saves side-by-side
comparison figures (GT, baseline, BAE, SVD, error maps).

Usage:
    source .venv/bin/activate && cd scripts
    uv run python plot_layer12.py [--samples 0 2 4] [--steps 2000]
"""

import argparse
import sys
import time
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inr_sos.utils.data import USDataset
from inr_sos.utils.config import ExperimentConfig
from inr_sos.utils.mismatch import compute_svd_weights
from inr_sos.training.engines import optimize_full_forward_operator
from inr_sos.evaluation.metrics import calculate_metrics
from inr_sos.visualization.plot_reconstruction import plot_method_comparison
from inr_sos.io.paths import DATA_DIR

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent


def load_dataset_config(key=None):
    cfg_path = SCRIPTS_DIR / "datasets.yaml"
    with open(cfg_path) as f:
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


def compute_loo_bae_stats(dataset, L_matrix, test_idx):
    N = len(dataset)
    epsilons, masks = [], []
    for i in range(N):
        if i == test_idx:
            continue
        sample = dataset[i]
        eps = sample['d_meas'] - L_matrix @ sample['s_gt_raw']
        epsilons.append(eps)
        masks.append(sample['mask'])

    epsilons = torch.stack(epsilons, dim=0)
    masks = torch.stack(masks, dim=0)
    mask_common = (masks.prod(dim=0) > 0.5).float()
    epsilons_masked = epsilons * mask_common.unsqueeze(0)
    eta = epsilons_masked.mean(dim=0)
    sigma2 = epsilons_masked.var(dim=0)
    sigma2 = sigma2 * mask_common + (1.0 - mask_common) * 1e10
    return {'eta': eta, 'sigma2': sigma2, 'n_samples': N - 1, 'mask_common': mask_common}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="layer12_results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    ds_cfg = load_dataset_config(args.dataset)
    logger.info(f"Dataset: {ds_cfg['name']} ({ds_cfg['key']})")
    dataset = load_dataset(ds_cfg)
    L_matrix = dataset.L_matrix

    # SVD weights (once)
    sample0 = dataset[0]
    svd_config = ExperimentConfig(svd_top_k=3800, svd_tail_damping=0.01)
    svd_info = compute_svd_weights(L_matrix, sample0['mask'], svd_config)

    base_kwargs = dict(
        model_type="SirenMLP", steps=args.steps, lr=1e-4,
        hidden_features=256, hidden_layers=3, omega=30.0,
        clamp_slowness=True, time_scale=1e6,
    )

    methods = {
        "Baseline (standard)": dict(loss_weighting="none"),
        "Layer 1: SVD-weighted": dict(loss_weighting="svd", svd_top_k=3800, svd_tail_damping=0.01),
        "Layer 2: BAE correction": dict(loss_weighting="bae"),
        "Layer 1+2: SVD+BAE": dict(loss_weighting="svd+bae", svd_top_k=3800, svd_tail_damping=0.01),
    }

    for sidx in args.samples:
        sample = dataset[sidx]
        logger.info(f"\n{'='*60}\nSample {sidx}\n{'='*60}")

        # LOO BAE stats
        bae_stats = compute_loo_bae_stats(dataset, L_matrix, sidx)

        results = {}
        for method_name, method_overrides in methods.items():
            config = ExperimentConfig(**base_kwargs, **method_overrides)
            model = build_model(config)

            _svd = svd_info if 'svd' in config.loss_weighting else None
            _bae = bae_stats if 'bae' in config.loss_weighting else None

            logger.info(f"Running: {method_name}")
            result = optimize_full_forward_operator(
                sample, L_matrix, model, method_name, config,
                use_wandb=False, bae_stats=_bae, svd_info=_svd,
            )
            metrics = calculate_metrics(result['s_phys'], sample['s_gt_raw'])
            logger.info(f"  MAE={metrics['MAE']:.2f}, SSIM={metrics['SSIM']:.4f}")
            results[method_name] = result

        # Generate comparison plot
        save_path = output_dir / f"layer12_comparison_sample{sidx}.png"
        plot_method_comparison(
            results, sample,
            title=f"Layer 1+2 Comparison — Sample {sidx} ({ds_cfg['key']})",
            save_path=str(save_path),
            show=False,
        )
        plt.close("all")
        logger.info(f"Plot saved: {save_path}")

    logger.info("\nAll plots saved.")


if __name__ == "__main__":
    main()
