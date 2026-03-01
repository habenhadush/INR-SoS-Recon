#!/usr/bin/env python3
"""
run_denoiser_test.py
--------------------
Proof-of-concept test for Architecture 1: INR displacement field denoiser.

Runs the two-stage pipeline on k-wave data and compares:
  1. No denoiser (raw d_meas → reconstruction)
  2. With denoiser (d_clean → reconstruction)
  3. Oracle denoiser (d_oracle = L @ s_true → reconstruction)
  4. L1/L2 baselines (from data file)

Usage:
    python scripts/run_denoiser_test.py
    python scripts/run_denoiser_test.py --dataset kwave_geom --indices 0 5 10
    python scripts/run_denoiser_test.py --omega 5 10 15 20 --indices 0
"""

import argparse
import json
import logging
import sys
import time
import numpy as np
import torch
import yaml
from datetime import datetime
from pathlib import Path

import inr_sos
from inr_sos import DATA_DIR
from inr_sos.utils.data import USDataset
from inr_sos.utils.config import ExperimentConfig
from inr_sos.evaluation.metrics import calculate_metrics, compute_model_error_floor
from inr_sos.denoising import DenoiserConfig, denoise_displacement, compute_ray_features
from inr_sos.denoising.engine import denoise_and_reconstruct
from inr_sos.models.siren import SirenMLP
from inr_sos.models.mlp import FourierMLP
from inr_sos.training.engines import optimize_full_forward_operator

SCRIPTS_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPTS_DIR / "denoiser_results"
RESULTS_DIR.mkdir(exist_ok=True)

log = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(RESULTS_DIR / "denoiser_test.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_dataset_config(key=None):
    cfg_path = SCRIPTS_DIR / "datasets.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    key = key or cfg["active"]
    ds = cfg["datasets"][key]
    ds["key"] = key
    ds["data_path"] = DATA_DIR + ds["data_file"]
    return ds


def compute_auto_time_scale(data_path):
    import h5py
    with h5py.File(data_path, "r") as f:
        pix2time = float(np.array(f["pix2time"]).flat[0])
    return 1.0 / pix2time


def build_recon_model(config):
    """Build the Stage 2 reconstruction model."""
    return SirenMLP(
        in_features=config.in_features,
        hidden_features=config.hidden_features,
        hidden_layers=config.hidden_layers,
        out_features=1,
        omega=config.omega,
    )


def run_single_experiment(
    sample, L_matrix, dataset, recon_config, denoiser_config, ray_features, label,
):
    """Run one experiment (with or without denoiser) and return metrics."""
    model = build_recon_model(recon_config)

    if denoiser_config is not None:
        result = denoise_and_reconstruct(
            sample=sample,
            L_matrix=L_matrix,
            model=model,
            label=recon_config.model_type,
            config=recon_config,
            engine_fn=optimize_full_forward_operator,
            denoiser_config=denoiser_config,
            ray_features=ray_features,
            use_wandb=False,
        )
    else:
        result = optimize_full_forward_operator(
            sample=sample,
            L_matrix=L_matrix,
            model=model,
            label=recon_config.model_type,
            config=recon_config,
            use_wandb=False,
        )

    metrics = calculate_metrics(
        s_phys_pred=result["s_phys"],
        s_gt_raw=sample["s_gt_raw"],
        grid_shape=(64, 64),
    )
    metrics["label"] = label
    return metrics, result


def run_oracle_experiment(sample, L_matrix, dataset, recon_config, ray_features):
    """Run reconstruction with oracle d_clean = L @ s_true."""
    s_gt = sample["s_gt_raw"].flatten()
    L_dense = L_matrix
    if hasattr(L_dense, "is_sparse") and L_dense.is_sparse:
        L_dense = L_dense.to_dense()
    d_oracle = (L_dense @ s_gt).unsqueeze(-1)

    oracle_sample = {**sample, "d_meas": d_oracle}
    model = build_recon_model(recon_config)
    result = optimize_full_forward_operator(
        sample=oracle_sample,
        L_matrix=L_matrix,
        model=model,
        label=recon_config.model_type,
        config=recon_config,
        use_wandb=False,
    )
    metrics = calculate_metrics(
        s_phys_pred=result["s_phys"],
        s_gt_raw=sample["s_gt_raw"],
        grid_shape=(64, 64),
    )
    metrics["label"] = "Oracle (L@s_true)"
    return metrics, result


def compute_baseline_metrics(sample, dataset, idx):
    """Extract L1/L2 baseline metrics if available."""
    rows = []
    for bname, attr in [("L1", "benchmarks_l1"), ("L2", "benchmarks_l2")]:
        bench = getattr(dataset, attr, None)
        if bench is None:
            continue
        if bench.ndim == 3:
            b_img = bench[idx]
        else:
            b_img = bench
        b_flat = torch.tensor(b_img, dtype=torch.float32).flatten()
        # benchmarks are in SoS (m/s), convert to slowness
        b_slowness = 1.0 / (b_flat + 1e-8)
        m = calculate_metrics(
            s_phys_pred=b_slowness.unsqueeze(-1),
            s_gt_raw=sample["s_gt_raw"],
            grid_shape=(64, 64),
        )
        m["label"] = f"{bname} baseline"
        rows.append(m)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Architecture 1: INR Denoiser PoC")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--indices", nargs="+", type=int, default=[0])
    parser.add_argument("--omega", nargs="+", type=float, default=[10.0],
                        help="Denoiser SIREN omega values to test")
    parser.add_argument("--steps", type=int, default=5000, help="Denoiser max steps")
    parser.add_argument("--patience", type=int, default=300, help="Denoiser patience")
    parser.add_argument("--recon_steps", type=int, default=500, help="Reconstruction steps")
    parser.add_argument("--recon_lr", type=float, default=1e-4)
    parser.add_argument("--skip_oracle", action="store_true",
                        help="Skip oracle experiment (saves time)")
    args = parser.parse_args()

    setup_logging()
    log.info("=" * 60)
    log.info("Architecture 1: INR Denoiser — Proof of Concept")
    log.info("=" * 60)

    # Load dataset
    ds_cfg = load_dataset_config(args.dataset)
    data_path = ds_cfg["data_path"]
    grid_file = DATA_DIR + "/DL-based-SoS/forward_model_lr/grid_parameters.mat"

    # Time scale
    sweep_section = ds_cfg.get("sweep", {})
    yaml_ts = sweep_section.get("time_scale", 1e6)
    if yaml_ts == "auto":
        time_scale = compute_auto_time_scale(data_path)
        log.info(f"time_scale: auto → {time_scale:.2e}")
    else:
        time_scale = float(yaml_ts)
        log.info(f"time_scale: {time_scale:.2e}")

    # Load dataset
    if not ds_cfg.get("has_A_matrix", True) and "matrix_file" in ds_cfg:
        matrix_path = DATA_DIR + ds_cfg["matrix_file"]
        dataset = USDataset(data_path, grid_file, matrix_path=matrix_path)
    else:
        dataset = USDataset(data_path, grid_file)
    log.info(f"Dataset: {ds_cfg['name']} — {len(dataset)} samples")

    # Precompute ray features (once for all experiments)
    ray_features = compute_ray_features(grid=dataset.grid)
    log.info(f"Ray features: {ray_features.shape}")

    # Stage 2 reconstruction config (reasonable defaults for k-wave)
    recon_config = ExperimentConfig(
        model_type="SirenMLP",
        in_features=2,
        hidden_features=256,
        hidden_layers=3,
        omega=30.0,
        lr=args.recon_lr,
        steps=args.recon_steps,
        time_scale=time_scale,
        tv_weight=1e-3,
        early_stopping=True,
        patience=100,
        val_fraction=0.1,
        clamp_slowness=True,
        loss_type="huber",
        huber_delta=1.0,
    )

    all_results = []

    for idx in args.indices:
        log.info(f"\n{'='*60}")
        log.info(f"Sample {idx}")
        log.info(f"{'='*60}")
        sample = dataset[idx]

        # --- Model error floor ---
        floor = compute_model_error_floor(sample, dataset.L_matrix, time_scale)
        log.info(f"Error floor: MSE={floor['floor_mse']:.4f}, "
                 f"RMSE={floor['floor_rmse']:.4f}, MAE={floor['floor_mae']:.4f}")

        # --- Baselines ---
        baseline_rows = compute_baseline_metrics(sample, dataset, idx)
        for br in baseline_rows:
            log.info(f"  {br['label']}: MAE={br['MAE']:.2f}, SSIM={br['SSIM']:.4f}")
            all_results.append({"sample_idx": idx, **br})

        # --- No-denoiser baseline ---
        log.info("\n--- No denoiser (raw d_meas) ---")
        m_raw, _ = run_single_experiment(
            sample, dataset.L_matrix, dataset, recon_config, None, ray_features,
            label="No denoiser",
        )
        log.info(f"  No denoiser: MAE={m_raw['MAE']:.2f}, SSIM={m_raw['SSIM']:.4f}")
        all_results.append({"sample_idx": idx, **m_raw})

        # --- Oracle ---
        if not args.skip_oracle:
            log.info("\n--- Oracle (d = L @ s_true) ---")
            m_oracle, _ = run_oracle_experiment(
                sample, dataset.L_matrix, dataset, recon_config, ray_features,
            )
            log.info(f"  Oracle: MAE={m_oracle['MAE']:.2f}, SSIM={m_oracle['SSIM']:.4f}")
            all_results.append({"sample_idx": idx, **m_oracle})

        # --- Denoiser with different omega values ---
        for omega in args.omega:
            log.info(f"\n--- Denoiser (omega={omega}) ---")
            d_config = DenoiserConfig(
                omega=omega,
                steps=args.steps,
                patience=args.patience,
                time_scale=time_scale,
            )
            m_den, result = run_single_experiment(
                sample, dataset.L_matrix, dataset, recon_config, d_config,
                ray_features, label=f"Denoiser omega={omega}",
            )
            log.info(f"  Denoiser omega={omega}: MAE={m_den['MAE']:.2f}, "
                     f"SSIM={m_den['SSIM']:.4f}")

            # Denoiser diagnostics
            if "denoiser_result" in result:
                dr = result["denoiser_result"]
                log.info(f"    best_step={dr['best_step']}, "
                         f"train_loss_final={dr['loss_history'][-1]:.4f}")
                # Denoised error floor
                denoised_sample = {**sample, "d_meas": dr["d_clean"]}
                d_floor = compute_model_error_floor(
                    denoised_sample, dataset.L_matrix, time_scale
                )
                log.info(f"    Denoised floor: MSE={d_floor['floor_mse']:.4f}, "
                         f"MAE={d_floor['floor_mae']:.4f}")
                m_den["denoiser_best_step"] = dr["best_step"]
                m_den["denoised_floor_mse"] = d_floor["floor_mse"]

            all_results.append({"sample_idx": idx, **m_den})

    # --- Save results ---
    results_file = RESULTS_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        # Convert non-serializable values
        clean = []
        for r in all_results:
            clean.append({k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v
                          for k, v in r.items()})
        json.dump(clean, f, indent=2)
    log.info(f"\nResults saved: {results_file}")

    # --- Summary table ---
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"{'Label':<25} {'MAE':>8} {'RMSE':>8} {'SSIM':>8} {'CNR':>8}")
    log.info("-" * 60)
    for r in all_results:
        log.info(f"{r['label']:<25} {r['MAE']:>8.2f} {r['RMSE']:>8.4f} "
                 f"{r['SSIM']:>8.4f} {r.get('CNR', 0):>8.4f}")


if __name__ == "__main__":
    main()
