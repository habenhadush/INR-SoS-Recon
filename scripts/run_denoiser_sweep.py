#!/usr/bin/env python3
"""
run_denoiser_sweep.py
---------------------
Sweep denoiser hyperparameters to find the optimal measurement-domain
regularization. Loads dataset ONCE, runs raw reconstruction ONCE per sample,
then sweeps denoiser configs and reconstructs from each.

Sweep axes:
  - scale:           [0.5, 1.0, 2.0, 5.0, 10.0]  (spectral bandwidth)
  - steps:           [100, 300, 500, 1000]          (fitting duration)
  - hidden_features: [32, 64, 128]                  (model capacity)

Usage:
    # Quick (2 samples, default sweep)
    python scripts/run_denoiser_sweep.py --n_samples 2

    # Custom axes
    python scripts/run_denoiser_sweep.py --scales 0.5 1.0 2.0 --steps 200 500

    # Specific samples
    python scripts/run_denoiser_sweep.py --indices 0 1 5 10
"""

import argparse
import copy
import json
import logging
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from inr_sos import DATA_DIR
from inr_sos.utils.data import USDataset
from inr_sos.utils.config import ExperimentConfig
from inr_sos.evaluation.metrics import calculate_metrics
from inr_sos.models.mlp import FourierMLP, ReluMLP
from inr_sos.training.engines import optimize_full_forward_operator
from inr_sos.training.denoise_engine import (
    denoise_measurements,
    DEFAULT_DENOISE_CFG,
)

SCRIPTS_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPTS_DIR / "data" / "denoiser_sweep"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("denoiser_sweep")


# ─── Reconstruction config (fixed across all sweep runs) ────────────────────

def make_recon_config():
    return ExperimentConfig(
        project_name="INR-SoS-Recon",
        experiment_group="Denoiser-Sweep",
        model_type="ReluMLP",
        hidden_features=256,
        hidden_layers=3,
        mapping_size=64,
        lr=1e-4,
        steps=2000,
        early_stopping=True,
        patience=100,
        clamp_slowness=True,
        loss_type="mse",
    )


def build_recon_model(cfg):
    return ReluMLP(
        in_features=cfg.in_features,
        hidden_features=cfg.hidden_features,
        hidden_layers=cfg.hidden_layers,
        mapping_size=cfg.mapping_size,
    )


def _to_sos(s_flat):
    if hasattr(s_flat, "detach"):
        s_flat = s_flat.detach().cpu().numpy()
    s_flat = np.asarray(s_flat).flatten()
    s_clamped = np.clip(s_flat, 1.0 / 1800.0, 1.0 / 1200.0)
    return (1.0 / s_clamped).reshape(64, 64, order="F")


# ─── Core: run one denoiser config on one sample ────────────────────────────

def run_denoised_recon(sample, dataset, grid, denoise_cfg, recon_cfg):
    """Denoise + reconstruct. Returns metrics dict + s_phys + loss_history."""
    L_matrix = dataset.L_matrix
    s_gt = sample["s_gt_raw"]

    # Stage 1: Denoise
    t0 = time.perf_counter()
    d_denoised, denoise_info = denoise_measurements(sample, grid, denoise_cfg)
    t_denoise = time.perf_counter() - t0

    # Stage 2: Reconstruct from denoised (keep original mask)
    sample_dn = copy.copy(sample)
    sample_dn["d_meas"] = d_denoised
    sample_dn["mask"] = sample["mask"]

    model = build_recon_model(recon_cfg)
    t0 = time.perf_counter()
    res = optimize_full_forward_operator(
        sample=sample_dn, L_matrix=L_matrix, model=model,
        label="ReluMLP_denoised", config=copy.deepcopy(recon_cfg), use_wandb=False,
    )
    t_recon = time.perf_counter() - t0

    m = calculate_metrics(res["s_phys"], s_gt)
    m["denoise_time_s"] = t_denoise
    m["recon_time_s"] = t_recon
    m["total_time_s"] = t_denoise + t_recon

    return {
        "metrics": m,
        "s_phys": res["s_phys"],
        "loss_history": res["loss_history"],
    }


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_sweep_summary(sweep_results, sample_baselines, out_dir):
    """Bar chart + table comparing all configs across samples."""
    configs = list(sweep_results.keys())
    n_cfgs = len(configs)

    # Aggregate metrics across samples
    summary = {}
    for cfg_label, per_sample in sweep_results.items():
        maes = [s["metrics"]["MAE"] for s in per_sample]
        cnrs = [s["metrics"]["CNR"] for s in per_sample]
        ssims = [s["metrics"]["SSIM"] for s in per_sample]
        summary[cfg_label] = {
            "MAE": np.mean(maes), "MAE_std": np.std(maes),
            "CNR": np.mean(cnrs), "CNR_std": np.std(cnrs),
            "SSIM": np.mean(ssims), "SSIM_std": np.std(ssims),
        }

    # Add baselines
    for bl_name in ["l1", "l2", "raw"]:
        if bl_name in sample_baselines:
            vals = sample_baselines[bl_name]
            maes = [v["MAE"] for v in vals]
            cnrs = [v["CNR"] for v in vals]
            ssims = [v["SSIM"] for v in vals]
            summary[bl_name] = {
                "MAE": np.mean(maes), "MAE_std": np.std(maes),
                "CNR": np.mean(cnrs), "CNR_std": np.std(cnrs),
                "SSIM": np.mean(ssims), "SSIM_std": np.std(ssims),
            }

    # ── Bar plot: MAE ────────────────────────────────────────────────────
    labels = list(summary.keys())
    mae_vals = [summary[l]["MAE"] for l in labels]
    mae_stds = [summary[l]["MAE_std"] for l in labels]

    # Color: baselines grey, sweep configs blue
    colors = ["#999999" if l in ("l1", "l2", "raw") else "#1f77b4" for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, n_cfgs * 1.5), 6))
    fig.suptitle("Denoiser Sweep — Mean Metrics Across Samples", fontsize=13)

    x = np.arange(len(labels))
    ax1.bar(x, mae_vals, yerr=mae_stds, color=colors, capsize=4, edgecolor="black",
            linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("MAE (m/s)", fontsize=10)
    ax1.set_title("MAE (lower is better)", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # ── Bar plot: CNR ────────────────────────────────────────────────────
    cnr_vals = [summary[l]["CNR"] for l in labels]
    cnr_stds = [summary[l]["CNR_std"] for l in labels]
    ax2.bar(x, cnr_vals, yerr=cnr_stds, color=colors, capsize=4, edgecolor="black",
            linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("CNR", fontsize=10)
    ax2.set_title("CNR (higher is better)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fp = out_dir / "sweep_summary.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Sweep summary plot → {fp}")


def plot_per_sample_grid(sweep_results, sample_baselines, sample_gts,
                         indices, out_dir):
    """Per-sample grid: rows=samples, cols=baselines+configs. Shows SoS maps."""
    configs = list(sweep_results.keys())
    baselines = [k for k in ["l1", "l2", "raw"] if k in sample_baselines]
    all_methods = baselines + configs
    n_cols = 1 + len(all_methods)  # GT + methods
    n_rows = len(indices)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.8 * n_cols, 3.2 * n_rows),
                              squeeze=False)
    fig.suptitle("Denoiser Sweep — Per-Sample Reconstructions", fontsize=13, y=1.01)

    for row, idx in enumerate(indices):
        v_gt = sample_gts[idx]

        # GT column
        im = axes[row, 0].imshow(v_gt, cmap="jet", vmin=1400, vmax=1600)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("GT", fontsize=9, fontweight="bold")
        axes[row, 0].set_ylabel(f"idx {idx}", fontsize=9, fontweight="bold")

        col = 1
        for method in all_methods:
            if method in baselines:
                m = sample_baselines[method][row]
                v_rec = _to_sos(
                    # baselines store metrics only, need s_phys from separate storage
                    sample_baselines[f"{method}_s_phys"][row]
                )
            else:
                res = sweep_results[method][row]
                v_rec = _to_sos(res["s_phys"])
                m = res["metrics"]

            err = float(np.mean(np.abs(v_gt - v_rec)))
            im = axes[row, col].imshow(v_rec, cmap="jet", vmin=1400, vmax=1600)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(method, fontsize=8, fontweight="bold")
            axes[row, col].text(
                0.02, 0.98, f"MAE={m['MAE']:.1f}\nCNR={m['CNR']:.2f}",
                transform=axes[row, col].transAxes, fontsize=7, va="top",
                color="white", bbox=dict(boxstyle="round", fc="black", alpha=0.6),
            )
            col += 1

    plt.tight_layout()
    fp = out_dir / "per_sample_grid.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Per-sample grid → {fp}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Denoiser hyperparameter sweep")
    parser.add_argument("--dataset", default="kwave_geom")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--indices", nargs="+", type=int, default=None)
    parser.add_argument("--scales", nargs="+", type=float,
                        default=[0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--steps", nargs="+", type=int,
                        default=[100, 300, 500])
    parser.add_argument("--hidden", nargs="+", type=int, default=[64])
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / args.dataset / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Build sweep grid ─────────────────────────────────────────────────
    sweep_grid = list(product(args.scales, args.steps, args.hidden))
    log.info("=" * 70)
    log.info("  Denoiser Hyperparameter Sweep")
    log.info(f"  Dataset: {args.dataset}")
    log.info(f"  Scales: {args.scales}")
    log.info(f"  Steps: {args.steps}")
    log.info(f"  Hidden: {args.hidden}")
    log.info(f"  Total configs: {len(sweep_grid)}")
    log.info(f"  Output: {run_dir}")
    log.info("=" * 70)

    # ── Load dataset ─────────────────────────────────────────────────────
    cfg_path = SCRIPTS_DIR / "datasets.yaml"
    with open(cfg_path) as f:
        ds_cfg = yaml.safe_load(f)["datasets"][args.dataset]

    data_path = DATA_DIR + ds_cfg["data_file"]
    grid_path = DATA_DIR + "/DL-based-SoS/forward_model_lr/grid_parameters.mat"

    ds_kwargs = {}
    if not ds_cfg.get("has_A_matrix", True):
        matrix_file = ds_cfg.get("matrix_file")
        if matrix_file:
            ds_kwargs["matrix_path"] = DATA_DIR + matrix_file
            ds_kwargs["use_external_L_matrix"] = True

    dataset = USDataset(data_path, grid_path, **ds_kwargs)
    grid = dataset.grid

    # ── Sample indices ───────────────────────────────────────────────────
    if args.indices:
        indices = args.indices
    else:
        n = args.n_samples if args.n_samples is not None else len(dataset)
        indices = list(range(min(n, len(dataset))))
    log.info(f"  Samples: {indices}")

    # ── Load samples and compute baselines (ONCE) ────────────────────────
    recon_cfg = make_recon_config()
    if hasattr(dataset, "pix2time") and dataset.pix2time is not None:
        recon_cfg.time_scale = 1.0 / dataset.pix2time

    samples = [dataset[idx] for idx in indices]
    sample_gts = {idx: _to_sos(samples[i]["s_gt_raw"]) for i, idx in enumerate(indices)}

    # Baselines: L1, L2, Raw INR
    sample_baselines = {}
    for key, label in [("s_l1_recon", "l1"), ("s_l2_recon", "l2")]:
        if key in samples[0]:
            sample_baselines[label] = []
            sample_baselines[f"{label}_s_phys"] = []
            for sample in samples:
                m = calculate_metrics(sample[key], sample["s_gt_raw"])
                sample_baselines[label].append(m)
                sample_baselines[f"{label}_s_phys"].append(sample[key])

    # Raw INR reconstruction (run once, reuse)
    log.info("\n── Computing raw INR baseline (once) ──")
    sample_baselines["raw"] = []
    sample_baselines["raw_s_phys"] = []
    raw_loss_histories = []
    for i, (idx, sample) in enumerate(zip(indices, samples)):
        log.info(f"  Raw INR: sample {i+1}/{len(indices)} (idx={idx})")
        model = build_recon_model(recon_cfg)
        res = optimize_full_forward_operator(
            sample=sample, L_matrix=dataset.L_matrix, model=model,
            label="ReluMLP_raw", config=copy.deepcopy(recon_cfg), use_wandb=False,
        )
        m = calculate_metrics(res["s_phys"], sample["s_gt_raw"])
        sample_baselines["raw"].append(m)
        sample_baselines["raw_s_phys"].append(res["s_phys"])
        raw_loss_histories.append(res["loss_history"])

    # ── Sweep ────────────────────────────────────────────────────────────
    sweep_results = {}  # {config_label: [per_sample_results]}

    for gi, (scale, steps, hidden) in enumerate(sweep_grid):
        cfg_label = f"s{scale}_st{steps}_h{hidden}"
        log.info(f"\n── Config {gi+1}/{len(sweep_grid)}: {cfg_label} ──")

        denoise_cfg = dict(DEFAULT_DENOISE_CFG)
        denoise_cfg.update({
            "scale": scale,
            "steps": steps,
            "hidden_features": hidden,
        })

        per_sample = []
        for i, (idx, sample) in enumerate(zip(indices, samples)):
            log.info(f"  {cfg_label}: sample {i+1}/{len(indices)} (idx={idx})")
            res = run_denoised_recon(sample, dataset, grid, denoise_cfg, recon_cfg)
            per_sample.append(res)

            m = res["metrics"]
            raw_m = sample_baselines["raw"][i]
            log.info(f"    Raw MAE={raw_m['MAE']:.1f} → Denoised MAE={m['MAE']:.1f} "
                     f"({(raw_m['MAE'] - m['MAE'])/raw_m['MAE']*100:+.1f}%)")

        sweep_results[cfg_label] = per_sample

    # ── Summary table ────────────────────────────────────────────────────
    log.info(f"\n{'='*90}")
    log.info(f"  DENOISER SWEEP RESULTS — {len(indices)} samples, "
             f"{len(sweep_grid)} configs")
    log.info(f"{'='*90}")
    log.info(f"  {'Config':<25} {'MAE±std':>14} {'RMSE±std':>14} "
             f"{'SSIM±std':>14} {'CNR±std':>14}")
    log.info(f"  {'─'*85}")

    # Baselines first
    for bl in ["l1", "l2", "raw"]:
        if bl not in sample_baselines:
            continue
        vals = sample_baselines[bl]
        label = {"l1": "L1 Baseline", "l2": "L2 Baseline", "raw": "Raw INR"}[bl]
        maes = [v["MAE"] for v in vals]
        rmses = [v["RMSE"] for v in vals]
        ssims = [v["SSIM"] for v in vals]
        cnrs = [v["CNR"] for v in vals]
        log.info(
            f"  {label:<25}"
            f"  {np.mean(maes):>6.2f}±{np.std(maes):<5.2f}"
            f"  {np.mean(rmses):>6.2f}±{np.std(rmses):<5.2f}"
            f"  {np.mean(ssims):>6.3f}±{np.std(ssims):<5.3f}"
            f"  {np.mean(cnrs):>6.3f}±{np.std(cnrs):<5.3f}"
        )
    log.info(f"  {'─'*85}")

    # Sweep configs
    for cfg_label, per_sample in sweep_results.items():
        maes = [s["metrics"]["MAE"] for s in per_sample]
        rmses = [s["metrics"]["RMSE"] for s in per_sample]
        ssims = [s["metrics"]["SSIM"] for s in per_sample]
        cnrs = [s["metrics"]["CNR"] for s in per_sample]
        log.info(
            f"  {cfg_label:<25}"
            f"  {np.mean(maes):>6.2f}±{np.std(maes):<5.2f}"
            f"  {np.mean(rmses):>6.2f}±{np.std(rmses):<5.2f}"
            f"  {np.mean(ssims):>6.3f}±{np.std(ssims):<5.3f}"
            f"  {np.mean(cnrs):>6.3f}±{np.std(cnrs):<5.3f}"
        )
    log.info(f"{'='*90}")

    # ── Plots ────────────────────────────────────────────────────────────
    plot_sweep_summary(sweep_results, sample_baselines, run_dir)
    plot_per_sample_grid(sweep_results, sample_baselines, sample_gts,
                         indices, run_dir)

    # ── Save results JSON ────────────────────────────────────────────────
    results_json = {
        "timestamp": timestamp,
        "dataset": args.dataset,
        "sweep_grid": {
            "scales": args.scales, "steps": args.steps, "hidden": args.hidden,
        },
        "n_samples": len(indices),
        "indices": indices,
        "baselines": {},
        "sweep": {},
    }
    for bl in ["l1", "l2", "raw"]:
        if bl in sample_baselines:
            results_json["baselines"][bl] = sample_baselines[bl]

    for cfg_label, per_sample in sweep_results.items():
        results_json["sweep"][cfg_label] = [
            s["metrics"] for s in per_sample
        ]

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    log.info(f"\n  Results saved → {results_path}")
    log.info(f"  Experiment complete. All outputs in {run_dir}")


if __name__ == "__main__":
    main()
