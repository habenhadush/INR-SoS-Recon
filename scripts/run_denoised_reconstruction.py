#!/usr/bin/env python3
"""
run_denoised_reconstruction.py
------------------------------
Two-stage pipeline: (1) denoise measurements with per-pair INR,
(2) reconstruct SoS using existing engine on denoised measurements.

Compares raw vs denoised vs oracle reconstruction quality.

Usage:
    # Quick test (2 samples, default denoiser)
    python scripts/run_denoised_reconstruction.py --n_samples 2

    # Sweep denoiser scale
    python scripts/run_denoised_reconstruction.py --scale 0.5

    # Full run
    python scripts/run_denoised_reconstruction.py --dataset kwave_geom
"""

import argparse
import copy
import io
import json
import logging
import sys
import time
from datetime import datetime
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
OUTPUT_DIR = SCRIPTS_DIR / "data" / "denoiser_experiment"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("denoiser_exp")


# ─── Reconstruction config (fixed, known good) ───────────────────────────────

def make_recon_config():
    return ExperimentConfig(
        project_name="INR-SoS-Recon",
        experiment_group="Denoiser-Experiment",
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


# ─── Run one sample through the two-stage pipeline ───────────────────────────

def run_single_sample(sample, dataset, grid, denoise_cfg, recon_cfg, idx):
    """Run raw, denoised, and oracle reconstruction for one sample.

    Returns dict with metrics for each condition.
    """
    L_matrix = dataset.L_matrix
    s_gt = sample["s_gt_raw"]

    # Set time_scale from dataset
    if hasattr(dataset, "pix2time") and dataset.pix2time is not None:
        recon_cfg.time_scale = 1.0 / dataset.pix2time

    results = {}

    # ── L1/L2 baselines (no optimization, just metrics) ─────────────────
    for key, label in [("s_l1_recon", "l1"), ("s_l2_recon", "l2")]:
        if key in sample:
            m = calculate_metrics(sample[key], s_gt)
            results[label] = {"metrics": m, "s_phys": sample[key]}

    # ── Condition 1: Raw measurements (baseline) ─────────────────────────
    log.info(f"  [idx={idx}] Raw reconstruction...")
    model_raw = build_recon_model(recon_cfg)
    t0 = time.perf_counter()
    res_raw = optimize_full_forward_operator(
        sample=sample, L_matrix=L_matrix, model=model_raw,
        label="ReluMLP_raw", config=copy.deepcopy(recon_cfg), use_wandb=False,
    )
    t_raw = time.perf_counter() - t0
    m_raw = calculate_metrics(res_raw["s_phys"], s_gt)
    m_raw["time_s"] = t_raw
    results["raw"] = {"metrics": m_raw, "s_phys": res_raw["s_phys"],
                      "loss_history": res_raw["loss_history"]}

    # ── Stage 1: Denoise measurements ────────────────────────────────────
    log.info(f"  [idx={idx}] Denoising measurements...")
    t0 = time.perf_counter()
    d_denoised, denoise_info = denoise_measurements(sample, grid, denoise_cfg)
    t_denoise = time.perf_counter() - t0

    # ── Stage 2: Reconstruct from denoised ───────────────────────────────
    log.info(f"  [idx={idx}] Denoised reconstruction...")
    sample_denoised = copy.copy(sample)
    sample_denoised["d_meas"] = d_denoised
    sample_denoised["mask"] = sample["mask"]  # keep original valid-ray mask

    model_dn = build_recon_model(recon_cfg)
    t0 = time.perf_counter()
    res_dn = optimize_full_forward_operator(
        sample=sample_denoised, L_matrix=L_matrix, model=model_dn,
        label="ReluMLP_denoised", config=copy.deepcopy(recon_cfg), use_wandb=False,
    )
    t_recon = time.perf_counter() - t0
    m_dn = calculate_metrics(res_dn["s_phys"], s_gt)
    m_dn["time_s"] = t_denoise + t_recon
    m_dn["denoise_time_s"] = t_denoise
    results["denoised"] = {"metrics": m_dn, "s_phys": res_dn["s_phys"],
                           "loss_history": res_dn["loss_history"],
                           "denoise_info": denoise_info,
                           "d_denoised": d_denoised}

    # ── Oracle comparison (diagnostic) ───────────────────────────────────
    d_raw_np = sample["d_meas"].numpy().flatten()
    d_dn_np = d_denoised.numpy().flatten()
    mask_np = sample["mask"].numpy().flatten()

    if hasattr(s_gt, "detach"):
        s_gt_np = s_gt.detach().cpu().numpy().flatten()
    else:
        s_gt_np = np.asarray(s_gt).flatten()

    if isinstance(L_matrix, torch.Tensor):
        L_np = L_matrix.cpu().numpy() if not L_matrix.is_sparse else L_matrix.to_dense().cpu().numpy()
    else:
        L_np = np.asarray(L_matrix)

    d_oracle = (L_np @ s_gt_np).flatten()

    valid = mask_np > 0.5
    raw_residual = np.abs(d_raw_np[valid] - d_oracle[valid]).mean()
    dn_residual = np.abs(d_dn_np[valid] - d_oracle[valid]).mean()
    results["oracle_comparison"] = {
        "raw_vs_oracle_mae": float(raw_residual),
        "denoised_vs_oracle_mae": float(dn_residual),
        "improvement_pct": float((raw_residual - dn_residual) / raw_residual * 100)
        if raw_residual > 0 else 0.0,
    }

    log.info(
        f"  [idx={idx}] Raw: MAE={m_raw['MAE']:.2f} CNR={m_raw['CNR']:.3f} | "
        f"Denoised: MAE={m_dn['MAE']:.2f} CNR={m_dn['CNR']:.3f} | "
        f"d_residual: raw={raw_residual:.2e} dn={dn_residual:.2e}"
    )

    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _to_sos(s_flat):
    if hasattr(s_flat, "detach"):
        s_flat = s_flat.detach().cpu().numpy()
    s_flat = np.asarray(s_flat).flatten()
    s_clamped = np.clip(s_flat, 1.0 / 1800.0, 1.0 / 1200.0)
    return (1.0 / s_clamped).reshape(64, 64, order="F")


def plot_sample_comparison(sample, results, idx, out_dir):
    """Comparison plot: GT | L1 | L2 | Raw INR | Denoised INR.

    Each method gets: SoS map + Abs Error side by side.
    INR methods also get a convergence column.
    """
    from inr_sos.evaluation.metrics import calculate_metrics

    v_gt = _to_sos(sample["s_gt_raw"])

    # Build list of (label, v_rec, metrics, loss_history)
    rows = []

    # L1/L2 baselines from dataset
    for key, label in [("s_l1_recon", "L1 Baseline"), ("s_l2_recon", "L2 Baseline")]:
        if key in sample:
            v = _to_sos(sample[key])
            m = calculate_metrics(sample[key], sample["s_gt_raw"])
            rows.append((label, v, m, None))

    # Raw and Denoised INR
    for label, res_key in [("Raw INR", "raw"), ("Denoised INR", "denoised")]:
        res = results[res_key]
        v = _to_sos(res["s_phys"])
        rows.append((label, v, res["metrics"], res.get("loss_history")))

    n_rows = 1 + len(rows)  # GT row + method rows
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.2 * n_rows))
    fig.suptitle(f"Sample {idx} — Reconstruction Comparison", fontsize=13)

    # Row 0: Ground Truth
    im = axes[0, 0].imshow(v_gt, cmap="jet", vmin=1400, vmax=1600)
    axes[0, 0].set_title("Ground Truth (m/s)", fontsize=10)
    axes[0, 0].axis("off")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")

    for row, (label, v_rec, m, loss_hist) in enumerate(rows, 1):
        err = np.abs(v_gt - v_rec)

        # SoS map
        im = axes[row, 0].imshow(v_rec, cmap="jet", vmin=1400, vmax=1600)
        axes[row, 0].set_title(
            f"{label}  MAE={m['MAE']:.1f}  CNR={m['CNR']:.2f}", fontsize=10
        )
        axes[row, 0].axis("off")
        plt.colorbar(im, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # Error map
        im_e = axes[row, 1].imshow(err, cmap="hot", vmin=0, vmax=50)
        axes[row, 1].set_title("Abs. Error", fontsize=10)
        axes[row, 1].axis("off")
        plt.colorbar(im_e, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # Convergence (only for INR methods)
        if loss_hist:
            axes[row, 2].plot(loss_hist, color="#1f77b4", linewidth=1)
            axes[row, 2].set_yscale("log")
            axes[row, 2].set_title("Convergence", fontsize=10)
            axes[row, 2].set_xlabel("Iteration", fontsize=9)
            axes[row, 2].grid(True, which="both", ls="--", alpha=0.4)
            axes[row, 2].spines["top"].set_visible(False)
            axes[row, 2].spines["right"].set_visible(False)
        else:
            axes[row, 2].axis("off")

    plt.tight_layout()
    fp = out_dir / f"sample_{idx}_comparison.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fp


def plot_displacement_fields(sample, d_denoised, idx, out_dir):
    """Visualize raw vs denoised displacement for all 8 firing pairs.

    Layout: 8 rows (one per pair) × 4 columns:
      Col 0: Raw (valid only)
      Col 1: Denoised (ALL pixels — shows hole-filling + cone extrapolation)
      Col 2: Difference on valid pixels (denoised - raw)
      Col 3: Filled holes only (pixels that were NaN in raw, now predicted)
    Orientation: transducer at top, depth downward (order="F").
    """
    d_raw = sample["d_meas"].numpy().flatten()
    d_dn = d_denoised.numpy().flatten()
    mask = sample["mask"].numpy().flatten()

    n_pairs = 8
    fig, axes = plt.subplots(n_pairs, 4, figsize=(20, 4 * n_pairs))
    fig.suptitle(f"Sample {idx} — Displacement Fields (all pairs)", fontsize=14, y=1.0)

    col_titles = ["Raw (valid only)", "Denoised (all pixels)",
                   "Diff on valid", "Filled holes"]

    for k in range(n_pairs):
        start = k * 16384
        end = (k + 1) * 16384

        # Reshape with order="F": depth (z) as rows, lateral (x) as cols
        raw_2d = d_raw[start:end].reshape(128, 128, order="F")
        dn_2d = d_dn[start:end].reshape(128, 128, order="F")
        mask_2d = mask[start:end].reshape(128, 128, order="F")

        # Masked versions
        raw_masked = np.where(mask_2d > 0.5, raw_2d, np.nan)
        holes_only = np.where(mask_2d < 0.5, dn_2d, np.nan)  # NaN pixels filled by denoiser

        vmin = np.nanmin(raw_masked)
        vmax = np.nanmax(raw_masked)

        # Col 0: Raw valid
        im0 = axes[k, 0].imshow(raw_masked, cmap="viridis", vmin=vmin, vmax=vmax,
                                  origin="upper")
        axes[k, 0].set_ylabel(f"Pair {k}", fontsize=10, fontweight="bold")
        axes[k, 0].set_xticks([]); axes[k, 0].set_yticks([])
        if k == 0:
            axes[k, 0].set_title(col_titles[0], fontsize=11)
        plt.colorbar(im0, ax=axes[k, 0], fraction=0.046, pad=0.04)

        # Col 1: Denoised ALL pixels (unmasked — shows the full INR prediction)
        im1 = axes[k, 1].imshow(dn_2d, cmap="viridis", vmin=vmin, vmax=vmax,
                                  origin="upper")
        axes[k, 1].axis("off")
        if k == 0:
            axes[k, 1].set_title(col_titles[1], fontsize=11)
        plt.colorbar(im1, ax=axes[k, 1], fraction=0.046, pad=0.04)

        # Col 2: Difference on valid pixels
        diff = np.where(mask_2d > 0.5, dn_2d - raw_2d, np.nan)
        abs_max = np.nanmax(np.abs(diff)) if not np.all(np.isnan(diff)) else 1.0
        im2 = axes[k, 2].imshow(diff, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max,
                                  origin="upper")
        axes[k, 2].axis("off")
        if k == 0:
            axes[k, 2].set_title(col_titles[2], fontsize=11)
        plt.colorbar(im2, ax=axes[k, 2], fraction=0.046, pad=0.04)

        # Col 3: Filled holes (only the pixels that were NaN)
        im3 = axes[k, 3].imshow(holes_only, cmap="viridis", vmin=vmin, vmax=vmax,
                                  origin="upper")
        axes[k, 3].axis("off")
        n_holes = int((mask_2d < 0.5).sum())
        if k == 0:
            axes[k, 3].set_title(col_titles[3], fontsize=11)
        axes[k, 3].text(0.02, 0.98, f"{n_holes} px", transform=axes[k, 3].transAxes,
                         fontsize=8, va="top", color="white",
                         bbox=dict(boxstyle="round", fc="black", alpha=0.6))
        plt.colorbar(im3, ax=axes[k, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fp = out_dir / f"sample_{idx}_displacement_all_pairs.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fp


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage denoised reconstruction"
    )
    parser.add_argument("--dataset", default="kwave_geom")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--indices", nargs="+", type=int, default=None)
    # Denoiser config overrides
    parser.add_argument("--model_type", default="FourierMLP")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--mapping_size", type=int, default=32)
    parser.add_argument("--denoise_steps", type=int, default=500)
    parser.add_argument("--denoise_lr", type=float, default=1e-3)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / args.dataset / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    denoise_cfg = dict(DEFAULT_DENOISE_CFG)
    denoise_cfg.update({
        "model_type": args.model_type,
        "scale": args.scale,
        "hidden_features": args.hidden_features,
        "hidden_layers": args.hidden_layers,
        "mapping_size": args.mapping_size,
        "steps": args.denoise_steps,
        "lr": args.denoise_lr,
    })

    log.info("=" * 70)
    log.info("  Measurement-Domain INR Denoiser Experiment")
    log.info(f"  Dataset: {args.dataset}")
    log.info(f"  Denoiser: {denoise_cfg['model_type']}, scale={denoise_cfg['scale']}, "
             f"hidden={denoise_cfg['hidden_features']}x{denoise_cfg['hidden_layers']}")
    log.info(f"  Output: {run_dir}")
    log.info("=" * 70)

    # ── Load dataset ──────────────────────────────────────────────────────
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
    log.info(f"  Loaded {len(dataset)} samples")

    # ── Sample indices ────────────────────────────────────────────────────
    if args.indices:
        indices = args.indices
    else:
        n = args.n_samples if args.n_samples is not None else len(dataset)
        indices = list(range(min(n, len(dataset))))
    log.info(f"  Using {len(indices)} samples: {indices}")

    # ── Run experiment ────────────────────────────────────────────────────
    recon_cfg = make_recon_config()
    all_results = []

    for i, idx in enumerate(indices):
        log.info(f"\n── Sample {i+1}/{len(indices)} (idx={idx}) ──")
        sample = dataset[idx]
        results = run_single_sample(
            sample, dataset, grid, denoise_cfg, recon_cfg, idx
        )
        results["idx"] = idx
        all_results.append(results)

        # Save plots
        plot_sample_comparison(sample, results, idx, run_dir)
        # Displacement fields for all 8 pairs
        d_dn_full = results["denoised"].get("d_denoised")
        if d_dn_full is not None:
            plot_displacement_fields(sample, d_dn_full, idx, run_dir)

    # ── Summary table ─────────────────────────────────────────────────────
    log.info(f"\n{'='*80}")
    log.info(f"  DENOISER EXPERIMENT RESULTS — {len(indices)} samples")
    log.info(f"{'='*80}")
    log.info(f"  {'Condition':<15} {'MAE±std':>14} {'RMSE±std':>14} "
             f"{'SSIM±std':>14} {'CNR±std':>14}")
    log.info(f"  {'─'*75}")

    conditions = ["l1", "l2", "raw", "denoised"]
    for cond in conditions:
        if cond not in all_results[0]:
            continue
        maes = [r[cond]["metrics"]["MAE"] for r in all_results]
        rmses = [r[cond]["metrics"]["RMSE"] for r in all_results]
        ssims = [r[cond]["metrics"]["SSIM"] for r in all_results]
        cnrs = [r[cond]["metrics"]["CNR"] for r in all_results]
        label = {"l1": "L1 Baseline", "l2": "L2 Baseline",
                 "raw": "Raw INR", "denoised": "Denoised INR"}[cond]
        log.info(
            f"  {label:<15}"
            f"  {np.mean(maes):>6.2f}±{np.std(maes):<5.2f}"
            f"  {np.mean(rmses):>6.2f}±{np.std(rmses):<5.2f}"
            f"  {np.mean(ssims):>6.3f}±{np.std(ssims):<5.3f}"
            f"  {np.mean(cnrs):>6.3f}±{np.std(cnrs):<5.3f}"
        )

    # Oracle comparison
    raw_oracle = [r["oracle_comparison"]["raw_vs_oracle_mae"] for r in all_results]
    dn_oracle = [r["oracle_comparison"]["denoised_vs_oracle_mae"] for r in all_results]
    log.info(f"\n  Oracle displacement MAE (d vs L@s_true):")
    log.info(f"    Raw:      {np.mean(raw_oracle):.2e} ± {np.std(raw_oracle):.2e}")
    log.info(f"    Denoised: {np.mean(dn_oracle):.2e} ± {np.std(dn_oracle):.2e}")
    log.info(f"{'='*80}")

    # ── Save results ──────────────────────────────────────────────────────
    results_json = {
        "timestamp": timestamp,
        "dataset": args.dataset,
        "denoise_cfg": denoise_cfg,
        "n_samples": len(indices),
        "indices": indices,
        "per_sample": [
            {
                "idx": r["idx"],
                **{k: r[k]["metrics"] for k in ["l1", "l2"] if k in r},
                "raw": r["raw"]["metrics"],
                "denoised": r["denoised"]["metrics"],
                "oracle_comparison": r["oracle_comparison"],
            }
            for r in all_results
        ],
    }
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    log.info(f"\n  Results saved → {results_path}")

    # ── W&B logging ───────────────────────────────────────────────────────
    if not args.no_wandb:
        import wandb

        wandb.init(
            project="INR-SoS-Recon",
            name=f"denoiser_{denoise_cfg['model_type']}_s{denoise_cfg['scale']}_{timestamp}",
            group="Denoiser-Experiment",
            tags=["denoiser", args.dataset, denoise_cfg["model_type"]],
            config={"denoise_cfg": denoise_cfg, "recon_cfg": recon_cfg.to_dict(),
                    "n_samples": len(indices)},
        )

        columns = ["Condition", "MAE_mean", "MAE_std", "CNR_mean", "CNR_std",
                    "SSIM_mean", "SSIM_std"]
        data = []
        for cond in ["l1", "l2", "raw", "denoised"]:
            if cond not in all_results[0]:
                continue
            maes = [r[cond]["metrics"]["MAE"] for r in all_results]
            cnrs = [r[cond]["metrics"]["CNR"] for r in all_results]
            ssims = [r[cond]["metrics"]["SSIM"] for r in all_results]
            data.append([cond, np.mean(maes), np.std(maes),
                         np.mean(cnrs), np.std(cnrs),
                         np.mean(ssims), np.std(ssims)])
        wandb.log({"comparison_table": wandb.Table(columns=columns, data=data)})

        # Log plots
        for fp in run_dir.glob("*.png"):
            wandb.log({fp.stem: wandb.Image(str(fp))})

        wandb.finish()

    log.info(f"\n  Experiment complete. All outputs in {run_dir}")


if __name__ == "__main__":
    main()
