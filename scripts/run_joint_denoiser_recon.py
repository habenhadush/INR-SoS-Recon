#!/usr/bin/env python3
"""
run_joint_denoiser_recon.py
---------------------------
Experiment 11: Joint denoiser + reconstructor (staged training).

Compares: L1 | L2 | Raw INR | Joint (staged, with adaptive λ strategies)

Usage:
    # Fixed λ (default)
    python scripts/run_joint_denoiser_recon.py --n_samples 5 --lambda_fit 0.07

    # Cosine decay
    python scripts/run_joint_denoiser_recon.py --n_samples 5 \
        --lambda_strategy cosine --lambda_max 1.0 --lambda_min 0.01

    # Loss-ratio balanced
    python scripts/run_joint_denoiser_recon.py --n_samples 5 \
        --lambda_strategy balanced --target_ratio 1.0

    # Residual-normalized (sweep α)
    python scripts/run_joint_denoiser_recon.py --n_samples 5 \
        --lambda_strategy residual --alpha 0.1 0.5 1.0
"""

import argparse
import copy
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
from inr_sos.models.mlp import ReluMLP
from inr_sos.training.engines import optimize_full_forward_operator
from inr_sos.training.denoise_engine import DEFAULT_DENOISE_CFG
from inr_sos.training.joint_engine import optimize_joint
from inr_sos.visualization.report_figures import (
    plot_method_grid,
    plot_metrics_comparison,
)

SCRIPTS_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPTS_DIR / "data" / "joint_denoiser_recon"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("joint_exp")


def make_recon_config():
    return ExperimentConfig(
        project_name="INR-SoS-Recon",
        experiment_group="Joint-Denoiser-Recon",
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


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_sample_comparison(sample, all_methods, idx, out_dir):
    """Multi-row comparison: GT + each method with SoS, error, convergence."""
    v_gt = _to_sos(sample["s_gt_raw"])

    n_rows = 1 + len(all_methods)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.2 * n_rows))
    fig.suptitle(f"Sample {idx} — Joint Denoiser+Recon Comparison", fontsize=13)

    # Row 0: GT
    im = axes[0, 0].imshow(v_gt, cmap="jet", vmin=1400, vmax=1600)
    axes[0, 0].set_title("Ground Truth (m/s)", fontsize=10)
    axes[0, 0].axis("off")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")

    for row, (label, res) in enumerate(all_methods, 1):
        v_rec = _to_sos(res["s_phys"])
        err = np.abs(v_gt - v_rec)
        m = res["metrics"]

        im = axes[row, 0].imshow(v_rec, cmap="jet", vmin=1400, vmax=1600)
        axes[row, 0].set_title(
            f"{label}  MAE={m['MAE']:.1f}  CNR={m['CNR']:.2f}", fontsize=10
        )
        axes[row, 0].axis("off")
        plt.colorbar(im, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im_e = axes[row, 1].imshow(err, cmap="hot", vmin=0, vmax=50)
        axes[row, 1].set_title("Abs. Error", fontsize=10)
        axes[row, 1].axis("off")
        plt.colorbar(im_e, ax=axes[row, 1], fraction=0.046, pad=0.04)

        loss_hist = res.get("loss_history")
        if loss_hist:
            axes[row, 2].plot(loss_hist, color="#1f77b4", linewidth=1)
            axes[row, 2].set_yscale("log")
            axes[row, 2].set_title("Convergence", fontsize=10)
            axes[row, 2].set_xlabel("Iteration", fontsize=9)
            axes[row, 2].grid(True, which="both", ls="--", alpha=0.4)
            axes[row, 2].spines["top"].set_visible(False)
            axes[row, 2].spines["right"].set_visible(False)

            # If we have separate recon/fit losses, overlay them
            if "recon_losses" in res and "fit_losses" in res:
                ax2 = axes[row, 2]
                ax2.clear()
                ax2.plot(res["recon_losses"], color="#1f77b4", linewidth=1,
                         label="L_recon", alpha=0.8)
                ax2.plot(res["fit_losses"], color="#ff7f0e", linewidth=1,
                         label="L_fit", alpha=0.8)
                ax2.set_yscale("log")
                ax2.set_title("Convergence", fontsize=10)
                ax2.set_xlabel("Iteration", fontsize=9)
                ax2.legend(fontsize=7)
                ax2.grid(True, which="both", ls="--", alpha=0.4)
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)

                # Overlay λ trajectory on secondary y-axis
                lam_traj = res.get("lambda_trajectory")
                if lam_traj and len(set(lam_traj)) > 1:
                    ax_lam = ax2.twinx()
                    ax_lam.plot(lam_traj, color="#2ca02c", linewidth=1,
                                label="λ(t)", alpha=0.6, linestyle="--")
                    ax_lam.set_ylabel("λ", fontsize=8, color="#2ca02c")
                    ax_lam.tick_params(axis="y", labelcolor="#2ca02c", labelsize=7)
        else:
            axes[row, 2].axis("off")

    plt.tight_layout()
    fp = out_dir / f"sample_{idx}_comparison.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fp


def plot_summary_bars(all_results, out_dir):
    """Bar chart comparing all methods across samples."""
    methods = list(all_results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(methods) * 2), 6))
    fig.suptitle("Joint Denoiser+Recon — Method Comparison", fontsize=13)

    mae_means, mae_stds, cnr_means, cnr_stds = [], [], [], []
    for method in methods:
        maes = [r["metrics"]["MAE"] for r in all_results[method]]
        cnrs = [r["metrics"]["CNR"] for r in all_results[method]]
        mae_means.append(np.mean(maes))
        mae_stds.append(np.std(maes))
        cnr_means.append(np.mean(cnrs))
        cnr_stds.append(np.std(cnrs))

    x = np.arange(len(methods))
    colors = ["#999999" if m in ("L1", "L2") else
              "#2ca02c" if m == "Raw INR" else
              "#1f77b4" for m in methods]

    ax1.bar(x, mae_means, yerr=mae_stds, color=colors, capsize=4,
            edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("MAE (m/s)")
    ax1.set_title("MAE (lower is better)")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, cnr_means, yerr=cnr_stds, color=colors, capsize=4,
            edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("CNR")
    ax2.set_title("CNR (higher is better)")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fp = out_dir / "summary_bars.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Joint denoiser+reconstructor (Exp 11)")
    parser.add_argument("--dataset", default="kwave_geom")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--indices", nargs="+", type=int, default=None)
    parser.add_argument("--lambda_fit", nargs="+", type=float, default=[0.1])
    parser.add_argument("--joint_steps", type=int, default=2000)
    parser.add_argument("--pretrain_dn_steps", type=int, default=300)
    parser.add_argument("--pretrain_rc_steps", type=int, default=500)
    # Lambda strategy
    parser.add_argument("--lambda_strategy", default="fixed",
                        choices=["fixed", "cosine", "balanced", "residual"],
                        help="Adaptive λ strategy for Stage 3")
    parser.add_argument("--lambda_max", type=float, default=1.0,
                        help="Cosine/balanced: starting λ (cosine) or upper bound (balanced)")
    parser.add_argument("--lambda_min", type=float, default=0.01,
                        help="Cosine/balanced: ending λ (cosine) or lower bound (balanced)")
    parser.add_argument("--target_ratio", type=float, default=1.0,
                        help="Balanced: target L_recon / L_fit ratio")
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.5],
                        help="Residual: scaling factor(s) for per-sample λ")
    # Denoiser config
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--hidden_features", type=int, default=128)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--report_plots",
        action="store_true",
        help="Generate thesis-quality comparison figures (SVG + PNG) after the run.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / args.dataset / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    denoise_cfg = dict(DEFAULT_DENOISE_CFG)
    denoise_cfg.update({
        "scale": args.scale,
        "hidden_features": args.hidden_features,
    })

    log.info("=" * 70)
    log.info("  Experiment 11: Joint Denoiser + Reconstructor (staged)")
    log.info(f"  Dataset: {args.dataset}")
    log.info(f"  λ strategy: {args.lambda_strategy}")
    if args.lambda_strategy == "fixed":
        log.info(f"  lambda_fit: {args.lambda_fit}")
    elif args.lambda_strategy == "cosine":
        log.info(f"  λ_max={args.lambda_max}, λ_min={args.lambda_min}")
    elif args.lambda_strategy == "balanced":
        log.info(f"  target_ratio={args.target_ratio}, λ_init={args.lambda_fit}")
    elif args.lambda_strategy == "residual":
        log.info(f"  α={args.alpha}")
    log.info(f"  Denoiser: scale={denoise_cfg['scale']}, h={denoise_cfg['hidden_features']}")
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

    if args.indices:
        indices = args.indices
    else:
        n = args.n_samples if args.n_samples is not None else len(dataset)
        indices = list(range(min(n, len(dataset))))
    log.info(f"  Samples: {indices}")

    recon_cfg = make_recon_config()
    if hasattr(dataset, "pix2time") and dataset.pix2time is not None:
        recon_cfg.time_scale = 1.0 / dataset.pix2time
    elif ds_cfg.get("pix2time") is not None:
        recon_cfg.time_scale = 1.0 / float(ds_cfg["pix2time"])
        log.info(f"  time_scale from yaml pix2time: {recon_cfg.time_scale:.4e}")

    samples = [dataset[idx] for idx in indices]

    # ── All results: {method_name: [per_sample_results]} ─────────────────
    all_results = {}

    # L1/L2 baselines
    for key, label in [("s_l1_recon", "L1"), ("s_l2_recon", "L2")]:
        if key in samples[0]:
            all_results[label] = []
            for sample in samples:
                m = calculate_metrics(sample[key], sample["s_gt_raw"])
                all_results[label].append({"metrics": m, "s_phys": sample[key]})

    # Raw INR baseline
    log.info("\n── Raw INR baseline ──")
    all_results["Raw INR"] = []
    for i, (idx, sample) in enumerate(zip(indices, samples)):
        log.info(f"  Raw INR: sample {i+1}/{len(indices)} (idx={idx})")
        model = build_recon_model(recon_cfg)
        res = optimize_full_forward_operator(
            sample=sample, L_matrix=dataset.L_matrix, model=model,
            label="ReluMLP_raw", config=copy.deepcopy(recon_cfg), use_wandb=False,
        )
        m = calculate_metrics(res["s_phys"], sample["s_gt_raw"])
        all_results["Raw INR"].append({
            "metrics": m, "s_phys": res["s_phys"],
            "loss_history": res["loss_history"],
        })

    # ── Build joint configs based on lambda strategy ───────────────────────
    joint_configs = []  # list of (method_label, lambda_fit, lambda_cfg)

    if args.lambda_strategy == "fixed":
        for lf in args.lambda_fit:
            joint_configs.append((f"Joint(fix,λ={lf})", lf, {}))

    elif args.lambda_strategy == "cosine":
        lam_cfg = {"lambda_max": args.lambda_max, "lambda_min": args.lambda_min}
        label = f"Joint(cos,{args.lambda_max}→{args.lambda_min})"
        joint_configs.append((label, args.lambda_max, lam_cfg))

    elif args.lambda_strategy == "balanced":
        for lf in args.lambda_fit:
            lam_cfg = {
                "target_ratio": args.target_ratio,
                "lambda_min": args.lambda_min,
                "lambda_max": args.lambda_max,
            }
            label = f"Joint(bal,r={args.target_ratio},λ₀={lf})"
            joint_configs.append((label, lf, lam_cfg))

    elif args.lambda_strategy == "residual":
        for a in args.alpha:
            lam_cfg = {"alpha": a}
            label = f"Joint(res,α={a})"
            joint_configs.append((label, 0.1, lam_cfg))  # base λ unused for residual

    # ── Joint optimization ───────────────────────────────────────────────
    for method_label, lf, lam_cfg in joint_configs:
        log.info(f"\n── {method_label} ──")
        all_results[method_label] = []

        for i, (idx, sample) in enumerate(zip(indices, samples)):
            log.info(f"  {method_label}: sample {i+1}/{len(indices)} (idx={idx})")
            model = build_recon_model(recon_cfg)
            t0 = time.perf_counter()
            res = optimize_joint(
                sample=sample,
                L_matrix=dataset.L_matrix,
                grid=grid,
                recon_model=model,
                recon_config=copy.deepcopy(recon_cfg),
                denoise_cfg=denoise_cfg,
                lambda_fit=lf,
                mode="staged",
                lambda_strategy=args.lambda_strategy,
                lambda_cfg=lam_cfg,
                pretrain_denoiser_steps=args.pretrain_dn_steps,
                pretrain_recon_steps=args.pretrain_rc_steps,
                joint_steps=args.joint_steps,
                use_wandb=False,
                label=method_label,
            )
            elapsed = time.perf_counter() - t0
            m = calculate_metrics(res["s_phys"], sample["s_gt_raw"])
            m["time_s"] = elapsed

            raw_m = all_results["Raw INR"][i]["metrics"]
            log.info(f"    Raw MAE={raw_m['MAE']:.1f} → Joint MAE={m['MAE']:.1f}")

            all_results[method_label].append({
                "metrics": m, "s_phys": res["s_phys"],
                "loss_history": res["loss_history"],
                "recon_losses": res.get("recon_losses"),
                "fit_losses": res.get("fit_losses"),
                "lambda_trajectory": res.get("lambda_trajectory"),
            })

    # ── Summary table ────────────────────────────────────────────────────
    log.info(f"\n{'='*90}")
    log.info(f"  JOINT DENOISER+RECON RESULTS — {len(indices)} samples")
    log.info(f"{'='*90}")
    log.info(f"  {'Method':<30} {'MAE±std':>14} {'RMSE±std':>14} "
             f"{'SSIM±std':>14} {'CNR±std':>14}")
    log.info(f"  {'─'*88}")

    for method, per_sample in all_results.items():
        maes = [r["metrics"]["MAE"] for r in per_sample]
        rmses = [r["metrics"]["RMSE"] for r in per_sample]
        ssims = [r["metrics"]["SSIM"] for r in per_sample]
        cnrs = [r["metrics"]["CNR"] for r in per_sample]
        log.info(
            f"  {method:<30}"
            f"  {np.mean(maes):>6.2f}±{np.std(maes):<5.2f}"
            f"  {np.mean(rmses):>6.2f}±{np.std(rmses):<5.2f}"
            f"  {np.mean(ssims):>6.3f}±{np.std(ssims):<5.3f}"
            f"  {np.mean(cnrs):>6.3f}±{np.std(cnrs):<5.3f}"
        )
    log.info(f"{'='*90}")

    # ── Thesis-quality report figures (optional, staged results only) ────
    if args.report_plots:
        # Filter to baselines + staged configs only (end/alt not promising)
        report_results = {
            method: per_sample
            for method, per_sample in all_results.items()
            if method in ("L1", "L2", "Raw INR") or method.startswith("Joint")
        }
        if not any(m.startswith("Joint") for m in report_results):
            log.warning("  No staged results to plot — skipping report figures")
        else:
            log.info("\n  Generating report figures (staged only) ...")
            grid_path_svg = run_dir / "report_comparison.svg"
            metrics_path_svg = run_dir / "report_metrics.svg"
            try:
                # Dataset title for the grid plot
                ds_titles = {
                    "kwave_geom": "GeomSet",
                    "kwave_blob": "BlobSet",
                    "inverse_crime": "InverseCrime",
                }
                plot_method_grid(
                    results=report_results,
                    samples=samples,
                    save_path=grid_path_svg,
                    dataset_title=ds_titles.get(args.dataset, args.dataset),
                    show=False,
                    png_fallback=True,
                )
                plot_metrics_comparison(
                    results=report_results,
                    save_path=metrics_path_svg,
                    show=False,
                    png_fallback=True,
                )
                log.info(f"  Report grid    -> {grid_path_svg}")
                log.info(f"  Report metrics -> {metrics_path_svg}")
            except Exception as exc:
                log.warning(f"  Report figure generation failed: {exc}")

    # ── Per-sample comparison plots ──────────────────────────────────────
    for i, idx in enumerate(indices):
        methods_for_plot = [
            (method, per_sample[i])
            for method, per_sample in all_results.items()
        ]
        plot_sample_comparison(samples[i], methods_for_plot, idx, run_dir)

    plot_summary_bars(all_results, run_dir)

    # ── Save results JSON ────────────────────────────────────────────────
    results_json = {
        "timestamp": timestamp,
        "dataset": args.dataset,
        "mode": "staged",
        "lambda_strategy": args.lambda_strategy,
        "lambda_fit_values": args.lambda_fit,
        "denoise_cfg": denoise_cfg,
        "n_samples": len(indices),
        "indices": indices,
        "methods": {},
    }
    for method, per_sample in all_results.items():
        results_json["methods"][method] = [r["metrics"] for r in per_sample]

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    log.info(f"\n  Results saved → {results_path}")
    log.info(f"  Experiment complete. All outputs in {run_dir}")


if __name__ == "__main__":
    main()
