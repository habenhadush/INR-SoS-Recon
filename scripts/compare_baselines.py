#!/usr/bin/env python3
"""
compare_baselines.py
--------------------
Compares the best INR config against L1 and L2 regularization baselines
from Bezek et al.'s dataset.

Two modes:

  --use_registry          (fast, no GPU)
      Reads INR results from sweep_registry.json (the 20 holdout samples
      already evaluated in run_best.py). Computes L1/L2 on the same indices.
      Use this for quick iteration.

  --fresh_n N             (full GPU run, strongest result for paper)
      Samples N indices disjoint from ALL previously used indices
      (sweep indices + holdout validation indices).
      Runs the best INR config fresh on those N samples.
      Computes L1/L2 on the same N samples.
      Use this for the final thesis/paper comparison.

Usage:
    # Quick mode — no GPU needed (~30 seconds)
    python scripts/compare_baselines.py --sweep_id hqt6bwmp --use_registry

    # Fresh mode — strongest result for paper (~N * 15min on GPU)
    python scripts/compare_baselines.py --sweep_id hqt6bwmp --fresh_n 50

    # Fresh mode, no W&B
    python scripts/compare_baselines.py --sweep_id hqt6bwmp --fresh_n 50 --no_wandb
"""

import argparse
import copy
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import wandb

import inr_sos
from inr_sos import DATA_DIR
from inr_sos.utils.data import USDataset
from inr_sos.utils.config import ExperimentConfig
from inr_sos.evaluation.metrics import calculate_metrics
from scripts.run_sweep import load_dataset_config

SCRIPTS_DIR   = Path(__file__).parent
REGISTRY_FILE = SCRIPTS_DIR / "sweep_registry.json"
LOG_DIR       = SCRIPTS_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


# Logging — named logger avoids basicConfig no-op if inr_sos configures root
def setup_logging(sweep_id: str, mode: str):
    log_path = LOG_DIR / f"comparison_{mode}_{sweep_id}.log"
    logger   = logging.getLogger("compare_baselines")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh  = logging.FileHandler(log_path)
        sh  = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return log_path, logger


# Registry helpers
def load_registry(sweep_id: str):
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    entry = next((e for e in registry if e["sweep_id"].startswith(sweep_id)), None)
    if entry is None:
        raise ValueError(f"Sweep {sweep_id} not found in registry.")
    return registry, entry


def save_registry(registry: list):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def get_used_indices(entry: dict) -> set:
    """All indices ever used for this sweep — excluded from fresh sampling."""
    used = set(entry.get("indices", []))
    used |= set(entry.get("validation", {}).get("holdout_indices", []))
    return used


def get_best_config(entry: dict) -> dict:
    """
    Returns rank4 Full_Matrix/ReluMLP — identified holdout winner.
    Falls back to best_on_holdout if rank4 not present.
    """
    top_k = entry.get("validation", {}).get("top_k_results", [])
    rank4 = next((r for r in top_k if r["rank"] == 4), None)
    return rank4 if rank4 else entry["validation"]["best_on_holdout"]


# Metrics
def compute_baseline_metrics(recons: np.ndarray,
                              gt: np.ndarray,
                              indices: list,
                              label: str,
                              log) -> dict:
    """
    Compute metrics for pre-computed reconstructions.
    recons / gt: (64, 64, N_total) — MATLAB axis order, sample is last dim.
    """
    all_mae, all_rmse, all_ssim, all_cnr = [], [], [], []

    for idx in indices:
        s_phys_pred = np.asarray(recons[:, :, idx].flatten(), dtype=np.float32)
        s_gt_raw    = np.asarray(gt[:, :, idx].flatten(),    dtype=np.float32)

        m = calculate_metrics(
            s_phys_pred=s_phys_pred,
            s_gt_raw=s_gt_raw,
            grid_shape=(64, 64),
        )
        all_mae.append(m["MAE"])
        all_rmse.append(m["RMSE"])
        all_ssim.append(m["SSIM"])
        all_cnr.append(m["CNR"])

    result = _aggregate(label, all_mae, all_rmse, all_ssim, all_cnr, len(indices))
    log.info(f"  {label:<28}  "
             f"MAE={result['MAE_mean']:.3f}±{result['MAE_std']:.3f}  "
             f"RMSE={result['RMSE_mean']:.3f}±{result['RMSE_std']:.3f}  "
             f"SSIM={result['SSIM_mean']:.3f}±{result['SSIM_std']:.3f}  "
             f"CNR={result['CNR_mean']:.3f}±{result['CNR_std']:.3f}")
    return result


def _aggregate(label, mae, rmse, ssim, cnr, n) -> dict:
    return {
        "method":    label,
        "MAE_mean":  float(np.mean(mae)),
        "MAE_std":   float(np.std(mae)),
        "RMSE_mean": float(np.mean(rmse)),
        "RMSE_std":  float(np.std(rmse)),
        "SSIM_mean": float(np.mean(ssim)),
        "SSIM_std":  float(np.std(ssim)),
        "CNR_mean":  float(np.mean(cnr)),
        "CNR_std":   float(np.std(cnr)),
        "n_samples": n,
    }


# INR fresh run
def run_inr_fresh(best_cfg_entry: dict,
                  dataset: USDataset,
                  indices: list,
                  base_config: ExperimentConfig,
                  log) -> dict:
    """Run the best INR config fresh on given indices. Returns metric dict."""
    from inr_sos.models.mlp import FourierMLP, ReluMLP
    from inr_sos.models.siren import SirenMLP
    from inr_sos.training.engines import (
        optimize_full_forward_operator,
        optimize_sequential_views,
        optimize_stochastic_ray_batching,
    )

    engine_map = {
        "Full_Matrix":    optimize_full_forward_operator,
        "Sequential_SGD": optimize_sequential_views,
        "Ray_Batching":   optimize_stochastic_ray_batching,
    }
    model_map = {
        "FourierMLP": FourierMLP,
        "ReluMLP":    ReluMLP,
        "SirenMLP":   SirenMLP,
    }

    method     = best_cfg_entry["method"]
    model_type = best_cfg_entry["model_type"]
    hparams    = best_cfg_entry["hyperparams"]

    log.info(f"\n  INR config: {method} / {model_type}")
    log.info(f"  Running on {len(indices)} fresh samples ...")

    cfg = copy.deepcopy(base_config)
    cfg.model_type      = model_type
    cfg.hidden_features = hparams.get("hidden_features", cfg.hidden_features)
    cfg.hidden_layers   = hparams.get("hidden_layers",   cfg.hidden_layers)
    cfg.lr              = hparams.get("lr",              cfg.lr)
    cfg.steps           = hparams.get("steps",           cfg.steps)
    cfg.mapping_size    = hparams.get("mapping_size",    cfg.mapping_size)
    cfg.scale           = hparams.get("scale",           cfg.scale)
    cfg.omega           = hparams.get("omega",           cfg.omega)
    cfg.tv_weight       = hparams.get("tv_weight",       cfg.tv_weight)
    cfg.reg_weight      = hparams.get("reg_weight",      cfg.reg_weight)
    if method == "Ray_Batching":
        cfg.epochs     = hparams.get("epochs",     cfg.epochs)
        cfg.batch_size = hparams.get("batch_size", cfg.batch_size)

    engine_fn = engine_map[method]
    model_cls = model_map[model_type]
    all_mae, all_rmse, all_ssim, all_cnr = [], [], [], []

    # Initialise W&B run now so per-sample metrics stream live
    wb_run = wandb.init(
        project=cfg.project_name,
        name=f"fresh_comparison_{method}_{model_type}",
        group="baseline_comparison",
        tags=["fresh_run", "comparison", method, model_type],
        config={
            "method":     method,
            "model_type": model_type,
            "n_samples":  len(indices),
            **hparams,
        },
        reinit=True,
    )

    for i, idx in enumerate(indices):
        log.info(f"    [{i+1:>3}/{len(indices)}]  sample idx={idx}")

        sample = dataset[idx]
        kwargs = dict(
            in_features=cfg.in_features,
            hidden_features=cfg.hidden_features,
            hidden_layers=cfg.hidden_layers,
            mapping_size=cfg.mapping_size,
        )
        if model_type == "FourierMLP":
            kwargs["scale"] = cfg.scale
        elif model_type == "SirenMLP":
            kwargs["omega"] = cfg.omega
        model = model_cls(**kwargs)

        result = engine_fn(
            sample=sample,
            L_matrix=dataset.L_matrix,
            model=model,
            label=model_type,
            config=cfg,
            use_wandb=False,
        )

        m = calculate_metrics(
            s_phys_pred=result["s_phys"],
            s_gt_raw=sample["s_gt_raw"],
            grid_shape=(64, 64),
        )
        all_mae.append(m["MAE"])
        all_rmse.append(m["RMSE"])
        all_ssim.append(m["SSIM"])
        all_cnr.append(m["CNR"])
        log.info(f"           MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  "
                 f"SSIM={m['SSIM']:.3f}  CNR={m['CNR']:.3f}")

        # Log per-sample metrics live — step=i is monotonically increasing
        wandb.log({
            "sample/MAE":  m["MAE"],
            "sample/RMSE": m["RMSE"],
            "sample/SSIM": m["SSIM"],
            "sample/CNR":  m["CNR"],
            "sample/idx":  idx,
            "running/MAE_mean":  float(np.mean(all_mae)),
            "running/RMSE_mean": float(np.mean(all_rmse)),
            "running/SSIM_mean": float(np.mean(all_ssim)),
            "running/CNR_mean":  float(np.mean(all_cnr)),
        }, step=i)

    label = f"INR_{method}_{model_type}"
    result = _aggregate(label, all_mae, all_rmse, all_ssim, all_cnr, len(indices))
    log.info(f"\n  INR aggregate → "
             f"MAE={result['MAE_mean']:.3f}±{result['MAE_std']:.3f}  "
             f"RMSE={result['RMSE_mean']:.3f}±{result['RMSE_std']:.3f}  "
             f"SSIM={result['SSIM_mean']:.3f}±{result['SSIM_std']:.3f}  "
             f"CNR={result['CNR_mean']:.3f}±{result['CNR_std']:.3f}")
    wandb.log({
        "final/MAE_mean":  result["MAE_mean"],
        "final/MAE_std":   result["MAE_std"],
        "final/RMSE_mean": result["RMSE_mean"],
        "final/RMSE_std":  result["RMSE_std"],
        "final/SSIM_mean": result["SSIM_mean"],
        "final/SSIM_std":  result["SSIM_std"],
        "final/CNR_mean":  result["CNR_mean"],
        "final/CNR_std":   result["CNR_std"],
    })
    wandb.finish()
    return result

# Reporting
def print_comparison_table(all_results: list, log):
    n = all_results[0]["n_samples"]
    log.info(f"\n{'═'*95}")
    log.info(f"  FINAL COMPARISON — {n} samples  (sorted by RMSE)")
    log.info(f"{'═'*95}")
    log.info(f"  {'Method':<32} {'MAE ± std':>14} {'RMSE ± std':>14} "
             f"{'SSIM ± std':>14} {'CNR ± std':>12}")
    log.info(f"  {'─'*93}")

    for r in sorted(all_results, key=lambda x: x["RMSE_mean"]):
        tag = "  ← INR (ours)" if "INR" in r["method"] else ""
        log.info(
            f"  {r['method']:<32}"
            f"  {r['MAE_mean']:>5.3f} ± {r['MAE_std']:<5.3f}"
            f"  {r['RMSE_mean']:>5.3f} ± {r['RMSE_std']:<5.3f}"
            f"  {r['SSIM_mean']:>5.3f} ± {r['SSIM_std']:<5.3f}"
            f"  {r['CNR_mean']:>5.3f} ± {r['CNR_std']:<5.3f}"
            f"{tag}"
        )

    log.info(f"{'═'*95}")

    inr = next((r for r in all_results if "INR" in r["method"]), None)
    l2  = next((r for r in all_results if r["method"] == "L2_regularization"), None)
    l1  = next((r for r in all_results if r["method"] == "L1_regularization"), None)

    for baseline, bl in [("L2 (classical)", l2), ("L1 (sparse)", l1)]:
        if inr and bl:
            log.info(f"\n  INR vs {baseline}:")
            for metric, key, higher_is_better in [
                ("RMSE", "RMSE_mean", False),
                ("SSIM", "SSIM_mean", True),
                ("CNR",  "CNR_mean",  True),
            ]:
                if higher_is_better:
                    pct = 100 * (inr[key] - bl[key]) / max(bl[key], 1e-9)
                else:
                    pct = 100 * (bl[key] - inr[key]) / max(bl[key], 1e-9)
                sign = "improvement ↑" if pct > 0 else "regression ↓"
                log.info(f"    {metric}: {bl[key]:.3f} → {inr[key]:.3f}  "
                         f"({pct:+.1f}% {sign})")


def log_to_wandb(all_results: list, entry: dict, indices: list, mode: str):
    run = wandb.init(
        project=entry["project"],
        entity=entry["entity"],
        name=f"baseline_comparison_{mode}",
        group="baseline_comparison",
        tags=["comparison", "baseline", "l1", "l2", "inr", mode],
        config={
            "sweep_id": entry["sweep_id"],
            "mode":     mode,
            "indices":  indices,
            "n":        len(indices),
        }
    )

    # Sortable W&B table
    table = wandb.Table(
        columns=["Method", "MAE_mean", "MAE_std",
                 "RMSE_mean", "RMSE_std",
                 "SSIM_mean", "SSIM_std",
                 "CNR_mean",  "CNR_std", "n_samples"],
        data=[
            [r["method"],
             r["MAE_mean"],  r["MAE_std"],
             r["RMSE_mean"], r["RMSE_std"],
             r["SSIM_mean"], r["SSIM_std"],
             r["CNR_mean"],  r["CNR_std"],
             r["n_samples"]]
            for r in sorted(all_results, key=lambda x: x["RMSE_mean"])
        ]
    )
    wandb.log({"comparison_table": table})

    # Flat metrics for bar charts
    for r in all_results:
        key = r["method"].replace(" ", "_").replace("/", "_")
        wandb.log({
            f"{key}/MAE_mean":  r["MAE_mean"],
            f"{key}/RMSE_mean": r["RMSE_mean"],
            f"{key}/SSIM_mean": r["SSIM_mean"],
            f"{key}/CNR_mean":  r["CNR_mean"],
        })

    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                    help="Dataset key from datasets.yaml (default: uses 'active' field)")
    parser.add_argument("--sweep_id",     required=True)
    parser.add_argument("--use_registry", action="store_true",
                        help="Use INR results already in registry (no GPU, fast)")
    parser.add_argument("--fresh_n",      type=int, default=None,
                        help="Sample N fresh disjoint indices and re-run INR (GPU)")
    parser.add_argument("--no_wandb",     action="store_true")
    args = parser.parse_args()

    if not args.use_registry and args.fresh_n is None:
        print("ERROR: specify either --use_registry or --fresh_n N")
        sys.exit(1)
    if args.use_registry and args.fresh_n is not None:
        print("ERROR: --use_registry and --fresh_n are mutually exclusive")
        sys.exit(1)

    mode = "registry" if args.use_registry else f"fresh_{args.fresh_n}"
    log_path, log = setup_logging(args.sweep_id, mode)

    log.info("=" * 65)
    log.info(f"  Baseline comparison")
    log.info(f"  Sweep   : {args.sweep_id}")
    log.info(f"  Mode    : {mode}")
    log.info(f"  Log     : {log_path}")
    log.info("=" * 65)

    registry, entry = load_registry(args.sweep_id)

    # ── Load analytical baseline data (fast, CPU) ─────────────────────────
    analytical_path = DATA_DIR + "/DL-based-SoS/train_IC_10k_l2rec_l1rec_imcon.mat"
    log.info("Loading L1/L2 baseline reconstructions ...")
    analytical = inr_sos.load_mat(analytical_path)
    l2_recons  = analytical["all_slowness_recons_l2"]   # (64, 64, 10000)
    l1_recons  = analytical["all_slowness_recons_l1"]   # (64, 64, 10000)
    gt         = analytical["imgs_gt"]                   # (64, 64, 10000)

    # ── Determine indices and INR results ─────────────────────────────────
    if args.use_registry:
        if "validation" not in entry:
            log.error("No validation found in registry. Run run_best.py first.")
            sys.exit(1)

        indices = entry["validation"]["holdout_indices"]
        log.info(f"Registry mode: using {len(indices)} holdout indices from registry")

        rank4 = next(
            (r for r in entry["validation"]["top_k_results"] if r["rank"] == 4),
            entry["validation"]["best_on_holdout"]
        )
        inr_results = {
            "method":    f"INR_{rank4['method']}_{rank4['model_type']}",
            "MAE_mean":  rank4["holdout_MAE_mean"],
            "MAE_std":   rank4["holdout_MAE_std"],
            "RMSE_mean": rank4["holdout_RMSE_mean"],
            "RMSE_std":  rank4["holdout_RMSE_std"],
            "SSIM_mean": rank4["holdout_SSIM_mean"],
            "SSIM_std":  rank4["holdout_SSIM_std"],
            "CNR_mean":  rank4["holdout_CNR_mean"],
            "CNR_std":   rank4["holdout_CNR_std"],
            "n_samples": len(indices),
        }
        log.info(f"INR results read from registry (rank#{rank4['rank']} "
                 f"{rank4['method']} / {rank4['model_type']})")

    else:
        used_indices = get_used_indices(entry)
        log.info(f"Previously used indices: {len(used_indices)} — all excluded")

        rng     = np.random.default_rng(seed=2026)
        pool    = [i for i in range(10000) if i not in used_indices]
        indices = rng.choice(pool, size=args.fresh_n, replace=False).tolist()
        log.info(f"Fresh indices ({len(indices)}): {indices}")

        ds_cfg = load_dataset_config(args.dataset)
        data_file  = ds_cfg["data_path"]
        grid_file = DATA_DIR + "/DL-based-SoS/forward_model_lr/grid_parameters.mat"
        log.info("Loading dataset for INR run ...")
        dataset = USDataset(data_file, grid_file)

        best_entry  = get_best_config(entry)
        base_config = ExperimentConfig(project_name=entry["project"])

        t0 = time.time()
        inr_results = run_inr_fresh(best_entry, dataset, indices, base_config, log)
        elapsed = (time.time() - t0) / 60
        log.info(f"\n  INR run complete in {elapsed:.1f} min")

    # ── Baseline metrics on same indices ──────────────────────────────────
    log.info(f"\nComputing baseline metrics on {len(indices)} samples ...")
    l2_results = compute_baseline_metrics(
        l2_recons, gt, indices, "L2_regularization", log
    )
    l1_results = compute_baseline_metrics(
        l1_recons, gt, indices, "L1_regularization", log
    )

    # ── Comparison table ──────────────────────────────────────────────────
    all_results = [l2_results, l1_results, inr_results]
    print_comparison_table(all_results, log)

    # ── W&B ──────────────────────────────────────────────────────────────
    if not args.no_wandb:
        log.info("\nLogging to W&B ...")
        log_to_wandb(all_results, entry, indices, mode)

    # ── Registry ──────────────────────────────────────────────────────────
    comparison_key = f"comparison_{mode}"
    for e in registry:
        if e["sweep_id"].startswith(args.sweep_id):
            e[comparison_key] = {
                "ran_at":    datetime.now().isoformat(),
                "mode":      mode,
                "indices":   indices,
                "n_samples": len(indices),
                "results":   all_results,
            }
    save_registry(registry)
    log.info(f"\n  Registry updated under key '{comparison_key}'")
    log.info(f"  Log: {log_path}")


if __name__ == "__main__":
    main()