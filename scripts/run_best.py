#!/usr/bin/env python3
"""
run_best.py
-----------
Fetches the top-K configs from a completed sweep and validates each on
a held-out sample set that was NOT used during the sweep.

Results are logged to W&B under group "best_config_validation" so they
appear alongside — but are clearly separated from — the original grid runs.
The registry entry for the sweep is updated with the best validated config.

Usage:
    python scripts/run_best.py --sweep_id hqt6bwmp
    python scripts/run_best.py --sweep_id hqt6bwmp --top_k 5 --n_holdout 20
    python scripts/run_best.py --sweep_id hqt6bwmp --holdout_indices 100 200 300 ...
"""

import argparse
import json
import logging
import sys
import time
import numpy as np
import wandb
import inr_sos

from datetime import datetime
from pathlib import Path

from inr_sos import DATA_DIR
from inr_sos.utils.data import USDataset
from inr_sos.utils.config import ExperimentConfig
from inr_sos.evaluation.pipeline import run_grid_comparison

SCRIPTS_DIR   = Path(__file__).parent
REGISTRY_FILE = SCRIPTS_DIR / "sweep_registry.json"
LOG_DIR       = SCRIPTS_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_logging(sweep_id: str) -> Path:
    log_path = LOG_DIR / f"best_validation_{sweep_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return log_path


def load_registry() -> list:
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_FILE}")
    with open(REGISTRY_FILE) as f:
        return json.load(f)


def save_registry(registry: list):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def get_sweep_entry(registry: list, sweep_id: str) -> dict:
    matches = [e for e in registry if e["sweep_id"].startswith(sweep_id)]
    if not matches:
        raise ValueError(f"Sweep {sweep_id} not found in registry.")
    return matches[-1]


def fetch_top_k(entry: dict, top_k: int) -> list:
    """Fetch top-K runs from W&B API, sorted by MAE_mean ascending."""
    api   = wandb.Api()
    sweep = api.sweep(f"{entry['entity']}/{entry['project']}/{entry['sweep_id']}")
    runs  = [r for r in sweep.runs if "MAE_mean" in r.summary]

    if not runs:
        raise RuntimeError("No completed runs with MAE_mean found in sweep.")

    runs_sorted = sorted(runs, key=lambda r: r.summary["MAE_mean"])[:top_k]

    configs = []
    for rank, run in enumerate(runs_sorted, 1):
        cfg = {
            "rank":       rank,
            "run_name":   run.name,
            "run_id":     run.id,
            "method":     run.config.get("method"),
            "model_type": run.config.get("model_type"),
            "MAE_mean":   run.summary["MAE_mean"],
            "RMSE_mean":  run.summary.get("RMSE_mean", float("nan")),
            "SSIM_mean":  run.summary.get("SSIM_mean", float("nan")),
            "CNR_mean":   run.summary.get("CNR_mean",  float("nan")),
            # full hyperparams
            **{k: v for k, v in run.config.items()
               if k not in {"method", "model_type", "_wandb"}},
        }
        configs.append(cfg)

    return configs


def config_to_experiment_config(sweep_cfg: dict, base: ExperimentConfig) -> ExperimentConfig:
    """Merge a sweep trial config dict onto the base ExperimentConfig."""
    return ExperimentConfig(
        project_name    = base.project_name,
        experiment_group= f"best_config_validation",
        model_type      = sweep_cfg["model_type"],
        in_features     = base.in_features,
        hidden_features = sweep_cfg.get("hidden_features", base.hidden_features),
        hidden_layers   = sweep_cfg.get("hidden_layers",   base.hidden_layers),
        mapping_size    = sweep_cfg.get("mapping_size",    base.mapping_size),
        scale           = sweep_cfg.get("scale",           base.scale),
        omega           = sweep_cfg.get("omega",           base.omega),
        lr              = sweep_cfg.get("lr",              base.lr),
        steps           = sweep_cfg.get("steps",           base.steps),
        epochs          = sweep_cfg.get("epochs",          base.epochs),
        batch_size      = sweep_cfg.get("batch_size",      base.batch_size),
        tv_weight       = sweep_cfg.get("tv_weight",       base.tv_weight),
        reg_weight      = sweep_cfg.get("reg_weight",      base.reg_weight),
        time_scale      = base.time_scale,
        log_interval    = base.log_interval,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id",         required=True)
    parser.add_argument("--top_k",            default=5,  type=int,
                        help="Number of top sweep configs to validate")
    parser.add_argument("--n_holdout",        default=20, type=int,
                        help="Holdout samples to evaluate on")
    parser.add_argument("--holdout_indices",  nargs="+",  type=int, default=None,
                        help="Explicit holdout indices (overrides --n_holdout)")
    parser.add_argument("--no_wandb",         action="store_true")
    args = parser.parse_args()

    log_path = setup_logging(args.sweep_id)
    log      = logging.getLogger(__name__)

    # ── Load registry ─────────────────────────────────────────────────────
    registry = load_registry()
    entry    = get_sweep_entry(registry, args.sweep_id)
    sweep_indices = set(entry.get("indices", []))

    log.info("=" * 60)
    log.info(f"  Best-config validation")
    log.info(f"  Sweep ID  : {args.sweep_id}")
    log.info(f"  Top-K     : {args.top_k}")
    log.info(f"  Log       : {log_path}")
    log.info("=" * 60)

    # ── Holdout indices — must be disjoint from sweep indices ─────────────
    if args.holdout_indices:
        holdout = args.holdout_indices
        overlap = set(holdout) & sweep_indices
        if overlap:
            raise ValueError(
                f"Holdout indices overlap with sweep indices: {overlap}. "
                f"Choose different samples."
            )
    else:
        log.info(f"Sweep used indices: {sorted(sweep_indices)}")
        log.info(f"Sampling {args.n_holdout} disjoint holdout indices ...")
        rng = np.random.default_rng(seed=99)   # fixed seed for reproducibility
        pool = [i for i in range(10000) if i not in sweep_indices]
        holdout = rng.choice(pool, size=args.n_holdout, replace=False).tolist()

    log.info(f"Holdout indices ({len(holdout)}): {holdout}")

    # ── Load dataset ──────────────────────────────────────────────────────
    data_file = DATA_DIR + "/DL-based-SoS/train-VS-8pairs-IC-081225.mat"
    grid_file = DATA_DIR + "/DL-based-SoS/forward_model_lr/grid_parameters.mat"
    log.info("Loading dataset ...")
    dataset = USDataset(data_file, grid_file)

    # ── Base config (defaults — sweep values override per trial) ──────────
    base_config = ExperimentConfig(project_name=entry["project"])

    # ── Fetch top-K configs from W&B ──────────────────────────────────────
    log.info(f"Fetching top-{args.top_k} configs from sweep {args.sweep_id} ...")
    top_configs = fetch_top_k(entry, args.top_k)

    log.info(f"\n  Sweep top-{args.top_k} (on sweep indices):")
    log.info(f"  {'Rank':<5} {'Method':<22} {'Model':<12} "
             f"{'MAE':>7} {'RMSE':>7} {'SSIM':>7} {'CNR':>6}")
    log.info(f"  {'─'*65}")
    for c in top_configs:
        log.info(f"  #{c['rank']:<4} {c['method']:<22} {c['model_type']:<12} "
                 f"{c['MAE_mean']:>7.3f} {c['RMSE_mean']:>7.3f} "
                 f"{c['SSIM_mean']:>7.3f} {c['CNR_mean']:>6.3f}")

    # ── Validate each top-K config on holdout ─────────────────────────────
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

    validation_results = []
    t_start = time.time()

    for sweep_cfg in top_configs:
        rank       = sweep_cfg["rank"]
        method     = sweep_cfg["method"]
        model_type = sweep_cfg["model_type"]
        log.info(f"\n{'─'*60}")
        log.info(f"  Validating rank #{rank}: {method} / {model_type}")
        log.info(f"  Sweep MAE: {sweep_cfg['MAE_mean']:.3f} "
                 f"SSIM: {sweep_cfg['SSIM_mean']:.3f}")
        log.info(f"{'─'*60}")

        cfg = config_to_experiment_config(sweep_cfg, base_config)
        # Tag with rank so W&B runs are distinguishable
        cfg.experiment_group = f"validation_rank{rank}_{method}"

        results = run_grid_comparison(
            dataset=dataset,
            target_indices=holdout,
            base_config=cfg,
            engines={method:     engine_map[method]},
            models= {model_type: model_map[model_type]},
            use_wandb=not args.no_wandb,
        )

        # Aggregate across holdout
        vals = results[(method, model_type)]
        summary = {
            "rank":       rank,
            "method":     method,
            "model_type": model_type,
            # sweep performance (on sweep indices)
            "sweep_MAE":  sweep_cfg["MAE_mean"],
            "sweep_SSIM": sweep_cfg["SSIM_mean"],
            "sweep_RMSE": sweep_cfg["RMSE_mean"],
            # holdout performance
            "holdout_MAE_mean":  float(np.mean(vals["MAE"])),
            "holdout_MAE_std":   float(np.std(vals["MAE"])),
            "holdout_RMSE_mean": float(np.mean(vals["RMSE"])),
            "holdout_RMSE_std":  float(np.std(vals["RMSE"])),
            "holdout_SSIM_mean": float(np.mean(vals["SSIM"])),
            "holdout_SSIM_std":  float(np.std(vals["SSIM"])),
            "holdout_CNR_mean":  float(np.mean(vals["CNR"])),
            "holdout_CNR_std":   float(np.std(vals["CNR"])),
            "hyperparams": {k: v for k, v in sweep_cfg.items()
                            if k not in {"rank","run_name","run_id",
                                         "method","model_type",
                                         "MAE_mean","RMSE_mean",
                                         "SSIM_mean","CNR_mean"}},
        }
        validation_results.append(summary)

        log.info(f"  Holdout → MAE: {summary['holdout_MAE_mean']:.3f} ± "
                 f"{summary['holdout_MAE_std']:.3f}  "
                 f"SSIM: {summary['holdout_SSIM_mean']:.3f} ± "
                 f"{summary['holdout_SSIM_std']:.3f}")

    # ── Final comparison table ────────────────────────────────────────────
    elapsed = (time.time() - t_start) / 3600
    log.info(f"\n{'═'*72}")
    log.info(f"  VALIDATION COMPLETE  ({elapsed:.2f} hrs)")
    log.info(f"{'═'*72}")
    log.info(f"  {'Rank':<5} {'Method':<22} {'Model':<12} "
             f"{'MAE_h':>8} {'SSIM_h':>8}  {'MAE_s':>8} {'SSIM_s':>8}  Gap")
    log.info(f"  {'─'*70}")

    for r in sorted(validation_results, key=lambda x: x["holdout_MAE_mean"]):
        mae_gap  = r["holdout_MAE_mean"]  - r["sweep_MAE"]
        ssim_gap = r["holdout_SSIM_mean"] - r["sweep_SSIM"]
        log.info(
            f"  #{r['rank']:<4} {r['method']:<22} {r['model_type']:<12} "
            f"{r['holdout_MAE_mean']:>8.3f} {r['holdout_SSIM_mean']:>8.3f}  "
            f"{r['sweep_MAE']:>8.3f} {r['sweep_SSIM']:>8.3f}  "
            f"ΔMAE={mae_gap:+.2f}"
        )

    # Best on holdout (by MAE)
    best_holdout = min(validation_results, key=lambda x: x["holdout_MAE_mean"])
    log.info(f"\n  Best on holdout: rank #{best_holdout['rank']} — "
             f"{best_holdout['method']} / {best_holdout['model_type']}")
    log.info(f"  MAE  {best_holdout['holdout_MAE_mean']:.3f} ± "
             f"{best_holdout['holdout_MAE_std']:.3f}")
    log.info(f"  SSIM {best_holdout['holdout_SSIM_mean']:.3f} ± "
             f"{best_holdout['holdout_SSIM_std']:.3f}")
    log.info(f"  RMSE {best_holdout['holdout_RMSE_mean']:.3f} ± "
             f"{best_holdout['holdout_RMSE_std']:.3f}")
    log.info(f"  CNR  {best_holdout['holdout_CNR_mean']:.3f} ± "
             f"{best_holdout['holdout_CNR_std']:.3f}")

    # ── Update registry ───────────────────────────────────────────────────
    for e in registry:
        if e["sweep_id"].startswith(args.sweep_id):
            e["validation"] = {
                "holdout_indices":  holdout,
                "ran_at":           datetime.now().isoformat(),
                "elapsed_hrs":      round(elapsed, 2),
                "top_k_results":    validation_results,
                "best_on_holdout":  best_holdout,
            }
    save_registry(registry)
    log.info(f"\n  Registry updated: {REGISTRY_FILE}")
    log.info(f"  Validation log:   {log_path}")


if __name__ == "__main__":
    main()