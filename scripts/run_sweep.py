#!/usr/bin/env python3
"""
run_sweep.py
------------
Run this inside a tmux session. Survives SSH disconnects.
Logs every trial result to  sweep_<ID>.log  and updates sweep_registry.json.

Workflow:
    # 1. Create a tmux session
    tmux new -s sweep_<first6ofID>

    # 2. Inside tmux, launch this script
    python run_sweep.py --sweep_id <ID> --n_runs 60

    # 3. Detach (keep it running)
    Ctrl+B  then  D

    # 4. Check progress anytime (new terminal, no need to re-attach)
    python check_sweep.py

    # 5. Re-attach if you want to see live output
    tmux attach -t sweep_<first6ofID>

Usage:
    python run_sweep.py --sweep_id abc123 --n_runs 60
    python run_sweep.py --sweep_id abc123 --n_runs 60 --indices 8052 7863 2923 1042 5501
"""

import argparse
import json
import logging
import os
import sys
import time
import inr_sos
import wandb
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from inr_sos import DATA_DIR
from inr_sos.utils.data import USDataset
from inr_sos.utils.config import ExperimentConfig
from inr_sos.evaluation.sweep_agent import run_sweep_agent


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


LOG_DIR       = Path(__file__).parent
REGISTRY_FILE = LOG_DIR / "sweep_registry.json"
SCRIPTS_DIR   = Path(__file__).parent


def load_dataset_config(key: str = None) -> dict:
    """
    Load dataset path from scripts/datasets.yaml.
    key overrides the 'active' field (use for --dataset CLI arg).
    """
    cfg_path = SCRIPTS_DIR / "datasets.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    key = key or cfg["active"]
    ds = cfg["datasets"][key]
    ds["key"] = key
    ds["data_path"] = DATA_DIR + ds["data_file"]
    return ds


def setup_logging(sweep_id: str) -> Path:
    log_path = LOG_DIR / f"sweep_{sweep_id}.log"
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


def update_registry(sweep_id: str, updates: dict):
    """Patch the registry entry for this sweep_id."""
    if not REGISTRY_FILE.exists():
        return
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    for entry in registry:
        if entry["sweep_id"] == sweep_id:
            entry.update(updates)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                    help="Dataset key from datasets.yaml (default: uses 'active' field)")
    parser.add_argument("--sweep_id",  required=True,
                        help="Sweep ID from create_sweep.py")
    parser.add_argument("--n_runs",    default=60, type=int,
                        help="Total Bayesian trials")
    parser.add_argument("--indices",   nargs="+", type=int, default=None,
                        help="Dataset sample indices to evaluate on. "
                             "If omitted, 10 random indices are chosen.")
    parser.add_argument("--project",   default="INR-SoS-Recon")
    args = parser.parse_args()

    log_path = setup_logging(args.sweep_id)
    log      = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info(f"  Sweep agent starting")
    log.info(f"  Sweep ID  : {args.sweep_id}")
    log.info(f"  N runs    : {args.n_runs}")
    log.info(f"  Log file  : {log_path}")
    log.info("=" * 60)

    # ── Update registry: mark as running ─────────────────────────────────
    update_registry(args.sweep_id, {
        "status":     "running",
        "started_at": datetime.now().isoformat(),
        "log_file":   str(log_path),
    })

    # ── Load dataset ──────────────────────────────────────────────────────
    grid_file  = DATA_DIR + "/DL-based-SoS/forward_model_lr/grid_parameters.mat"

    log.info("Loading dataset ...")
    ds_cfg = load_dataset_config(args.dataset)
    data_file  = ds_cfg["data_path"]
    dataset = USDataset(data_file, grid_file)
    log.info(f"Dataset loaded — {len(dataset)} samples")

    # ── Sample indices ────────────────────────────────────────────────────
    if args.indices:
        indices = args.indices
    else:
        np.random.seed(42)   # reproducible default
        indices = np.random.choice(len(dataset), size=10, replace=False).tolist()

    log.info(f"Evaluating on indices: {indices}")
    update_registry(args.sweep_id, {"indices": indices})

    # ── Base config ───────────────────────────────────────────────────────
    base_config = ExperimentConfig(
        project_name="INR-SoS-Recon",
        in_features=2,
        hidden_features=256,
        hidden_layers=3,
        mapping_size=64,
        scale=0.6,
        omega=30.0,
        lr=1e-4,
        steps=2000,
        epochs=150,
        batch_size=4096,
        tv_weight=0.0,
        reg_weight=0.0,
    )

    # ── Read entity/project from registry (set by create_sweep.py) ───────
    registry = []
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            registry = json.load(f)
    reg_entry = next(
        (e for e in registry if e["sweep_id"] == args.sweep_id), {}
    )
    entity  = reg_entry.get("entity")   # e.g. "habenhadush-uppsala-universitet"
    project = reg_entry.get("project", args.project)
    log.info(f"Entity  : {entity}")
    log.info(f"Project : {project}")

    # ── Run the agent ─────────────────────────────────────────────────────
    log.info("Launching W&B sweep agent ...")
    t_start = time.time()

    try:
        run_sweep_agent(
            sweep_id=args.sweep_id,
            dataset=dataset,
            target_indices=indices,
            base_config=base_config,
            n_runs=args.n_runs,
            entity=entity,
            project=project,
        )
        elapsed = (time.time() - t_start) / 3600
        log.info(f"Sweep finished — {args.n_runs} runs in {elapsed:.1f} hours")
        update_registry(args.sweep_id, {
            "status":      "done",
            "finished_at": datetime.now().isoformat(),
            "elapsed_hrs": round(elapsed, 2),
        })

    except KeyboardInterrupt:
        log.warning("Interrupted by user (Ctrl+C)")
        update_registry(args.sweep_id, {"status": "interrupted"})

    except Exception as e:
        log.error(f"Sweep failed: {e}", exc_info=True)
        update_registry(args.sweep_id, {
            "status": "failed",
            "error":  str(e),
        })
        sys.exit(1)


if __name__ == "__main__":
    main()