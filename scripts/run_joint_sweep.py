#!/usr/bin/env python3
"""
Run a joint denoiser+reconstructor Bayesian sweep agent.

Workflow:
    # 1. Create sweep
    python create_joint_sweep.py --dataset kwave_geom --n_runs 150

    # 2. Launch in tmux
    tmux new -s jsweep_<first6ofID>
    python run_joint_sweep.py --sweep_id <ID> --n_runs 150 --dataset kwave_geom

    # 3. Detach: Ctrl+B D
    # 4. Check: python check_sweep.py
"""

import argparse
import json
import logging
import sys
import time
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path

from inr_sos import DATA_DIR
from inr_sos.utils.data import USDataset
from inr_sos.evaluation.joint_sweep_agent import run_joint_sweep_agent

SCRIPTS_DIR = Path(__file__).parent
REGISTRY_FILE = SCRIPTS_DIR / "sweep_registry.json"


def setup_logging(sweep_id: str) -> Path:
    log_path = SCRIPTS_DIR / f"sweep_{sweep_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


def update_registry(sweep_id: str, updates: dict):
    if not REGISTRY_FILE.exists():
        return
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    for entry in registry:
        if entry.get("sweep_id") == sweep_id:
            entry.update(updates)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", required=True)
    parser.add_argument("--dataset", default=None,
                        help="Dataset key (default: uses 'active' field)")
    parser.add_argument("--n_runs", default=150, type=int)
    parser.add_argument("--indices", nargs="+", type=int, default=None)
    parser.add_argument("--project", default="INR-SoS-Recon")
    args = parser.parse_args()

    log_path = setup_logging(args.sweep_id)
    log = logging.getLogger(__name__)

    # ── Load dataset config ──────────────────────────────────────────────
    cfg_path = SCRIPTS_DIR / "datasets.yaml"
    with open(cfg_path) as f:
        all_cfg = yaml.safe_load(f)
    dataset_key = args.dataset or all_cfg["active"]
    ds_cfg = all_cfg["datasets"][dataset_key]

    joint_section = ds_cfg.get("joint_sweep", {})
    n_eval_samples = joint_section.get("n_eval_samples", 5)

    log.info("=" * 60)
    log.info(f"  Joint sweep agent starting")
    log.info(f"  Sweep ID : {args.sweep_id}")
    log.info(f"  Dataset  : {ds_cfg['name']} ({dataset_key})")
    log.info(f"  N runs   : {args.n_runs}")
    log.info(f"  Log file : {log_path}")
    log.info("=" * 60)

    update_registry(args.sweep_id, {
        "status": "running",
        "started_at": datetime.now().isoformat(),
    })

    # ── Resolve time_scale ───────────────────────────────────────────────
    yaml_time_scale = joint_section.get("time_scale", "auto")
    if yaml_time_scale == "auto":
        yaml_pix2time = ds_cfg.get("pix2time")
        if yaml_pix2time is not None:
            base_time_scale = 1.0 / float(yaml_pix2time)
        else:
            import h5py
            data_path = DATA_DIR + ds_cfg["data_file"]
            with h5py.File(data_path, "r") as f:
                pix2time = float(np.array(f["pix2time"]).flat[0])
            base_time_scale = 1.0 / pix2time
    else:
        base_time_scale = float(yaml_time_scale)
    log.info(f"  time_scale: {base_time_scale:.4e}")

    # ── Load dataset ─────────────────────────────────────────────────────
    data_path = DATA_DIR + ds_cfg["data_file"]
    grid_path = DATA_DIR + "/DL-based-SoS/forward_model_lr/grid_parameters.mat"

    ds_kwargs = {}
    if not ds_cfg.get("has_A_matrix", True):
        matrix_file = ds_cfg.get("matrix_file")
        if matrix_file:
            ds_kwargs["matrix_path"] = DATA_DIR + matrix_file
            ds_kwargs["use_external_L_matrix"] = True

    log.info("  Loading dataset ...")
    dataset = USDataset(data_path, grid_path, **ds_kwargs)
    grid = dataset.grid
    log.info(f"  Dataset loaded — {len(dataset)} samples")

    # ── Sample indices ───────────────────────────────────────────────────
    if args.indices:
        indices = args.indices
    else:
        np.random.seed(42)
        n = min(n_eval_samples, len(dataset))
        indices = np.random.choice(len(dataset), size=n, replace=False).tolist()

    log.info(f"  Eval samples: {len(indices)} indices = {indices}")
    update_registry(args.sweep_id, {"indices": indices, "dataset": dataset_key})

    # ── Read entity/project from registry ────────────────────────────────
    registry = []
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            registry = json.load(f)
    reg_entry = next(
        (e for e in registry if e.get("sweep_id") == args.sweep_id), {}
    )
    entity = reg_entry.get("entity")
    project = reg_entry.get("project", args.project)

    # ── Run the agent ────────────────────────────────────────────────────
    log.info("  Launching W&B sweep agent ...")
    t_start = time.time()

    try:
        run_joint_sweep_agent(
            sweep_id=args.sweep_id,
            dataset=dataset,
            grid=grid,
            target_indices=indices,
            base_time_scale=base_time_scale,
            n_runs=args.n_runs,
            entity=entity,
            project=project,
        )
        elapsed = (time.time() - t_start) / 3600
        log.info(f"  Sweep finished — {args.n_runs} runs in {elapsed:.1f} hours")
        update_registry(args.sweep_id, {
            "status": "done",
            "finished_at": datetime.now().isoformat(),
            "elapsed_hrs": round(elapsed, 2),
        })

    except KeyboardInterrupt:
        log.warning("  Interrupted by user (Ctrl+C)")
        update_registry(args.sweep_id, {"status": "interrupted"})

    except Exception as e:
        log.error(f"  Sweep failed: {e}", exc_info=True)
        update_registry(args.sweep_id, {"status": "failed", "error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
