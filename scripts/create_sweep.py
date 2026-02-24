#!/usr/bin/env python3

"""
bootstrap a new sweep and log it in the registry

Usage:
    python create_sweep.py
    python create_sweep.py --project INR-SoS-Recon --n_runs 60
"""

import argparse
import json
import os
import sys
import wandb
from datetime import datetime
from pathlib import Path
from inr_sos.evaluation.sweep_agent import get_sweep_config

_LOG_FILE = Path(__file__).parent / "sweep_registry.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",   default="INR-SoS-Recon")
    parser.add_argument("--n_runs",    default=60,  type=int)
    parser.add_argument("--metric",    default="MAE_mean")
    parser.add_argument("--direction", default="minimize")
    args = parser.parse_args()

    cfg = get_sweep_config(
        metric_goal=args.metric,
        metric_direction=args.direction,
    )

    sweep_id = wandb.sweep(cfg, project=args.project)
    entity   = wandb.api.default_entity

    url = f"https://wandb.ai/{entity}/{args.project}/sweeps/{sweep_id}"

    entry = {
        "sweep_id":   sweep_id,
        "project":    args.project,
        "entity":     entity,
        "n_runs":     args.n_runs,
        "metric":     args.metric,
        "direction":  args.direction,
        "created_at": datetime.now().isoformat(),
        "status":     "created",   # created → running → done / failed
        "url":        url,
        "log_file":   str(Path(__file__).parent / f"sweep_{sweep_id}.log"),
    }

    # Load or create the registry
    registry = []
    if _LOG_FILE.exists():
        with open(_LOG_FILE) as f:
            registry = json.load(f)

    registry.append(entry)
    with open(_LOG_FILE, "w") as f:
        json.dump(registry, f, indent=2)

    print("\n" + "═" * 60)
    print(f"  Sweep created successfully")
    print(f"  ID      : {sweep_id}")
    print(f"  Project : {args.project}")
    print(f"  N runs  : {args.n_runs}")
    print(f"  URL     : {url}")
    print(f"  Registry: {_LOG_FILE}")
    print("═" * 60)
    print(f"\nNext step — launch the agent in tmux:")
    print(f"\n  tmux new -s sweep_{sweep_id[:6]}")
    print(f"  python run_sweep.py --sweep_id {sweep_id} --n_runs {args.n_runs}")
    print(f"  # Then Ctrl+B D to detach")
    print(f"\nCheck progress anytime:")
    print(f"  python check_sweep.py")
    print()


if __name__ == "__main__":
    main()