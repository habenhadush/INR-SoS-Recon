#!/usr/bin/env python3
"""
check_sweep.py
--------------
Run anytime from any terminal (no tmux, no W&B login needed for basic info).
Shows all sweeps from sweep_registry.json + live best results from W&B API.

Usage:
    python check_sweep.py               # show all sweeps
    python check_sweep.py --id abc123   # show one sweep in detail
    python check_sweep.py --tail 20     # tail the log of the latest running sweep
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR  = Path(__file__).parent 
LOG_DIR.mkdir(exist_ok=True)
REGISTRY_FILE = LOG_DIR / "sweep_registry.json" 


def load_registry():
    if not REGISTRY_FILE.exists():
        print("No sweeps registered yet. Run create_sweep.py first.")
        sys.exit(0)
    with open(REGISTRY_FILE) as f:
        return json.load(f)


def status_icon(status):
    return {
        "created":     "⏳",
        "running":     "🔄",
        "done":        "✅",
        "failed":      "❌",
        "interrupted": "⚠️ ",
    }.get(status, "❓")


def show_all(registry):
    print(f"\n{'═'*78}")
    print(f"  Sweep Registry — {REGISTRY_FILE}")
    print(f"{'═'*78}")
    print(f"  {'#':<3} {'Status':<14} {'Sweep ID':<14} {'Created':<20} {'Runs':<6} {'URL'}")
    print(f"  {'─'*74}")
    for i, e in enumerate(registry):
        icon    = status_icon(e.get("status", "?"))
        created = e.get("created_at", "")[:16]
        n_runs  = e.get("n_runs", "?")
        sid     = e.get("sweep_id", "?")[:12]
        url     = e.get("url", "")
        elapsed = f"  [{e['elapsed_hrs']:.1f}h]" if "elapsed_hrs" in e else ""
        print(f"  {i:<3} {icon} {e.get('status','?'):<12} {sid:<14} {created:<20} {n_runs:<6} {url}{elapsed}")
    print(f"{'═'*78}\n")


def show_detail(entry):
    import wandb
    sid = entry["sweep_id"]
    print(f"\n{'═'*60}")
    print(f"  Sweep: {sid}")
    print(f"  Status: {status_icon(entry.get('status'))} {entry.get('status')}")
    print(f"  URL:    {entry.get('url')}")
    print(f"  Log:    {entry.get('log_file', 'N/A')}")
    if "started_at" in entry:
        print(f"  Started:  {entry['started_at'][:19]}")
    if "finished_at" in entry:
        print(f"  Finished: {entry['finished_at'][:19]}  ({entry.get('elapsed_hrs', '?')} hrs)")
    if "indices" in entry:
        print(f"  Indices:  {entry['indices']}")
    print()

    # Pull live results from W&B API
    try:
        api    = wandb.Api()
        sweep  = api.sweep(f"{entry['entity']}/{entry['project']}/{sid}")
        runs   = [r for r in sweep.runs if "MAE_mean" in r.summary]

        if not runs:
            print("  No completed trials yet.")
        else:
            print(f"  Completed trials: {len(runs)} / {entry.get('n_runs', '?')}")
            print()

            # Best run
            best = min(runs, key=lambda r: r.summary["MAE_mean"])
            print(f"  Best so far:")
            print(f"    method     : {best.config.get('method', '?')}")
            print(f"    model_type : {best.config.get('model_type', '?')}")
            print(f"    MAE_mean   : {best.summary.get('MAE_mean',  float('nan')):.4f}")
            print(f"    RMSE_mean  : {best.summary.get('RMSE_mean', float('nan')):.4f}")
            print(f"    SSIM_mean  : {best.summary.get('SSIM_mean', float('nan')):.4f}")
            print(f"    CNR_mean   : {best.summary.get('CNR_mean',  float('nan')):.4f}")
            ignore = {"method", "model_type", "_wandb"}
            print(f"    hyperparams:")
            for k, v in sorted(best.config.items()):
                if k not in ignore:
                    print(f"      {k:<18}: {v}")

            # Top 5 table
            top5 = sorted(runs, key=lambda r: r.summary["MAE_mean"])[:5]
            print(f"\n  Top-5 trials:")
            print(f"  {'Method':<20} {'Model':<12} {'MAE':>7} {'RMSE':>7} {'SSIM':>7} {'CNR':>6}")
            print(f"  {'─'*62}")
            for r in top5:
                print(f"  {r.config.get('method','?'):<20} "
                      f"{r.config.get('model_type','?'):<12} "
                      f"{r.summary.get('MAE_mean', float('nan')):>7.3f} "
                      f"{r.summary.get('RMSE_mean',float('nan')):>7.3f} "
                      f"{r.summary.get('SSIM_mean',float('nan')):>7.3f} "
                      f"{r.summary.get('CNR_mean', float('nan')):>6.3f}")

            # Method breakdown
            print(f"\n  By method (mean MAE across completed trials):")
            method_mae = {}
            for r in runs:
                m = r.config.get("method", "?")
                method_mae.setdefault(m, []).append(r.summary.get("MAE_mean", float("nan")))
            for m, vals in sorted(method_mae.items(), key=lambda x: sum(x[1])/len(x[1])):
                import numpy as np
                print(f"    {m:<22}: MAE {np.mean(vals):.3f} ± {np.std(vals):.3f}  (n={len(vals)})")

    except Exception as e:
        print(f"  Could not fetch W&B results: {e}")
        print(f"  Check the log file: {entry.get('log_file', 'N/A')}")

    print(f"{'═'*60}\n")


def tail_log(entry, n_lines):
    log_file = entry.get("log_file")
    if not log_file or not Path(log_file).exists():
        print(f"Log file not found: {log_file}")
        return
    print(f"\nTailing {log_file} (last {n_lines} lines):\n")
    subprocess.run(["tail", f"-{n_lines}", log_file])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id",   default=None, help="Sweep ID to inspect in detail")
    parser.add_argument("--tail", default=None, type=int,
                        help="Tail last N lines of the log for latest running sweep")
    args = parser.parse_args()

    registry = load_registry()

    if args.id is None and args.tail is None:
        show_all(registry)
        # If any sweep is running, show a quick summary
        running = [e for e in registry if e.get("status") == "running"]
        if running:
            print(f"  {len(running)} sweep(s) running. Use --id <sweep_id> for details.")
            print(f"  Use --tail 30 to tail the latest running sweep log.\n")
        return

    # Find the target entry
    if args.id:
        matches = [e for e in registry if e["sweep_id"].startswith(args.id)]
    else:
        # Default to latest running sweep for --tail
        matches = [e for e in registry if e.get("status") == "running"]
        if not matches:
            matches = registry[-1:]   # fall back to most recent

    if not matches:
        print(f"No sweep found matching '{args.id}'")
        sys.exit(1)

    entry = matches[-1]

    if args.tail:
        tail_log(entry, args.tail)
    else:
        show_detail(entry)


if __name__ == "__main__":
    main()