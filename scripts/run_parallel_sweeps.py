#!/usr/bin/env python3
"""
run_parallel_sweeps.py
----------------------
Orchestrator that creates and runs W&B Bayesian sweeps across multiple
datasets in parallel on a single GPU.

Phase 1 (serial):   create_sweep for each dataset → {dataset: sweep_id}
Phase 1b:           copy each sweep's registry entry to a local file
Phase 2 (parallel): run_sweep subprocesses with --registry_file pointing to local copy
Phase 3 (serial):   merge local registries back into main, print summary

Each subprocess writes to its own local registry file to avoid race conditions.
Serial usage (run_sweep.py without --registry_file) still writes to main registry.

Usage:
    # Sweep 4 datasets, all concurrent on GPU 0
    python run_parallel_sweeps.py \
        --datasets kwave_geom kwave_blob kwave_geom_unet kwave_blob_unet \
        --n_runs 100

    # Limit concurrency
    python run_parallel_sweeps.py \
        --datasets kwave_geom kwave_blob kwave_geom_unet kwave_blob_unet \
        --n_runs 100 --max_concurrent 2

    # Dry run — create sweeps only, print commands
    python run_parallel_sweeps.py \
        --datasets kwave_geom kwave_blob --n_runs 100 --dry_run
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

SCRIPTS_DIR   = Path(__file__).parent
REGISTRY_FILE = SCRIPTS_DIR / "sweep_registry.json"
DATASETS_YAML = SCRIPTS_DIR / "datasets.yaml"
LOG_DIR       = SCRIPTS_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOCAL_REG_DIR = LOG_DIR / "local_registries"
LOCAL_REG_DIR.mkdir(exist_ok=True)


# ─── Phase 1: Create sweeps serially ─────────────────────────────────────────

def create_sweep(dataset_key, n_runs, project):
    """Call create_sweep.py as subprocess, return sweep_id."""
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "create_sweep.py"),
        "--dataset", dataset_key,
        "--n_runs", str(n_runs),
        "--project", project,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPTS_DIR))

    if result.returncode != 0:
        print(f"  [FAIL] create_sweep for {dataset_key}:")
        print(result.stderr)
        return None

    # Parse sweep_id from registry (most reliable)
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    for entry in reversed(registry):
        if entry["dataset"] == dataset_key and entry["status"] == "created":
            return entry["sweep_id"]

    # Fallback: parse from stdout
    for line in result.stdout.splitlines():
        if "ID" in line and ":" in line:
            return line.split(":")[-1].strip()
    return None


def create_local_registry(sweep_id):
    """Copy the sweep's entry from main registry into a local file.
    run_sweep.py will read/write only this file during execution."""
    local_path = LOCAL_REG_DIR / f"registry_{sweep_id[:8]}.json"

    with open(REGISTRY_FILE) as f:
        main_registry = json.load(f)

    entry = next((e for e in main_registry if e.get("sweep_id") == sweep_id), None)
    if entry is None:
        print(f"  WARNING: sweep {sweep_id} not found in main registry")
        return None

    with open(local_path, "w") as f:
        json.dump([entry], f, indent=2)

    return local_path


# ─── Phase 2: Run sweeps in parallel ─────────────────────────────────────────

def launch_sweep(dataset_key, sweep_id, n_runs, project, local_reg_path, gpu_id=0):
    """Launch run_sweep.py as a background subprocess with its own registry."""
    log_file = LOG_DIR / f"parallel_{dataset_key}_{sweep_id[:8]}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "run_sweep.py"),
        "--sweep_id", sweep_id,
        "--n_runs", str(n_runs),
        "--dataset", dataset_key,
        "--project", project,
        "--registry_file", str(local_reg_path),
    ]

    fh = open(log_file, "w")
    proc = subprocess.Popen(
        cmd, stdout=fh, stderr=subprocess.STDOUT,
        env=env, cwd=str(SCRIPTS_DIR),
    )
    return proc, log_file, fh


def run_parallel(sweep_map, local_reg_map, n_runs, project, max_concurrent, gpu_id):
    """Run sweeps with a max-concurrent pool on a single GPU."""
    queue    = list(sweep_map.items())  # [(dataset_key, sweep_id), ...]
    active   = {}  # {dataset_key: (proc, log_file, fh, sweep_id, start_time)}
    finished = {}  # {dataset_key: (sweep_id, returncode, elapsed_min)}

    print(f"\n{'─'*60}")
    print(f"  Phase 2: Running {len(queue)} sweeps "
          f"(max {max_concurrent} concurrent, GPU {gpu_id})")
    print(f"{'─'*60}")

    while queue or active:
        # Fill slots
        while queue and len(active) < max_concurrent:
            ds_key, sweep_id = queue.pop(0)
            local_reg = local_reg_map[ds_key]
            proc, log_file, fh = launch_sweep(
                ds_key, sweep_id, n_runs, project, local_reg, gpu_id
            )
            active[ds_key] = (proc, log_file, fh, sweep_id, time.time())
            print(f"  [{len(active)}/{max_concurrent}] STARTED  "
                  f"{ds_key:<30} sweep={sweep_id[:8]}  "
                  f"log={log_file.name}")

        # Check for completions
        done_keys = []
        for ds_key, (proc, log_file, fh, sweep_id, t0) in active.items():
            ret = proc.poll()
            if ret is not None:
                elapsed = (time.time() - t0) / 60
                fh.close()
                done_keys.append(ds_key)
                status = "DONE" if ret == 0 else f"FAIL(rc={ret})"
                finished[ds_key] = (sweep_id, ret, elapsed)
                print(f"  [{len(active)-1}/{max_concurrent}] {status:<8} "
                      f"{ds_key:<30} sweep={sweep_id[:8]}  "
                      f"{elapsed:.1f} min")

        for k in done_keys:
            del active[k]

        if active:
            time.sleep(10)  # poll every 10s

    return finished


# ─── Phase 3: Merge & Summary ────────────────────────────────────────────────

def merge_local_registries(sweep_map, local_reg_map):
    """Merge local registry updates back into the main registry.
    Each local file has exactly one entry (the sweep's entry with updates).
    We patch the corresponding entry in the main registry."""
    with open(REGISTRY_FILE) as f:
        main_registry = json.load(f)

    merged_count = 0
    for ds_key, sweep_id in sweep_map.items():
        local_path = local_reg_map.get(ds_key)
        if not local_path or not local_path.exists():
            continue

        with open(local_path) as f:
            local_entries = json.load(f)

        # Find the updated entry
        local_entry = next(
            (e for e in local_entries if e.get("sweep_id") == sweep_id), None
        )
        if local_entry is None:
            continue

        # Patch the main registry entry
        for i, entry in enumerate(main_registry):
            if entry.get("sweep_id") == sweep_id:
                main_registry[i] = local_entry
                merged_count += 1
                break

    with open(REGISTRY_FILE, "w") as f:
        json.dump(main_registry, f, indent=2)

    print(f"  Merged {merged_count} local registries → {REGISTRY_FILE.name}")


def print_summary(sweep_map, finished):
    """Print final summary table."""
    print(f"\n{'═'*80}")
    print(f"  PARALLEL SWEEP SUMMARY")
    print(f"{'═'*80}")
    print(f"  {'Dataset':<30} {'Sweep ID':<14} {'Status':<10} {'Time (min)':>10}")
    print(f"  {'─'*70}")

    for ds_key, sweep_id in sweep_map.items():
        if ds_key in finished:
            sid, rc, elapsed = finished[ds_key]
            status = "done" if rc == 0 else f"fail({rc})"
            print(f"  {ds_key:<30} {sweep_id[:12]:<14} "
                  f"{status:<10} {elapsed:>10.1f}")
        else:
            print(f"  {ds_key:<30} {sweep_id[:12]:<14} "
                  f"{'skipped':<10} {'—':>10}")

    print(f"{'═'*80}")
    print(f"\n  Next steps:")
    for ds_key, sweep_id in sweep_map.items():
        print(f"    python compare_topk.py --sweep_id {sweep_id} "
              f"--dataset {ds_key} --n_samples 10")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Create and run W&B sweeps across multiple datasets in parallel"
    )
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Dataset keys from datasets.yaml")
    parser.add_argument("--n_runs", default=100, type=int,
                        help="Bayesian trials per sweep")
    parser.add_argument("--max_concurrent", default=None, type=int,
                        help="Max concurrent sweep processes "
                             "(default: number of datasets)")
    parser.add_argument("--gpu_id", default=0, type=int,
                        help="GPU device ID (default: 0)")
    parser.add_argument("--project", default="INR-SoS-Recon",
                        help="W&B project name")
    parser.add_argument("--dry_run", action="store_true",
                        help="Create sweeps only, print run commands")
    args = parser.parse_args()

    # Default max_concurrent = number of datasets
    if args.max_concurrent is None:
        args.max_concurrent = len(args.datasets)

    # Validate dataset keys
    with open(DATASETS_YAML) as f:
        ds_yaml = yaml.safe_load(f)
    for key in args.datasets:
        if key not in ds_yaml["datasets"]:
            print(f"ERROR: '{key}' not found in datasets.yaml. "
                  f"Available: {list(ds_yaml['datasets'].keys())}")
            sys.exit(1)

    # ── Phase 1: Create sweeps (serial) ───────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Phase 1: Creating {len(args.datasets)} sweeps")
    print(f"{'═'*60}")

    sweep_map = {}      # {dataset_key: sweep_id}
    local_reg_map = {}  # {dataset_key: Path to local registry}

    for ds_key in args.datasets:
        ds_name = ds_yaml["datasets"][ds_key]["name"]
        print(f"  Creating sweep for {ds_key} ({ds_name}) ...",
              end=" ", flush=True)
        sweep_id = create_sweep(ds_key, args.n_runs, args.project)
        if sweep_id:
            sweep_map[ds_key] = sweep_id
            # Create isolated local registry for this sweep
            local_path = create_local_registry(sweep_id)
            local_reg_map[ds_key] = local_path
            print(f"→ {sweep_id}")
        else:
            print("→ FAILED")

    if not sweep_map:
        print("\nNo sweeps created. Exiting.")
        sys.exit(1)

    print(f"\n  {len(sweep_map)}/{len(args.datasets)} sweeps created.")

    # ── Dry run: just print commands ──────────────────────────────────────
    if args.dry_run:
        print(f"\n  Dry run — launch manually:")
        for ds_key, sweep_id in sweep_map.items():
            print(f"    CUDA_VISIBLE_DEVICES={args.gpu_id} "
                  f"python run_sweep.py --sweep_id {sweep_id} "
                  f"--n_runs {args.n_runs} --dataset {ds_key}")
        return

    # ── Phase 2: Run in parallel ──────────────────────────────────────────
    t_total = time.time()
    finished = run_parallel(sweep_map, local_reg_map, args.n_runs,
                            args.project, args.max_concurrent, args.gpu_id)
    total_hrs = (time.time() - t_total) / 3600

    # ── Phase 3: Merge local registries + summary ─────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Phase 3: Merging results")
    print(f"{'─'*60}")
    merge_local_registries(sweep_map, local_reg_map)

    print_summary(sweep_map, finished)
    print(f"  Total wall time: {total_hrs:.2f} hours")

    # Append parallel run metadata to main registry
    parallel_entry = {
        "type":           "parallel_sweep",
        "datasets":       list(sweep_map.keys()),
        "sweep_ids":      sweep_map,
        "n_runs":         args.n_runs,
        "max_concurrent": args.max_concurrent,
        "gpu_id":         args.gpu_id,
        "created_at":     datetime.now().isoformat(),
        "total_hrs":      round(total_hrs, 2),
        "results":        {
            ds: {"sweep_id": sid, "returncode": rc,
                 "elapsed_min": round(el, 1)}
            for ds, (sid, rc, el) in finished.items()
        },
    }
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    registry.append(parallel_entry)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


if __name__ == "__main__":
    main()
