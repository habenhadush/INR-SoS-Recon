"""Persist per-sample reconstructed slowness arrays alongside results.json.

results.json carries metrics only. Figure builders (thesis reconstruction
grids) need the actual per-sample reconstructed fields, often composed across
several run dirs. This module writes a compact `recons.npz` next to
results.json so figures are a pure read-and-plot job — no re-running, no
training randomness, and the figure always matches the run's reported metrics.

Stored arrays are RAW flattened slowness (length 4096), exactly as the
engines produce `s_phys`. Convert to SoS and reshape at plot time:
    sos = np.clip(1.0 / (s + 1e-10), 1200, 1800)
    img = sos.reshape((64, 64), order="F")
(see inr_sos.visualization.report_figures._slowness_to_sos / _reshape).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _stack(arrays) -> np.ndarray:
    """Stack a list of per-sample arrays into (n_samples, 4096) float32."""
    flat = []
    for a in arrays:
        if hasattr(a, "detach"):
            a = a.detach().cpu().numpy()
        flat.append(np.asarray(a, dtype=np.float32).flatten())
    return np.stack(flat, axis=0)


def save_recons_npz(path, indices, gt_slowness, recons_by_method) -> Path:
    """Write per-sample reconstructed slowness arrays to a compressed .npz.

    Args:
        path:            output .npz path.
        indices:         list[int] — sample indices, in order.
        gt_slowness:     list of flat slowness arrays (one per sample) — GT.
        recons_by_method: dict[label -> list of flat slowness arrays], one
                          array per sample, same order as `indices`.

    npz layout:
        indices : (n,) int64
        gt      : (n, 4096) float32
        labels  : 0-d str  — json list of method labels, index-aligned
        recon_000, recon_001, ... : (n, 4096) float32  — per method
    """
    path = Path(path)
    labels = list(recons_by_method.keys())
    payload = {
        "indices": np.asarray(indices, dtype=np.int64),
        "gt": _stack(gt_slowness),
        "labels": np.asarray(json.dumps(labels)),
    }
    for i, lbl in enumerate(labels):
        payload[f"recon_{i:03d}"] = _stack(recons_by_method[lbl])
    np.savez_compressed(path, **payload)
    return path


def load_recons_npz(path) -> dict:
    """Load a recons.npz written by save_recons_npz.

    Returns:
        {"indices": (n,) int64,
         "gt": (n, 4096) float32,
         "recons": {label: (n, 4096) float32}}
    """
    z = np.load(Path(path), allow_pickle=False)
    labels = json.loads(str(z["labels"]))
    return {
        "indices": z["indices"],
        "gt": z["gt"],
        "recons": {lbl: z[f"recon_{i:03d}"] for i, lbl in enumerate(labels)},
    }
