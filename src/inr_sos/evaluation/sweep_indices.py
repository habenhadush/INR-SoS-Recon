"""
sweep_indices.py
----------------
Utility to load the sample indices that were consumed during a Bayesian sweep
so that reconstruction runner scripts can exclude them by default.

All three sweep types (standard INR, joint, denoiser) write their target
indices into ``scripts/sweep_registry.json`` under the ``"indices"`` key
of their respective entry.  ``run_best.py`` additionally writes
``validation.holdout_indices`` into the same entry.

Public API
----------
load_sweep_indices(dataset_key, sweep_id=None, registry_path=None)
    Return the set of all indices that were seen during the sweep phase and
    (optionally) any subsequent held-out validation phase.  Returns an empty
    set when the registry is absent or the entry has no recorded indices.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Default location — scripts/ directory sits one level above src/
_DEFAULT_REGISTRY = Path(__file__).parent.parent.parent.parent / "scripts" / "sweep_registry.json"


def load_sweep_indices(
    dataset_key: str,
    sweep_id: Optional[str] = None,
    registry_path: Optional[Path] = None,
) -> set[int]:
    """Return the union of all sample indices used during the sweep phase.

    Parameters
    ----------
    dataset_key:
        Dataset key as it appears in datasets.yaml, e.g. ``"kwave_geom"``.
        Used to select the most recent matching registry entry when
        ``sweep_id`` is not given.
    sweep_id:
        Optional prefix of the W&B sweep ID (e.g. ``"hqt6bwmp"``).  When
        given, the matching entry is selected by ID rather than by dataset key.
    registry_path:
        Path to ``sweep_registry.json``.  Defaults to
        ``<repo_root>/scripts/sweep_registry.json``.

    Returns
    -------
    set[int]
        All indices that appeared in ``entry["indices"]`` or
        ``entry["validation"]["holdout_indices"]`` for the matching entry.
        Returns an empty set if the registry does not exist, the key is not
        found, or no indices were recorded.
    """
    path = Path(registry_path) if registry_path is not None else _DEFAULT_REGISTRY

    if not path.exists():
        log.warning("sweep_registry.json not found at %s — no indices excluded", path)
        return set()

    with open(path) as fh:
        registry: list[dict] = json.load(fh)

    entry = _find_entry(registry, dataset_key, sweep_id)
    if entry is None:
        log.warning(
            "No sweep entry found for dataset=%r sweep_id=%r — no indices excluded",
            dataset_key,
            sweep_id,
        )
        return set()

    indices: set[int] = set()

    # Primary sweep indices
    indices.update(int(i) for i in entry.get("indices", []))

    # Holdout indices used in run_best.py validation
    validation = entry.get("validation", {})
    indices.update(int(i) for i in validation.get("holdout_indices", []))

    # Any other sub-keys that record indices (e.g. "reconstruction", "comparison")
    for key, value in entry.items():
        if isinstance(value, dict) and "indices" in value:
            indices.update(int(i) for i in value["indices"])

    log.info(
        "Sweep exclusion: %d indices from entry sweep_id=%s dataset=%s",
        len(indices),
        entry.get("sweep_id", "?"),
        entry.get("dataset", "?"),
    )
    return indices


def _find_entry(
    registry: list[dict],
    dataset_key: str,
    sweep_id: Optional[str],
) -> Optional[dict]:
    """Select a registry entry.

    Priority:
    1. If ``sweep_id`` is given, match by sweep_id prefix (most-recent first).
    2. Otherwise, find the most-recent entry whose ``dataset`` field matches
       ``dataset_key``.
    """
    if sweep_id is not None:
        matches = [e for e in registry if e.get("sweep_id", "").startswith(sweep_id)]
        return matches[-1] if matches else None

    # Most-recent (last) entry whose dataset matches
    matches = [e for e in registry if e.get("dataset") == dataset_key]
    return matches[-1] if matches else None
