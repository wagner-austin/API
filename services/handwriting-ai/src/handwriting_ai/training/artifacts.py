"""Artifact pruning utilities for model directory cleanup.

This module provides functions for cleaning up old model snapshots.
Artifact creation is handled by the jobs layer using platform_ml.ArtifactStore.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger


def _run_id_from_name(name: str) -> tuple[str, str] | None:
    """Return (kind, run_id) where kind is 'model' or 'manifest' for unique snapshot files.

    - model-<run_id>.pt -> ("model", <run_id>)
    - manifest-<run_id>.json -> ("manifest", <run_id>)
    Returns None for canonical files and non-matching names.
    """
    if name.startswith("model-") and name.endswith(".pt"):
        rid = name[len("model-") : -len(".pt")]
        return ("model", rid) if rid else None
    if name.startswith("manifest-") and name.endswith(".json"):
        rid = name[len("manifest-") : -len(".json")]
        return ("manifest", rid) if rid else None
    return None


def _delete_paths(paths: list[Path]) -> list[Path]:
    deleted: list[Path] = []
    for p in paths:
        try:
            p.unlink()
            deleted.append(p)
        except OSError as exc:
            get_logger("handwriting_ai").error("prune_delete_failed path=%s error=%s", p, exc)
            raise
    return deleted


def prune_model_artifacts(model_dir: Path, keep_runs: int) -> list[Path]:
    """Remove older unique snapshot files, keeping the newest N runs.

    Preserves canonical files (model.pt, manifest.json).
    Keeps the newest `keep_runs` of unique snapshots identified by run_id in filenames.
    Returns a list of deleted file paths.
    """
    keep = max(0, int(keep_runs))
    try:
        entries = list(model_dir.iterdir())
    except OSError as exc:
        get_logger("handwriting_ai").error("prune_list_failed dir=%s error=%s", model_dir, exc)
        raise

    # Collect all run_ids present in either side
    run_ids = {m[1] for p in entries if (m := _run_id_from_name(p.name)) is not None}

    if keep <= 0:
        to_delete = [p for p in entries if _run_id_from_name(p.name) is not None]
        return _delete_paths(to_delete)

    if not run_ids:
        return []

    # Sort run ids lexicographically (timestamp-first names sort correctly)
    sorted_ids = sorted(run_ids)
    keep_set = set(sorted_ids[-keep:])
    to_delete = [
        p for p in entries if (m := _run_id_from_name(p.name)) is not None and m[1] not in keep_set
    ]
    return _delete_paths(to_delete)
