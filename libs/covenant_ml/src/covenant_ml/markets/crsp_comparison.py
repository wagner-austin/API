"""Exact semantic row differences for private CRSP vintage diagnostics.

This module performs no I/O. Its rows are licensed evidence: callers must
retain them only inside private evaluator storage or encrypted artifacts.
It neither judges economic completeness nor admits a dataset for study use.
"""

from __future__ import annotations

from typing import NamedTuple

from platform_core.json_utils import JSONObject

from .crsp_extraction_records import CRSP_RELATIONS
from .crsp_rows import crsp_row_key


class CrspRowDifference(NamedTuple):
    """Internal row accounting; changed pairs contain the old row then the new row."""

    unchanged: int
    added: tuple[JSONObject, ...]
    removed: tuple[JSONObject, ...]
    changed: tuple[tuple[JSONObject, JSONObject], ...]


def _indexed(rows: tuple[JSONObject, ...], relation: str) -> dict[tuple[str, ...], JSONObject]:
    """Refuse duplicate keys instead of allowing dictionary overwrite to hide evidence."""
    indexed = {crsp_row_key(row, relation): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("CRSP_COMPARISON_KEY: duplicate source key in a comparison input")
    return indexed


def compare_crsp_rows(
    old: tuple[JSONObject, ...], fresh: tuple[JSONObject, ...], relation: str
) -> CrspRowDifference:
    """Account for every decoded row under the canonical reader's natural keys.

    Args:
        old: Frozen-vintage rows already validated by the source reader.
        fresh: Fresh-vintage rows validated under the same protocol projection.
        relation: One of the four protocol relation roles.

    Returns:
        Unchanged count, additions, removals and old/new changed pairs in
        deterministic key order. Null and zero differ. No tolerance, rounding
        or economic interpretation is applied. Editing a key removes the old
        row and adds the new one. Empty inputs have explicit zero counts.

    Raises:
        ValueError: CRSP_COMPARISON_RELATION for an unknown role, or
            CRSP_COMPARISON_KEY for duplicate keys on either side.
    """
    if relation not in CRSP_RELATIONS:
        raise ValueError("CRSP_COMPARISON_RELATION: require one of the four source relations")
    before, after = _indexed(old, relation), _indexed(fresh, relation)
    shared = sorted(before.keys() & after.keys())
    changed = tuple((before[key], after[key]) for key in shared if before[key] != after[key])
    return CrspRowDifference(
        unchanged=len(shared) - len(changed),
        added=tuple(after[key] for key in sorted(after.keys() - before.keys())),
        removed=tuple(before[key] for key in sorted(before.keys() - after.keys())),
        changed=changed,
    )


__all__ = ["CrspRowDifference", "compare_crsp_rows"]
