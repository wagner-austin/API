"""Explicit chronological fit/calibrate/validate partitions for warning research."""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.json_utils import JSONObject, narrow_json_to_dict, require_list
from typing_extensions import ReadOnly, TypedDict

from .warning_history import require_warning_history
from .warning_panel import (
    WarningRow,
    decode_warning_row,
    encode_warning_row,
    require_warning_split,
)


class WarningFold(TypedDict):
    """Immutable explicit partitions; calibration never shares fitting borrowers."""

    fitting: ReadOnly[tuple[WarningRow, ...]]
    calibration: ReadOnly[tuple[WarningRow, ...]]
    validation: ReadOnly[tuple[WarningRow, ...]]


def decode_warning_fold(document: JSONObject) -> WarningFold:
    """Decode fold evidence; study validation separately checks chronology.

    Args:
        document: Three lists of archived warning opportunities.

    Returns:
        WarningFold: Typed immutable partitions.

    Raises:
        TypeError: Required lists or nested row types are invalid.
        ValueError: Nested evidence violates the warning row contract.
    """
    return WarningFold(
        fitting=tuple(
            decode_warning_row(narrow_json_to_dict(v)) for v in require_list(document, "fitting")
        ),
        calibration=tuple(
            decode_warning_row(narrow_json_to_dict(v))
            for v in require_list(document, "calibration")
        ),
        validation=tuple(
            decode_warning_row(narrow_json_to_dict(v)) for v in require_list(document, "validation")
        ),
    )


def encode_warning_fold(fold: WarningFold) -> JSONObject:
    """Encode all partition assignments and their source-pinned evidence.

    Args:
        fold: Explicit fold definition.

    Returns:
        JSONObject: Replayable partition record.
    """
    return {
        "fitting": [encode_warning_row(row) for row in fold["fitting"]],
        "calibration": [encode_warning_row(row) for row in fold["calibration"]],
        "validation": [encode_warning_row(row) for row in fold["validation"]],
    }


def require_warning_fold(fold: WarningFold, as_of_ms: int) -> None:
    """Require three borrower-disjoint stages with mature earlier-stage labels.

    Args:
        fold: Fitting, calibration and later validation opportunities.
        as_of_ms: Analysis cutoff for validation outcomes.

    Returns:
        None: All stages satisfy chronological and borrower exclusion constraints.

    Raises:
        ValueError: Existing WARN-008/009 partition validation errors propagate;
            WARN-024 rejects inconsistent evidence; WARN-030 rejects borrower re-entry.
    """
    require_warning_split(fold["fitting"], fold["calibration"], as_of_ms)
    require_warning_split((*fold["fitting"], *fold["calibration"]), fold["validation"], as_of_ms)
    require_warning_history((*fold["fitting"], *fold["calibration"], *fold["validation"]))


def require_warning_study(folds: Sequence[WarningFold], final: WarningFold, as_of_ms: int) -> None:
    """Keep final calibration and holdout borrowers out of every tuning fold.

    Args:
        folds: Ordered development folds; validation opportunities must not repeat.
        final: Final fitting, untouched calibration, and later holdout partitions.
        as_of_ms: Latest permitted label availability.

    Returns:
        None: Explicit partitions are suitable for the implemented study workflow.

    Raises:
        ValueError: WARN-015 for absent folds, reused holdout identities, repeated
            validation opportunities or development later than final calibration.
            Existing partition validation errors propagate.
    """
    if not folds:
        raise ValueError("[WARN-015] chronological tuning requires development folds")
    require_warning_fold(final, as_of_ms)
    reserved = (*final["calibration"], *final["validation"])
    previous_end = -1
    for fold in folds:
        require_warning_fold(fold, as_of_ms)
        development = (*fold["fitting"], *fold["calibration"], *fold["validation"])
        require_warning_split(development, reserved, as_of_ms)
        start = min(row["cutoff_ms"] for row in fold["validation"])
        if start <= previous_end:
            raise ValueError("[WARN-015] validation windows must be strictly ordered")
        previous_end = max(row["cutoff_ms"] for row in fold["validation"])
    require_warning_history(
        tuple(
            row
            for fold in (*folds, final)
            for rows in (fold["fitting"], fold["calibration"], fold["validation"])
            for row in rows
        )
    )


__all__ = [
    "WarningFold",
    "decode_warning_fold",
    "encode_warning_fold",
    "require_warning_fold",
    "require_warning_study",
]
