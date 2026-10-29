"""Exclude labels unavailable before the next chronological partition."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..benchmarking.splitting import SplitMasks
from .types import ERR_INVALID_ARGUMENT, ERR_LENGTH_MISMATCH, ERR_TOO_FEW_UNITS


def purge_label_overlap(
    times: NDArray[np.int64], available_at: NDArray[np.int64], masks: SplitMasks
) -> SplitMasks:
    """Purge by actual label availability, with strict chronological boundaries.

    Args:
        times: Decision time per row, in one consistently ordered time unit.
        available_at: Label observation/valuation availability in that same unit.
        masks: Disjoint train, early-stopping and scoring membership.

    Returns:
        Masks with training labels strictly before the first stopping decision
        and stopping labels strictly before the first scoring decision. Scoring
        labels are never used to decide whether a training row is retained.

    Raises:
        ValueError: CVML-EVAL-001 for shapes; CVML-EVAL-018 for reversed labels,
            overlapping or nonchronological folds; CVML-EVAL-003 for empty folds.
    """
    arrays = (times, available_at, masks.train, masks.val, masks.test)
    if any(values.ndim != 1 or len(values) != len(times) for values in arrays):
        raise ValueError(f"[{ERR_LENGTH_MISMATCH}] times, availability and masks must align")
    for mask in (masks.train, masks.val, masks.test):
        if not bool(np.any(mask)):
            raise ValueError(f"[{ERR_TOO_FEW_UNITS}] chronological folds must be nonempty")
    reversed_labels: NDArray[np.bool_] = available_at < times
    membership: NDArray[np.int64] = (
        masks.train.astype(np.int64) + masks.val.astype(np.int64) + masks.test.astype(np.int64)
    )
    overlapping: NDArray[np.bool_] = membership > 1
    if bool(np.any(reversed_labels)) or bool(np.any(overlapping)):
        raise ValueError(f"[{ERR_INVALID_ARGUMENT}] reversed label intervals or overlapping folds")
    stopping = int(np.min(times[masks.val]))
    scoring = int(np.min(times[masks.test]))
    if int(np.max(times[masks.train])) >= stopping or int(np.max(times[masks.val])) >= scoring:
        raise ValueError(f"[{ERR_INVALID_ARGUMENT}] folds must be strictly chronological")
    known_train: NDArray[np.bool_] = available_at < stopping
    known_val: NDArray[np.bool_] = available_at < scoring
    train: NDArray[np.bool_] = masks.train & known_train
    val: NDArray[np.bool_] = masks.val & known_val
    if not bool(np.any(train)) or not bool(np.any(val)):
        raise ValueError(f"[{ERR_TOO_FEW_UNITS}] label overlap exclusion emptied a fold")
    return SplitMasks(train=train, val=val, test=masks.test.copy())
