"""Which rows move together, and the bootstrap that respects it.

A resample of company-years must draw whole companies, and a resample of
trading days must draw whole runs of days; otherwise it treats rows that move
together as independent and the interval comes out too narrow. This module
reduces each :class:`ResamplingUnit` to ONE representation -- a dense unit id
per row -- so every statistic downstream handles all three cases with the
same code.

A bootstrap resample is represented as a WEIGHT per row: the number of times
that row's unit was drawn. Drawing ``G`` units with replacement and gathering
their rows is the same multiset as weighting each row by its unit's draw
count, and the weighted form never materialises a resampled array, so the
statistic can reuse anything it precomputed on the original order.

No ``_test_hooks.py``: the only randomness is a ``numpy`` generator seeded by
an argument, so every draw is a pure function of the inputs.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .types import (
    ERR_INVALID_BLOCK,
    ERR_INVALID_RESAMPLES,
    ERR_LENGTH_MISMATCH,
    ERR_TOO_FEW_UNITS,
    ResamplingUnit,
)


class Resampling(NamedTuple):
    """The independent units of one held-out fold.

    Args:
        unit: What one independent observation is.
        groups: Dense unit id per row, in ``[0, n_units)``.
        n_units: Number of independent units.
        block_length: Rows per block for ``block``, else 1.
    """

    unit: ResamplingUnit
    groups: NDArray[np.int64]
    n_units: int
    block_length: int


def build_resampling(
    unit: ResamplingUnit,
    n_rows: int,
    clusters: NDArray[np.int64],
    block_length: int,
) -> Resampling:
    """Assign every held-out row to its independent unit.

    Args:
        unit: What one independent observation is.
        n_rows: Held-out rows.
        clusters: Group code per row. Read only for ``cluster``; for the
            other units it must still cover the rows, so a caller cannot
            pass one fold's codes beside another fold's rows unnoticed.
        block_length: Rows per block. Must be at least 2 for ``block`` (a
            block of one row is a row) and exactly 1 otherwise.

    Returns:
        The unit assignment.

    Raises:
        ValueError: If ``clusters`` does not cover the rows, if
            ``block_length`` does not fit the unit, or if fewer than two
            independent units result, which leaves no spread to measure.
    """
    if len(clusters) != n_rows:
        raise ValueError(
            f"[{ERR_LENGTH_MISMATCH}] clusters must cover {n_rows} rows, got {len(clusters)}"
        )
    if unit is ResamplingUnit.BLOCK:
        if block_length < 2:
            raise ValueError(
                f"[{ERR_INVALID_BLOCK}] a block must hold at least 2 rows, got {block_length}"
            )
        groups: NDArray[np.int64] = np.arange(n_rows, dtype=np.int64) // block_length
    elif block_length != 1:
        raise ValueError(
            f"[{ERR_INVALID_BLOCK}] block_length applies to the block unit only; "
            f"got {block_length} for {unit.value}"
        )
    elif unit is ResamplingUnit.CLUSTER:
        inverse: NDArray[np.intp] = np.unique(clusters, return_inverse=True)[1]
        groups = inverse.astype(np.int64)
    else:
        groups = np.arange(n_rows, dtype=np.int64)

    n_units = int(np.max(groups)) + 1 if n_rows > 0 else 0
    if n_units < 2:
        raise ValueError(
            f"[{ERR_TOO_FEW_UNITS}] at least 2 independent {unit.value} units are needed, "
            f"got {n_units} from {n_rows} rows"
        )
    return Resampling(unit=unit, groups=groups, n_units=n_units, block_length=block_length)


def unit_sums(values: NDArray[np.float64], resampling: Resampling) -> NDArray[np.float64]:
    """Sum a per-row quantity within each unit.

    Args:
        values: One value per held-out row.
        resampling: The unit assignment of those rows.

    Returns:
        One sum per unit, shape (n_units,).

    Raises:
        ValueError: If ``values`` does not cover the rows.
    """
    if len(values) != len(resampling.groups):
        raise ValueError(
            f"[{ERR_LENGTH_MISMATCH}] values must cover {len(resampling.groups)} rows, "
            f"got {len(values)}"
        )
    # A weighted bincount returns float64 at run time, but numpy's stubs type
    # every bincount as integer; astype states the real dtype rather than
    # casting the annotation.
    sums: NDArray[np.float64] = np.bincount(
        resampling.groups, weights=values, minlength=resampling.n_units
    ).astype(np.float64)
    return sums


def bootstrap_weights(
    resampling: Resampling,
    n_resamples: int,
    seed: int,
) -> Generator[NDArray[np.float64], None, None]:
    """Yield one row-weight vector per bootstrap resample.

    Each resample draws ``n_units`` units with replacement; a row's weight is
    how many times its unit was drawn.

    Args:
        resampling: The unit assignment of the held-out rows.
        n_resamples: Resamples to draw, at least 2.
        seed: Seed of the draws.

    Yields:
        Row weights, shape (n_rows,), each summing to the row count of the
        drawn units.

    Raises:
        ValueError: If fewer than two resamples are requested.
    """
    if n_resamples < 2:
        raise ValueError(
            f"[{ERR_INVALID_RESAMPLES}] at least 2 resamples are needed, got {n_resamples}"
        )
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        drawn: NDArray[np.int64] = rng.integers(
            0, resampling.n_units, size=resampling.n_units, dtype=np.int64
        )
        counts: NDArray[np.intp] = np.bincount(drawn, minlength=resampling.n_units)
        weights: NDArray[np.float64] = counts[resampling.groups].astype(np.float64)
        yield weights


__all__ = ["Resampling", "bootstrap_weights", "build_resampling", "unit_sums"]
