"""Measuring a sweep's seeds in blocks, across several jobs, and assembling them.

WHY THIS EXISTS. A variance pilot can demand more seeds than one job can
measure. The trait arc's pilot did: its paired spread put the smallest effect
this arc has acted on 138 seeds away, on a CPU partition that measures about
ten minutes of grid per seed and preempts without warning. One job of 138
seeds is a day of wall time with whole cells -- every seed of a compartment
count -- at risk on each eviction.

THE BLOCK IS THE UNIT. A plan's seeds are cut into consecutive blocks of the
minimum replicate count, and every cell a sweep checkpoints is one block of one
measurement (``n4@b17``). That is the whole change to what a sweep measures,
and it changes no number, for two reasons that are each enforced elsewhere:
every arm is a function of its own seed (``train_cartridge`` re-seeds after
the geometry probe and says why), and the spacing between one replicate's
cartridge draws is the PLAN's seed count, passed as ``seed_stride``, rather
than the count of whatever call measured it. A block therefore draws exactly
the cartridges a straight run of the whole plan would have drawn.

SHARDS ARE A PARTITION OF CELLS, AND A MERGE IS A RESUME. A shard job runs the
cells whose position in the sweep's list falls to it and leaves its completed
checkpoint in its own directory; that file is its output. A merge copies every
shard's cells into one checkpoint and then runs the sweep as usual, so every
cell RESUMES and the reduction is the one an unsharded run performs on its own
cells. There is no second assembly path to drift from the first, which is the
only way "sharded and unsharded records are identical" stays true.

WHY NO ``_test_hooks.py``: the only seam is the filesystem, for the reason
:mod:`~model_trainer.core.services.model.cartridge_sweep_checkpoint` gives.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.run_record import Observation
from typing_extensions import TypedDict

from model_trainer.core.contracts.replicated_measurement import MIN_SEEDS
from model_trainer.core.contracts.sweep_checkpoint import (
    SWEEP_CHECKPOINT_SCHEMA_VERSION,
    CellRecord,
    SweepCheckpoint,
    sweep_checkpoint_mismatches,
)
from model_trainer.core.services.model.cartridge_sweep_checkpoint import (
    checkpoint_exists,
    checkpointed_cells,
    load_sweep_checkpoint,
    require_distinct_cells,
    resume_or_start,
    run_cells,
    save_sweep_checkpoint,
)

#: Seeds per block: the fewest any measurement here accepts, so the finest
#: grain a preemption can cost and a shard can be handed.
SEED_BLOCK_SIZE: Final[int] = MIN_SEEDS


class SweepShard(TypedDict):
    """One job's share of a sharded sweep.

    Attributes:
        root: Directory every shard of this sweep writes beneath.
        count: How many shards the sweep is cut into.
        index: Which one this is, from zero.
    """

    root: Path
    count: int
    index: int


class SweepMerge(TypedDict):
    """Where a merge finds the shards it assembles.

    Attributes:
        root: Directory every shard of the sweep wrote beneath.
        count: How many shards the sweep was cut into.
    """

    root: Path
    count: int


def seed_blocks(seeds: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Cut a plan's seeds into consecutive blocks of :data:`SEED_BLOCK_SIZE`.

    Args:
        seeds: The plan's seeds, in order.

    Returns:
        The blocks, in seed order.

    Raises:
        AppError: With ``CARTRIDGE_SEED_BLOCKS_UNEVEN`` when the count is not
            a positive multiple of the block size: a ragged last block would
            be a replicate set smaller than any measurement accepts.
    """
    if not seeds or len(seeds) % SEED_BLOCK_SIZE:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_SEED_BLOCKS_UNEVEN,
            (
                f"{len(seeds)} seed(s) do not cut into whole blocks of {SEED_BLOCK_SIZE}; "
                f"declare a multiple of {SEED_BLOCK_SIZE} so every block is a replicate set "
                f"the measurements accept"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_SEED_BLOCKS_UNEVEN),
        )
    return tuple(
        tuple(seeds[start : start + SEED_BLOCK_SIZE])
        for start in range(0, len(seeds), SEED_BLOCK_SIZE)
    )


def block_cell(name: str, index: int) -> str:
    """Name one block of one measurement as a checkpoint cell.

    Args:
        name: The measurement's own cell name, e.g. ``"n4"``.
        index: The block's position in :func:`seed_blocks`.

    Returns:
        The cell name, e.g. ``"n4@b17"``.
    """
    return f"{name}@b{index}"


def gather_blocks(
    produced: Mapping[str, tuple[Observation, ...]], name: str, *, blocks: int
) -> tuple[Observation, ...]:
    """Concatenate one measurement's block cells, for its arms to be rebuilt.

    Only the per-seed rows of the result are meaningful across blocks; the
    per-block means and spreads in it describe one block each, and a caller
    rebuilds every arm over all seeds rather than reading them.

    Args:
        produced: The sweep's cells, as :func:`checkpointed_cells` returns them.
        name: The measurement's own cell name.
        blocks: How many blocks the plan's seeds were cut into.

    Returns:
        Every block's rows, in block order.
    """
    return tuple(row for index in range(blocks) for row in produced[block_cell(name, index)])


def shard_directory(root: Path, *, index: int, count: int) -> Path:
    """Locate one shard's checkpoint directory.

    Args:
        root: Directory every shard of the sweep writes beneath.
        index: The shard, from zero.
        count: How many shards the sweep is cut into, so two cuts of one
            sweep never share a directory.

    Returns:
        The directory, whether or not it exists.
    """
    return root / f"shard-{index}-of-{count}"


def owned_cells(
    cells: Sequence[tuple[str, Callable[[], Sequence[Observation]]]], shard: SweepShard
) -> list[tuple[str, Callable[[], Sequence[Observation]]]]:
    """Select the cells one shard measures: every ``count``-th, from ``index``.

    By POSITION, so a shard never needs to know what a cell measures, and a
    sweep's expensive cells -- which it lists together -- spread across shards
    rather than landing on one.

    Args:
        cells: The sweep's whole cell list, in order.
        shard: The shard.

    Returns:
        This shard's cells, in order.
    """
    return [
        cell for position, cell in enumerate(cells) if position % shard["count"] == shard["index"]
    ]


def measure_shard(
    shard: SweepShard,
    *,
    measurement: str,
    inputs_digest: str,
    cells: Sequence[tuple[str, Callable[[], Sequence[Observation]]]],
) -> Path:
    """Run one shard's cells and publish its checkpoint as its output.

    PUBLISHED BEFORE THE FIRST CELL, so a shard that owns no cell of this
    measurement still leaves a file a merge can tell from a shard that never
    ran. The whole list is checked for distinct names, not just this shard's
    share, because a collision between two shards' cells would only surface
    at the merge.

    Args:
        shard: Which share to run.
        measurement: Names the sweep.
        inputs_digest: Digest of what the sweep measures over.
        cells: The sweep's WHOLE cell list; this shard runs its share.

    Returns:
        The shard's checkpoint directory.

    Raises:
        AppError: With ``CARTRIDGE_CHECKPOINT_DUPLICATE_CELL`` when two cells
            share a name, or ``CARTRIDGE_CHECKPOINT_FOREIGN`` when the shard's
            directory holds a checkpoint over other inputs.
    """
    require_distinct_cells(cells)
    directory = shard_directory(shard["root"], index=shard["index"], count=shard["count"])
    save_sweep_checkpoint(
        directory,
        resume_or_start(directory, measurement=measurement, inputs_digest=inputs_digest),
    )
    run_cells(
        directory,
        measurement=measurement,
        inputs_digest=inputs_digest,
        cells=owned_cells(cells, shard),
    )
    return directory


def adopt_shards(
    directory: Path,
    *,
    root: Path,
    count: int,
    measurement: str,
    inputs_digest: str,
    cells: Sequence[str],
) -> None:
    """Copy every shard's completed cells into one checkpoint for a merge to resume.

    REFUSES RATHER THAN ASSEMBLES FROM WHAT IS THERE. A missing shard or a
    missing cell would otherwise surface as a rebuild failure deep in the
    reduction, or -- worse -- not at all, for a cell whose rows nobody reads.
    The shards' own directories are left untouched; they are the evidence of
    which node measured which block.

    Args:
        directory: The merge run's checkpoint directory.
        root: Directory every shard of the sweep wrote beneath.
        count: How many shards the sweep was cut into.
        measurement: Names the sweep.
        inputs_digest: Digest of what the merge measures over; every shard
            must have measured over the same.
        cells: Every cell name the sweep needs, in the order it runs them.

    Raises:
        AppError: With ``CARTRIDGE_SHARD_INCOMPLETE`` when a shard has no
            checkpoint or a needed cell is in none of them,
            ``CARTRIDGE_CHECKPOINT_FOREIGN`` when a shard measured other
            inputs or a cell this sweep does not have, and
            ``CARTRIDGE_CHECKPOINT_DUPLICATE_CELL`` when two shards hold the
            same cell.
    """
    found: dict[str, CellRecord] = {}
    absent: list[str] = []
    duplicated: list[str] = []
    for index in range(count):
        shard_dir = shard_directory(root, index=index, count=count)
        if not checkpoint_exists(shard_dir, measurement):
            absent.append(str(shard_dir))
            continue
        checkpoint = load_sweep_checkpoint(shard_dir, measurement)
        mismatches = sweep_checkpoint_mismatches(
            checkpoint, measurement=measurement, inputs_digest=inputs_digest
        )
        if mismatches:
            raise AppError(
                ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_FOREIGN,
                f"shard {shard_dir} measured something else: {'; '.join(mismatches)}",
                model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_FOREIGN),
            )
        for record in checkpoint["cells"]:
            if record["cell"] in found:
                duplicated.append(record["cell"])
            found[record["cell"]] = record
    if duplicated:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_DUPLICATE_CELL,
            f"cell(s) {', '.join(duplicated)} were measured by more than one shard",
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_DUPLICATE_CELL),
        )
    foreign = [name for name in found if name not in cells]
    if foreign:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_FOREIGN,
            f"shards hold cell(s) {', '.join(foreign)} that this sweep does not have",
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_FOREIGN),
        )
    missing = [name for name in cells if name not in found]
    if absent or missing:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_SHARD_INCOMPLETE,
            (
                f"{measurement} cannot be assembled: {len(absent)} shard(s) have no "
                f"checkpoint ({', '.join(absent) or 'none'}) and {len(missing)} cell(s) "
                f"are in no shard ({', '.join(missing) or 'none'}); run those shards first"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_SHARD_INCOMPLETE),
        )
    save_sweep_checkpoint(
        directory,
        SweepCheckpoint(
            schema_version=SWEEP_CHECKPOINT_SCHEMA_VERSION,
            measurement=measurement,
            inputs_digest=inputs_digest,
            cells=[found[name] for name in cells],
        ),
    )


def assembled_cells(
    directory: Path,
    *,
    merge: SweepMerge | None,
    measurement: str,
    inputs_digest: str,
    cells: Sequence[tuple[str, Callable[[], Sequence[Observation]]]],
) -> Mapping[str, tuple[Observation, ...]]:
    """Run a phase's cells straight, or adopt them from shards and resume them.

    The one call a sweep makes for a phase it REDUCES. With no merge it is
    :func:`checkpointed_cells`; with one, the shards' cells are adopted first,
    so every cell resumes and the reduction that follows is the straight
    run's own.

    Args:
        directory: This run's checkpoint directory.
        merge: The shards to assemble from, or None for a straight run.
        measurement: Names the phase.
        inputs_digest: Digest of what the sweep measures over.
        cells: The phase's whole cell list, in order.

    Returns:
        Each cell's observations, keyed by cell name, in the order given.

    Raises:
        AppError: Propagated from :func:`adopt_shards` and
            :func:`checkpointed_cells`.
    """
    if merge is not None:
        adopt_shards(
            directory,
            root=merge["root"],
            count=merge["count"],
            measurement=measurement,
            inputs_digest=inputs_digest,
            cells=[name for name, _measure in cells],
        )
    return checkpointed_cells(
        directory, measurement=measurement, inputs_digest=inputs_digest, cells=cells
    )


__all__ = [
    "SEED_BLOCK_SIZE",
    "SweepMerge",
    "SweepShard",
    "adopt_shards",
    "assembled_cells",
    "block_cell",
    "gather_blocks",
    "measure_shard",
    "owned_cells",
    "seed_blocks",
    "shard_directory",
]
