"""Typed contract for a cartridge sweep interrupted between cells.

WHY THIS EXISTS. ``free-gpu`` carries ``PreemptMode=CANCEL``: an evicted job
is killed outright and Slurm resubmits nothing. Twenty-seven committed ``mi``
run documents -- four of them at 2400 minutes, forty hours -- were relying on
a project-wide ``deterministic: true`` to make that survivable. It says the
RESULT is reproducible, not that the hours are recoverable, and none of the
sweep payloads wrote a checkpoint of any kind.

ONE CONTRACT FOR SIX SWEEPS, and the reason is that they already agree.
``measure_solo_seeds``, ``measure_solo_grid``, ``measure_headroom`` and the
three ``measure_grid`` variants all return
``tuple[tuple[Observation, ...], str]`` -- observations and a digest. They
differ in what a unit of work IS (a seed, a grid cell, a base model, a
compartment count) but not in what one produces, so a checkpoint keyed by
cell name and carrying observations fits every one of them.

WHY A CELL AND NOT AN ITEM. A cell is the smallest unit whose observations
are COMPLETE. Half a cell is not a partial measurement, it is an unfinished
average: ``replicate`` reduces per-seed gains into a mean and a spread, and a
checkpoint holding some of the seeds would let a resumed run report a spread
over fewer draws than the record claims. So a cell is written only once it
has produced every observation it is going to.

WHY NOT THE QUESTION-SET CHECKPOINT. That one carries per-item outcomes
because its comparison is McNemar-paired and needs to know WHICH items each
arm got right. No sweep here does that -- their arms reduce to gains and
spreads -- so carrying outcome vectors would persist megabytes nothing reads.
The two contracts share the atomic write in
:mod:`model_trainer.core.services.model.checkpoint_file` and nothing else,
which is the honest amount of overlap.

THE FINGERPRINT IS THE POINT OF THE FILE, not the cells in it. A resume is
valid only when the run being resumed is the same measurement, and a
mismatch is reported field by field rather than as a boolean, because an
operator told only "refused" deletes the checkpoint and loses exactly the
hours it existed to save.
"""

from __future__ import annotations

from typing import Final

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_list,
    require_str,
)
from platform_core.run_record import Observation, decode_observation, encode_observation
from typing_extensions import TypedDict

#: Version stamp written into every sweep checkpoint. A decoder meeting a
#: different version refuses to resume rather than guessing at field
#: semantics; bump this when the shape below changes.
SWEEP_CHECKPOINT_SCHEMA_VERSION: Final[int] = 1


class CellRecord(TypedDict):
    """One completed unit of a sweep, and everything it contributed.

    Attributes:
        cell: What this unit is, named by the sweep that ran it -- a seed, a
            grid cell's token, a base model, a compartment count. The sweep
            chooses the spelling; this contract only requires that the same
            unit produces the same name on a later run, because that is what
            a resume matches on.
        observations: Every observation this cell contributed, complete. A
            cell is recorded only once it has produced all of them.
    """

    cell: str
    observations: list[Observation]


class SweepCheckpoint(TypedDict):
    """A sweep's completed cells and what identifies the measurement.

    Attributes:
        schema_version: Must equal :data:`SWEEP_CHECKPOINT_SCHEMA_VERSION`.
        measurement: Which sweep this is, so one plan's cells can never be
            adopted by another.
        inputs_digest: Digest of what the sweep was run over, as the sweep's
            own ``measure_*`` computes and returns it. Two runs over
            different corpora are different measurements however alike their
            plans look.
        cells: Completed cells, in the order they finished.
    """

    schema_version: int
    measurement: str
    inputs_digest: str
    cells: list[CellRecord]


def encode_cell_record(record: CellRecord) -> JSONObject:
    """Encode one completed cell.

    Args:
        record: The cell to encode.

    Returns:
        A JSON object.
    """
    observations: list[JSONValue] = [
        encode_observation(observation) for observation in record["observations"]
    ]
    return {"cell": record["cell"], "observations": observations}


def decode_cell_record(obj: JSONObject) -> CellRecord:
    """Decode one completed cell, validating every field.

    Args:
        obj: The JSON object to decode.

    Returns:
        The cell record.

    Raises:
        JSONTypeError: If a field is missing or mistyped, or if an
            observation fails its own validation.
    """
    return CellRecord(
        cell=require_str(obj, "cell"),
        observations=[decode_observation(entry) for entry in require_list(obj, "observations")],
    )


def encode_sweep_checkpoint(checkpoint: SweepCheckpoint) -> JSONObject:
    """Encode a sweep checkpoint.

    Args:
        checkpoint: The checkpoint to encode.

    Returns:
        A JSON object ready to be written.
    """
    cells: list[JSONValue] = [encode_cell_record(record) for record in checkpoint["cells"]]
    return {
        "schema_version": checkpoint["schema_version"],
        "measurement": checkpoint["measurement"],
        "inputs_digest": checkpoint["inputs_digest"],
        "cells": cells,
    }


def decode_sweep_checkpoint(obj: JSONObject) -> SweepCheckpoint:
    """Decode a sweep checkpoint, validating every field.

    Args:
        obj: The JSON object to decode.

    Returns:
        The checkpoint.

    Raises:
        JSONTypeError: If a field is missing or mistyped, or if the schema
            version is not the one this code understands. The version is
            checked HERE rather than by the caller, so no reader can obtain a
            decoded checkpoint without the version having been agreed -- a
            check the caller may forget is not a check.
    """
    version = require_int(obj, "schema_version")
    if version != SWEEP_CHECKPOINT_SCHEMA_VERSION:
        raise JSONTypeError(
            f"schema_version {version} is not {SWEEP_CHECKPOINT_SCHEMA_VERSION}; this "
            f"checkpoint was written by a different version of the sweep and its cells "
            f"cannot be assumed to mean the same thing. Re-run rather than resume."
        )
    cells: list[CellRecord] = []
    for position, entry in enumerate(require_list(obj, "cells")):
        # The POSITION is in the message because a cell is anonymous until it
        # decodes: its own name lives inside the object that would not.
        if not isinstance(entry, dict):
            raise JSONTypeError(f"'cells[{position}]' must be an object")
        cells.append(decode_cell_record(entry))
    return SweepCheckpoint(
        schema_version=version,
        measurement=require_str(obj, "measurement"),
        inputs_digest=require_str(obj, "inputs_digest"),
        cells=cells,
    )


def sweep_checkpoint_mismatches(
    checkpoint: SweepCheckpoint, *, measurement: str, inputs_digest: str
) -> list[str]:
    """Name every way this checkpoint describes a different measurement.

    A LIST RATHER THAN A BOOLEAN. An operator told only that a resume was
    refused deletes the checkpoint; one told ``inputs_digest: checkpoint
    'a1b2' != current 'c3d4'`` knows they pointed the run at a different
    corpus, which is usually the actual mistake and is recoverable.

    Args:
        checkpoint: The checkpoint found on disk.
        measurement: The sweep the current run is performing.
        inputs_digest: Digest of what the current run is measuring over.

    Returns:
        One line per disagreeing field, empty when the checkpoint describes
        this exact measurement.
    """
    expected: tuple[tuple[str, str, str], ...] = (
        ("measurement", checkpoint["measurement"], measurement),
        ("inputs_digest", checkpoint["inputs_digest"], inputs_digest),
    )
    return [
        f"{field}: checkpoint {found!r} != current {wanted!r}"
        for field, found, wanted in expected
        if found != wanted
    ]


def completed_cells(checkpoint: SweepCheckpoint) -> frozenset[str]:
    """Name the cells this checkpoint already holds observations for.

    Args:
        checkpoint: The checkpoint.

    Returns:
        The cell names, for the caller to skip.
    """
    return frozenset(record["cell"] for record in checkpoint["cells"])


def with_cell(checkpoint: SweepCheckpoint, record: CellRecord) -> SweepCheckpoint:
    """Record one more completed cell, without mutating the original.

    A NEW VALUE RATHER THAN AN APPEND. The caller holds a checkpoint that has
    already been written to disk; mutating it in place makes the in-memory
    object and the published file diverge between the append and the next
    save, and a failure in that window leaves a checkpoint claiming a cell
    whose file does not record it -- which on the next run skips work nobody
    did.

    Args:
        checkpoint: The checkpoint so far.
        record: The cell that just finished.

    Returns:
        A checkpoint holding it.
    """
    return SweepCheckpoint(
        schema_version=checkpoint["schema_version"],
        measurement=checkpoint["measurement"],
        inputs_digest=checkpoint["inputs_digest"],
        cells=[*checkpoint["cells"], record],
    )


__all__ = [
    "SWEEP_CHECKPOINT_SCHEMA_VERSION",
    "CellRecord",
    "SweepCheckpoint",
    "completed_cells",
    "decode_cell_record",
    "decode_sweep_checkpoint",
    "encode_cell_record",
    "encode_sweep_checkpoint",
    "sweep_checkpoint_mismatches",
    "with_cell",
]
