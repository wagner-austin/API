"""Persisting a cartridge sweep's completed cells across an eviction.

One rolling file per measurement, rewritten after every completed cell and
published through
:mod:`model_trainer.core.services.model.checkpoint_file`, so a sweep killed
mid-cell resumes at the last cell boundary instead of at zero. On
``free-gpu`` -- ``PreemptMode=CANCEL``, nothing resubmits -- that is the
difference between losing one cell and losing forty hours.

:func:`checkpointed_cells` IS THE ONLY THING A SWEEP SHOULD CALL, and the
functions under it are what it is built from. Six sweeps need this and the
sequence they each need is identical: resume, run the cells that are not
recorded yet, save each one the moment it finishes, and delete the file only
once the last observation exists. Every step of that has a wrong version that
still runs to completion --

* deleting the checkpoint before the reduction leaves a window in which a
  failure loses both the run and the record of what it had already done;
* deleting it at the top of the sweep instead of the bottom makes every
  resume a fresh start, silently;
* not deleting it at all makes the NEXT submission resume a completed run and
  report an earlier execution's numbers as its own;
* two cells sharing a name makes the second one inherit the first one's
  observations -- but only after an eviction, so it passes every test that
  runs the sweep once.

-- which is why the sequence is written once here rather than six times in
the CLIs. The last of those four is refused outright, before any work starts.

WHAT A RESUME COSTS, STATED PLAINLY. This checkpoint carries OBSERVATIONS,
not weights. A resumed sweep rebuilds whatever its cells are measured
against -- the tokenised corpus, the loaded base, and for the LoRA sweeps the
adapter trained over the crowding pool -- because that setup is deterministic
and reproducing it is cheaper than serialising it. So a resume pays the setup
again and skips the measurement, and it is the measurement that is the hours.
It is not free and this module does not pretend it is.

WHY A RESUME REPRODUCES A STRAIGHT RUN'S NUMBERS, which is the property the
whole mechanism rests on and would be worthless without: every training path
these sweeps use re-seeds the process-wide generator from its own seed before
it trains -- ``train_cartridge`` does it deliberately and says why -- so a
cell is a function of its seed and nothing else. Skipping cells therefore
cannot move the cells that follow. If that ever stops being true, resuming
starts producing complete tables of subtly wrong numbers, and nothing
downstream could tell.

LOADING IS STRICT AND REFUSES RATHER THAN RECOVERS. A file that does not
decode, or that decodes to a different measurement, is not repaired and not
ignored: the caller is told which field disagrees and re-runs deliberately.

WHY NO ``_test_hooks.py``. The only seam is the filesystem, and a test that
writes a real file into ``tmp_path`` exercises the real ``os.replace``,
including the atomicity that is the whole point, while a faked filesystem
would exercise the fake. Same reasoning
:mod:`model_trainer.core.services.model.cartridge_qa_power` states for
itself.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeVar

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger
from platform_core.run_record import Observation

from model_trainer.core.contracts.sweep_checkpoint import (
    SWEEP_CHECKPOINT_SCHEMA_VERSION,
    CellRecord,
    SweepCheckpoint,
    decode_sweep_checkpoint,
    encode_sweep_checkpoint,
    sweep_checkpoint_mismatches,
    with_cell,
)
from model_trainer.core.services.model.checkpoint_file import (
    publish_atomically,
    read_checkpoint_object,
    resume_or_refuse,
)

_log = get_logger(__name__)

#: Whatever one sweep's unit of work is described by -- a seed, a compartment
#: count, a base model id, a (cell, seed) pair. This module never inspects
#: one; it only hands it back to the caller's own measuring function.
_UnitT = TypeVar("_UnitT")


def checkpoint_path(directory: Path, measurement: str) -> Path:
    """Locate the checkpoint file for one measurement.

    Args:
        directory: Where checkpoints live for this run.
        measurement: The sweep being run.

    Returns:
        The file's path, whether or not it exists.
    """
    return directory / f"sweep-checkpoint-{measurement}.json"


def checkpoint_exists(directory: Path, measurement: str) -> bool:
    """Report whether a resumable checkpoint is present.

    ``is_file`` rather than ``exists``: a directory at that path would pass an
    existence check and then fail to read.

    Args:
        directory: Where checkpoints live for this run.
        measurement: The sweep being run.

    Returns:
        True when the file is present and is a file.
    """
    return checkpoint_path(directory, measurement).is_file()


def save_sweep_checkpoint(directory: Path, checkpoint: SweepCheckpoint) -> Path:
    """Write a checkpoint atomically, replacing any previous one.

    Args:
        directory: Where checkpoints live for this run.
        checkpoint: The state to persist.

    Returns:
        Path of the published file.
    """
    return publish_atomically(
        checkpoint_path(directory, checkpoint["measurement"]),
        dump_json_str(encode_sweep_checkpoint(checkpoint)),
    )


def load_sweep_checkpoint(directory: Path, measurement: str) -> SweepCheckpoint:
    """Read and validate the checkpoint for one measurement.

    Args:
        directory: Where checkpoints live for this run.
        measurement: The sweep being run.

    Returns:
        The decoded checkpoint.

    Raises:
        JSONTypeError: If the file does not decode, including when its schema
            version is not the one this code understands.
        TypeError: If the file does not hold a JSON object.
        OSError: If the file is absent. Callers ask :func:`checkpoint_exists`
            first; a missing checkpoint is the ordinary case on a first run
            and is not an error this module invents a value for.
    """
    return decode_sweep_checkpoint(read_checkpoint_object(checkpoint_path(directory, measurement)))


def delete_sweep_checkpoint(directory: Path, measurement: str) -> None:
    """Remove a checkpoint once its sweep has completed.

    A COMPLETED RUN DELETES ITS OWN CHECKPOINT. A leftover file is
    indistinguishable from an interrupted run, so the next submission of the
    same sweep would skip cells it should have re-measured and report an
    earlier execution's numbers as its own.

    Args:
        directory: Where checkpoints live for this run.
        measurement: The sweep that completed.
    """
    checkpoint_path(directory, measurement).unlink(missing_ok=True)


def resume_or_start(directory: Path, *, measurement: str, inputs_digest: str) -> SweepCheckpoint:
    """Decide what a starting sweep may reuse, refusing a foreign checkpoint.

    The policy itself lives in
    :func:`~model_trainer.core.services.model.checkpoint_file.resume_or_refuse`,
    which the question-set checkpoints drive too. What is decided HERE is the
    only thing that differs: which fields identify this measurement, and what
    a wrongly resumed run would end up reporting.

    Args:
        directory: Where checkpoints live for this run.
        measurement: The sweep about to run.
        inputs_digest: Digest of what this run is measuring over.

    Returns:
        A checkpoint to accumulate into: the one on disk when it describes
        this measurement, otherwise an empty one.

    Raises:
        AppError: With ``CARTRIDGE_CHECKPOINT_FOREIGN`` when a checkpoint is
            present and describes a different measurement.
    """

    def _load() -> SweepCheckpoint:
        """Read the checkpoint this sweep would resume.

        Returns:
            The decoded checkpoint.
        """
        return load_sweep_checkpoint(directory, measurement)

    def _mismatches(found: SweepCheckpoint) -> list[str]:
        """Name every way the found checkpoint measures something else.

        Args:
            found: The checkpoint on disk.

        Returns:
            One line per disagreeing field.
        """
        return sweep_checkpoint_mismatches(
            found, measurement=measurement, inputs_digest=inputs_digest
        )

    return resume_or_refuse(
        checkpoint_path(directory, measurement),
        fresh=SweepCheckpoint(
            schema_version=SWEEP_CHECKPOINT_SCHEMA_VERSION,
            measurement=measurement,
            inputs_digest=inputs_digest,
            cells=[],
        ),
        load=_load,
        mismatches=_mismatches,
        adopting="cells measured over other inputs",
    )


def cell_or_resume(
    checkpoint: SweepCheckpoint,
    directory: Path,
    cell: str,
    measure: Callable[[], Sequence[Observation]],
) -> tuple[SweepCheckpoint, tuple[Observation, ...]]:
    """Measure one cell, or hand back what an earlier run already measured.

    SAVED IMMEDIATELY AFTER MEASURING, not at the end of the sweep. The file
    is the only thing an eviction cannot take, so the window between
    finishing a cell and recording it is exactly the work at risk.

    Args:
        checkpoint: Work completed so far.
        directory: Where the checkpoint file lives.
        cell: What this unit is. Must be spelled the same way on a later run,
            because that is what a resume matches on.
        measure: Runs this cell and returns every observation it produces.

    Returns:
        The checkpoint including this cell, and the cell's observations.
    """
    for record in checkpoint["cells"]:
        if record["cell"] == cell:
            _log.info(
                "resuming cell %s from checkpoint, %d observation(s)",
                cell,
                len(record["observations"]),
            )
            return checkpoint, tuple(record["observations"])
    produced = tuple(measure())
    recorded = with_cell(checkpoint, CellRecord(cell=cell, observations=list(produced)))
    save_sweep_checkpoint(directory, recorded)
    return recorded, produced


def bind_cells(
    units: Sequence[tuple[str, _UnitT]],
    measure: Callable[[_UnitT], Sequence[Observation]],
) -> list[tuple[str, Callable[[], Sequence[Observation]]]]:
    """Pair each named unit with a callable that measures that unit.

    THE BINDING IS THE POINT, and it is here because every sweep needs it and
    the hand-written version has a bug that survives review. Writing
    ``lambda: measure(unit)`` inside a loop captures the VARIABLE, not its
    value, and a closure reads it when it RUNS -- which is later, inside the
    driver. Every cell would measure the last unit, and the record would come
    out with one number under N labels: complete, plausible, and wrong.
    ``functools.partial`` binds the value at build time, and doing it once
    here means no sweep has to remember why.

    Args:
        units: Each unit paired with the cell name that identifies it. The
            name sits beside the unit so the two cannot drift apart.
        measure: Runs one unit and returns every observation it produces.

    Returns:
        Cells ready for :func:`checkpointed_cells`, in the order given.
    """
    return [(name, functools.partial(measure, unit)) for name, unit in units]


def require_distinct_cells(
    cells: Sequence[tuple[str, Callable[[], Sequence[Observation]]]],
) -> None:
    """Refuse a sweep whose cells would collide in the checkpoint.

    CHECKED BEFORE ANY WORK STARTS, because of when the damage would
    otherwise appear. A resume matches cells by name, so two cells sharing
    one would hand the second the first's observations and never run it. On a
    first run that is invisible -- both execute, both are recorded -- and it
    only becomes wrong after an eviction. Failing in seconds beats reporting
    one cell's numbers twice under two labels after forty hours.

    Args:
        cells: The sweep's cells, in the order they will run.

    Raises:
        AppError: With ``CARTRIDGE_CHECKPOINT_DUPLICATE_CELL`` when two cells
            share a name.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for name, _measure in cells:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_DUPLICATE_CELL,
            (
                f"cell name(s) {', '.join(duplicates)} are used more than once in this "
                f"sweep. A resume matches cells by name, so the later one would inherit "
                f"the earlier one's observations and never run -- reporting one cell's "
                f"numbers under two labels. Name each unit for what makes it different."
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_DUPLICATE_CELL),
        )


def checkpointed_cells(
    directory: Path,
    *,
    measurement: str,
    inputs_digest: str,
    cells: Sequence[tuple[str, Callable[[], Sequence[Observation]]]],
) -> Mapping[str, tuple[Observation, ...]]:
    """Run a sweep's cells, skipping any an earlier run already finished.

    THE WHOLE RESUME DISCIPLINE, IN ONE PLACE. Every sweep needs the same
    sequence and every step of it has a wrong version that still exits zero;
    the module docstring lists them. Writing it once means a new sweep gets
    the discipline by calling this, rather than by remembering it.

    KEYED BY CELL NAME RATHER THAN RETURNED FLAT. Most sweeps reduce their
    cells -- a mean and a spread over seeds, a noise floor over arms -- and
    the reduction has to find the cells it reduces. A flat sequence would make
    that index arithmetic against the order the cells were built in, which is
    correct until somebody inserts a cell. A caller looks its cells up by the
    same names it just supplied.

    THE REDUCTIONS STAY WITH THE CALLER, and are recomputed on every run
    rather than checkpointed. A resumed sweep therefore reduces over the same
    draws a straight run would have; persisting a reduction instead would let
    a resume report a spread over fewer seeds than its own record claims.

    Args:
        directory: Where this sweep's checkpoint file lives.
        measurement: Names the sweep, so one sweep's cells can never be
            adopted by another.
        inputs_digest: Digest of what this run measures over, so a resume
            against a different corpus is refused rather than silently mixed.
        cells: Every unit of work, in the order to run them, each a
            ``(name, measure)`` pair. Names must be distinct. ``measure``
            takes no arguments so a cell already in the checkpoint costs
            nothing -- a signature taking finished observations would have
            made the skip cosmetic, because the work would already have
            happened.

    Returns:
        Each cell's observations, keyed by cell name, in the order given --
        so a caller that wants them flat can take the values, and one that
        reduces per cell can look them up.

    Raises:
        AppError: With ``CARTRIDGE_CHECKPOINT_DUPLICATE_CELL`` when two cells
            share a name, or ``CARTRIDGE_CHECKPOINT_FOREIGN`` when the
            checkpoint on disk describes a different measurement.
    """
    require_distinct_cells(cells)
    checkpoint = resume_or_start(directory, measurement=measurement, inputs_digest=inputs_digest)
    produced: dict[str, tuple[Observation, ...]] = {}
    for name, measure in cells:
        checkpoint, observations = cell_or_resume(checkpoint, directory, name, measure)
        produced[name] = observations

    # DELETED LAST, once every cell this sweep will run has run. A leftover
    # file is indistinguishable from an interrupted run, so the next
    # submission would skip cells it should have re-measured -- but deleting
    # it any earlier would leave a window where a failure loses both the run
    # and the record of what it had already done.
    delete_sweep_checkpoint(directory, measurement)
    return produced


__all__ = [
    "bind_cells",
    "cell_or_resume",
    "checkpoint_exists",
    "checkpoint_path",
    "checkpointed_cells",
    "delete_sweep_checkpoint",
    "load_sweep_checkpoint",
    "require_distinct_cells",
    "resume_or_start",
    "save_sweep_checkpoint",
]
