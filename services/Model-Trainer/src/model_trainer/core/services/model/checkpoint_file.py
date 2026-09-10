"""What every kind of checkpoint file must do the same way.

TWO KINDS OF CHECKPOINT NEEDED THIS AND NEITHER SHOULD OWN IT. The
question-set measurement records completed arms and seeds; the cartridge
sweeps record completed cells. They carry different payloads and validate
them differently, but three things must not differ between them, and each is
here because getting it wrong in only one of the two would be invisible:

1. WHAT A READER FINDS IF THE PROCESS DIES MID-WRITE -- ``publish_atomically``.
2. WHAT A FILE THAT IS NOT A JSON OBJECT DOES -- ``read_checkpoint_object``.
3. WHAT HAPPENS WHEN THE CHECKPOINT ON DISK DESCRIBES A DIFFERENT
   MEASUREMENT -- ``resume_or_refuse``, which is the one that matters.

WRITE A SIBLING, THEN RENAME. ``os.replace`` either publishes a whole file or
leaves the previous one untouched, so a reader never meets a truncated
checkpoint and an interrupted save costs the newest cell rather than every
cell before it.

THE TEMPORARY IS A SIBLING, DELIBERATELY, AND THIS IS THE PART THAT WOULD
NEVER FAIL ON ONE MACHINE. ``os.replace`` is atomic only WITHIN a
filesystem. On the cluster a checkpoint lives on ``/pub`` scratch while
``/tmp`` is routinely a different device, and a cross-device rename degrades
to a copy -- which is precisely the non-atomic publish this module exists to
prevent. A developer testing on one disk would never see it; a preempted job
on the cluster would see nothing else.

WHY THE REFUSAL IS HERE RATHER THAN IN EACH CHECKPOINT SERVICE. It is not
the code that would drift, it is the MESSAGE. An operator meeting a refused
resume has one decision to make -- fix the flag that disagrees, or spend the
hours again -- and they make it from this text. Two copies of it means one
copy improves, and the run that meets the other copy is the one that loses
the night. What genuinely differs between the two kinds is a single noun
phrase, so that is the only thing a caller passes.

WHY NO ``_test_hooks.py``. The only seam is the filesystem, and a test that
writes a real file into ``tmp_path`` exercises the real ``os.replace``,
including the atomicity that is the whole point, while a faked filesystem
would exercise the fake. Same reasoning
:mod:`model_trainer.core.services.model.cartridge_qa_power` states for
itself.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import JSONObject, load_json_str

#: Suffix of the file a save writes before publishing it under the real name.
_PENDING_SUFFIX = ".pending"

#: One decoded checkpoint, whatever its contract. This module never inspects
#: it: it decides whether the caller may resume, and the caller's own
#: ``mismatches`` function decides what "the same measurement" means.
_CheckpointT = TypeVar("_CheckpointT")


def publish_atomically(target: Path, text: str) -> Path:
    """Write text to a path so that no reader ever sees it half-written.

    Args:
        target: Where the file should end up. Its parent is created if
            absent, because a resumed run may be the first thing to touch a
            scratch directory after the previous node went away.
        text: The whole file's contents.

    Returns:
        The published path, for a caller that wants to log or assert on it.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + _PENDING_SUFFIX)
    pending.write_text(text, encoding="utf-8")
    os.replace(pending, target)
    return target


def read_checkpoint_object(path: Path) -> JSONObject:
    """Read a checkpoint file and confirm it holds a JSON object.

    The object check is here rather than in each decoder because a decoder
    receives a ``JSONObject`` and so cannot make it: a JSON array reaching one
    would fail somewhere inside a field lookup, naming a field rather than the
    file.

    Args:
        path: The checkpoint file to read.

    Returns:
        The decoded top-level object, for a contract decoder to validate.

    Raises:
        OSError: If the file is absent. Callers ask whether it exists first;
            a missing checkpoint is the ordinary case on a first run and is
            not an error this module invents a value for.
        InvalidJsonError: If the file does not parse as JSON.
        TypeError: If it parses to something other than an object.
    """
    decoded = load_json_str(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError(f"{path} does not hold a JSON object")
    return decoded


def resume_or_refuse(
    path: Path,
    *,
    fresh: _CheckpointT,
    load: Callable[[], _CheckpointT],
    mismatches: Callable[[_CheckpointT], list[str]],
    adopting: str,
) -> _CheckpointT:
    """Decide what a starting run may reuse, refusing a foreign checkpoint.

    THE THREE STATES, AND ONLY ONE IS INTERESTING. No file is the ordinary
    first run. A file describing THIS measurement yields its completed work.
    A file describing a different one is refused, loudly -- resuming across it
    would produce a complete table in which some of the numbers came from
    other inputs, and nothing downstream could tell.

    REFUSED RATHER THAN QUIETLY RESTARTED, which is the tempting third
    option. Discarding a checkpoint an operator believed in spends exactly the
    hours it existed to save, and does it silently. The disagreeing field is
    nearly always something they can fix -- a corpus path pointing at last
    week's directory -- so naming it turns a lost night into a corrected
    command.

    Args:
        path: The checkpoint file, named in the refusal so an operator can go
            and look at it.
        fresh: The empty checkpoint to accumulate into when there is nothing
            to resume.
        load: Reads and validates the file. Called only when it exists.
        mismatches: Names every way the loaded checkpoint describes a
            different measurement, one line per disagreeing field. An empty
            list means this run may resume it.
        adopting: What resuming a foreign checkpoint would wrongly report,
            as a noun phrase completing "would report ..." -- the one part of
            the refusal that is not the same for every kind of checkpoint.

    Returns:
        The checkpoint on disk when it describes this measurement, otherwise
        ``fresh``.

    Raises:
        AppError: With ``CARTRIDGE_CHECKPOINT_FOREIGN`` when a checkpoint is
            present and describes a different measurement.
    """
    if not path.is_file():
        # ``is_file`` rather than ``exists``: a directory at that path would
        # pass an existence check and then fail to read.
        return fresh
    found = load()
    disagreements = mismatches(found)
    if disagreements:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_FOREIGN,
            (
                f"the checkpoint at {path} describes a different measurement and "
                f"resuming from it would report {adopting}: {'; '.join(disagreements)}. "
                f"Fix whichever of those is the mistake, or delete the checkpoint to "
                f"measure it again from the start."
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_FOREIGN),
        )
    return found


__all__ = ["publish_atomically", "read_checkpoint_object", "resume_or_refuse"]
