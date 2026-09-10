"""Persisting a question-set measurement's completed arms across an eviction.

One rolling file per plan, rewritten after every completed arm and published
with ``os.replace``, so a run killed mid-arm resumes at the last arm boundary
instead of at item zero. The pattern is lifted from
:mod:`model_trainer.core.services.training.checkpoint`, which solved the same
problem for epoch-structured training: write a sibling temporary, rename it
over the target, and a crash mid-write leaves the previous checkpoint intact
and readable rather than leaving a half-written one that decodes to nonsense.

WHAT IS DIFFERENT FROM THE TRAINING CHECKPOINT, AND WHY IT IS SIMPLER. That
one carries tensors -- model weights, optimizer state, RNG -- so its payload
is a torch file. This one carries only scored results, which are already
JSON-encodable, so the file is plain JSON and can be read by a person
debugging a resume without loading torch. Nothing here needs to reproduce the
model's internal state, because an arm is scored from the base model and the
items, both of which the resumed process rebuilds identically.

LOADING IS STRICT AND REFUSES RATHER THAN RECOVERS. A file that does not
decode, or that decodes to a different measurement, is not repaired and not
ignored: the caller is told which field disagrees and re-runs deliberately.
The failure this guards against is subtle and expensive -- a checkpoint from
a DIFFERENT corpus would resume into a run that reports arms measured on two
different question sets, and every arm would look plausible. There is no
outcome of that worth having, so there is no path to it.

WHY NO ``_test_hooks.py``. Following the reasoning
:mod:`model_trainer.core.services.model.cartridge_qa_power` states for
itself. The only seam here is the filesystem, and a test that writes a real
file into ``tmp_path`` exercises the real ``os.replace`` -- including the
atomicity that is the whole point of the module -- while a faked filesystem
would exercise the fake. A hook here would make the tests weaker, not the
code more testable.
"""

from __future__ import annotations

import os
from pathlib import Path

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import dump_json_str, load_json_str

from model_trainer.core.contracts.qa_checkpoint import (
    QA_CHECKPOINT_SCHEMA_VERSION,
    QaCheckpoint,
    decode_qa_checkpoint,
    encode_qa_checkpoint,
    qa_checkpoint_mismatches,
)

#: Suffix of the temporary file a save writes before publishing it. A sibling
#: rather than a system temporary directory, because ``os.replace`` is only
#: atomic within one filesystem and ``/tmp`` is routinely a different one on
#: the cluster -- a cross-device rename falls back to a copy, which is
#: precisely the non-atomic write this module exists to avoid.
_PENDING_SUFFIX = ".pending"


def checkpoint_path(directory: Path, plan_name: str) -> Path:
    """Locate the checkpoint file for one plan.

    Args:
        directory: Where checkpoints live for this run.
        plan_name: The plan being measured.

    Returns:
        The file's path, whether or not it exists.
    """
    return directory / f"qa-checkpoint-{plan_name}.json"


def checkpoint_exists(directory: Path, plan_name: str) -> bool:
    """Report whether a resumable checkpoint is present.

    Args:
        directory: Where checkpoints live for this run.
        plan_name: The plan being measured.

    Returns:
        True when the file is present and is a file.
    """
    return checkpoint_path(directory, plan_name).is_file()


def save_qa_checkpoint(directory: Path, checkpoint: QaCheckpoint) -> Path:
    """Write a checkpoint atomically, replacing any previous one.

    Args:
        directory: Where checkpoints live for this run. Created if absent.
        checkpoint: The state to persist.

    Returns:
        Path of the published file.
    """
    target = checkpoint_path(directory, checkpoint["plan_name"])
    directory.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + _PENDING_SUFFIX)
    pending.write_text(dump_json_str(encode_qa_checkpoint(checkpoint)), encoding="utf-8")
    os.replace(pending, target)
    return target


def load_qa_checkpoint(directory: Path, plan_name: str) -> QaCheckpoint:
    """Read and validate the checkpoint for one plan.

    Args:
        directory: Where checkpoints live for this run.
        plan_name: The plan being measured.

    Returns:
        The decoded checkpoint.

    Raises:
        JSONTypeError: If the file does not decode, including when its schema
            version is not the one this code understands.
        OSError: If the file is absent. Callers ask
            :func:`checkpoint_exists` first; a missing checkpoint is the
            ordinary case on a first run and is not an error this module
            invents a value for.
    """
    path = checkpoint_path(directory, plan_name)
    decoded = load_json_str(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError(f"{path} does not hold a JSON object")
    return decode_qa_checkpoint(decoded)


def delete_qa_checkpoint(directory: Path, plan_name: str) -> None:
    """Remove a checkpoint once its measurement has completed.

    A COMPLETED RUN DELETES ITS OWN CHECKPOINT, following the training
    service's rule and for the same reason: a leftover file is
    indistinguishable from an interrupted run, so the next submission of the
    same plan would resume arms it should have re-measured. A failed or
    evicted run leaves the file in place, and that file is exactly what the
    resubmission continues from.

    Args:
        directory: Where checkpoints live for this run.
        plan_name: The plan that completed.
    """
    checkpoint_path(directory, plan_name).unlink(missing_ok=True)


def resume_or_start(
    directory: Path,
    *,
    plan_name: str,
    corpus_digest: str,
    item_count: int,
    model_id: str,
) -> QaCheckpoint:
    """Decide what a starting run may reuse, refusing a foreign checkpoint.

    THE THREE STATES, AND ONLY ONE OF THEM IS INTERESTING. No file is the
    ordinary first run and yields an empty checkpoint to accumulate into. A
    file describing THIS measurement yields its completed arms. A file
    describing a different one is refused, loudly, and this is the case the
    function exists for -- resuming across it would produce a complete arms
    table in which some arms were scored on another question set, and nothing
    downstream could tell.

    WHY REFUSE RATHER THAN QUIETLY START FRESH, which is the tempting third
    option. Discarding a checkpoint an operator believed in spends exactly the
    hours the checkpoint existed to save, and does it silently. The
    disagreeing field is nearly always something they can fix -- a corpus path
    pointing at last week's directory -- so naming it turns a lost night into
    a corrected command.

    Args:
        directory: Where checkpoints live for this run.
        plan_name: The plan about to be measured.
        corpus_digest: Digest of the corpus this run read.
        item_count: Items this run's corpus realised, which the power gate
            has already ruled on.
        model_id: Base model this run will score against.

    Returns:
        A checkpoint to accumulate into: the one on disk when it describes
        this measurement, otherwise an empty one.

    Raises:
        AppError: With ``CARTRIDGE_CHECKPOINT_FOREIGN`` when a checkpoint is
            present and describes a different measurement.
    """
    fresh = QaCheckpoint(
        schema_version=QA_CHECKPOINT_SCHEMA_VERSION,
        plan_name=plan_name,
        corpus_digest=corpus_digest,
        item_count=item_count,
        model_id=model_id,
        arms=[],
        seeds=[],
    )
    if not checkpoint_exists(directory, plan_name):
        return fresh
    found = load_qa_checkpoint(directory, plan_name)
    mismatches = qa_checkpoint_mismatches(
        found,
        plan_name=plan_name,
        corpus_digest=corpus_digest,
        item_count=item_count,
        model_id=model_id,
    )
    if mismatches:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_FOREIGN,
            (
                f"the checkpoint at {checkpoint_path(directory, plan_name)} describes a "
                f"different measurement and resuming from it would report arms scored on "
                f"another question set: {'; '.join(mismatches)}. Fix whichever of those is "
                f"the mistake, or delete the checkpoint to re-measure from the first arm."
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_FOREIGN),
        )
    return found


__all__ = [
    "checkpoint_exists",
    "checkpoint_path",
    "delete_qa_checkpoint",
    "load_qa_checkpoint",
    "resume_or_start",
    "save_qa_checkpoint",
]
