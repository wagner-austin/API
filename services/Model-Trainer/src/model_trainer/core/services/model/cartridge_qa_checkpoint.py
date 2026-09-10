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

from pathlib import Path

from platform_core.json_utils import dump_json_str

from model_trainer.core.contracts.qa_checkpoint import (
    QA_CHECKPOINT_SCHEMA_VERSION,
    QaCheckpoint,
    decode_qa_checkpoint,
    encode_qa_checkpoint,
    qa_checkpoint_mismatches,
)
from model_trainer.core.services.model.checkpoint_file import (
    publish_atomically,
    read_checkpoint_object,
    resume_or_refuse,
)


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
    return publish_atomically(
        checkpoint_path(directory, checkpoint["plan_name"]),
        dump_json_str(encode_qa_checkpoint(checkpoint)),
    )


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
        TypeError: If the file does not hold a JSON object.
        OSError: If the file is absent. Callers ask
            :func:`checkpoint_exists` first; a missing checkpoint is the
            ordinary case on a first run and is not an error this module
            invents a value for.
    """
    return decode_qa_checkpoint(read_checkpoint_object(checkpoint_path(directory, plan_name)))


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

    THE POLICY ITSELF LIVES IN
    :func:`~model_trainer.core.services.model.checkpoint_file.resume_or_refuse`,
    which the sweep checkpoints drive too. What is decided HERE is the only
    thing that differs between them: which fields identify this measurement,
    and what a wrongly resumed run would end up reporting.

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

    def _load() -> QaCheckpoint:
        """Read the checkpoint this run would resume.

        Returns:
            The decoded checkpoint.
        """
        return load_qa_checkpoint(directory, plan_name)

    def _mismatches(found: QaCheckpoint) -> list[str]:
        """Name every way the found checkpoint measures something else.

        Args:
            found: The checkpoint on disk.

        Returns:
            One line per disagreeing field.
        """
        return qa_checkpoint_mismatches(
            found,
            plan_name=plan_name,
            corpus_digest=corpus_digest,
            item_count=item_count,
            model_id=model_id,
        )

    return resume_or_refuse(
        checkpoint_path(directory, plan_name),
        fresh=QaCheckpoint(
            schema_version=QA_CHECKPOINT_SCHEMA_VERSION,
            plan_name=plan_name,
            corpus_digest=corpus_digest,
            item_count=item_count,
            model_id=model_id,
            arms=[],
            seeds=[],
        ),
        load=_load,
        mismatches=_mismatches,
        adopting="arms scored on another question set",
    )


__all__ = [
    "checkpoint_exists",
    "checkpoint_path",
    "delete_qa_checkpoint",
    "load_qa_checkpoint",
    "resume_or_start",
    "save_qa_checkpoint",
]
