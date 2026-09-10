"""Typed contract for a question-set measurement interrupted between arms.

WHY THIS EXISTS, AND IT IS A COST NOBODY HAD PRICED. ``free-gpu`` on HPC3
carries ``PreemptMode=CANCEL``: an evicted job is killed outright and Slurm
resubmits nothing. The submission guard admits a long run there anyway when
its workspace declares ``deterministic: true``, on the reasoning that a
replayable run is its own checkpoint at step zero. That reasoning is sound
and it protects the RESULT rather than the HOURS -- a nine-arm measurement
evicted in its eighth arm has lost no science and most of a night.

Twenty-seven committed ``mi`` run documents were relying on exactly that,
four of them at 2400 minutes, and none of the cartridge payloads wrote a
checkpoint of any kind. This module is the first half of making the
declaration true instead of asserted.

WHY THE ARM IS THE UNIT, rather than the item or the seed. The arms are
scored one after another and each is worth roughly twenty minutes at corpus
scale -- BM25 over 15607 chunks took 23m35s in job 55901956. An arm is
therefore both the largest unit an eviction can destroy and the smallest one
worth the machinery, and it is the only boundary at which a partial result is
meaningful: half an arm is not a number, it is an unfinished average.

WHY THE OUTCOMES TRAVEL AND ARE NOT RECOMPUTED. The comparison this
programme exists to make is McNemar-paired, which needs to know WHICH items
each arm got right, not how many. A checkpoint carrying only accuracies would
resume into a run that could report every arm's score and none of the
pairings -- the exact statistic the 2026-09-09 retraction turned on. So each
arm's full :class:`~model_trainer.core.contracts.cloze.ClozeEvalResult`
persists, per-item outcomes included, through the codec that type already
owns rather than a second one written here.

THE FINGERPRINT IS THE POINT OF THE FILE, not the results in it. A resume is
only valid when the run being resumed is the same measurement, and "same" has
four parts: the plan, the corpus, the realised item count and the base model.
The corpus digest is not decoration -- ``corpus_digest`` hashes document
BOUNDARIES as well as text, so a corpus regrouped into different files is a
different measurement and must not be resumed into. A mismatch is reported
field by field rather than as a boolean, because an operator who is told only
"refused" will delete the checkpoint, and the field that disagrees is usually
the thing they actually want to know.

Following :mod:`model_trainer.core.contracts.checkpoint`, which owns the same
problem one layer down for epoch-structured training: schema version stamped
into the file, ``require_*`` validation on every decoded field, and a typed
refusal rather than a recovered guess.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, TypeVar

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_dict,
    require_float,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from model_trainer.core.contracts.cloze import (
    ClozeEvalResult,
    decode_cloze_eval_result,
    encode_cloze_eval_result,
)
from model_trainer.core.contracts.paired_comparison import (
    PairedComparison,
    decode_paired_comparison,
    encode_paired_comparison,
)

#: Version stamp written into every question-set checkpoint. A decoder that
#: meets a different version refuses to resume rather than guessing at field
#: semantics; bump this when the shape below changes.
QA_CHECKPOINT_SCHEMA_VERSION: Final[int] = 1

#: One decoded entry of a checkpoint's arm or seed list. Named so
#: :func:`_decode_each` can serve both without either list's decoder leaking
#: into the other's type.
_RecordT = TypeVar("_RecordT")


class ArmRecord(TypedDict):
    """One completed arm, with the timings that were measured for it.

    THE TIMINGS PERSIST BECAUSE THEY CANNOT BE RE-DERIVED. A resumed process
    could re-score an arm to recover its accuracy; it cannot recover how long
    the original scoring took, and this programme reports latency beside
    accuracy because the cartridge's claim is partly a latency claim. An arm
    whose seconds were lost to an eviction would have to be re-run to be
    reported honestly, which is the cost the checkpoint exists to avoid.

    Attributes:
        arm: Which arm this is, matching the field name in
            :class:`~model_trainer.core.services.model.cartridge_qa_report.ArmScores`.
        result: The full scored result, per-item outcomes included, because
            the comparison is paired.
        select_seconds: Time spent choosing evidence for this arm, separate
            from scoring it. Zero for arms that select nothing.
        score_seconds: Time spent scoring the items once evidence was chosen.
        index_seconds: Time spent building whatever index this arm queries,
            which a deployment pays once per corpus rather than per request.
            Zero for arms that build no index.
        evidence_fraction: Share of the corpus this arm's evidence carried,
            for the arms that report one; zero otherwise.
    """

    arm: str
    result: ClozeEvalResult
    select_seconds: float
    score_seconds: float
    index_seconds: float
    evidence_fraction: float


class SeedRecord(TypedDict):
    """One cartridge seed's completed result.

    A SEPARATE SHAPE FROM :class:`ArmRecord`, AND THE DIFFERENCE IS REAL
    RATHER THAN TIDINESS. The retrieval arms are scored and compared on
    accuracy alone. A cartridge seed additionally carries a PAIRED
    NEGATIVE-LOG-LIKELIHOOD comparison against the untreated base -- the
    measurement that says whether the cartridge made the answer more likely,
    not merely more often chosen -- and there is no field on an arm record
    where that belongs. Forcing one shape to serve both would mean seven arms
    carrying an empty comparison so that three seeds could carry a real one.

    It is also the most expensive unit in the run: each seed TRAINS a
    cartridge over the whole corpus before it scores anything, so a seed lost
    to an eviction costs more than any retrieval arm.

    Attributes:
        seed: The initialisation seed, which is what makes this replicate
            distinct from its siblings.
        result: The scored result for the cartridge trained at this seed.
        answer_nll: Paired NLL of the answer under base versus cartridge.
        score_seconds: Time spent scoring, excluding the training that
            preceded it -- the benchmark charges serving cost only, and the
            capacity benchmark already records training.
    """

    seed: int
    result: ClozeEvalResult
    answer_nll: PairedComparison
    score_seconds: float


class QaCheckpoint(TypedDict):
    """A question-set measurement's completed work and what identifies it.

    Attributes:
        schema_version: Must equal :data:`QA_CHECKPOINT_SCHEMA_VERSION`.
        plan_name: The plan being measured, as named in ``QA_PLANS``.
        corpus_digest: Digest of the exact documents the items were built
            from, boundaries included.
        item_count: How many items the corpus REALISED, which is not the
            plan's ``max_items`` cap and is what the power gate ruled on.
        model_id: The base model every arm was scored against.
        arms: Completed retrieval arms, in the order they finished.
        seeds: Completed cartridge replicates, in the order they finished.
    """

    schema_version: int
    plan_name: str
    corpus_digest: str
    item_count: int
    model_id: str
    arms: list[ArmRecord]
    seeds: list[SeedRecord]


def encode_arm_record(record: ArmRecord) -> JSONObject:
    """Encode one completed arm.

    Args:
        record: The arm to encode.

    Returns:
        A JSON object.
    """
    return {
        "arm": record["arm"],
        "result": encode_cloze_eval_result(record["result"]),
        "select_seconds": record["select_seconds"],
        "score_seconds": record["score_seconds"],
        "index_seconds": record["index_seconds"],
        "evidence_fraction": record["evidence_fraction"],
    }


def decode_arm_record(obj: JSONObject) -> ArmRecord:
    """Decode one completed arm, validating every field.

    Args:
        obj: The JSON object to decode.

    Returns:
        The arm record.

    Raises:
        JSONTypeError: If any field is missing or of the wrong type.
    """
    return ArmRecord(
        arm=require_str(obj, "arm"),
        result=decode_cloze_eval_result(require_dict(obj, "result")),
        select_seconds=require_float(obj, "select_seconds"),
        score_seconds=require_float(obj, "score_seconds"),
        index_seconds=require_float(obj, "index_seconds"),
        evidence_fraction=require_float(obj, "evidence_fraction"),
    )


def encode_seed_record(record: SeedRecord) -> JSONObject:
    """Encode one completed cartridge seed.

    Args:
        record: The seed to encode.

    Returns:
        A JSON object.
    """
    return {
        "seed": record["seed"],
        "result": encode_cloze_eval_result(record["result"]),
        "answer_nll": encode_paired_comparison(record["answer_nll"]),
        "score_seconds": record["score_seconds"],
    }


def decode_seed_record(obj: JSONObject) -> SeedRecord:
    """Decode one completed cartridge seed, validating every field.

    Args:
        obj: The JSON object to decode.

    Returns:
        The seed record.

    Raises:
        JSONTypeError: If any field is missing or of the wrong type.
    """
    return SeedRecord(
        seed=require_int(obj, "seed"),
        result=decode_cloze_eval_result(require_dict(obj, "result")),
        answer_nll=decode_paired_comparison(require_dict(obj, "answer_nll")),
        score_seconds=require_float(obj, "score_seconds"),
    )


def _decode_each(
    obj: JSONObject, key: str, decode: Callable[[JSONObject], _RecordT]
) -> list[_RecordT]:
    """Decode every entry of a required list field, naming a bad one by index.

    ONE FUNCTION FOR BOTH LISTS. The arm and seed loops were written out
    separately first and differed only in the decoder and the field name,
    which is a fork waiting to drift: a fix to the arm loop's message would
    not reach the seed loop's.

    THE POSITION IS IN THE MESSAGE because these entries are anonymous until
    they decode. An operator told only "must be an object" cannot find which
    of nine entries is malformed, and the entry's own identifier -- the arm's
    name, the seed's number -- lives inside the object that would not decode.

    Args:
        obj: The enclosing JSON object.
        key: Field holding the list.
        decode: Decoder for one entry.

    Returns:
        Every entry, decoded, in order.

    Raises:
        JSONTypeError: If the field is absent, is not a list, or holds an
            entry that is not an object.
    """
    raw = obj.get(key)
    if not isinstance(raw, list):
        raise JSONTypeError(f"'{key}' must be a list")
    decoded: list[_RecordT] = []
    for position, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise JSONTypeError(f"'{key}[{position}]' must be an object")
        decoded.append(decode(entry))
    return decoded


def _replacing(
    checkpoint: QaCheckpoint, *, arms: list[ArmRecord], seeds: list[SeedRecord]
) -> QaCheckpoint:
    """Build a checkpoint carrying new completed work and the same identity.

    Both lists are REQUIRED rather than optional, so that a caller replacing
    one must state what happens to the other. A default would let a future
    edit drop the seeds by omission and look correct.

    Args:
        checkpoint: The checkpoint whose identity to keep.
        arms: Completed retrieval arms for the new checkpoint.
        seeds: Completed cartridge replicates for the new checkpoint.

    Returns:
        The new checkpoint.
    """
    return QaCheckpoint(
        schema_version=checkpoint["schema_version"],
        plan_name=checkpoint["plan_name"],
        corpus_digest=checkpoint["corpus_digest"],
        item_count=checkpoint["item_count"],
        model_id=checkpoint["model_id"],
        arms=arms,
        seeds=seeds,
    )


def with_arm(checkpoint: QaCheckpoint, record: ArmRecord) -> QaCheckpoint:
    """Record one more completed arm, without mutating the original.

    A NEW VALUE RATHER THAN AN APPEND. The caller holds a checkpoint that has
    already been written to disk; mutating it in place makes the in-memory
    object and the published file silently diverge between the append and the
    next save, and any failure in that window leaves a checkpoint claiming
    work whose file does not record it.

    Args:
        checkpoint: The checkpoint so far.
        record: The arm that just finished.

    Returns:
        A checkpoint holding it.
    """
    return _replacing(checkpoint, arms=[*checkpoint["arms"], record], seeds=checkpoint["seeds"])


def with_seed(checkpoint: QaCheckpoint, record: SeedRecord) -> QaCheckpoint:
    """Record one more completed cartridge replicate, without mutating.

    Args:
        checkpoint: The checkpoint so far.
        record: The seed that just finished.

    Returns:
        A checkpoint holding it.
    """
    return _replacing(checkpoint, arms=checkpoint["arms"], seeds=[*checkpoint["seeds"], record])


def encode_qa_checkpoint(checkpoint: QaCheckpoint) -> JSONObject:
    """Encode a question-set checkpoint.

    Args:
        checkpoint: The checkpoint to encode.

    Returns:
        A JSON object ready to be written.
    """
    arms: list[JSONValue] = [encode_arm_record(record) for record in checkpoint["arms"]]
    seeds: list[JSONValue] = [encode_seed_record(record) for record in checkpoint["seeds"]]
    return {
        "schema_version": checkpoint["schema_version"],
        "plan_name": checkpoint["plan_name"],
        "corpus_digest": checkpoint["corpus_digest"],
        "item_count": checkpoint["item_count"],
        "model_id": checkpoint["model_id"],
        "arms": arms,
        "seeds": seeds,
    }


def decode_qa_checkpoint(obj: JSONObject) -> QaCheckpoint:
    """Decode a question-set checkpoint, validating every field.

    Args:
        obj: The JSON object to decode.

    Returns:
        The checkpoint.

    Raises:
        JSONTypeError: If a field is missing, of the wrong type, or if the
            schema version is not one this code understands. The version is
            checked HERE rather than by the caller so that no reader can
            obtain a decoded checkpoint without the version having been
            agreed -- a check the caller may forget is not a check.
    """
    version = require_int(obj, "schema_version")
    if version != QA_CHECKPOINT_SCHEMA_VERSION:
        raise JSONTypeError(
            f"schema_version {version} is not {QA_CHECKPOINT_SCHEMA_VERSION}; this "
            f"checkpoint was written by a different version of the measurement and "
            f"its fields cannot be assumed to mean the same thing. Re-run rather "
            f"than resume."
        )
    return QaCheckpoint(
        schema_version=version,
        plan_name=require_str(obj, "plan_name"),
        corpus_digest=require_str(obj, "corpus_digest"),
        item_count=require_int(obj, "item_count"),
        model_id=require_str(obj, "model_id"),
        arms=_decode_each(obj, "arms", decode_arm_record),
        seeds=_decode_each(obj, "seeds", decode_seed_record),
    )


def qa_checkpoint_mismatches(
    checkpoint: QaCheckpoint,
    *,
    plan_name: str,
    corpus_digest: str,
    item_count: int,
    model_id: str,
) -> list[str]:
    """Name every way this checkpoint describes a different measurement.

    A LIST RATHER THAN A BOOLEAN, following
    :func:`~model_trainer.core.contracts.checkpoint.model_train_config_mismatches`.
    An operator told only that a resume was refused deletes the checkpoint;
    an operator told ``corpus_digest: checkpoint dccd3375f54d != current
    9f21ab0c771e`` knows they pointed the run at a different corpus, which is
    usually the actual mistake and is recoverable.

    Args:
        checkpoint: The checkpoint found on disk.
        plan_name: Plan the current run is measuring.
        corpus_digest: Digest of the corpus the current run read.
        item_count: Items the current run's corpus realised.
        model_id: Base model the current run will score against.

    Returns:
        One human-readable line per disagreeing field, empty when the
        checkpoint describes this exact measurement.
    """
    expected: tuple[tuple[str, str, str], ...] = (
        ("plan_name", checkpoint["plan_name"], plan_name),
        ("corpus_digest", checkpoint["corpus_digest"], corpus_digest),
        ("item_count", str(checkpoint["item_count"]), str(item_count)),
        ("model_id", checkpoint["model_id"], model_id),
    )
    return [
        f"{field}: checkpoint {found!r} != current {wanted!r}"
        for field, found, wanted in expected
        if found != wanted
    ]


def completed_arms(checkpoint: QaCheckpoint) -> frozenset[str]:
    """Name the arms this checkpoint already holds a result for.

    Args:
        checkpoint: The checkpoint.

    Returns:
        The arm names, for the caller to skip.
    """
    return frozenset(record["arm"] for record in checkpoint["arms"])


def completed_seeds(checkpoint: QaCheckpoint) -> frozenset[int]:
    """Name the cartridge seeds this checkpoint already holds a result for.

    Args:
        checkpoint: The checkpoint.

    Returns:
        The seeds, for the caller to skip retraining.
    """
    return frozenset(record["seed"] for record in checkpoint["seeds"])


__all__ = [
    "QA_CHECKPOINT_SCHEMA_VERSION",
    "ArmRecord",
    "QaCheckpoint",
    "SeedRecord",
    "completed_arms",
    "completed_seeds",
    "decode_arm_record",
    "decode_qa_checkpoint",
    "decode_seed_record",
    "encode_arm_record",
    "encode_qa_checkpoint",
    "encode_seed_record",
    "qa_checkpoint_mismatches",
    "with_arm",
    "with_seed",
]
