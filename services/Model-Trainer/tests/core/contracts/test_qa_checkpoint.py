"""The question-set checkpoint contract, and the resumes it must refuse.

WHAT THESE TESTS ARE REALLY GUARDING. A checkpoint is read exactly once, by a
process that is about to skip work on its authority, and every mistake it can
make is silent: a resume against the wrong corpus produces a full arms table
in which every number looks reasonable and two of them were measured on a
different question set. So the refusals matter more than the round trip, and
most of what is below is about what does NOT decode.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from model_trainer.core.contracts.cloze import ClozeEvalResult, ClozeItemOutcome
from model_trainer.core.contracts.paired_comparison import PairedComparison
from model_trainer.core.contracts.qa_checkpoint import (
    QA_CHECKPOINT_SCHEMA_VERSION,
    ArmRecord,
    QaCheckpoint,
    SeedRecord,
    completed_arms,
    completed_seeds,
    decode_qa_checkpoint,
    encode_arm_record,
    encode_qa_checkpoint,
    encode_seed_record,
    qa_checkpoint_mismatches,
    with_arm,
    with_seed,
)

_PLAN = "gpt2-full-wiki-qa"
_DIGEST = "dccd3375f54d"
_ITEMS = 3740
_MODEL = "gpt2"


def _result(*, correct: int) -> ClozeEvalResult:
    """Build a scored result whose outcomes agree with its counts.

    Args:
        correct: How many of two items were answered correctly.

    Returns:
        The result.
    """
    # THE SCORES MUST AGREE WITH `correct`, because `decode_cloze_eval_result`
    # checks it: the answer sits at index 0 and must be the strict minimum for
    # an item to count as correct. A fixture that ignored that decoded fine
    # here and would have been refused by the real codec, so the constraint is
    # inherited rather than restated -- which is the point of embedding
    # `ClozeEvalResult` instead of copying its fields.
    outcomes: list[ClozeItemOutcome] = [
        {
            "item_id": f"item-{index}",
            "correct": index < correct,
            "scores": [1.0, 2.0] if index < correct else [2.0, 1.0],
        }
        for index in range(2)
    ]
    return ClozeEvalResult(
        total=2, correct=correct, accuracy=correct / 2, chance=0.25, outcomes=outcomes
    )


def _arm(name: str, *, correct: int = 1) -> ArmRecord:
    """Build one completed arm record.

    Args:
        name: The arm's name.
        correct: How many items it got right.

    Returns:
        The record.
    """
    return ArmRecord(
        arm=name,
        result=_result(correct=correct),
        select_seconds=1.5,
        score_seconds=20.25,
        index_seconds=0.0,
        evidence_fraction=0.5,
    )


def _seed(number: int) -> SeedRecord:
    """Build one completed cartridge replicate.

    Args:
        number: The initialisation seed.

    Returns:
        The record.
    """
    return SeedRecord(
        seed=number,
        result=_result(correct=1),
        answer_nll=PairedComparison(
            items=2,
            mean_baseline=3.5,
            mean_treatment=3.1,
            improved=2,
            worsened=0,
            tied=0,
            p_value=0.5,
            outcomes_digest="a1b2c3d4",
        ),
        score_seconds=41.0,
    )


def _checkpoint(*arms: ArmRecord, seeds: tuple[SeedRecord, ...] = ()) -> QaCheckpoint:
    """Build a checkpoint holding the given arms and seeds.

    Args:
        *arms: Completed retrieval arms.
        seeds: Completed cartridge replicates.

    Returns:
        The checkpoint.
    """
    return QaCheckpoint(
        schema_version=QA_CHECKPOINT_SCHEMA_VERSION,
        plan_name=_PLAN,
        corpus_digest=_DIGEST,
        item_count=_ITEMS,
        model_id=_MODEL,
        arms=list(arms),
        seeds=list(seeds),
    )


class TestTheRoundTrip:
    def test_a_checkpoint_survives_encoding_and_decoding(self) -> None:
        original = _checkpoint(_arm("bm25"), _arm("dense", correct=2))

        assert decode_qa_checkpoint(encode_qa_checkpoint(original)) == original

    def test_the_per_item_outcomes_survive_it(self) -> None:
        """THE FIELD THE WHOLE COMPARISON DEPENDS ON.

        This programme's claims are McNemar-paired, which needs to know WHICH
        items each arm got right rather than how many. A checkpoint that
        round-tripped counts but dropped outcomes would resume into a run
        able to report every arm's accuracy and not one pairing -- the exact
        statistic the 2026-09-09 retraction turned on.
        """
        decoded = decode_qa_checkpoint(encode_qa_checkpoint(_checkpoint(_arm("bm25"))))

        outcomes = decoded["arms"][0]["result"]["outcomes"]
        assert [outcome["item_id"] for outcome in outcomes] == ["item-0", "item-1"]
        assert [outcome["correct"] for outcome in outcomes] == [True, False]

    def test_an_empty_checkpoint_is_valid(self) -> None:
        """A run evicted before its first arm finished has nothing to skip,
        which is a legitimate state and not a corrupt file."""
        assert decode_qa_checkpoint(encode_qa_checkpoint(_checkpoint()))["arms"] == []


class TestTheSchemaVersionIsCheckedInsideDecode:
    """Not by the caller, so that no reader can obtain a decoded checkpoint
    without the version having been agreed. A check the caller may forget is
    not a check."""

    def test_a_different_version_is_refused(self) -> None:
        payload = encode_qa_checkpoint(_checkpoint(_arm("bm25")))
        payload["schema_version"] = QA_CHECKPOINT_SCHEMA_VERSION + 1

        with pytest.raises(JSONTypeError, match="cannot be assumed to mean the same thing"):
            decode_qa_checkpoint(payload)

    def test_the_refusal_names_re_running_rather_than_repair(self) -> None:
        payload = encode_qa_checkpoint(_checkpoint())
        payload["schema_version"] = 99

        with pytest.raises(JSONTypeError, match="Re-run rather than resume"):
            decode_qa_checkpoint(payload)


class TestMalformedArms:
    def test_arms_that_are_not_a_list_are_refused(self) -> None:
        payload = encode_qa_checkpoint(_checkpoint())
        payload["arms"] = "bm25"

        with pytest.raises(JSONTypeError, match="'arms' must be a list"):
            decode_qa_checkpoint(payload)

    def test_a_malformed_entry_is_named_by_its_position(self) -> None:
        """The entries are anonymous until they decode.

        An arm's name lives inside the object that will not decode, so a
        message naming the field cannot tell an operator which of nine
        entries is broken. The index can.
        """
        # BUILT rather than read back and narrowed. Reaching into an encoded
        # payload needs an isinstance to satisfy the type checker, and an
        # isinstance in a test is a weak assertion the guards refuse -- it
        # asserts a type where the test means to assert a value.
        payload: JSONObject = {
            **encode_qa_checkpoint(_checkpoint(_arm("bm25"))),
            "arms": [encode_arm_record(_arm("bm25")), "not an object"],
        }

        with pytest.raises(JSONTypeError, match=r"'arms\[1\]' must be an object"):
            decode_qa_checkpoint(payload)

    def test_an_arm_missing_a_timing_is_refused(self) -> None:
        """Timings cannot be re-derived by a resumed process, so an arm
        record without them is not a partial record, it is an unusable one."""
        entry = encode_arm_record(_arm("bm25"))
        del entry["score_seconds"]
        payload: JSONObject = {**encode_qa_checkpoint(_checkpoint()), "arms": [entry]}

        with pytest.raises(JSONTypeError, match="score_seconds"):
            decode_qa_checkpoint(payload)


class TestTheFingerprintRefusesADifferentMeasurement:
    def test_the_same_measurement_reports_no_mismatch(self) -> None:
        assert (
            qa_checkpoint_mismatches(
                _checkpoint(_arm("bm25")),
                plan_name=_PLAN,
                corpus_digest=_DIGEST,
                item_count=_ITEMS,
                model_id=_MODEL,
            )
            == []
        )

    def test_a_different_corpus_is_named_with_both_digests(self) -> None:
        """THE EXPENSIVE ONE. `corpus_digest` hashes document BOUNDARIES as
        well as text, so a corpus regrouped into different files is a
        different measurement -- and resuming into it would report arms
        scored on two different question sets, every one of them plausible.
        """
        mismatches = qa_checkpoint_mismatches(
            _checkpoint(_arm("bm25")),
            plan_name=_PLAN,
            corpus_digest="9f21ab0c771e",
            item_count=_ITEMS,
            model_id=_MODEL,
        )

        assert mismatches == ["corpus_digest: checkpoint 'dccd3375f54d' != current '9f21ab0c771e'"]

    def test_every_fingerprint_field_is_checked(self) -> None:
        """A LIST RATHER THAN A BOOLEAN, and all four at once rather than the
        first: an operator who fixes the corpus path and resubmits should not
        then discover the item count also disagreed."""
        mismatches = qa_checkpoint_mismatches(
            _checkpoint(),
            plan_name="other-plan",
            corpus_digest="other-digest",
            item_count=32,
            model_id="gpt2-large",
        )

        assert len(mismatches) == 4
        assert [line.split(":")[0] for line in mismatches] == [
            "plan_name",
            "corpus_digest",
            "item_count",
            "model_id",
        ]

    def test_a_realised_item_count_that_shifted_is_a_mismatch(self) -> None:
        """The count the power gate ruled on. A corpus that yielded 3740
        items once and 3739 later is not the same question set, whatever the
        plan's `max_items` cap says."""
        mismatches = qa_checkpoint_mismatches(
            _checkpoint(),
            plan_name=_PLAN,
            corpus_digest=_DIGEST,
            item_count=_ITEMS - 1,
            model_id=_MODEL,
        )

        assert mismatches == ["item_count: checkpoint '3740' != current '3739'"]


class TestCompletedArms:
    def test_it_names_what_a_resume_may_skip(self) -> None:
        assert completed_arms(_checkpoint(_arm("bm25"), _arm("dense"))) == frozenset(
            {"bm25", "dense"}
        )

    def test_an_empty_checkpoint_skips_nothing(self) -> None:
        assert completed_arms(_checkpoint()) == frozenset()


class TestAccumulatingCompletedWork:
    """A NEW VALUE EACH TIME, and the immutability is the point rather than
    a style preference.

    The caller holds a checkpoint that has already been written to disk.
    Appending to it in place makes the in-memory object and the published
    file diverge between the append and the next save, so a failure in that
    window leaves a checkpoint claiming work whose file does not record it --
    which on the next run is a resume that skips an arm nobody scored.
    """

    def test_adding_an_arm_leaves_the_original_untouched(self) -> None:
        before = _checkpoint(_arm("bm25"))

        after = with_arm(before, _arm("dense"))

        assert completed_arms(before) == frozenset({"bm25"})
        assert completed_arms(after) == frozenset({"bm25", "dense"})

    def test_adding_a_seed_leaves_the_original_untouched(self) -> None:
        before = _checkpoint(seeds=(_seed(7),))

        after = with_seed(before, _seed(8))

        assert completed_seeds(before) == frozenset({7})
        assert completed_seeds(after) == frozenset({7, 8})

    def test_adding_an_arm_keeps_the_seeds(self) -> None:
        """The shared constructor takes both lists as REQUIRED arguments so
        that neither can be dropped by omission. This is the assertion that
        would fail if a default were ever introduced."""
        after = with_arm(_checkpoint(seeds=(_seed(7),)), _arm("bm25"))

        assert completed_seeds(after) == frozenset({7})

    def test_adding_a_seed_keeps_the_arms(self) -> None:
        after = with_seed(_checkpoint(_arm("bm25")), _seed(7))

        assert completed_arms(after) == frozenset({"bm25"})

    def test_the_identity_travels_unchanged(self) -> None:
        """Accumulating work must never restate what the measurement IS --
        the fingerprint is what a resume is checked against, and a copy that
        rebuilt it from anywhere but the original could drift from the file
        it is about to replace."""
        after = with_arm(_checkpoint(), _arm("bm25"))

        assert (after["plan_name"], after["corpus_digest"]) == (_PLAN, _DIGEST)
        assert (after["item_count"], after["model_id"]) == (_ITEMS, _MODEL)
        assert after["schema_version"] == QA_CHECKPOINT_SCHEMA_VERSION


class TestTheCartridgeSeeds:
    """The most expensive unit in the run, and the one a resume most wants.

    Each seed TRAINS a cartridge over the whole corpus before it scores
    anything, so a seed lost to an eviction costs more than any retrieval
    arm. It is also shaped differently: a seed carries a paired NLL
    comparison against the untreated base, which no retrieval arm has and
    which no field on an arm record could hold.
    """

    def test_a_seed_survives_encoding_and_decoding(self) -> None:
        original = _checkpoint(_arm("bm25"), seeds=(_seed(7), _seed(8)))

        assert decode_qa_checkpoint(encode_qa_checkpoint(original)) == original

    def test_the_paired_nll_survives_it(self) -> None:
        """The measurement that says the cartridge made the answer more
        LIKELY rather than merely more often chosen. Losing it to a resume
        would leave the accuracy half of the cartridge claim standing with
        nothing underneath it."""
        decoded = decode_qa_checkpoint(encode_qa_checkpoint(_checkpoint(seeds=(_seed(7),))))

        nll = decoded["seeds"][0]["answer_nll"]
        assert (nll["mean_baseline"], nll["mean_treatment"]) == (3.5, 3.1)
        assert nll["outcomes_digest"] == "a1b2c3d4"

    def test_completed_seeds_names_what_need_not_be_retrained(self) -> None:
        assert completed_seeds(_checkpoint(seeds=(_seed(7), _seed(9)))) == frozenset({7, 9})

    def test_a_checkpoint_with_no_seeds_retrains_everything(self) -> None:
        assert completed_seeds(_checkpoint(_arm("bm25"))) == frozenset()

    def test_seeds_that_are_not_a_list_are_refused(self) -> None:
        payload = encode_qa_checkpoint(_checkpoint())
        payload["seeds"] = 7

        with pytest.raises(JSONTypeError, match="'seeds' must be a list"):
            decode_qa_checkpoint(payload)

    def test_a_malformed_seed_is_named_by_its_position(self) -> None:
        payload: JSONObject = {
            **encode_qa_checkpoint(_checkpoint()),
            "seeds": [encode_seed_record(_seed(7)), "not an object"],
        }

        with pytest.raises(JSONTypeError, match=r"'seeds\[1\]' must be an object"):
            decode_qa_checkpoint(payload)

    def test_a_seed_missing_its_nll_is_refused(self) -> None:
        """Half a seed record is not a partial result. A resume that adopted
        it would report a cartridge replicate with no evidence about answer
        likelihood, which is the wrong half to lose silently."""
        entry = encode_seed_record(_seed(7))
        del entry["answer_nll"]
        payload: JSONObject = {**encode_qa_checkpoint(_checkpoint()), "seeds": [entry]}

        with pytest.raises(JSONTypeError, match="answer_nll"):
            decode_qa_checkpoint(payload)


def test_the_encoded_form_is_json_serialisable() -> None:
    """The file is plain JSON so a person debugging a resume can read it
    without loading torch, which is the one thing this checkpoint has that
    the training checkpoint beside it does not."""
    encoded: JSONObject = encode_qa_checkpoint(_checkpoint(_arm("bm25")))

    assert sorted(encoded.keys()) == [
        "arms",
        "corpus_digest",
        "item_count",
        "model_id",
        "plan_name",
        "schema_version",
        "seeds",
    ]
