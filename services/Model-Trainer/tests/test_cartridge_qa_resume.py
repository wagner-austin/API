"""What a resumed question-set measurement reuses, and what it re-derives.

THE PROPERTY THESE TESTS PIN IS THAT RESUME IS NOT COSMETIC. A skip that
re-runs the work and then throws the result away costs exactly as much as no
skip at all and looks identical from outside, so every test here plants a
result the real arm could not possibly produce and asserts that the PLANTED
value reaches the record. If the arm were re-scored, the planted value would
be overwritten by a real one and the assertion would fail.

The corpus and the plan are the shared fakes; the checkpoint is built by hand
against the same question set the run will build, because a fingerprint that
did not match would be refused rather than resumed and the test would pass for
the wrong reason.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator

import pytest

from model_trainer.cli import cartridge_qa_benchmark as bench
from model_trainer.core.contracts.cloze import ClozeEvalResult, ClozeItemOutcome
from model_trainer.core.contracts.paired_comparison import PairedComparison
from model_trainer.core.contracts.qa_checkpoint import (
    QA_CHECKPOINT_SCHEMA_VERSION,
    ArmRecord,
    QaCheckpoint,
    SeedRecord,
)
from model_trainer.core.services.model.cartridge_plans import corpus_digest
from model_trainer.core.services.model.cartridge_qa_checkpoint import (
    checkpoint_exists,
    save_qa_checkpoint,
)
from model_trainer.core.services.model.cartridge_question_set import build_question_set
from tests._qa_benchmark_support import (
    DOCUMENTS,
    TINY_PLAN,
    Tokenizer,
    install_fakes,
    restore_fakes,
    values,
)

_PLAN_NAME = "tiny"

#: An accuracy no scored arm can reach on this corpus, so its appearance in
#: the record proves the value was RESUMED rather than recomputed. The fake
#: embedder and the tiny plan cannot produce a perfect arm.
_PLANTED_ACCURACY = 1.0


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the shared fakes, and put the real hooks back afterwards."""
    install_fakes()
    yield None
    restore_fakes()


def _question_set_identity() -> tuple[str, tuple[str, ...]]:
    """Build the same question set the run will, and report what identifies it.

    A checkpoint whose fingerprint disagrees is REFUSED rather than resumed,
    so a test planting a mismatched one would exercise the refusal and pass
    while proving nothing about resume.

    THE ITEM IDS COME BACK TOO, and they are not decoration. ``compare_arms``
    pairs two arms' outcomes BY ITEM ID -- that is what makes the comparison
    paired rather than unpaired -- so a resumed arm carrying invented ids
    cannot be compared against anything. The first version of this fixture
    used ``planted-0``, ``planted-1`` and died on a KeyError, which is the
    real contract asserting itself: a checkpoint holds the ids of the items
    it actually scored.

    Returns:
        The corpus digest and every item's id, in order.
    """
    tokenizer = Tokenizer()
    encoder = bench.HFTokenizerEncoder(tokenizer)
    encoded = [tokenizer.encode(document) for document in DOCUMENTS]
    items, _training = build_question_set(DOCUMENTS, encoded, encoder, TINY_PLAN)
    return corpus_digest(DOCUMENTS), tuple(item["item_id"] for item in items)


def _perfect(item_ids: tuple[str, ...]) -> ClozeEvalResult:
    """Build a scored result no real arm on this corpus would produce.

    Args:
        item_ids: The question set's item ids, which the outcomes must carry
            so the paired comparison can line two arms up.

    Returns:
        A result where every item was answered correctly, with scores that
        agree with that -- the answer sits at index 0 and must be the strict
        minimum, which ``decode_cloze_eval_result`` enforces.
    """
    outcomes: list[ClozeItemOutcome] = [
        {"item_id": item_id, "correct": True, "scores": [1.0, 2.0]} for item_id in item_ids
    ]
    return ClozeEvalResult(
        total=len(item_ids),
        correct=len(item_ids),
        accuracy=_PLANTED_ACCURACY,
        chance=0.25,
        outcomes=outcomes,
    )


def _plant(
    directory: pathlib.Path, *, arms: tuple[str, ...] = (), seeds: tuple[int, ...] = ()
) -> None:
    """Write a checkpoint claiming the named arms and seeds are already done.

    Args:
        directory: Where the checkpoint file goes.
        arms: Retrieval arms to claim, each with a perfect planted result.
        seeds: Cartridge seeds to claim.
    """
    digest, item_ids = _question_set_identity()
    item_count = len(item_ids)
    save_qa_checkpoint(
        directory,
        QaCheckpoint(
            schema_version=QA_CHECKPOINT_SCHEMA_VERSION,
            plan_name=_PLAN_NAME,
            corpus_digest=digest,
            item_count=item_count,
            model_id=TINY_PLAN["model_id"],
            arms=[
                ArmRecord(
                    arm=arm,
                    result=_perfect(item_ids),
                    select_seconds=0.0,
                    score_seconds=99.0,
                    index_seconds=0.0,
                    evidence_fraction=0.0,
                )
                for arm in arms
            ],
            seeds=[
                SeedRecord(
                    seed=seed,
                    result=_perfect(item_ids),
                    answer_nll=PairedComparison(
                        items=item_count,
                        mean_baseline=9.0,
                        mean_treatment=1.0,
                        improved=item_count,
                        worsened=0,
                        tied=0,
                        p_value=0.5,
                        outcomes_digest="planted",
                    ),
                    score_seconds=99.0,
                )
                for seed in seeds
            ],
        ),
    )


def _measure(tmp_path: pathlib.Path) -> dict[str, float]:
    """Run the measurement against a checkpoint directory under ``tmp_path``.

    Args:
        tmp_path: The test's scratch directory.

    Returns:
        The observations, by name.
    """
    measured = bench.measure_qa_plan(
        _PLAN_NAME,
        TINY_PLAN,
        corpus=tmp_path,
        device="cpu",
        checkpoints=tmp_path / "ckpt",
    )
    return values(measured["observations"])


class TestResumingARetrievalArm:
    def test_a_checkpointed_arm_is_not_scored_again(self, tmp_path: pathlib.Path) -> None:
        """The planted accuracy reaching the record IS the proof of the skip.

        A perfect BM25 arm cannot happen on this corpus, so if the arm were
        re-scored the record would carry a real number instead.
        """
        _plant(tmp_path / "ckpt", arms=("bm25",))

        named = _measure(tmp_path)

        assert named["bm25_accuracy"] == pytest.approx(_PLANTED_ACCURACY)

    def test_the_arms_around_it_are_still_measured(self, tmp_path: pathlib.Path) -> None:
        """Resume is per-arm, so a run that skips one still scores the rest.

        Asserted as "not the planted value" rather than against a fixed
        number, because what matters is that these arms were computed, not
        what they scored.
        """
        _plant(tmp_path / "ckpt", arms=("bm25",))

        named = _measure(tmp_path)

        assert named["dense_accuracy"] != pytest.approx(_PLANTED_ACCURACY)
        assert named["long_context_accuracy"] != pytest.approx(_PLANTED_ACCURACY)

    def test_several_arms_resume_together(self, tmp_path: pathlib.Path) -> None:
        _plant(tmp_path / "ckpt", arms=("bm25", "dense", "long_context"))

        named = _measure(tmp_path)

        for arm in ("bm25", "dense", "long_context"):
            assert named[f"{arm}_accuracy"] == pytest.approx(_PLANTED_ACCURACY)


class TestResumingACartridgeSeed:
    def test_a_checkpointed_seed_is_not_retrained(self, tmp_path: pathlib.Path) -> None:
        """THE MOST EXPENSIVE SKIP IN THE RUN.

        Every seed trains a cartridge over the whole corpus before it scores
        anything. The planted seed reports a perfect accuracy, so its gain
        over the base arm is the largest the plan could ever record -- a
        retrained seed would report a real, far smaller one.
        """
        _plant(tmp_path / "ckpt", seeds=(TINY_PLAN["seeds"][0],))

        named = _measure(tmp_path)

        base = named["base_accuracy"]
        assert named["cartridge-accuracy-gain_seed7_gain"] == pytest.approx(
            _PLANTED_ACCURACY - base
        )

    def test_a_resumed_seed_carries_its_own_answer_nll(self, tmp_path: pathlib.Path) -> None:
        """The paired comparison travels with the seed or the cartridge claim
        loses the half that says the answer became more LIKELY.

        The planted comparison improves by 8.0, which the real fake model
        cannot produce.
        """
        _plant(tmp_path / "ckpt", seeds=(TINY_PLAN["seeds"][0],))

        named = _measure(tmp_path)

        assert named["cartridge-answer-nll-gain_seed7_gain"] == pytest.approx(8.0)


class TestTheCheckpointIsCleanedUp:
    def test_a_completed_run_deletes_its_checkpoint(self, tmp_path: pathlib.Path) -> None:
        """A leftover file is indistinguishable from an interrupted run, so
        the NEXT submission of this plan would skip arms it should have
        re-measured and report an earlier execution's numbers as its own."""
        _plant(tmp_path / "ckpt", arms=("bm25",))

        _measure(tmp_path)

        assert checkpoint_exists(tmp_path / "ckpt", _PLAN_NAME) is False

    def test_a_run_with_no_checkpoint_still_finishes_clean(self, tmp_path: pathlib.Path) -> None:
        """The ordinary first run: nothing to resume, nothing left behind."""
        named = _measure(tmp_path)

        assert "bm25_accuracy" in named
        assert checkpoint_exists(tmp_path / "ckpt", _PLAN_NAME) is False
