"""The question-set plan table, and what its label has to name.

The label exists so two records cannot be differenced unless they measured the
same thing. For a question set that means naming ``distractor_count``, because
that field has been measured to invert the answer: over one 24-item set on
gpt2, a repeated distractor triple put the base model at chance and made the
cartridge look significant, and rotating distractors moved the base to 0.5417
and removed the effect.
"""

from __future__ import annotations

import pytest

from model_trainer.core.contracts.qa_plan import QA_EXPERIMENT, QaPlan
from model_trainer.core.contracts.replicated_measurement import MIN_SEEDS
from model_trainer.core.services.model.cartridge_plans import (
    CARTRIDGE_EXPERIMENT,
    corpus_digest,
    require_cartridge_plan,
)
from model_trainer.core.services.model.cartridge_qa_plans import QA_PLANS
from model_trainer.core.services.model.cartridge_qa_power import resolvable_floor
from model_trainer.core.services.model.cloze.identity import qa_plan_label

#: The scale ladder: one field moves, and it is the base model.
_SCALE_LADDER: tuple[str, ...] = (
    "gpt2-wiki-qa",
    "gpt2-medium-wiki-qa",
    "gpt2-large-wiki-qa",
    "gpt2-xl-wiki-qa",
)

#: The capacity axis: one field moves, and it is the cartridge's slot count.
_SLOT_AXIS: tuple[str, ...] = (
    "gpt2-large-api-wiki-qa-slots-32",
    "gpt2-large-api-wiki-qa-slots-64",
    "gpt2-large-api-wiki-qa-slots-128",
    "gpt2-large-api-wiki-qa-slots-256",
)


def _redrawn(
    plan: QaPlan, *, max_seq_len: int | None = None, distractor_count: int | None = None
) -> QaPlan:
    """Build a variant plan without mutating the table's own entry.

    Each varying field is a named, typed parameter rather than ``**changes``.
    A keyword bag would have to be typed ``object`` and cast back on the way
    out, which is exactly the unchecked hop the strictness rules exclude --
    and it would silently accept a misspelled field name, leaving a test that
    varies nothing and passes.

    Args:
        plan: The plan to vary.
        max_seq_len: Replacement token budget, or None to keep the plan's.
        distractor_count: Replacement distractor count, or None to keep it.

    Returns:
        The varied plan.
    """
    return QaPlan(
        model_id=plan["model_id"],
        window=plan["window"],
        held_out_stride=plan["held_out_stride"],
        num_slots=plan["num_slots"],
        max_seq_len=plan["max_seq_len"] if max_seq_len is None else max_seq_len,
        seeds=plan["seeds"],
        epochs=plan["epochs"],
        learning_rate=plan["learning_rate"],
        distractor_count=(
            plan["distractor_count"] if distractor_count is None else distractor_count
        ),
        max_items=plan["max_items"],
        smallest_effect_of_interest=plan["smallest_effect_of_interest"],
        alpha=plan["alpha"],
        mcnemar_test=plan["mcnemar_test"],
        bm25_k1=plan["bm25_k1"],
        bm25_b=plan["bm25_b"],
        retrieved_chunks=plan["retrieved_chunks"],
        expansion_feedback_chunks=plan["expansion_feedback_chunks"],
        expansion_terms=plan["expansion_terms"],
        rerank_candidates=plan["rerank_candidates"],
    )


class TestTheSharedPlanLookupServesQaPlansToo:
    def test_it_returns_the_named_plan(self) -> None:
        assert require_cartridge_plan(QA_PLANS, "gpt2-wiki-qa") is QA_PLANS["gpt2-wiki-qa"]

    def test_an_unknown_plan_names_the_known_ones(self) -> None:
        with pytest.raises(KeyError, match="gpt2-wiki-qa"):
            require_cartridge_plan(QA_PLANS, "gpt2-wiki-q")

    def test_it_looks_in_the_table_it_is_given(self) -> None:
        """The table is a parameter so a suite can reach this path cheaply."""
        plan = QA_PLANS["gpt2-wiki-qa"]

        assert require_cartridge_plan({"only": plan}, "only") is plan


class TestThePlanTable:
    def test_every_plan_names_enough_seeds_to_have_a_spread(self) -> None:
        for name, plan in QA_PLANS.items():
            assert len(plan["seeds"]) >= MIN_SEEDS, name

    def test_every_plan_uses_distinct_seeds(self) -> None:
        for name, plan in QA_PLANS.items():
            assert len(set(plan["seeds"])) == len(plan["seeds"]), name

    def test_the_window_leaves_room_for_the_cartridge(self) -> None:
        """The prefix occupies positions the scored item can no longer use.

        A budget at or above the model's own window would put the cartridge
        arm past the position embedding and raise torch's `IndexError` from
        somewhere that names neither the cartridge nor the limit.
        """
        for name, plan in QA_PLANS.items():
            assert plan["max_seq_len"] > plan["window"], name
            assert plan["num_slots"] < plan["max_seq_len"], name

    def test_the_ladder_varies_the_model_and_nothing_else(self) -> None:
        """A ladder that moved a second field would confound what it measures.

        The rungs exist to answer whether "a cartridge loses to BM25" is a
        fact about cartridges or a fact about gpt2's 124M. That question is
        only answerable if `model_id` is the ONLY difference -- a rung that
        also nudged `epochs` or `num_slots` would produce a number nobody
        could attribute, and the attribution is the entire point.
        """
        # Built by SPREADING the base and overriding only `model_id`, rather
        # than by comparing field names. Naming the fields would leave a
        # field added to QaPlan later silently unchecked; this way the
        # equality covers whatever the type holds on the day it runs.
        base = QA_PLANS["gpt2-wiki-qa"]
        rungs = ["gpt2-medium-wiki-qa", "gpt2-large-wiki-qa", "gpt2-xl-wiki-qa"]

        for name in rungs:
            rung = QA_PLANS[name]
            expected: QaPlan = {**base, "model_id": rung["model_id"]}
            assert rung == expected, name

    def test_the_ladder_holds_the_window_fixed_and_so_tests_only_scale(self) -> None:
        """Stated as an assertion because it BOUNDS what the ladder can claim.

        `max_seq_len` is constant across the rungs, so a bigger model reads
        the same short, truncated prompt. That makes this a test of model
        scale and NOT of context length -- which is the axis a cartridge is
        supposed to win on, since it exists to compress a long context. If
        someone later varies the window here, this test fails and the ladder's
        claim has to be restated.

        SCOPED TO THE SCALE LADDER rather than to the whole table, since the
        table gained a second family on 2026-09-09. The invariant is a
        property of an AXIS -- hold everything but one field -- and asserting
        it across two axes at once would only be satisfiable by a table with
        one axis in it.
        """
        windows = {QA_PLANS[name]["max_seq_len"] for name in _SCALE_LADDER}

        assert windows == {896}

    def test_the_slot_axis_holds_the_window_fixed_and_so_tests_only_capacity(self) -> None:
        """The same discipline, for the axis that had never been varied.

        Every cell is held to 768 rather than to its own ``1024 - num_slots``.
        Sizing each cell to its own prefix would hand the 32-slot cell 992
        tokens of evidence and the 256-slot cell 768, so the cell with the
        smallest cartridge would also have the largest retrieval budget and
        the axis would measure two things moving in opposite directions.
        """
        cells = [QA_PLANS[name] for name in _SLOT_AXIS]

        assert {cell["max_seq_len"] for cell in cells} == {768}
        assert {cell["num_slots"] for cell in cells} == {32, 64, 128, 256}

    def test_the_slot_axis_varies_nothing_but_its_slot_count(self) -> None:
        """Anything else moving with it would confound the capacity reading."""
        base = QA_PLANS["gpt2-large-api-wiki-qa-slots-128"]

        for name in _SLOT_AXIS:
            cell = QA_PLANS[name]
            expected: QaPlan = {**base, "num_slots": cell["num_slots"]}
            assert cell == expected, name

    def test_the_ladder_reaches_past_the_crossing_it_reports(self) -> None:
        """A ladder that stops at its own finding cannot test it.

        The crossing this programme reports is at ~774M, and until
        2026-09-09 the table stopped at gpt2-xl -- one rung later. The base
        added is the one the cartridge sweeps already run on an A30, so the
        GPU profile is proven rather than assumed.
        """
        assert QA_PLANS["pythia-6.9b-api-wiki-qa"]["model_id"] == "EleutherAI/pythia-6.9b"

    def test_the_experiment_is_not_the_loss_experiment_s(self) -> None:
        """A loss record and a question-set record must never be differenced.

        `compare_run_records` refuses records from different experiments,
        which is the behaviour wanted: one says how surprising the prose was,
        the other whether the model could use it.
        """
        assert QA_EXPERIMENT == "cartridge-question-set"
        assert QA_EXPERIMENT != CARTRIDGE_EXPERIMENT


class TestTheFullWikiPlan:
    """The plan whose cap does not bind, and why that is the point.

    Every other plan in the table declares a `max_items` at or below what its
    corpus yields, so the CAP is the instrument's limit and the corpus is
    hidden behind it. This one is the reverse, and the tests assert that
    property rather than the number.
    """

    def test_its_cap_exceeds_what_its_corpus_can_yield(self) -> None:
        """Measured 2026-09-09: ~/PROJECTS/wiki offers 3,735 items at gpt2.

        A cap at or under that would make this plan's item count a fact about
        this file rather than about the corpus, which is exactly how a
        240-item cap came to be read as an unavailable instrument.
        """
        assert QA_PLANS["gpt2-full-wiki-qa"]["max_items"] > 3735

    def test_it_resolves_the_effect_every_plan_declares(self) -> None:
        """The corpus clears the declared effect with real margin.

        At 3,735 items the floor is 0.00134 against a declared 0.02. Asserted
        through the gate rather than by arithmetic written here, so the two
        cannot drift.
        """
        plan = QA_PLANS["gpt2-full-wiki-qa"]

        assert (
            resolvable_floor(3735, plan["alpha"], plan["mcnemar_test"])
            < (plan["smallest_effect_of_interest"])
        )

    def test_it_re_runs_the_retracted_comparison_rather_than_a_new_one(self) -> None:
        """gpt2 is where the cartridge was reported to lose to every retriever.

        A larger rung on an untrusted instrument answers a question nobody
        asked; the same rung on a fifteen-times finer one answers the question
        that was withdrawn.
        """
        plan = QA_PLANS["gpt2-full-wiki-qa"]
        retracted = QA_PLANS["gpt2-wiki-qa"]

        assert plan["model_id"] == retracted["model_id"] == "gpt2"
        assert plan["num_slots"] == retracted["num_slots"]
        assert plan["max_seq_len"] == retracted["max_seq_len"]
        assert plan["distractor_count"] == retracted["distractor_count"]


class TestQaPlanLabel:
    def test_every_field_that_moves_a_number_appears(self) -> None:
        label = qa_plan_label("gpt2-wiki-qa", QA_PLANS["gpt2-wiki-qa"], digest="0123456789abcdef")

        assert label == (
            "gpt2-wiki-qa-gpt2-w256-s4-c128-m896-e12-lr0.01-d3-n120-seeds7.8.9-0123456789ab"
        )

    def test_changing_the_distractor_count_changes_the_label(self) -> None:
        """The field measured to invert the answer.

        A record built with three distractors and one built with seven are not
        the same measurement, and a shared label would let them be subtracted.
        """
        plan = QA_PLANS["gpt2-wiki-qa"]

        assert qa_plan_label("p", plan, digest="d" * 16) != qa_plan_label(
            "p", _redrawn(plan, distractor_count=7), digest="d" * 16
        )

    def test_changing_the_token_budget_changes_the_label(self) -> None:
        """The budget bounds how much evidence the retrieval arm can carry."""
        plan = QA_PLANS["gpt2-wiki-qa"]

        assert qa_plan_label("p", plan, digest="d" * 16) != qa_plan_label(
            "p", _redrawn(plan, max_seq_len=512), digest="d" * 16
        )

    def test_two_corpora_get_two_labels(self) -> None:
        plan = QA_PLANS["gpt2-wiki-qa"]

        assert qa_plan_label("p", plan, digest=corpus_digest(["a"])) != qa_plan_label(
            "p", plan, digest=corpus_digest(["b"])
        )
