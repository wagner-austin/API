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

from model_trainer.core.contracts.replicated_measurement import MIN_SEEDS
from model_trainer.core.services.model.cartridge_plans import (
    CARTRIDGE_EXPERIMENT,
    corpus_digest,
    require_cartridge_plan,
)
from model_trainer.core.services.model.cartridge_qa_plans import (
    QA_EXPERIMENT,
    QA_PLANS,
    QaPlan,
    qa_plan_label,
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

    def test_the_experiment_is_not_the_loss_experiment_s(self) -> None:
        """A loss record and a question-set record must never be differenced.

        `compare_run_records` refuses records from different experiments,
        which is the behaviour wanted: one says how surprising the prose was,
        the other whether the model could use it.
        """
        assert QA_EXPERIMENT == "cartridge-question-set"
        assert QA_EXPERIMENT != CARTRIDGE_EXPERIMENT


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
