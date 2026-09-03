"""The plan table, and the label that keeps two corpora from being differenced.

A cartridge gain means nothing apart from the corpus, window, schedule and
seeds that produced it. The plan holds all of those but the corpus, which
stays a path because it is data and lives somewhere different on every
machine -- so the label folds in a digest of the corpus text, and that is the
part worth testing hardest.
"""

from __future__ import annotations

import pytest

from model_trainer.core.contracts.replicated_measurement import MIN_SEEDS
from model_trainer.core.services.model.cartridge_plans import (
    CARTRIDGE_EXPERIMENT,
    CARTRIDGE_PLANS,
    CartridgePlan,
    corpus_digest,
    plan_label,
    require_cartridge_plan,
)


class TestRequirePlan:
    def test_it_returns_the_named_plan(self) -> None:
        assert require_cartridge_plan(CARTRIDGE_PLANS, "gpt2-wiki") is CARTRIDGE_PLANS["gpt2-wiki"]

    def test_an_unknown_plan_names_the_known_ones(self) -> None:
        """A bare dict index raises the same class carrying only the missing key.

        The answer to a mistyped plan name is nearly always the list.
        """
        with pytest.raises(KeyError, match="gpt2-wiki"):
            require_cartridge_plan(CARTRIDGE_PLANS, "gpt2-wik")

    def test_it_looks_in_the_table_it_is_given(self) -> None:
        """The table is a parameter so a suite can reach this path at all.

        Every plan in the real table is minutes of GPU per arm; a lookup that
        could only consult it would leave the CLI's plan resolution untested
        or would run it.
        """
        plan = CARTRIDGE_PLANS["gpt2-wiki"]

        assert require_cartridge_plan({"only": plan}, "only") is plan


class TestPlanTable:
    def test_every_plan_names_enough_seeds_to_have_a_floor(self) -> None:
        """A plan is the thing that decides replication, so it must clear the bar.

        Failing here rather than at run time matters: the arms are the
        expensive part, and a plan that cannot be replicated would be found
        out after paying for them.
        """
        for name, plan in CARTRIDGE_PLANS.items():
            assert len(plan["seeds"]) >= MIN_SEEDS, name

    def test_every_plan_sweeps_in_increasing_slot_order(self) -> None:
        """`sweep_observations` names each step by the pair it compares.

        Out of order, the record would carry a step called
        `slots-128_to_slots-8` reporting a negative difference, which reads as
        a finding rather than as a table sorted wrongly.
        """
        for name, plan in CARTRIDGE_PLANS.items():
            counts = plan["slot_counts"]
            assert list(counts) == sorted(counts), name

    def test_every_plan_uses_distinct_seeds(self) -> None:
        """A repeated seed is a replicate of nothing -- it measures the same draw twice.

        The spread would come back smaller than the true one, and the floor
        built from it would license claims.
        """
        for name, plan in CARTRIDGE_PLANS.items():
            assert len(set(plan["seeds"])) == len(plan["seeds"]), name

    def test_the_experiment_name_is_fixed(self) -> None:
        """It is what makes two records comparable at all.

        A run under a caller-supplied experiment could not be compared with
        the record it was meant to check.
        """
        assert CARTRIDGE_EXPERIMENT == "cartridge-capacity-and-composition"


class TestCorpusDigest:
    def test_the_same_text_digests_the_same(self) -> None:
        assert corpus_digest(["alpha", "beta"]) == corpus_digest(["alpha", "beta"])

    def test_different_text_digests_differently(self) -> None:
        assert corpus_digest(["alpha", "beta"]) != corpus_digest(["alpha", "gamma"])

    def test_document_boundaries_change_the_digest(self) -> None:
        """The whole point of hashing lengths as well as bytes.

        `build_windows` never chunks across a document boundary, so one
        document of "alphabeta" and two of "alpha" and "beta" produce
        different windows. They are different corpora and must not share a
        name.
        """
        assert corpus_digest(["alphabeta"]) != corpus_digest(["alpha", "beta"])

    def test_document_order_changes_the_digest(self) -> None:
        """Order decides which windows the stride holds out."""
        assert corpus_digest(["alpha", "beta"]) != corpus_digest(["beta", "alpha"])


class TestPlanLabel:
    def test_every_field_that_moves_a_gain_appears(self) -> None:
        plan = require_cartridge_plan(CARTRIDGE_PLANS, "gpt2-wiki")

        label = plan_label("gpt2-wiki", plan, digest="0123456789abcdef")

        assert label == (
            "gpt2-wiki-gpt2-w256-s4-e12-lr0.01-slots2.8.32.128.512-c128-seeds7.8.9-0123456789ab"
        )

    def test_two_corpora_get_two_labels(self) -> None:
        """The hole the digest exists to close.

        Without it a run against different bytes would register under an
        existing name and be differenced against numbers it cannot reproduce.
        """
        plan = require_cartridge_plan(CARTRIDGE_PLANS, "gpt2-wiki")

        first = plan_label("gpt2-wiki", plan, digest=corpus_digest(["alpha"]))
        second = plan_label("gpt2-wiki", plan, digest=corpus_digest(["beta"]))

        assert first != second

    def test_changing_the_seeds_changes_the_label(self) -> None:
        """The seeds are part of what a plan measures, not a detail of running it.

        Every arm's spread comes from that draw, so a different draw reports a
        different floor and must carry a different name.
        """
        plan = require_cartridge_plan(CARTRIDGE_PLANS, "gpt2-wiki")
        redrawn = CartridgePlan(
            model_id=plan["model_id"],
            window=plan["window"],
            held_out_stride=plan["held_out_stride"],
            slot_counts=plan["slot_counts"],
            composition_slots=plan["composition_slots"],
            seeds=(1, 2, 3),
            epochs=plan["epochs"],
            learning_rate=plan["learning_rate"],
        )

        assert plan_label("p", plan, digest="d" * 16) != plan_label("p", redrawn, digest="d" * 16)
