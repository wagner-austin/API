"""The committed trait plans, checked against what they commit to.

A PLAN TABLE IS A SET OF PROMISES THAT NOTHING ELSE RE-READS. Every row below
declares a roster, a compartment ladder and an effect size, and each of those
can be internally inconsistent in a way that produces a complete run: a roster
shorter than the largest cell composes fewer compartments than its label
claims, and a declared effect the corpus cannot resolve is refused only once a
job has reached the cluster and read its corpus.

So the invariants are checked here, in seconds, against the same arithmetic
the gate uses at run time.
"""

from __future__ import annotations

import pathlib

import pytest

from model_trainer.core.contracts.trait_corpus import CONSISTENTLY_EFFECTIVE_TRAITS
from model_trainer.core.contracts.trait_plan import TraitPlan, trait_plan_label
from model_trainer.core.services.model.cartridge_qa_power import resolvable_floor
from model_trainer.core.services.model.cartridge_trait_plans import TRAIT_SWEEP_PLANS
from model_trainer.core.services.model.trait_corpus import load_trait_corpora, split_trait_pairs

#: The committed corpus, read rather than assumed.
#:
#: THE POINT OF READING IT IS THAT THE PLAN AND THE DATA CAN DISAGREE. A plan
#: declares an effect size; the corpus decides how many pairs are held out;
#: and the floor is a function of the second. Adding pairs to one trait and
#: not another, or tightening an effect without growing the corpus, produces a
#: plan that is refused on the cluster after a job has been scheduled. Reading
#: the real files here moves that failure to a second in this suite.
_CORPUS_ROOT = pathlib.Path(__file__).resolve().parents[4] / "corpus" / "traits"


@pytest.mark.parametrize("name", sorted(TRAIT_SWEEP_PLANS))
class TestEveryPlanIsInternallyConsistent:
    """Walked rather than sampled, so a new row cannot skip these."""

    def test_the_roster_reaches_the_largest_compartment_count(self, name: str) -> None:
        """An n4 cell consumes four traits; a shorter roster composes fewer.

        The run would still complete and its label would still say n4, so
        nothing downstream could see it.

        Args:
            name: The plan under test.
        """
        plan = TRAIT_SWEEP_PLANS[name]
        assert len(plan["traits"]) >= max(plan["compartment_counts"])

    def test_every_trait_is_admissible(self, name: str) -> None:
        """A positionally concentrated trait gives a null about trait CHOICE.

        The decoder refuses one, but only once a file has been read on a
        compute node; the plan is where the roster is chosen.

        Args:
            name: The plan under test.
        """
        plan = TRAIT_SWEEP_PLANS[name]
        assert set(plan["traits"]) <= CONSISTENTLY_EFFECTIVE_TRAITS

    def test_the_roster_names_each_trait_once(self, name: str) -> None:
        """A repeated trait composes a cartridge with a copy of itself.

        Which measures self-interference and reports it as composition.

        Args:
            name: The plan under test.
        """
        plan = TRAIT_SWEEP_PLANS[name]
        assert len(set(plan["traits"])) == len(plan["traits"])

    def test_the_declared_effect_is_resolvable_by_the_committed_corpus(self, name: str) -> None:
        """The gate would refuse this plan on the cluster otherwise.

        Computed with the same function the gate calls, against the pair count
        the COMMITTED corpus actually yields -- so this fails here, in
        seconds, rather than after a job has been scheduled.

        Args:
            name: The plan under test.
        """
        plan = TRAIT_SWEEP_PLANS[name]
        corpora = load_trait_corpora(_CORPUS_ROOT, plan["traits"])
        _train, held_out = split_trait_pairs(
            corpora[0]["pairs"], held_out_stride=plan["held_out_stride"]
        )
        floor = resolvable_floor(len(held_out), plan["alpha"], plan["mcnemar_test"])
        assert floor <= plan["smallest_effect_of_interest"]

    def test_every_trait_in_the_roster_is_committed_and_the_same_size(self, name: str) -> None:
        """A roster whose traits differ in size makes the cross arms uneven.

        Each cross arm scores one trait's cartridge on the PRIMARY trait's
        pairs, so an unequal roster does not break the scoring -- it changes
        how much text each compartment was trained on, which is a difference
        between compartments that the grid would report as composition.

        Args:
            name: The plan under test.
        """
        plan = TRAIT_SWEEP_PLANS[name]
        corpora = load_trait_corpora(_CORPUS_ROOT, plan["traits"])
        sizes = {len(corpus["pairs"]) for corpus in corpora}
        assert len(sizes) == 1

    def test_the_compartment_counts_increase(self, name: str) -> None:
        """The step verdicts subtract adjacent cells, so order is the meaning.

        Args:
            name: The plan under test.
        """
        plan = TRAIT_SWEEP_PLANS[name]
        counts = plan["compartment_counts"]
        assert list(counts) == sorted(counts)

    def test_the_label_carries_every_field_that_moves_a_number(self, name: str) -> None:
        """Two plans that differ anywhere must not share a label.

        Args:
            name: The plan under test.
        """
        plan = TRAIT_SWEEP_PLANS[name]
        label = trait_plan_label(name, plan, digest="0123456789abcdef")
        assert plan["model_id"] in label
        assert all(trait in label for trait in plan["traits"])
        assert all(str(seed) in label for seed in plan["seeds"])
        assert plan["steering_module"] in label
        assert label.endswith("0123456789ab")


class TestTheRosterControl:
    """A rotation exists so roster IDENTITY can be told from roster ORDER."""

    def test_the_rotated_plan_differs_only_in_its_roster_order(self) -> None:
        """Anything else moving would make a difference attributable to two things.

        Asserted field by field rather than by eye, because this is exactly
        the kind of pair that drifts when one of them is edited.
        """
        straight = TRAIT_SWEEP_PLANS["gpt2-traits"]
        rotated = TRAIT_SWEEP_PLANS["gpt2-traits-rotated"]
        assert straight["traits"] != rotated["traits"]
        assert sorted(straight["traits"]) == sorted(rotated["traits"])
        rebuilt: TraitPlan = {**rotated, "traits": straight["traits"]}
        assert rebuilt == straight

    def test_the_two_rosters_produce_different_labels(self) -> None:
        """Otherwise a rotation would register under the straight run's name."""
        straight = TRAIT_SWEEP_PLANS["gpt2-traits"]
        rotated = TRAIT_SWEEP_PLANS["gpt2-traits-rotated"]
        assert trait_plan_label("a", straight, digest="d" * 16) != trait_plan_label(
            "a", rotated, digest="d" * 16
        )


class TestTheDepthRung:
    """The second rung is where this arc's surprises have historically been."""

    def test_the_medium_plan_steers_deeper_than_the_small_one(self) -> None:
        """One layer index cannot serve two depths.

        A site declared for gpt2 does not exist on gpt2-medium's graph in the
        same place, and a plan that shared one would either fail at the site
        or read a direction from a different fraction of the network.
        """
        small = TRAIT_SWEEP_PLANS["gpt2-traits"]
        medium = TRAIT_SWEEP_PLANS["gpt2-medium-traits"]
        assert medium["model_id"] == "gpt2-medium"
        assert medium["steering_module"] != small["steering_module"]
