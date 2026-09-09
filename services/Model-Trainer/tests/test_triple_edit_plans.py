"""The triple-edit plan table, checked against what a plan has to promise.

A plan is the whole description of one measurement here: which question set it
is comparable to, which corpus its triples were curated against, where the
edit is written and how hard it is pushed. Every assertion below is about one
of those promises being keepable rather than about a value being pretty.
"""

from __future__ import annotations

from model_trainer.core.services.model.cartridge_qa_plans import QA_PLANS
from model_trainer.core.services.model.editing.curated_triples import (
    CORPUS_DIGEST,
    ME_WIKI_PUBLIC_TRIPLES,
)
from model_trainer.core.services.model.editing.triple_edit_plans import (
    TRIPLE_EDIT_EXPERIMENT,
    TRIPLE_EDIT_PLANS,
    triple_edit_plan_label,
)

_REFERENCE = "gpt2-triples"

#: The rungs that hold the base fixed and move the dose.
_DOSE_RUNGS = (
    _REFERENCE,
    "gpt2-triples-dose-25s-lr005",
    "gpt2-triples-dose-10s-lr005",
    "gpt2-triples-dose-5s-lr005",
    "gpt2-triples-dose-5s-lr001",
)

#: The rungs that hold the dose fixed and move the base.
_LADDER_RUNGS = ("gpt2-medium-triples", "gpt2-large-triples", "gpt2-xl-triples")


class TestTheTable:
    def test_every_plan_names_a_question_set_that_exists(self) -> None:
        """The arm's whole claim to comparability. A plan naming a question
        set that is not in the table would produce an accuracy nothing can be
        read beside.
        """
        for name, plan in TRIPLE_EDIT_PLANS.items():
            assert plan["qa_plan"] in QA_PLANS, name

    def test_the_table_is_exactly_the_two_families(self) -> None:
        """A plan in neither family belongs to no comparison, and would be
        reported beside numbers it cannot be differenced against.
        """
        assert set(TRIPLE_EDIT_PLANS) == set(_DOSE_RUNGS) | set(_LADDER_RUNGS)

    def test_every_plan_carries_the_curation_it_is_judged_on(self) -> None:
        for name, plan in TRIPLE_EDIT_PLANS.items():
            assert plan["corpus_digest"] == CORPUS_DIGEST, name
            assert plan["candidates"] == ME_WIKI_PUBLIC_TRIPLES, name

    def test_the_dose_rungs_differ_from_the_reference_only_in_the_dose(self) -> None:
        """THE PROPERTY THAT MAKES A DOSE CURVE A DOSE CURVE.

        A rung that also moved the layer, the fact token or the question set
        would be a second experiment, and the difference between two rungs
        would stop being the dose.
        """
        reference = TRIPLE_EDIT_PLANS[_REFERENCE]
        for name in _DOSE_RUNGS:
            plan = TRIPLE_EDIT_PLANS[name]
            assert plan["site"] == reference["site"], name
            assert plan["qa_plan"] == reference["qa_plan"], name

    def test_the_curve_spans_more_than_one_order_of_magnitude(self) -> None:
        """A dose curve over a narrow band cannot separate 'this cannot work'
        from 'this was pushed too hard'.
        """
        doses = [
            TRIPLE_EDIT_PLANS[name]["value_steps"] * TRIPLE_EDIT_PLANS[name]["value_learning_rate"]
            for name in _DOSE_RUNGS
        ]

        assert max(doses) / min(doses) >= 100.0

    def test_the_ladder_rungs_differ_from_each_other_only_in_the_base(self) -> None:
        """THE PROPERTY THAT MAKES A SCALE LADDER A SCALE LADDER.

        Fact token, module template, dose, corpus and triples fixed; the
        question set changes only because each base needs its own plan, and
        those plans are themselves copies differing in ``model_id``. A rung
        that also moved the dose would confound scale with over-driving.
        """
        for name in _LADDER_RUNGS:
            plan = TRIPLE_EDIT_PLANS[name]
            assert plan["value_steps"] == 10, name
            assert plan["value_learning_rate"] == 0.05, name
            assert plan["site"]["fact_token"] == "subject_last", name
            assert plan["site"]["module_template"] == "transformer.h.{}.mlp.c_proj", name

    def test_every_ladder_rung_edits_at_the_same_relative_depth(self) -> None:
        """Half way down, at every scale.

        The same ABSOLUTE index would be a shallower site on every larger
        base, so a difference between rungs would be a difference in where the
        edit went as much as in how big the model is.
        """
        depths = {"gpt2-medium-triples": 24, "gpt2-large-triples": 36, "gpt2-xl-triples": 48}
        for name, layers in depths.items():
            assert TRIPLE_EDIT_PLANS[name]["site"]["layer"] * 2 == layers, name
        # And the reference the ladder extends sits at the same fraction.
        assert TRIPLE_EDIT_PLANS[_REFERENCE]["site"]["layer"] * 2 == 12

    def test_the_experiment_name_is_its_own(self) -> None:
        """Two experiments that share a name are two records the comparability
        layer will happily subtract.
        """
        assert TRIPLE_EDIT_EXPERIMENT == "corpus-representation-triple-edit"


class TestTheLabel:
    def test_it_carries_everything_that_makes_a_run_that_run(self) -> None:
        label = triple_edit_plan_label(
            _REFERENCE, TRIPLE_EDIT_PLANS[_REFERENCE], digest="abcdef0123456789"
        )

        assert label.startswith("gpt2-triples-gpt2-wiki-qa-L6-subject_last")
        assert "vs25" in label
        assert "vlr0.5" in label
        assert label.endswith("abcdef012345")

    def test_two_doses_do_not_share_a_label(self) -> None:
        """Otherwise two rungs of the curve are one run as far as any reader
        of the records is concerned.
        """
        labels = {
            triple_edit_plan_label(name, plan, digest="0" * 64)
            for name, plan in TRIPLE_EDIT_PLANS.items()
        }

        assert len(labels) == len(TRIPLE_EDIT_PLANS)
