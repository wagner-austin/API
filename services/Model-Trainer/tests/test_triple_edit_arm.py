"""The corpus-representation arm, on a real model with real edits.

WHAT IS REAL. The model is the production probe builder's `tiny` GPT-2, the
edits are really solved and really written into its weights, the question set
is really re-scored afterwards, and the gate is the shipped one. Nothing about
the arm's arithmetic is stood in for, because every number it reports is a
difference between two scorings of a model that was actually edited.

WHAT IS SUPPLIED RATHER THAN BUILT. The items, the candidates and the training
text are literals here instead of being derived from a corpus. That is not a
shortcut: the arm's contract is that the triples are grounded in the TRAINING
half, and a test that derived both from one builder could not tell a grounded
triple from an ungrounded one -- it would assert whatever the builder
produced. Supplying them makes the grounding a claim the test states and the
gate checks.
"""

from __future__ import annotations

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.contracts.cloze import BLANK_MARKER, ClozeItem
from model_trainer.core.contracts.knowledge_edit import EditSite
from model_trainer.core.encoding import Encoder
from model_trainer.core.services.model.backends.hf_lm.encoding import HFTokenizerEncoder
from model_trainer.core.services.model.editing.grounding import (
    GATE_CLAUSES,
    GateReport,
    TripleCandidate,
    gate_candidates,
)
from model_trainer.core.services.model.editing.triple_edit_arm import (
    TargetLikelihood,
    TripleEditOutcome,
    accepted_candidates,
    edit_one_triple,
    run_triple_edit_arm,
    target_likelihoods,
    triple_edit_observations,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import TracedLMModelProto
from tests._qa_benchmark_support import Tokenizer

_SITE: EditSite = {
    "layer": 0,
    "module_template": "transformer.h.{}.mlp.c_proj",
    "fact_token": "subject_last",
}

#: One sentence naming two entities, which is what a triple needs and what
#: most prose does not offer.
_SENTENCE = "The measured run put NavProbe beside ClearGBM in one harness and reported both."

_TRAINING = f"A first sentence with nothing in it. {_SENTENCE}"

_GOOD = TripleCandidate(
    item_id="t00",
    subject="NavProbe",
    relation="was measured beside",
    object="ClearGBM",
    source_document="fake.md",
    source_sentence=_SENTENCE,
)

#: Rejected because its sentence is not in the training half. Present so the
#: arm's accept/reject split is exercised rather than assumed.
_UNGROUNDED = TripleCandidate(
    item_id="t01",
    subject="NavProbe",
    relation="was never measured beside",
    object="CoverGate",
    source_document="fake.md",
    source_sentence="A sentence that appears in no training text at all here.",
)

_ITEMS: tuple[ClozeItem, ...] = (
    ClozeItem(
        item_id="i00",
        template=f"The measured run put NavProbe beside {BLANK_MARKER} in one harness.",
        answer="ClearGBM",
        distractors=["CoverGate", "TankpitBot"],
    ),
    ClozeItem(
        item_id="i01",
        template=f"A later pass moved {BLANK_MARKER} onto a faster route for speed.",
        answer="NavProbe",
        distractors=["ClearGBM", "CoverGate"],
    ),
)


@pytest.fixture()
def encoder() -> Encoder:
    """A reversible word tokenizer inside the tiny rung's vocabulary.

    Returns:
        The tokenizer.
    """
    return HFTokenizerEncoder(Tokenizer())


@pytest.fixture()
def model() -> TracedLMModelProto:
    """Build the tiny rung.

    Returns:
        A real GPT-2 with the probe's deterministic initialisation.
    """
    built, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return built


def _run(model: TracedLMModelProto, encoder: Encoder, *, steps: int = 3) -> TripleEditOutcome:
    """Run the arm over the two candidates.

    Args:
        model: The model to edit.
        encoder: The tokenizer.
        steps: Value-optimisation steps, kept small so the test is about the
            plumbing rather than about convergence.

    Returns:
        What the arm produced.
    """
    return run_triple_edit_arm(
        model=model,
        encoder=encoder,
        items=_ITEMS,
        candidates=[_GOOD, _UNGROUNDED],
        training_text=_TRAINING,
        site=_SITE,
        value_steps=steps,
        value_learning_rate=0.1,
        device="cpu",
        max_seq_len=48,
    )


class TestAcceptedCandidates:
    def test_it_keeps_only_the_accepted_ones_in_order(self) -> None:
        report = gate_candidates([_GOOD, _UNGROUNDED], training_text=_TRAINING)

        assert accepted_candidates([_GOOD, _UNGROUNDED], report) == (_GOOD,)

    def test_it_keeps_nothing_when_nothing_passed(self) -> None:
        report = gate_candidates([_UNGROUNDED], training_text=_TRAINING)

        assert accepted_candidates([_UNGROUNDED], report) == ()


class TestEditOneTriple:
    def test_it_writes_into_the_weight_the_site_names(
        self, model: TracedLMModelProto, encoder: Encoder
    ) -> None:
        record = edit_one_triple(
            model=model,
            encoder=encoder,
            candidate=_GOOD,
            site=_SITE,
            value_steps=3,
            value_learning_rate=0.1,
            device="cpu",
        )

        assert record["item_id"] == "t00"
        assert record["module"] == "transformer.h.0.mlp.c_proj.weight"
        assert record["update_norm"] > 0.0

    def test_it_actually_changes_the_weight(
        self, model: TracedLMModelProto, encoder: Encoder
    ) -> None:
        """A record describing an edit that did not land would be the worst
        possible outcome: a run that looks complete and measured nothing.
        """
        parameter = dict(model.named_parameters())["transformer.h.0.mlp.c_proj.weight"]
        before = parameter.detach().clone()

        edit_one_triple(
            model=model,
            encoder=encoder,
            candidate=_GOOD,
            site=_SITE,
            value_steps=3,
            value_learning_rate=0.1,
            device="cpu",
        )

        assert not torch.equal(before, parameter.detach())


class TestTargetLikelihoods:
    def test_one_score_per_candidate_in_order(
        self, model: TracedLMModelProto, encoder: Encoder
    ) -> None:
        scores = target_likelihoods(model=model, encoder=encoder, candidates=[_GOOD], device="cpu")

        assert len(scores) == 1
        assert scores[0] > 0.0

    def test_no_candidates_scores_nothing(
        self, model: TracedLMModelProto, encoder: Encoder
    ) -> None:
        assert target_likelihoods(model=model, encoder=encoder, candidates=[], device="cpu") == ()


class TestRunTripleEditArm:
    def test_the_base_accuracy_is_measured_before_any_edit(
        self, model: TracedLMModelProto, encoder: Encoder
    ) -> None:
        """THE ORDER THE ARM CANNOT GET WRONG SILENTLY.

        The edits change the weights in place and there is no second copy of
        the base, so a base measured afterwards would be the edited model
        wearing the base's name. Checked by scoring the untouched model here
        and requiring the arm's own base to match it.
        """
        from model_trainer.core.services.model.cloze.score import score_cloze_items

        expected = score_cloze_items(
            items=_ITEMS, model=model, encoder=encoder, device="cpu", max_seq_len=48
        )["accuracy"]

        outcome = _run(model, encoder)

        assert outcome["base_accuracy"] == expected

    def test_the_curve_has_one_point_per_applied_edit(
        self, model: TracedLMModelProto, encoder: Encoder
    ) -> None:
        outcome = _run(model, encoder)

        assert len(outcome["edits"]) == 1
        assert len(outcome["accuracy_curve"]) == 1

    def test_the_rejected_candidate_is_counted_and_not_edited(
        self, model: TracedLMModelProto, encoder: Encoder
    ) -> None:
        outcome = _run(model, encoder)

        assert outcome["gate"]["accepted"] == 1
        assert outcome["gate"]["rejected"] == 1
        assert [record["item_id"] for record in outcome["edits"]] == ["t00"]

    def test_every_accepted_triple_gets_a_before_and_an_after(
        self, model: TracedLMModelProto, encoder: Encoder
    ) -> None:
        outcome = _run(model, encoder)

        assert len(outcome["likelihoods"]) == 1
        assert outcome["likelihoods"][0]["item_id"] == "t00"
        assert outcome["likelihoods"][0]["before"] != outcome["likelihoods"][0]["after"]

    def test_nothing_accepted_leaves_the_model_untouched(
        self, model: TracedLMModelProto, encoder: Encoder
    ) -> None:
        """The empty-curve case is a measurement, not a missing one: the
        edited model IS the base model and the record has to say so.
        """
        outcome = run_triple_edit_arm(
            model=model,
            encoder=encoder,
            items=_ITEMS,
            candidates=[_UNGROUNDED],
            training_text=_TRAINING,
            site=_SITE,
            value_steps=3,
            value_learning_rate=0.1,
            device="cpu",
            max_seq_len=48,
        )

        assert outcome["edits"] == ()
        assert outcome["accuracy_curve"] == ()
        assert outcome["likelihoods"] == ()


def _report(accepted: int, rejected: int) -> GateReport:
    """Build a gate report with the counts a test needs.

    Args:
        accepted: How many passed.
        rejected: How many failed.

    Returns:
        The report, every clause present, the first carrying the rejections.
        Every clause, because that is the shipped report's own guarantee and a
        stand-in that dropped one would test a shape production never emits.
    """
    counts = dict.fromkeys(GATE_CLAUSES, 0)
    counts["sentence_is_a_training_sentence"] = rejected
    return GateReport(
        verdicts=(),
        accepted=accepted,
        rejected=rejected,
        failures_by_clause=tuple((clause, counts[clause]) for clause in GATE_CLAUSES),
    )


class TestObservations:
    def test_the_cost_line_travels_with_the_accuracy(
        self, model: TracedLMModelProto, encoder: Encoder
    ) -> None:
        """A representation that works and costs more than writing the answer
        by hand is a negative result, and only a record carrying both can be
        read as one.
        """
        outcome = _run(model, encoder)
        named = {
            observation["name"]: observation["value"]
            for observation in triple_edit_observations(outcome)
        }
        likelihood = outcome["likelihoods"][0]

        assert named["triples_curated"] == 2.0
        assert named["triples_accepted"] == 1.0
        assert named["triples_rejected"] == 1.0
        assert named["triples_reject_rate"] == 0.5
        assert named["edits_applied"] == 1.0
        assert named["accuracy_after_1_edits"] == named["edited_accuracy"]
        assert named["edited_accuracy_gain"] == pytest.approx(
            named["edited_accuracy"] - named["base_accuracy"]
        )
        # Compared against the outcome's own likelihoods rather than pinned to
        # a constant: the rate is a fact about this edit, and pinning it would
        # make the test a statement about one initialisation.
        assert named["edit_success_rate"] == float(likelihood["after"] < likelihood["before"])

    def test_every_gate_clause_is_named_even_at_zero(
        self, model: TracedLMModelProto, encoder: Encoder
    ) -> None:
        named = {
            observation["name"] for observation in triple_edit_observations(_run(model, encoder))
        }

        assert "gate_rejects_subject_is_a_corpus_term" in named
        assert "gate_rejects_object_after_subject" in named

    def test_an_empty_curve_reports_the_base_as_the_edited_accuracy(self) -> None:
        """Rather than omitting the field: a run that accepted nothing did
        measure an edited accuracy, and it is the base's.
        """
        outcome = TripleEditOutcome(
            gate=_report(0, 3),
            edits=(),
            likelihoods=(),
            base_accuracy=0.25,
            accuracy_curve=(),
        )

        named = {
            observation["name"]: observation["value"]
            for observation in triple_edit_observations(outcome)
        }

        assert named["edited_accuracy"] == 0.25
        assert named["edited_accuracy_gain"] == 0.0
        assert "edit_success_rate" not in named

    def test_a_curated_set_of_nothing_reports_a_zero_reject_rate(self) -> None:
        """A division by zero here would be a NaN in a record, which reads as
        a measurement and is not one.
        """
        outcome = TripleEditOutcome(
            gate=_report(0, 0),
            edits=(),
            likelihoods=(),
            base_accuracy=0.5,
            accuracy_curve=(),
        )

        named = {
            observation["name"]: observation["value"]
            for observation in triple_edit_observations(outcome)
        }

        assert named["triples_reject_rate"] == 0.0

    def test_the_success_rate_counts_targets_that_got_likelier(self) -> None:
        outcome = TripleEditOutcome(
            gate=_report(3, 0),
            edits=(),
            likelihoods=(
                TargetLikelihood(item_id="a", before=10.0, after=1.0),
                TargetLikelihood(item_id="b", before=10.0, after=11.0),
                TargetLikelihood(item_id="c", before=10.0, after=2.0),
            ),
            base_accuracy=0.5,
            accuracy_curve=(0.5,),
        )

        named = {
            observation["name"]: observation["value"]
            for observation in triple_edit_observations(outcome)
        }

        assert named["edit_success_rate"] == pytest.approx(2.0 / 3.0)
        assert named["target_nll_before"] == pytest.approx(10.0)


class TestFailuresPropagate:
    def test_a_site_naming_no_module_is_not_swallowed(
        self, model: TracedLMModelProto, encoder: Encoder
    ) -> None:
        """An arm that skipped a failed edit would report the accuracy of a
        different experiment.
        """
        missing: EditSite = {
            "layer": 99,
            "module_template": "transformer.h.{}.mlp.c_proj",
            "fact_token": "subject_last",
        }

        with pytest.raises(AppError) as raised:
            edit_one_triple(
                model=model,
                encoder=encoder,
                candidate=_GOOD,
                site=missing,
                value_steps=3,
                value_learning_rate=0.1,
                device="cpu",
            )

        assert raised.value.code is ModelTrainerErrorCode.EDIT_MODULE_NOT_FOUND
