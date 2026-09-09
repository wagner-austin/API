"""The precision bar a curated triple has to clear, clause by clause.

NOTHING IS FAKED. Every subject here is a pure function of literal strings,
and the one dependency -- the corpus term extractor -- is the real one,
because a test that stubbed it would stop checking the thing the clause is
about: that the triples and the question set share a definition of "entity".

Each clause gets a test that fires it ALONE. A candidate that fails three
clauses proves nothing about which line rejected it, and the reject rate is
reported per clause, so a per-clause test is what makes those counts mean
anything.
"""

from __future__ import annotations

from model_trainer.core.contracts.knowledge_edit import (
    SUBJECT_MARKER,
    render_edit_prompt,
)
from model_trainer.core.services.model.editing.grounding import (
    GATE_CLAUSES,
    TripleCandidate,
    as_edit_request,
    gate_candidates,
    gate_triple,
    normalise_whitespace,
)

_SENTENCE = (
    "Seven classifiers -- XGBoost, LightGBM, ClearGBM, logistic regression and a "
    "bidirectional LSTM -- sit behind one interface."
)

#: A candidate that clears every clause, and the base for the failing ones.
_GOOD = TripleCandidate(
    item_id="t00",
    subject="XGBoost",
    relation="sits in the same backend list as the from-scratch booster",
    object="ClearGBM",
    source_document="covenant-radar.md",
    source_sentence=_SENTENCE,
)


def _variant(
    *,
    item_id: str = "t00",
    subject: str = "XGBoost",
    relation: str = "sits in the same backend list as the from-scratch booster",
    obj: str = "ClearGBM",
    source_sentence: str = _SENTENCE,
) -> TripleCandidate:
    """Build a candidate differing from the good one in named fields.

    Spelled out rather than built by merging dictionaries: a ``**`` spread
    into a TypedDict is typed ``Any``, and an Any reaching an assertion is
    what this repository refuses.

    Args:
        item_id: Its identifier.
        subject: The entity.
        relation: The curator's phrasing.
        obj: The answer. Named ``obj`` because ``object`` is a builtin.
        source_sentence: The sentence it claims to come from.

    Returns:
        The candidate.
    """
    return TripleCandidate(
        item_id=item_id,
        subject=subject,
        relation=relation,
        object=obj,
        source_document="covenant-radar.md",
        source_sentence=source_sentence,
    )


def _verdict(candidate: TripleCandidate) -> tuple[str, ...]:
    """Gate one candidate against the good candidate's own sentence.

    Args:
        candidate: The candidate to judge.

    Returns:
        Its failed clauses.
    """
    return gate_triple(candidate, training_sentences=[_SENTENCE])["failed_clauses"]


class TestNormaliseWhitespace:
    def test_a_wrapped_sentence_matches_an_unwrapped_one(self) -> None:
        """Markdown wraps prose, and a wrap is not a difference in content."""
        assert normalise_whitespace("one\n  two\tthree ") == "one two three"

    def test_case_is_left_alone(self) -> None:
        """A tokenizer treats 'clearGBM' and 'ClearGBM' as different strings,
        so a gate that folded case would accept a triple about neither.
        """
        assert normalise_whitespace(" ClearGBM ") == "ClearGBM"


class TestTheAcceptedCase:
    def test_a_grounded_triple_clears_every_clause(self) -> None:
        verdict = gate_triple(_GOOD, training_sentences=[_SENTENCE])

        assert verdict["accepted"]
        assert verdict["failed_clauses"] == ()
        assert verdict["item_id"] == "t00"

    def test_a_line_wrapped_source_sentence_still_matches(self) -> None:
        """The curated rows are Python literals joined across source lines."""
        wrapped = _variant(source_sentence=_SENTENCE.replace(" -- ", " --\n    "))

        assert _verdict(wrapped) == ()


class TestEachClauseFiresAlone:
    def test_a_sentence_from_outside_the_training_split(self) -> None:
        """THE CLAUSE THAT KEEPS THE ARMS COMPARABLE.

        A triple sourced from a held-out sentence would hand the edit the
        answer to a question the cartridge had to infer.
        """
        assert gate_triple(_GOOD, training_sentences=["something else entirely."])[
            "failed_clauses"
        ] == ("sentence_is_a_training_sentence",)

    def test_a_subject_that_is_not_in_the_sentence(self) -> None:
        """One clause, not two: the term clause is guarded by presence, so an
        absent subject is one problem rather than a pair of them.
        """
        assert _verdict(_variant(subject="PyTorch")) == ("subject_in_sentence",)

    def test_a_subject_that_is_present_but_is_not_a_term(self) -> None:
        """THE CLAUSE THAT DOES THE MOST WORK, on the shape it was added for.

        'Seven classifiers' is verbatim in the sentence, before the object,
        and gives nothing away -- so every positional clause passes and the
        triple still asserts nothing about an entity.
        """
        assert _verdict(_variant(subject="Seven", relation="classifiers include")) == (
            "subject_is_a_corpus_term",
        )

    def test_an_object_that_is_not_in_the_sentence(self) -> None:
        assert _verdict(_variant(obj="PyTorch")) == ("object_in_sentence",)

    def test_an_object_that_precedes_its_subject(self) -> None:
        """The direction a causal continuation reads.

        `LightGBM` is in the sentence, is a term, and comes BEFORE `LSTM`
        would make sense as its answer -- reversing them is a triple the
        source does not support in that order.
        """
        assert _verdict(_variant(subject="LSTM", obj="LightGBM")) == ("object_after_subject",)

    def test_a_prompt_that_contains_its_own_answer(self) -> None:
        assert _verdict(_variant(relation="is listed beside ClearGBM, namely")) == (
            "prompt_does_not_give_away_object",
        )

    def test_a_subject_the_object_is_part_of(self) -> None:
        """`GBM` inside `LightGBM` is not an association between two things.

        Two clauses fire together and both are right: a substring of the
        subject is also a substring of the prompt, so the prompt gives the
        answer away as well.
        """
        assert _verdict(_variant(subject="LightGBM", obj="GBM")) == (
            "prompt_does_not_give_away_object",
            "subject_and_object_are_distinct",
        )

    def test_a_subject_equal_to_its_object(self) -> None:
        assert _verdict(_variant(subject="ClearGBM", relation="is also called")) == (
            "object_after_subject",
            "prompt_does_not_give_away_object",
            "subject_and_object_are_distinct",
        )


class TestOrderIsNotDoubleCounted:
    def test_a_missing_end_does_not_also_fail_the_order_clause(self) -> None:
        """Reporting a missing end twice would inflate the reject rate this
        function exists to measure.
        """
        failed = _verdict(_variant(subject="PyTorch", obj="Keras"))

        assert "object_after_subject" not in failed
        assert failed == ("subject_in_sentence", "object_in_sentence")


class TestTheReport:
    def test_it_counts_what_passed_and_what_did_not(self) -> None:
        report = gate_candidates(
            [_GOOD, _variant(item_id="t01", obj="PyTorch")], training_text=_SENTENCE
        )

        assert report["accepted"] == 1
        assert report["rejected"] == 1
        assert tuple(verdict["item_id"] for verdict in report["verdicts"]) == ("t00", "t01")

    def test_every_clause_is_present_even_at_zero(self) -> None:
        """A zero that is present is a measurement; a missing key is not."""
        report = gate_candidates([_GOOD], training_text=_SENTENCE)

        assert tuple(clause for clause, _ in report["failures_by_clause"]) == GATE_CLAUSES
        assert all(count == 0 for _, count in report["failures_by_clause"])

    def test_a_clause_failed_twice_is_counted_twice(self) -> None:
        report = gate_candidates(
            [
                _variant(item_id="t00", obj="PyTorch"),
                _variant(item_id="t01", obj="Keras"),
            ],
            training_text=_SENTENCE,
        )
        counts = dict(report["failures_by_clause"])

        assert counts["object_in_sentence"] == 2
        assert report["accepted"] == 0

    def test_it_splits_the_training_text_the_way_the_item_builder_does(self) -> None:
        """The gate is handed raw text and must reach the same sentences the
        question set was built from, or the two disagree about what the
        training half contains.
        """
        report = gate_candidates([_GOOD], training_text=f"A first sentence here. {_SENTENCE}")

        assert report["accepted"] == 1

    def test_an_empty_candidate_set_reports_zeroes_rather_than_nothing(self) -> None:
        report = gate_candidates([], training_text=_SENTENCE)

        assert report["accepted"] == 0
        assert report["rejected"] == 0
        assert report["verdicts"] == ()


class TestAsEditRequest:
    def test_it_builds_the_prompt_the_editor_renders(self) -> None:
        request = as_edit_request(_GOOD)

        assert request["item_id"] == "t00"
        assert request["subject"] == "XGBoost"
        assert request["target_new"] == "ClearGBM"
        assert request["prompt"].count(SUBJECT_MARKER) == 1
        assert render_edit_prompt(request) == (
            "XGBoost sits in the same backend list as the from-scratch booster"
        )
