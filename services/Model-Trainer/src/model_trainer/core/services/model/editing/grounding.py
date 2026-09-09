"""Whether a curated triple is grounded in the text it claims to come from.

WHY THIS EXISTS AT ALL, and it is not a quality filter bolted onto a pipeline.
AKEW (Wu et al. 2024) measures locate-then-edit methods at 93-99% on clean
subject-relation-object triples and 2.25-4.78% on the same facts reached by
automatic triplet extraction from prose. The gap is representation, not
method. So an arm that runs an LLM over prose and edits whatever comes out is
buying a published negative result; the only version worth measuring is
triples held to a precision bar, with the bar stated and its REJECT RATE
reported as the cost.

This module is that bar, and every clause of it is mechanical. Nothing here
asks a model whether a triple looks right, because a judge that can be
persuaded is not a bar -- it is the extraction step wearing a second hat.

WHAT IS DELIBERATELY NOT REQUIRED. The relation is free text and is not
checked against the source. That is the curator's contribution: turning
"ClearGBM, a Rust reimplementation of LightGBM's histogram loop" into a
question a causal model can continue. What IS required is that both ENDS of
the association -- the subject and the answer -- come from one named sentence,
in the order a continuation reads them.

WHY THE SUBJECT MUST BE A CORPUS TERM, which is the clause that does the most
work. Without it every other clause is satisfiable by a fragment: "A closed
PR is worth stating plainly" yields subject "A closed", object "PR", both
verbatim, in the right order, no giveaway -- a triple that passes a purely
positional bar and asserts nothing. A locate-then-edit method writes an
association between ENTITIES, and this corpus already has a definition of what
counts as one: :func:`~model_trainer.core.services.model.corpus_cloze.terms_in`,
the same extractor that decided which terms the question set is allowed to ask
about. Reusing it rather than inventing a second notion of entity keeps the
triples and the questions talking about the same vocabulary.

WHERE THE SENTENCE MUST COME FROM. Training text only. The cartridge arm
trains on the training windows and is examined on items built from the
held-out ones; a triple sourced from a held-out sentence would be handed the
answer to a question the cartridge had to infer, and the two arms would no
longer be comparable. :func:`gate_triple` takes the training text as an
argument rather than the whole corpus so that this is a requirement rather
than a convention.
"""

from __future__ import annotations

from collections.abc import Sequence

from typing_extensions import TypedDict

from model_trainer.core.contracts.knowledge_edit import (
    SUBJECT_MARKER,
    EditRequest,
)
from model_trainer.core.services.model.corpus_cloze import sentences, terms_in

GATE_CLAUSES: tuple[str, ...] = (
    "sentence_is_a_training_sentence",
    "subject_in_sentence",
    "subject_is_a_corpus_term",
    "object_in_sentence",
    "object_after_subject",
    "prompt_does_not_give_away_object",
    "subject_and_object_are_distinct",
)
"""Every reason a candidate can be rejected, in the order they are checked.

Named and ordered so a reject rate can be reported per clause rather than as
one number. Which clause a corpus fails on is the finding: "no sentence states
both ends" and "the prompt kept leaking the answer" are different problems
with different costs to fix.
"""


class TripleCandidate(TypedDict):
    """One curated association, and the sentence it is claimed to come from.

    Attributes:
        item_id: Stable identifier, unique within a candidate set.
        subject: The entity, exactly as it appears in the source sentence.
        relation: The curator's phrasing that turns the subject into a prompt.
            Free text, checked only for what it must NOT contain.
        object: The answer, exactly as it appears in the source sentence.
        source_document: Which corpus document the sentence was taken from,
            recorded so a reader can find it without searching every page.
        source_sentence: The sentence itself, verbatim.
    """

    item_id: str
    subject: str
    relation: str
    object: str
    source_document: str
    source_sentence: str


class GateVerdict(TypedDict):
    """Whether one candidate passed, and which clauses it failed.

    A LIST OF FAILURES RATHER THAN THE FIRST ONE, because the reject rate is
    the deliverable and a first-failure-only verdict undercounts every clause
    after it. A candidate that fails three clauses is evidence about three.

    Attributes:
        item_id: The candidate this judges.
        accepted: True when ``failed_clauses`` is empty.
        failed_clauses: Names from :data:`GATE_CLAUSES`, in that order.
    """

    item_id: str
    accepted: bool
    failed_clauses: tuple[str, ...]


class GateReport(TypedDict):
    """What a whole candidate set cost, in the terms the arm is judged on.

    Attributes:
        verdicts: One per candidate, in the order given.
        accepted: How many passed every clause.
        rejected: How many failed at least one.
        failures_by_clause: Count per clause name, covering every name in
            :data:`GATE_CLAUSES` including the ones nothing failed -- a zero
            that is present is a measurement, and a missing key is not.
    """

    verdicts: tuple[GateVerdict, ...]
    accepted: int
    rejected: int
    failures_by_clause: tuple[tuple[str, int], ...]


def normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace so a line break cannot fail a match.

    Markdown wraps sentences across lines, and a sentence that spans a wrap
    is the same sentence. Nothing else is normalised: case, punctuation and
    spelling all stay significant, because a triple that gets the subject's
    capitalisation wrong is a triple about a different string as far as a
    tokenizer is concerned.

    Args:
        text: Text to normalise.

    Returns:
        The text with every whitespace run replaced by one space, stripped.
    """
    return " ".join(text.split())


def gate_triple(candidate: TripleCandidate, *, training_sentences: Sequence[str]) -> GateVerdict:
    """Judge one candidate against every clause of the bar.

    Args:
        candidate: The curated triple and its claimed source.
        training_sentences: The sentences of the text the cartridge arm
            trained on, from
            :func:`~model_trainer.core.services.model.corpus_cloze.sentences`.
            MEMBERSHIP IN THIS LIST rather than a substring test of the raw
            text, and the difference is not pedantry: that splitter removes
            fenced code, markdown table rows and URLs, so a sentence it
            returns often does NOT appear verbatim in the text it came from.
            Testing against the raw text rejected six of the first twenty
            curated rows for a reason that had nothing to do with their
            grounding. Membership is also stricter in the way that matters --
            a fragment of a sentence is not a sentence.

    Returns:
        The verdict, naming every clause the candidate failed.
    """
    sentence = normalise_whitespace(candidate["source_sentence"])
    subject = normalise_whitespace(candidate["subject"])
    obj = normalise_whitespace(candidate["object"])
    prompt = normalise_whitespace(f"{subject} {candidate['relation']}")
    subject_at = sentence.find(subject)
    object_at = sentence.find(obj)

    failed: list[str] = []
    if sentence not in {normalise_whitespace(known) for known in training_sentences}:
        failed.append("sentence_is_a_training_sentence")
    # A MISSING END IS REPORTED ONCE. Every clause below that depends on the
    # subject or the object being present is guarded by that presence, because
    # the per-clause counts are the arm's cost line: a subject that is simply
    # absent would otherwise be counted again as "not a corpus term" and again
    # as "out of order", and the report would name three problems where the
    # curator has one.
    if subject_at < 0:
        failed.append("subject_in_sentence")
    elif subject not in terms_in(sentence):
        failed.append("subject_is_a_corpus_term")
    if object_at < 0:
        failed.append("object_in_sentence")
    if subject_at >= 0 and object_at >= 0 and object_at <= subject_at:
        failed.append("object_after_subject")
    if obj in prompt:
        failed.append("prompt_does_not_give_away_object")
    if subject == obj or subject in obj or obj in subject:
        failed.append("subject_and_object_are_distinct")
    return GateVerdict(
        item_id=candidate["item_id"], accepted=not failed, failed_clauses=tuple(failed)
    )


def gate_candidates(candidates: Sequence[TripleCandidate], *, training_text: str) -> GateReport:
    """Judge a whole candidate set and count what it cost.

    Args:
        candidates: The curated triples, in curation order.
        training_text: The text the cartridge arm trained on. Split once here
            rather than once per candidate.

    Returns:
        The report, whose counts are the arm's cost line.
    """
    training_sentences = sentences(training_text)
    verdicts = tuple(
        gate_triple(candidate, training_sentences=training_sentences) for candidate in candidates
    )
    counts = dict.fromkeys(GATE_CLAUSES, 0)
    for verdict in verdicts:
        for clause in verdict["failed_clauses"]:
            counts[clause] += 1
    accepted = sum(1 for verdict in verdicts if verdict["accepted"])
    return GateReport(
        verdicts=verdicts,
        accepted=accepted,
        rejected=len(verdicts) - accepted,
        failures_by_clause=tuple((clause, counts[clause]) for clause in GATE_CLAUSES),
    )


def as_edit_request(candidate: TripleCandidate) -> EditRequest:
    """Turn an accepted candidate into the request the editor consumes.

    ONLY CALL THIS ON AN ACCEPTED CANDIDATE. Nothing here re-checks the gate,
    and that is deliberate: a function that silently re-judged would let a
    caller skip the report and lose the reject rate, which is the arm's actual
    deliverable rather than a side effect of it.

    Args:
        candidate: A candidate whose verdict was ``accepted``.

    Returns:
        The edit request, its prompt carrying exactly one
        :data:`~model_trainer.core.contracts.knowledge_edit.SUBJECT_MARKER`.
    """
    return EditRequest(
        item_id=candidate["item_id"],
        subject=candidate["subject"],
        prompt=f"{SUBJECT_MARKER} {candidate['relation']}",
        target_new=candidate["object"],
    )


__all__ = [
    "GATE_CLAUSES",
    "GateReport",
    "GateVerdict",
    "TripleCandidate",
    "as_edit_request",
    "gate_candidates",
    "gate_triple",
    "normalise_whitespace",
]
