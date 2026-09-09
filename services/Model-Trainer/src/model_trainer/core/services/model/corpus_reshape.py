"""Reshape a wiki corpus into the prose its own question set is built from.

THE MISMATCH THIS EXISTS TO TEST, and nobody chose it. The item builder
already refuses to make questions out of raw markdown:
:func:`~model_trainer.core.services.model.corpus_cloze.sentences` strips
fenced code, markdown table rows and bare URLs before splitting, because an
item blanking a token out of a URL asks the model to recall a path rather
than a fact. But the CARTRIDGE trains on the raw document, scaffolding and
all. So the arm reads one representation and is examined on another, and that
asymmetry arrived by accident rather than by decision.

Measured on the twelve public me-wiki pages: the scaffolding is **22.5% of the
corpus** -- 3,510 of 15,602 gpt2 tokens. Nine of 115 corpus terms exist only
inside it (``Model-Trainer``, ``DTypeLike``, ``hsp2026``, ``tu11``, ``v2`` and
four more), which is the ghost-term class in its original habitat.

WHY THIS IS THE ARM WORTH TESTING, and it is a prediction rather than a hope.
A cartridge has a FIXED budget -- 128 prefix slots, whatever the corpus costs
-- while a retriever's index is unbounded and pays for scaffolding only in
storage. If representation matters anywhere, it should matter most to the arm
whose capacity is the binding constraint. That makes the comparison a real
test rather than a fishing expedition: the two arms are predicted to respond
DIFFERENTLY, and a result where both move together is evidence against the
prediction rather than a null.

WHAT THIS IS NOT. It is not the automatic triple extraction AKEW measured in
the single digits, and the distinction matters because the last arm died on
it. That extraction had to CREATE a subject-relation-object structure the
prose did not contain, and creating structure is where extraction goes wrong.
This only DELETES -- every surviving sentence is a sentence of the source,
byte for byte, and :func:`gate_reshape` proves it rather than assuming it. A
rule that can only remove cannot invent a fact.
"""

from __future__ import annotations

from collections.abc import Sequence

from typing_extensions import TypedDict

from model_trainer.core.services.model.corpus_cloze import sentences, terms_in

RESHAPE_CLAUSES: tuple[str, ...] = (
    "every_sentence_came_from_the_source",
    "no_term_was_invented",
    "document_is_not_empty",
)
"""Every way a reshaped document can fail its source, in check order.

Three clauses and no more, because the rule only deletes. A transformation
that rewrote text would need clauses about meaning; this one needs only that
what survived was already there.
"""


class ReshapedDocument(TypedDict):
    """One document before and after, with what the reshaping cost it.

    Attributes:
        source: The document as the corpus reader returned it.
        reshaped: Its sentences, rejoined with single spaces.
        failed_clauses: Names from :data:`RESHAPE_CLAUSES`, in that order.
            Empty when the reshape is faithful.
        terms_lost: Corpus terms present in the source and absent after,
            sorted. These are the facts the reshaped corpus can no longer
            teach, and they are the price of the manipulation.
    """

    source: str
    reshaped: str
    failed_clauses: tuple[str, ...]
    terms_lost: tuple[str, ...]


class ReshapeReport(TypedDict):
    """What reshaping a whole corpus produced and cost.

    Attributes:
        documents: One entry per input document, in order.
        accepted: How many passed every clause.
        rejected: How many failed at least one.
        terms_lost: Terms lost across the corpus as a whole, sorted. NOT the
            sum of the per-document losses: a term dropped from one page's
            table may survive in another page's prose, and a corpus-level
            count is what a question set is built against.
    """

    documents: tuple[ReshapedDocument, ...]
    accepted: int
    rejected: int
    terms_lost: tuple[str, ...]


def reshape_document(source: str) -> str:
    """Reduce one document to the prose its question set is built from.

    Args:
        source: The document body, as the corpus reader returns it.

    Returns:
        Its sentences joined by single spaces. Empty when the document
        carries no prose at all, which is a fact about the document and is
        reported by the gate rather than raised here.
    """
    return " ".join(sentences(source))


def gate_reshape(source: str, reshaped: str) -> ReshapedDocument:
    """Judge one reshaped document against the source it claims to preserve.

    Args:
        source: The original document body.
        reshaped: What :func:`reshape_document` produced from it.

    Returns:
        The pair, the clauses it failed, and the terms it cost.
    """
    known = set(sentences(source))
    failed: list[str] = []
    if any(sentence not in known for sentence in sentences(reshaped)):
        failed.append("every_sentence_came_from_the_source")
    source_terms = terms_in(source)
    if terms_in(reshaped) - source_terms:
        failed.append("no_term_was_invented")
    if not reshaped.strip():
        failed.append("document_is_not_empty")
    return ReshapedDocument(
        source=source,
        reshaped=reshaped,
        failed_clauses=tuple(failed),
        terms_lost=tuple(sorted(source_terms - terms_in(reshaped))),
    )


def reshape_corpus(documents: Sequence[str]) -> ReshapeReport:
    """Reshape every document and report what the corpus lost.

    Args:
        documents: The corpus, in the order the reader returned it. That
            order is load-bearing downstream -- it decides which windows the
            stride holds out -- so it is preserved rather than sorted again.

    Returns:
        The reshaped corpus and its cost line.
    """
    judged = tuple(gate_reshape(document, reshape_document(document)) for document in documents)
    before: set[str] = set()
    after: set[str] = set()
    for entry in judged:
        before |= terms_in(entry["source"])
        after |= terms_in(entry["reshaped"])
    accepted = sum(1 for entry in judged if not entry["failed_clauses"])
    return ReshapeReport(
        documents=judged,
        accepted=accepted,
        rejected=len(judged) - accepted,
        terms_lost=tuple(sorted(before - after)),
    )


__all__ = [
    "RESHAPE_CLAUSES",
    "ReshapeReport",
    "ReshapedDocument",
    "gate_reshape",
    "reshape_corpus",
    "reshape_document",
]
