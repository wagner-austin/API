"""Which text each arm is given, and nothing about how it is scored.

SPLIT FROM :mod:`cartridge_qa` ON 2026-09-09, when adding the long-context
arm pushed that module through the 600-line ceiling. The boundary is the one
this experiment's TESTS had already drawn a fortnight earlier --
``test_cartridge_qa_arms.py`` describes itself as being about "WHICH text
each arm sees", against a sibling covering the record the run emits -- so the
source is now split where the suite already was, rather than at whatever line
the ceiling happened to fall on.

What is left behind is the machinery every arm shares: fitting evidence into
an item's remaining window, and scoring an answer once it is there. What
moved here is the CHOICE of evidence, which is the only thing that
distinguishes one arm from another. An oracle, a lexical retriever, a dense
one and a reader handed the whole corpus differ in nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for

from model_trainer.core.contracts.cloze import BLANK_MARKER, ClozeItem
from model_trainer.core.encoding import Encoder
from model_trainer.core.services.model.cartridge_qa import (
    evidence_budget_tokens,
    evidence_for,
    with_evidence,
)
from model_trainer.core.services.model.cartridge_retrieval import (
    Bm25Index,
    join_chunks,
    retrieve,
)


def retrieval_items(
    items: Sequence[ClozeItem],
    documents: Sequence[str],
    encoder: Encoder,
    *,
    max_seq_len: int,
) -> list[ClozeItem]:
    """Build the retrieval arm's item set.

    Args:
        items: The shared question set.
        documents: Training documents the evidence is drawn from.
        encoder: Tokenizer the scorer will use.
        max_seq_len: The scorer's token budget.

    Returns:
        One item per input, each carrying whatever evidence fits.
    """
    return [
        with_evidence(
            item, evidence_for(item["answer"], documents), encoder, max_seq_len=max_seq_len
        )
        for item in items
    ]


def long_context_items(
    items: Sequence[ClozeItem],
    training_text: str,
    encoder: Encoder,
    *,
    max_seq_len: int,
) -> tuple[list[ClozeItem], float]:
    """Build the arm that skips retrieval and just puts the corpus in the window.

    THE COMPETITOR THIS PROGRAMME HAD NEVER RUN. A cartridge exists to
    compress a corpus into a fixed prefix, and every verdict this benchmark
    has produced compares it against RETRIEVERS -- which answers "is a
    cartridge better than searching" and leaves "is a cartridge better than
    simply reading" unasked. That second question is the one a reader assumes
    has been settled.

    IT IS ONLY A FAIR TEST WHERE THE CORPUS FITS, which is why the returned
    fraction is not optional. :func:`with_evidence` keeps the opening of the
    evidence and drops the rest, so where the corpus overflows the window this
    arm is not "the corpus in context" but "the first few per cent of the
    corpus, chosen by document order". At a fraction near 1 it is the honest
    long-context baseline; at 0.06 it is a statement about where the answer
    happened to sit. Reporting the number is what stops the two being read as
    the same arm -- and a cartridge beating a 6%-coverage arm has beaten
    almost nothing.

    Args:
        items: The shared question set.
        training_text: The text the cartridge trained on, whole and
            unretrieved, so both arms are given the same corpus.
        encoder: Tokenizer the scorer will use.
        max_seq_len: The scorer's token budget.

    Returns:
        ``(items, fraction)`` -- one item per input carrying as much of the
        corpus as fits, and the mean share of the corpus's tokens they
        carried. A plain pair rather than a record, for the reason
        :func:`~model_trainer.core.services.model.cartridge_question_set.build_question_set`
        gives: the two have different types, so nothing can transpose them.

    Raises:
        AppError: With ``CARTRIDGE_CORPUS_UNUSABLE`` when the training text
            holds no tokens to carry, or ``CLOZE_ITEM_UNSCOREABLE`` via
            :func:`with_evidence` when an item leaves no room for evidence.
    """
    corpus_tokens = len(encoder.encode(training_text).ids)
    if corpus_tokens == 0:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE,
            (
                "the training text encodes to zero tokens, so the long-context arm has "
                "nothing to put in the window; the arm cannot report a coverage fraction "
                "over an empty corpus"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE),
        )
    built = [with_evidence(item, training_text, encoder, max_seq_len=max_seq_len) for item in items]
    carried = [
        min(evidence_budget_tokens(item, encoder, max_seq_len=max_seq_len), corpus_tokens)
        for item in items
    ]
    return built, sum(carried) / (len(carried) * corpus_tokens)


def bm25_retrieval_items(
    items: Sequence[ClozeItem],
    index: Bm25Index,
    encoder: Encoder,
    *,
    max_seq_len: int,
) -> list[ClozeItem]:
    """Build the REAL retrieval arm's item set, from the questions alone.

    The counterpart to :func:`retrieval_items`. That one searches each item's
    own answer and bounds what retrieval could ever do; this one searches the
    question, gets some of them wrong, and is therefore the arm a cartridge
    can lose to informatively.

    THE BLANK MARKER IS REMOVED BEFORE QUERYING, and not because it currently
    matters. ``<<BLANK>>`` yields the term ``blank``, which scores nothing
    while no corpus sentence happens to contain that word -- and silently
    starts retrieving on it the day one does. The marker is a rendering
    artifact of how the item is posed, not part of what was asked.

    Args:
        items: The shared question set.
        index: A BM25 index over the same training documents the oracle arm
            draws its evidence from.
        encoder: Tokenizer the scorer will use.
        max_seq_len: The scorer's token budget.

    Returns:
        One item per input, each carrying what the retriever chose for it.

    Raises:
        AppError: With ``CLOZE_ITEM_UNSCOREABLE`` via :func:`with_evidence`
            when an item leaves no room for evidence. Not caught: a window
            too small for the plan is a misconfiguration, and the real arm
            refuses for the same reason the oracle arm does.
    """
    return [
        with_evidence(
            item,
            retrieve(index, item["template"].replace(BLANK_MARKER, " ")),
            encoder,
            max_seq_len=max_seq_len,
        )
        for item in items
    ]


def ranked_retrieval_items(
    items: Sequence[ClozeItem],
    index: Bm25Index,
    encoder: Encoder,
    rankings: Sequence[Sequence[int]],
    *,
    max_seq_len: int,
) -> list[ClozeItem]:
    """Build an arm's item set from a pre-computed ranking per item.

    Takes the RANKING rather than computing it, so the caller can time the
    retrieval separately from the item assembly and can feed the same shape
    from a dense arm, a fused arm, or anything else that orders chunks. The
    BM25 arm has its own entry point because it also owns its ranking.

    Args:
        items: The shared question set.
        index: The index the rankings refer to.
        encoder: Tokenizer the scorer will use.
        rankings: Per item, chunk indices best first, already truncated to
            however many the arm retrieves.
        max_seq_len: The scorer's token budget.

    Returns:
        One item per input, carrying that item's chosen chunks.

    Raises:
        AppError: With ``CLOZE_ITEM_UNSCOREABLE`` via :func:`with_evidence`
            when an item leaves no room for evidence.
    """
    return [
        with_evidence(item, join_chunks(index, chosen), encoder, max_seq_len=max_seq_len)
        for item, chosen in zip(items, rankings, strict=True)
    ]


__all__ = [
    "bm25_retrieval_items",
    "long_context_items",
    "ranked_retrieval_items",
    "retrieval_items",
]
