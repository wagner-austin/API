"""Score every retrieval arm of a question set against one base model.

WHAT THIS IS AND WHY IT MOVED. Seven arms -- oracle, BM25, dense, fused,
expanded, reranked and long-context -- built and scored over one shared index
and one shared question set, which is what makes their differences readable.
It lived inside :mod:`model_trainer.cli.cartridge_qa_benchmark` beside the
argument parsing, and it is measurement logic: what an arm is, what is timed
apart from what, and which cost a deployment actually pays.

The CLI keeps the orchestration -- corpus, power gate, base model, cartridge
seeds, record assembly -- because that is wiring. This side keeps the arms.

WHY THE ARMS AND NOT THE WHOLE MEASUREMENT. Taking the enclosing function
would have changed its signature at eleven call sites across two test files
and a support module, for no benefit to the arms themselves. Taking the block
that is genuinely a unit leaves every existing caller untouched, which is the
smaller and more reversible half of the same refactor.

THE CLOCK AND THE WAIT ARE PLAIN CALLABLES, matching
:func:`~model_trainer.core.services.model.cloze.score.scored_and_timed` in
this same layer rather than introducing a second name for a port the layer
has already spelled. The embedder factory is
:class:`~model_trainer.core.services.model.cartridge_dense.EmbedderFactoryProto`,
which already lives here. Nothing in this module reaches up into ``cli``, and
no module under ``core`` does -- measured across the tree.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from platform_core.logging import get_logger
from typing_extensions import TypedDict

from model_trainer.core.contracts.cloze import BLANK_MARKER, ClozeEvalResult, ClozeItem
from model_trainer.core.contracts.qa_plan import QaPlan
from model_trainer.core.services.model.backends.hf_lm.encoding import HFTokenizerEncoder
from model_trainer.core.services.model.cartridge_dense import (
    EmbedderFactoryProto,
    dense_ranking,
    embed_chunks,
)
from model_trainer.core.services.model.cartridge_qa_arms import (
    bm25_retrieval_items,
    expanded_retrieval_items,
    long_context_items,
    ranked_retrieval_items,
    reranked_retrieval_items,
    retrieval_items,
)
from model_trainer.core.services.model.cartridge_retrieval import (
    build_index,
    fuse_by_reciprocal_rank,
    rank_chunks,
)
from model_trainer.core.services.model.cloze.score import score_cloze_items, scored_and_timed
from model_trainer.core.types import CacheCapableLMProto

_log = get_logger(__name__)


class RetrievalArms(TypedDict):
    """Every retrieval arm's score, its costs, and what it carried.

    ONE FLAT RECORD RATHER THAN A TUPLE OF TWENTY-ONE VALUES. Seven of these
    are ``ClozeEvalResult`` and thirteen are ``float``, so a positional return
    would be transposable without a type error -- which is the argument
    :class:`~model_trainer.core.services.model.cartridge_qa_report.ArmScores`
    already makes for the same reason one layer up.

    NO ``encode``/``decode`` PAIR, deliberately, and stated rather than
    omitted: this record never leaves the process. It is handed straight to
    the caller that assembles the run record, and the things that DO get
    persisted -- the observations, and each arm's result inside a checkpoint
    -- already own codecs. A pair written here would be uncovered code whose
    only test would be its own round trip.

    Attributes:
        oracle: Evidence selected by knowing the answer -- the upper bound.
        bm25: Lexical retrieval from the question alone.
        dense: Embedding retrieval from the question alone.
        fused: Reciprocal-rank fusion of the lexical and dense rankings.
        expanded: Lexical retrieval after pseudo-relevance feedback.
        reranked: The lexical shortlist re-ordered by the model itself.
        long_context: The corpus handed over whole, with no retrieval.
        long_context_corpus_fraction: Share of the corpus the long-context
            arm actually carried. Recorded beside its accuracy because the
            two cannot be read apart -- an arm carrying 6% of the corpus has
            not tested long context.
        retrieval_build_seconds: Oracle SELECTION, which no real retriever
            can do, so it belongs in the record and not in the comparison.
        retrieval_seconds: Oracle scoring.
        real_index_seconds: Building the BM25 index, paid once per corpus.
        real_select_seconds: Querying it, paid per request.
        real_seconds: Scoring the BM25 arm.
        dense_index_seconds: Embedding the corpus, paid once per corpus.
        dense_select_seconds: Embedding the queries and ranking.
        dense_seconds: Scoring the dense arm.
        fused_select_seconds: Fusing, plus the lexical ranking it still needs.
        fused_seconds: Scoring the fused arm.
        expanded_seconds: Scoring the expanded arm.
        reranked_seconds: Scoring the reranked arm.
        long_context_seconds: Scoring the long-context arm.
    """

    oracle: ClozeEvalResult
    bm25: ClozeEvalResult
    dense: ClozeEvalResult
    fused: ClozeEvalResult
    expanded: ClozeEvalResult
    reranked: ClozeEvalResult
    long_context: ClozeEvalResult
    long_context_corpus_fraction: float
    retrieval_build_seconds: float
    retrieval_seconds: float
    real_index_seconds: float
    real_select_seconds: float
    real_seconds: float
    dense_index_seconds: float
    dense_select_seconds: float
    dense_seconds: float
    fused_select_seconds: float
    fused_seconds: float
    expanded_seconds: float
    reranked_seconds: float
    long_context_seconds: float


def score_retrieval_arms(
    items: Sequence[ClozeItem],
    *,
    plan: QaPlan,
    training_text: str,
    base: CacheCapableLMProto,
    encoder: HFTokenizerEncoder,
    device: str,
    wait: Callable[[], None],
    clock: Callable[[], float],
    make_embedder: EmbedderFactoryProto,
) -> RetrievalArms:
    """Build and score every retrieval arm over one shared index.

    ONE INDEX AND ONE QUESTION SET FOR ALL OF THEM, which is the whole point:
    the arms differ only in HOW they choose, so a difference between them is
    about selection rather than about what they were given.

    Args:
        items: The question set every arm is scored on.
        plan: Supplies the BM25 parameters, the retrieved-chunk count, the
            expansion settings and the rerank shortlist size.
        training_text: The corpus text the index and the long-context arm
            are built from.
        base: The model every arm is scored against.
        encoder: Tokenizer for building evidence windows.
        device: Device to score on.
        wait: Blocks until queued device work has finished, so a clock read
            times the work rather than its queueing.
        clock: Monotonic seconds, read only as a difference.
        make_embedder: Builds the dense arm's encoder, bound to a device.

    Returns:
        Every arm's result, its costs, and the long-context arm's coverage.

    Raises:
        AppError: With ``CLOZE_ITEM_UNSCOREABLE`` when an item cannot carry
            evidence, or ``CARTRIDGE_CORPUS_UNUSABLE`` when the corpus yields
            no tokens for the long-context arm.
    """
    max_seq = plan["max_seq_len"]

    # The oracle's SELECTION is timed apart from the scoring it feeds. It
    # searches each item's own answer, which no real retriever can do, so its
    # cost belongs in the record but not in the comparison.
    started = clock()
    retrieval_set = retrieval_items(items, [training_text], encoder, max_seq_len=max_seq)
    retrieval_build_seconds = clock() - started

    wait()
    started = clock()
    scored_retrieval = score_cloze_items(
        items=retrieval_set,
        model=base,
        encoder=encoder,
        device=device,
        max_seq_len=max_seq,
    )
    wait()
    retrieval_seconds = clock() - started

    # THE REAL ARM. Indexing is timed apart from querying because a
    # deployment pays them at different times -- the index is built once when
    # the corpus changes, the query runs per request. Unlike the oracle's
    # selection, the query time here IS chargeable: searching an index from
    # the question is work every real retriever does.
    started = clock()
    index = build_index(
        [training_text],
        k1=plan["bm25_k1"],
        b=plan["bm25_b"],
        retrieved_chunks=plan["retrieved_chunks"],
    )
    real_index_seconds = clock() - started

    started = clock()
    real_set = bm25_retrieval_items(items, index, encoder, max_seq_len=max_seq)
    real_select_seconds = clock() - started

    wait()
    started = clock()
    scored_real = score_cloze_items(
        items=real_set,
        model=base,
        encoder=encoder,
        device=device,
        max_seq_len=max_seq,
    )
    wait()
    real_seconds = clock() - started
    _log.info("bm25 retrieval %.4f over %d chunks", scored_real["accuracy"], len(index["chunks"]))

    # THE DENSE ARM, and the FUSION of it with BM25. All three rank the same
    # chunks, so they differ only in HOW they choose -- which is the
    # comparison worth making. Queries strip the blank marker for the reason
    # `bm25_retrieval_items` documents.
    queries = [item["template"].replace(BLANK_MARKER, " ") for item in items]

    # OFFLINE, exactly as the BM25 index build is. The first version of this
    # embedded the whole corpus inside every query and recorded 17452 ms/item
    # against BM25's 72 -- real arithmetic over a design nobody deploys.
    # Built ONCE, before any clock starts. A deployment loads its encoder at
    # startup; the first version reloaded gte on every call and left 300 ms
    # of model loading inside each query's measured cost.
    embedder = make_embedder(device)

    started = clock()
    dense_vectors = embed_chunks(index, embedder)
    dense_index_seconds = clock() - started

    started = clock()
    dense_ranks = [dense_ranking(dense_vectors, query, embedder) for query in queries]
    dense_select_seconds = clock() - started

    dense_set = ranked_retrieval_items(
        items,
        index,
        encoder,
        [ranking[: plan["retrieved_chunks"]] for ranking in dense_ranks],
        max_seq_len=max_seq,
    )
    scored_dense, dense_seconds = scored_and_timed(
        dense_set, base, encoder, device=device, max_seq_len=max_seq, wait=wait, clock=clock
    )

    # Fusion re-uses the dense ranking rather than recomputing it, so this
    # times the FUSION plus the lexical ranking it still needs. A deployment
    # pays the dense arm on top; the record carries the numbers separately
    # so a reader can add whichever total they mean.
    started = clock()
    fused_ranks = [
        fuse_by_reciprocal_rank(ranking, rank_chunks(index, query), limit=plan["retrieved_chunks"])
        for ranking, query in zip(dense_ranks, queries, strict=True)
    ]
    fused_select_seconds = clock() - started

    fused_set = ranked_retrieval_items(items, index, encoder, fused_ranks, max_seq_len=max_seq)
    scored_fused, fused_seconds = scored_and_timed(
        fused_set, base, encoder, device=device, max_seq_len=max_seq, wait=wait, clock=clock
    )
    # SEARCHING TWICE, the second time with terms mined from the first pass.
    # Reported beside plain BM25 rather than replacing it: expansion assumes
    # its feedback set is relevant and never checks, so where the first
    # search was wrong it adds the wrong vocabulary and the second search is
    # more confidently wrong. The two arms share one index and one question
    # set, which is what makes the difference readable.
    expanded_set = expanded_retrieval_items(
        items,
        index,
        encoder,
        max_seq_len=max_seq,
        feedback_chunks=plan["expansion_feedback_chunks"],
        expansion_terms=plan["expansion_terms"],
    )
    scored_expanded, expanded_seconds = scored_and_timed(
        expanded_set, base, encoder, device=device, max_seq_len=max_seq, wait=wait, clock=clock
    )
    _log.info(
        "expanded %.4f against bm25 %.4f",
        scored_expanded["accuracy"],
        scored_real["accuracy"],
    )

    # THE HALF OF A REAL PIPELINE THIS AXIS HAS BEEN MISSING. A deployment
    # over-retrieves cheaply and reranks the shortlist with something that
    # reads; comparing a cartridge against unranked BM25 compares it against
    # a system nobody ships. Costed separately because the cost is the trade:
    # rerank_candidates forward passes per item, where BM25 pays none.
    reranked_set = reranked_retrieval_items(
        items,
        index,
        encoder,
        base,
        device=device,
        max_seq_len=max_seq,
        candidates=plan["rerank_candidates"],
    )
    scored_reranked, reranked_seconds = scored_and_timed(
        reranked_set, base, encoder, device=device, max_seq_len=max_seq, wait=wait, clock=clock
    )
    _log.info(
        "reranked %.4f against bm25 %.4f",
        scored_reranked["accuracy"],
        scored_real["accuracy"],
    )

    # THE ARM THAT SKIPS RETRIEVAL ENTIRELY, and the one a reader assumes
    # has already been run. Everything above searches; this simply hands the
    # model the corpus and lets the window truncate it. Its coverage is
    # recorded beside its accuracy because the two cannot be read apart: an
    # arm carrying 6% of the corpus has not tested long context, and a
    # cartridge beating it has not beaten reading.
    long_set, long_context_fraction = long_context_items(
        items, training_text, encoder, max_seq_len=max_seq
    )
    scored_long_context, long_context_seconds = scored_and_timed(
        long_set, base, encoder, device=device, max_seq_len=max_seq, wait=wait, clock=clock
    )
    _log.info(
        "long-context %.4f over %.4f of the corpus",
        scored_long_context["accuracy"],
        long_context_fraction,
    )

    _log.info("dense %.4f, fused %.4f", scored_dense["accuracy"], scored_fused["accuracy"])

    return RetrievalArms(
        oracle=scored_retrieval,
        bm25=scored_real,
        dense=scored_dense,
        fused=scored_fused,
        expanded=scored_expanded,
        reranked=scored_reranked,
        long_context=scored_long_context,
        long_context_corpus_fraction=long_context_fraction,
        retrieval_build_seconds=retrieval_build_seconds,
        retrieval_seconds=retrieval_seconds,
        real_index_seconds=real_index_seconds,
        real_select_seconds=real_select_seconds,
        real_seconds=real_seconds,
        dense_index_seconds=dense_index_seconds,
        dense_select_seconds=dense_select_seconds,
        dense_seconds=dense_seconds,
        fused_select_seconds=fused_select_seconds,
        fused_seconds=fused_seconds,
        expanded_seconds=expanded_seconds,
        reranked_seconds=reranked_seconds,
        long_context_seconds=long_context_seconds,
    )


__all__ = ["RetrievalArms", "score_retrieval_arms"]
