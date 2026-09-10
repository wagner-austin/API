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
from pathlib import Path

from platform_core.logging import get_logger
from typing_extensions import TypedDict

from model_trainer.core.contracts.cloze import BLANK_MARKER, ClozeEvalResult, ClozeItem
from model_trainer.core.contracts.qa_checkpoint import ArmRecord, QaCheckpoint, with_arm
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
from model_trainer.core.services.model.cartridge_qa_checkpoint import save_qa_checkpoint
from model_trainer.core.services.model.cartridge_retrieval import (
    build_index,
    fuse_by_reciprocal_rank,
    rank_chunks,
)
from model_trainer.core.services.model.cloze.score import scored_and_timed
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


def _arm_or_resume(
    checkpoint: QaCheckpoint,
    directory: Path,
    name: str,
    build: Callable[[], tuple[Sequence[ClozeItem], float, float]],
    *,
    base: CacheCapableLMProto,
    encoder: HFTokenizerEncoder,
    device: str,
    max_seq: int,
    wait: Callable[[], None],
    clock: Callable[[], float],
    index_seconds: float,
) -> tuple[QaCheckpoint, ArmRecord]:
    """Score one arm, or hand back the result an earlier run already got.

    THE EVIDENCE SET IS BUILT LAZILY, which is the reason ``build`` is a
    callable rather than a value. Selecting evidence costs real time --
    retrieval runs per item -- and a resumed arm must not pay it again to
    produce a set nothing will score. Passing the built set would have made
    the skip cosmetic.

    SAVED IMMEDIATELY AFTER SCORING, not at the end of the run. The file is
    the only thing an eviction cannot take, so the window between finishing an
    arm and recording it is exactly the work at risk, and it should be as
    short as the code can make it.

    Args:
        checkpoint: Work completed so far.
        directory: Where the checkpoint file lives.
        name: This arm's name, matching its field in :class:`RetrievalArms`.
        build: Selects this arm's evidence, returning the item set, the
            seconds that selection took, and the share of the corpus the
            evidence carried -- zero for the arms that do not report one.
            The FRACTION comes back from the build rather than being passed
            in because only the long-context arm knows it, and it knows it
            only once it has selected.
        base: The model to score against.
        encoder: Tokenizer for evidence windows.
        device: Device to score on.
        max_seq: Evidence window size.
        wait: Blocks until queued device work has finished.
        clock: Monotonic seconds.
        index_seconds: Cost of the index this arm queries, or zero when it
            builds none.

    Returns:
        The checkpoint including this arm, and the arm's record.
    """
    for record in checkpoint["arms"]:
        if record["arm"] == name:
            _log.info(
                "resuming %s from checkpoint, accuracy %.4f",
                name,
                record["result"]["accuracy"],
            )
            return checkpoint, record
    item_set, select_seconds, evidence_fraction = build()
    result, score_seconds = scored_and_timed(
        item_set, base, encoder, device=device, max_seq_len=max_seq, wait=wait, clock=clock
    )
    scored = ArmRecord(
        arm=name,
        result=result,
        select_seconds=select_seconds,
        score_seconds=score_seconds,
        index_seconds=index_seconds,
        evidence_fraction=evidence_fraction,
    )
    recorded = with_arm(checkpoint, scored)
    save_qa_checkpoint(directory, recorded)
    return recorded, scored


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
    checkpoint: QaCheckpoint,
    checkpoints: Path,
) -> tuple[QaCheckpoint, RetrievalArms]:
    """Build and score every retrieval arm over one shared index.

    ONE INDEX AND ONE QUESTION SET FOR ALL OF THEM, which is the whole point:
    the arms differ only in HOW they choose, so a difference between them is
    about selection rather than about what they were given.

    RESUMES AT THE ARM BOUNDARY. Each arm's scoring is skipped when the
    checkpoint already holds it, and the checkpoint is written the moment an
    arm finishes. What is NOT skipped is the shared structure the later arms
    need -- the BM25 index and the dense vectors and rankings feed the fused,
    expanded and reranked arms, so a resumed run rebuilds them and skips only
    the scoring. That is the right side of the trade: the index costs seconds
    and embedding the corpus costs minutes, while scoring one arm over a
    corpus-scale question set costs twenty of them.

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
        checkpoint: Work an earlier execution of this measurement completed.
        checkpoints: Directory the checkpoint file lives in.

    Returns:
        The checkpoint including every arm, and the arms themselves.

    Raises:
        AppError: With ``CLOZE_ITEM_UNSCOREABLE`` when an item cannot carry
            evidence, or ``CARTRIDGE_CORPUS_UNUSABLE`` when the corpus yields
            no tokens for the long-context arm.
    """
    max_seq = plan["max_seq_len"]

    def _select_oracle() -> tuple[Sequence[ClozeItem], float, float]:
        """Select each item's own answer, which no real retriever can do.

        Returns:
            The evidence set and the seconds selecting it took.
        """
        started = clock()
        built = retrieval_items(items, [training_text], encoder, max_seq_len=max_seq)
        return built, clock() - started, 0.0

    # The oracle's SELECTION is timed apart from the scoring it feeds, so its
    # cost belongs in the record but not in the comparison.
    checkpoint, oracle = _arm_or_resume(
        checkpoint,
        checkpoints,
        "oracle",
        _select_oracle,
        base=base,
        encoder=encoder,
        device=device,
        max_seq=max_seq,
        wait=wait,
        clock=clock,
        index_seconds=0.0,
    )

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

    def _select_bm25() -> tuple[Sequence[ClozeItem], float, float]:
        """Search the index with the question's own terms.

        Returns:
            The evidence set and the seconds selecting it took.
        """
        started_at = clock()
        built = bm25_retrieval_items(items, index, encoder, max_seq_len=max_seq)
        return built, clock() - started_at, 0.0

    checkpoint, bm25 = _arm_or_resume(
        checkpoint,
        checkpoints,
        "bm25",
        _select_bm25,
        base=base,
        encoder=encoder,
        device=device,
        max_seq=max_seq,
        wait=wait,
        clock=clock,
        index_seconds=real_index_seconds,
    )
    _log.info(
        "bm25 retrieval %.4f over %d chunks", bm25["result"]["accuracy"], len(index["chunks"])
    )

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

    def _select_dense() -> tuple[Sequence[ClozeItem], float, float]:
        """Take the top chunks by meaning.

        The ranking itself is timed above as ``dense_select_seconds`` and is
        SHARED with the fused arm, so it is not folded in here: it is
        recomputed on every run because the fused arm needs the rankings
        whether or not the dense arm resumed, which makes the reported figure
        a real measurement of this execution either way. What this reports is
        the windowing alone -- which this arm does NOT report, so it is not
        timed. Timing it would be a change of measurement smuggled into a
        refactor: the reported `dense_select_seconds` has always been the
        ranking, and adding the windowing to it would move a published number
        for a reason no reader could see.

        Returns:
            The evidence set, zero seconds, and zero fraction.
        """
        built = ranked_retrieval_items(
            items,
            index,
            encoder,
            [ranking[: plan["retrieved_chunks"]] for ranking in dense_ranks],
            max_seq_len=max_seq,
        )
        return built, 0.0, 0.0

    checkpoint, dense = _arm_or_resume(
        checkpoint,
        checkpoints,
        "dense",
        _select_dense,
        base=base,
        encoder=encoder,
        device=device,
        max_seq=max_seq,
        wait=wait,
        clock=clock,
        index_seconds=dense_index_seconds,
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

    def _select_fused() -> tuple[Sequence[ClozeItem], float, float]:
        """Window the fused ranking into evidence.

        Untimed for the reason the dense build gives: `fused_select_seconds`
        is the fusion, measured above, and this arm reports no second figure.

        Returns:
            The evidence set, zero seconds, and zero fraction.
        """
        built = ranked_retrieval_items(items, index, encoder, fused_ranks, max_seq_len=max_seq)
        return built, 0.0, 0.0

    checkpoint, fused = _arm_or_resume(
        checkpoint,
        checkpoints,
        "fused",
        _select_fused,
        base=base,
        encoder=encoder,
        device=device,
        max_seq=max_seq,
        wait=wait,
        clock=clock,
        index_seconds=0.0,
    )

    # SEARCHING TWICE, the second time with terms mined from the first pass.
    # Reported beside plain BM25 rather than replacing it: expansion assumes
    # its feedback set is relevant and never checks, so where the first
    # search was wrong it adds the wrong vocabulary and the second search is
    # more confidently wrong. The two arms share one index and one question
    # set, which is what makes the difference readable.
    def _select_expanded() -> tuple[Sequence[ClozeItem], float, float]:
        """Search twice, the second time with terms mined from the first.

        Untimed, as it has always been: this arm reports only a scoring cost.

        Returns:
            The evidence set, zero seconds, and zero fraction.
        """
        built = expanded_retrieval_items(
            items,
            index,
            encoder,
            max_seq_len=max_seq,
            feedback_chunks=plan["expansion_feedback_chunks"],
            expansion_terms=plan["expansion_terms"],
        )
        return built, 0.0, 0.0

    checkpoint, expanded = _arm_or_resume(
        checkpoint,
        checkpoints,
        "expanded",
        _select_expanded,
        base=base,
        encoder=encoder,
        device=device,
        max_seq=max_seq,
        wait=wait,
        clock=clock,
        index_seconds=0.0,
    )
    _log.info(
        "expanded %.4f against bm25 %.4f",
        expanded["result"]["accuracy"],
        bm25["result"]["accuracy"],
    )

    # THE HALF OF A REAL PIPELINE THIS AXIS HAS BEEN MISSING. A deployment
    # over-retrieves cheaply and reranks the shortlist with something that
    # reads; comparing a cartridge against unranked BM25 compares it against
    # a system nobody ships. Costed separately because the cost is the trade:
    # rerank_candidates forward passes per item, where BM25 pays none.
    def _select_reranked() -> tuple[Sequence[ClozeItem], float, float]:
        """Re-order the lexical shortlist with the model itself.

        UNTIMED, AND THAT IS A PRE-EXISTING GAP RATHER THAN A CHOICE MADE
        HERE. This arm's selection is `rerank_candidates` forward passes per
        item -- the expensive half, and the trade the arm exists to measure --
        and the reported `reranked_seconds` has never included it. Timing it
        would change a published number inside a refactor, so it is recorded
        as a gap and left for a commit that is about it.

        Returns:
            The evidence set, zero seconds, and zero fraction.
        """
        built = reranked_retrieval_items(
            items,
            index,
            encoder,
            base,
            device=device,
            max_seq_len=max_seq,
            candidates=plan["rerank_candidates"],
        )
        return built, 0.0, 0.0

    checkpoint, reranked = _arm_or_resume(
        checkpoint,
        checkpoints,
        "reranked",
        _select_reranked,
        base=base,
        encoder=encoder,
        device=device,
        max_seq=max_seq,
        wait=wait,
        clock=clock,
        index_seconds=0.0,
    )
    _log.info(
        "reranked %.4f against bm25 %.4f",
        reranked["result"]["accuracy"],
        bm25["result"]["accuracy"],
    )

    # THE ARM THAT SKIPS RETRIEVAL ENTIRELY, and the one a reader assumes
    # has already been run. Everything above searches; this simply hands the
    # model the corpus and lets the window truncate it. Its coverage is
    # recorded beside its accuracy because the two cannot be read apart: an
    # arm carrying 6% of the corpus has not tested long context, and a
    # cartridge beating it has not beaten reading.
    def _select_long_context() -> tuple[Sequence[ClozeItem], float, float]:
        """Hand the model the corpus and let the window truncate it.

        THE ONLY ARM THAT REPORTS A FRACTION, which is why the build returns
        one at all. Coverage cannot be read apart from accuracy here: an arm
        carrying 6% of the corpus has not tested long context, and a cartridge
        beating it has not beaten reading.

        Returns:
            The evidence set, zero seconds -- this arm reports no selection
            cost and never has -- and the share of the corpus it carried.
        """
        built, fraction = long_context_items(items, training_text, encoder, max_seq_len=max_seq)
        return built, 0.0, fraction

    checkpoint, long_context = _arm_or_resume(
        checkpoint,
        checkpoints,
        "long_context",
        _select_long_context,
        base=base,
        encoder=encoder,
        device=device,
        max_seq=max_seq,
        wait=wait,
        clock=clock,
        index_seconds=0.0,
    )
    _log.info(
        "long-context %.4f over %.4f of the corpus",
        long_context["result"]["accuracy"],
        long_context["evidence_fraction"],
    )

    _log.info("dense %.4f, fused %.4f", dense["result"]["accuracy"], fused["result"]["accuracy"])

    # EVERY PER-ARM FIGURE IS READ OUT OF ITS RECORD, so a resumed arm
    # reports the seconds that were actually measured for it rather than a
    # zero or a re-timing of work that did not happen. The two figures NOT
    # read from a record -- `dense_select_seconds` and `fused_select_seconds`
    # -- are the rankings, which every run recomputes because the fused arm
    # needs them regardless, so a fresh measurement is the honest one there.
    return checkpoint, RetrievalArms(
        oracle=oracle["result"],
        bm25=bm25["result"],
        dense=dense["result"],
        fused=fused["result"],
        expanded=expanded["result"],
        reranked=reranked["result"],
        long_context=long_context["result"],
        long_context_corpus_fraction=long_context["evidence_fraction"],
        retrieval_build_seconds=oracle["select_seconds"],
        retrieval_seconds=oracle["score_seconds"],
        real_index_seconds=bm25["index_seconds"],
        real_select_seconds=bm25["select_seconds"],
        real_seconds=bm25["score_seconds"],
        dense_index_seconds=dense["index_seconds"],
        dense_select_seconds=dense_select_seconds,
        dense_seconds=dense["score_seconds"],
        fused_select_seconds=fused_select_seconds,
        fused_seconds=fused["score_seconds"],
        expanded_seconds=expanded["score_seconds"],
        reranked_seconds=reranked["score_seconds"],
        long_context_seconds=long_context["score_seconds"],
    )


__all__ = ["RetrievalArms", "score_retrieval_arms"]
