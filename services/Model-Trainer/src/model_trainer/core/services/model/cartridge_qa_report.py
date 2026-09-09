"""What one question-set run reports: its two identities and its costs.

Split from :mod:`~model_trainer.cli.cartridge_qa_benchmark` by role when that
module passed the 600-line ceiling, the same way ``build_question_set`` was.
The role here is the SHAPE of a run's output -- what a reader receives and
what each number in it means -- as against running the arms that produce it.

Nothing in this module touches a model, a clock or a device. Every function is
arithmetic over durations already measured, which is what lets the cost
accounting be asserted directly rather than inferred from a timed run.
"""

from __future__ import annotations

from platform_core.run_record import Observation
from typing_extensions import TypedDict


class QaMeasurement(TypedDict):
    """One run of every arm, and the two identities that separate it.

    TWO DIGESTS BECAUSE THE RUN HAS TWO INPUTS and only one of them is the
    corpus. The label carries the corpus digest, which answers "was this the
    same text"; the payload carries the question-set digest, which answers
    "were these the same questions". They came apart on this experiment's own
    records -- see :mod:`~model_trainer.core.services.model.cloze.identity` --
    and a single digest cannot say which one moved.

    Attributes:
        observations: Every arm's named numbers, in the order they were
            appended.
        corpus_digest: Digest of the documents that went in, from
            :func:`~model_trainer.core.services.model.cartridge_plans.corpus_digest`.
        question_set_digest: Digest of the items derived from them, from
            :func:`~model_trainer.core.services.model.cloze.identity.question_set_digest`.
    """

    observations: tuple[Observation, ...]
    corpus_digest: str
    question_set_digest: str


def latency_observations(
    *,
    base_seconds: float,
    retrieval_seconds: float,
    cartridge_seconds: float,
    retrieval_build_seconds: float,
    real_seconds: float,
    real_select_seconds: float,
    real_index_seconds: float,
    dense_seconds: float,
    dense_select_seconds: float,
    dense_index_seconds: float,
    fused_seconds: float,
    fused_select_seconds: float,
) -> tuple[Observation, ...]:
    """Name what each arm cost to SERVE, per pass over the question set.

    WHAT IS BEING COMPARED, precisely, because the arms are not symmetric.
    All three run the same scorer over the same items; they differ only in
    what precedes the question -- nothing, retrieved evidence, or a trained
    prefix. So the difference between them is prefill, which is the thing a
    serving comparison is actually about: the retrieval arm re-encodes its
    evidence on every query, and the cartridge arm does not.

    THE ORACLE'S SELECTION IS MEASURED AND THEN EXCLUDED FROM THE COMPARISON,
    rather than quietly left out. ``retrieval_build_seconds`` is the time to
    pick each item's evidence by searching for its own ANSWER -- something no
    real retriever can do, so charging it to retrieval would invent a cost,
    and dropping it silently would hide that a step happened at all. It is
    recorded so a reader can see both the number and the argument.

    WHICH DIRECTION THIS BOUND CUTS. The oracle arm pays no embedding, no
    index search and no ranking, so it is the CHEAPEST any retrieval could
    be. A cartridge that beats it beats a real pipeline by more; a cartridge
    that loses to it has proven nothing about real pipelines. Only the first
    direction is conclusive, and the write-up has to say so.

    Args:
        base_seconds: Scoring the question set with no context added.
        retrieval_seconds: Scoring it with evidence in the prompt.
        cartridge_seconds: Scoring it behind a trained prefix, MEAN over the
            plan's seeds so it is one pass like the other two rather than a
            sum over however many seeds the plan happens to declare.
        retrieval_build_seconds: Assembling the oracle's evidence. Reported,
            not charged.
        real_seconds: Scoring behind BM25-retrieved evidence.
        real_select_seconds: Querying the index. CHARGED, unlike the oracle's
            selection, because searching from the question is work every
            deployment does per request. ``bm25_total_serve_seconds`` is the
            sum, and it is the number to compare against the cartridge.
        real_index_seconds: Building the index. Reported separately and NOT
            in the total: a deployment pays it once when its corpus changes,
            so charging it per query would overstate retrieval exactly as
            charging the oracle's cheating would.
        dense_seconds: Scoring behind embedding-retrieved evidence.
        dense_select_seconds: Embedding the QUERY and ranking pre-embedded
            chunks against it. Charged, for the reason BM25's select is.
        dense_index_seconds: Embedding the corpus. Offline and excluded from
            the total, symmetric with ``real_index_seconds`` -- and the
            asymmetry the first version of this got wrong, by embedding the
            corpus inside every query and recording 17452 ms/item.
        fused_seconds: Scoring behind reciprocal-rank-fused evidence.
        fused_select_seconds: Fusing, plus the lexical ranking fusion needs.
            The dense ranking is an INPUT to it, so the fused total adds
            ``dense_select_seconds`` as well -- a hybrid cannot cost less
            than an arm it is built on.

    Returns:
        The named durations. Every arm's per-request total is present as its
        own name, so a reader compares totals without re-deriving which
        components belong to which arm.
    """
    return tuple(
        Observation(name=name, value=value)
        for name, value in (
            ("base_serve_seconds", base_seconds),
            ("retrieval_serve_seconds", retrieval_seconds),
            ("cartridge_serve_seconds", cartridge_seconds),
            ("retrieval_oracle_build_seconds", retrieval_build_seconds),
            ("bm25_serve_seconds", real_seconds),
            ("bm25_select_seconds", real_select_seconds),
            ("bm25_total_serve_seconds", real_seconds + real_select_seconds),
            ("bm25_index_seconds", real_index_seconds),
            ("dense_serve_seconds", dense_seconds),
            ("dense_select_seconds", dense_select_seconds),
            ("dense_total_serve_seconds", dense_seconds + dense_select_seconds),
            ("dense_index_seconds", dense_index_seconds),
            ("fused_serve_seconds", fused_seconds),
            ("fused_select_seconds", fused_select_seconds),
            (
                "fused_total_serve_seconds",
                fused_seconds + fused_select_seconds + dense_select_seconds,
            ),
        )
    )


__all__ = [
    "QaMeasurement",
    "latency_observations",
]
