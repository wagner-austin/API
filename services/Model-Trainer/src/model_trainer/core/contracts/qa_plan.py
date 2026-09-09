"""What one question-set measurement DECLARES, as against which ones exist.

SPLIT FROM :mod:`~model_trainer.core.services.model.cartridge_qa_plans` on
2026-09-09, when adding the full-wiki plan pushed that module through the
600-line ceiling. The boundary is the one this package already draws
everywhere else: ``core/contracts`` holds the SHAPES a measurement is
described by, and the service module holds the instances. The table kept
growing -- a scale ladder, a capacity axis, a powered pair, a full-corpus
plan -- while the type it is typed by grew only when the measurement learned
to declare something new.

WHAT THIS TYPE IS FOR, and it is not documentation. Every field here is
either an input a number cannot be reproduced without, or a commitment the
run is checked against before it starts. ``smallest_effect_of_interest``,
``alpha`` and ``mcnemar_test`` are the second kind: they are read by
:func:`~model_trainer.core.services.model.cartridge_qa_power.require_resolvable_question_set`
before any model loads, and a plan that cannot resolve what it declares is
refused rather than run and regretted.
"""

from __future__ import annotations

from platform_core.power_distributions import McNemarTest
from typing_extensions import TypedDict


class QaPlan(TypedDict):
    """One complete, reproducible question-set measurement.

    Attributes:
        model_id: HuggingFace id of the base to measure against.
        window: Tokens per training window.
        held_out_stride: One window in this many is held out. Items are built
            from the held-out windows and the cartridge trains on the rest, so
            this is what keeps the cartridge from being scored on sentences it
            read.
        num_slots: Prefix length for the cartridge arm.
        max_seq_len: Token budget every arm is scored under, INCLUDING the
            evidence the retrieval arm carries.

            Declared rather than read off the model. A model's context window
            is a fact about the model; the budget a measurement spends is a
            choice, and it has to be the same choice in every arm or the arms
            are not comparable. Reading ``config.n_positions`` would also mean
            widening :class:`~model_trainer.core.types.ConfigLike`, which is
            memberless precisely because not every backend has that field.

            For ``gpt2-wiki-qa`` this is 896: gpt2's 1024 positions less the
            128 the cartridge occupies, so the base and retrieval arms are
            held to the same room the cartridge arm actually has.
        seeds: Initialisation seeds; every arm runs once per seed.
        epochs: Passes over the training windows.
        learning_rate: Step size for AdamW.
        distractor_count: Wrong candidates per item. Chance accuracy is
            ``1 / (distractor_count + 1)``.
        max_items: Cap on the question set's size. An UPPER BOUND the corpus
            is free to fall short of, which is why it is not what the power
            gate reads: the 32-item set behind the retracted
            cartridge-beats-retrieval headline came from a plan whose cap said
            120.
        smallest_effect_of_interest: The smallest accuracy difference between
            two arms this measurement exists to resolve, in the units the arms
            report. Declared HERE, before the run, and checked against the
            realised question set by
            :func:`~model_trainer.core.services.model.cartridge_qa_power.require_resolvable_question_set`.

            This field is the whole lesson of 2026-09-09. Every plan below had
            an implicit answer to this question and none of them stated it, so
            a 0.05 difference over 32 items -- four times below what those
            items can resolve -- was published, reached a wiki hub, and had to
            be withdrawn. A plan that cannot say what size of effect it is
            hunting cannot be told it failed to find one.

            A DECLARED VALUE MUST SAY WHETHER IT WAS MEASURED OR CHOSEN, and
            the table's comment does. The second lesson of the same day is
            that a threshold can be wrong while looking derived: this field's
            first value was averaged from real anchors and then described as
            a floor. Erring small is the safe direction -- a too-large value
            licenses a verdict, a too-small one only refuses a run.
        alpha: Two-sided significance level the rejection region is fixed at.
        mcnemar_test: Which McNemar variant the arms are judged under. Carried
            rather than assumed because the exact and mid-p rejection regions
            differ, so a power statement computed against the wrong one
            describes a test nobody ran.
        bm25_k1: BM25 term-frequency saturation for the retrieval arms.
        bm25_b: BM25 length normalisation for the retrieval arms.
        retrieved_chunks: How many chunks the retrieval arms return per
            question, and the cutoff the dense and fused arms take too.
        expansion_feedback_chunks: How many first-pass results the
            query-expansion arm treats as relevant and mines for terms.
        expansion_terms: How many terms that arm adds to the query.
        rerank_candidates: How many BM25 results the reranking arm hands
            to the model to re-score. Must exceed ``retrieved_chunks``,
            or the cutoff does the selecting and the reranker has nothing
            to choose between. It is also the arm's cost: this many
            forward passes per item, where BM25 pays none.

            Declared for the same reason the BM25 knobs are. Expansion has
            no standard setting -- it trades recall for the risk of
            amplifying a bad first search -- so an arm reported without
            them says nothing a reader can reproduce.

            THESE THREE WERE MODULE CONSTANTS in ``cartridge_retrieval`` --
            1.5, 0.75 and 5 -- which is how "the cartridge beats BM25" came
            to be reported of a single arbitrary point in BM25's parameter
            space. BM25 is a family, and all three move the arm the cartridge
            is being compared against. They are declared here so a record
            carries the configuration it was measured under and so the
            retrieval side can be swept rather than assumed. The values below
            are the constants they replaced, so nothing already measured
            moves.
    """

    model_id: str
    window: int
    held_out_stride: int
    num_slots: int
    max_seq_len: int
    seeds: tuple[int, ...]
    epochs: int
    learning_rate: float
    distractor_count: int
    max_items: int
    smallest_effect_of_interest: float
    alpha: float
    mcnemar_test: McNemarTest
    bm25_k1: float
    bm25_b: float
    retrieved_chunks: int
    expansion_feedback_chunks: int
    expansion_terms: int
    rerank_candidates: int


#: Fixed rather than a flag, and distinct from the loss experiment's name.
QA_EXPERIMENT = "cartridge-question-set"

__all__ = [
    "QA_EXPERIMENT",
    "QaPlan",
]
