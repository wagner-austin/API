"""Which question-set measurements exist, and what identifies each one's numbers.

A SEPARATE PLAN FROM :mod:`cartridge_plans`, DELIBERATELY. That table's plans
sweep slot counts and compose cartridges, and are identified by a label naming
those fields. This one runs a fixed cartridge against a question set and is
identified by the question set's shape -- how many distractors, how many items.
Folding both into one TypedDict would give every plan fields the other half
does not use, and a label that names them anyway.

The two also answer different questions and must not be differenced: a loss
plan reports how surprising held-out prose is, and this reports whether the
model can use what the prose said. :data:`QA_EXPERIMENT` differs from
``CARTRIDGE_EXPERIMENT`` for exactly that reason -- the comparability layer
refuses to subtract records from different experiments, which is the behaviour
wanted here.

THERE IS NO ``require_qa_plan`` HERE, DELIBERATELY.
:func:`~cartridge_plans.require_cartridge_plan` is generic over the plan type,
so a second lookup beside it would be six lines that differ only in the noun
in their error message -- and would be the copy that stops matching when the
first one learns something.

WHAT MOVED THE ANSWER MOST, and why ``distractor_count`` is in the label. The
multiple-choice arms turned out to be dominated by which wrong candidates were
offered: measured on gpt2 over 24 items, a set built with one repeated
distractor triple put the base model at chance (0.2500) and the cartridge at
0.5417; rotating distractors per item put the base at 0.5417 and the cartridge
at 0.5833. Same corpus, same items, same models. Any field that can do that
belongs in the identity of the measurement.
"""

from __future__ import annotations

from typing import Final

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
        alpha: Two-sided significance level the rejection region is fixed at.
        mcnemar_test: Which McNemar variant the arms are judged under. Carried
            rather than assumed because the exact and mid-p rejection regions
            differ, so a power statement computed against the wrong one
            describes a test nobody ran.
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


#: Fixed rather than a flag, and distinct from the loss experiment's name.
QA_EXPERIMENT = "cartridge-question-set"


#: The plans. ``gpt2-wiki-qa`` mirrors ``gpt2-wiki``'s corpus, window, split
#: and schedule so the two measurements describe the same cartridge, and adds
#: only what a question set needs.
#:
#: WHY EVERY PLAN DECLARES 0.05, AND WHY THAT NUMBER IS NOT A PREFERENCE. It
#: is the size of effect this literature actually reports for interventions of
#: this kind, so it is the smallest difference worth building an instrument
#: for:
#:
#:   WRAP, LLM rephrasing of C4 into four styles, 1:1 synthetic mix   +0.020
#:     (Maini et al., arXiv 2401.16380, 13 zero-shot QA benchmarks)
#:   this machine's extraction ablation, 7:1 OSCAR dilution removed   +0.061
#:   this machine's extraction ablation, sentence-permuted copies     +0.029
#:   this machine's extraction ablation, hub-slug markers             +0.004  (noise)
#:
#: A plan hunting something smaller than 0.05 needs a bigger corpus, and a
#: plan claiming something larger should say so rather than inherit this.
#:
#: WHAT IT COSTS, STATED PLAINLY BECAUSE IT IS THE POINT. At alpha 0.05 under
#: mid-p the fewest disagreements that can ever reject is 5, so resolving 0.05
#: needs at least 100 items. THE FOUR ME-WIKI PLANS BELOW YIELD ABOUT 32 AND
#: ARE THEREFORE REFUSED BY
#: :func:`~model_trainer.core.services.model.cartridge_qa_power.require_resolvable_question_set`
#: BEFORE THEY RUN. That is not a regression; it is the whole change. Those
#: plans produced the cartridge-beats-retrieval headline that was retracted on
#: 2026-09-09 for resting on 1.3 to 1.7 items against a 5-item floor, and they
#: will keep being refused until the corpus they read is large enough for the
#: question they are asked.
QA_PLANS: Final[dict[str, QaPlan]] = {
    "gpt2-wiki-qa": {
        "model_id": "gpt2",
        "window": 256,
        "held_out_stride": 4,
        "num_slots": 128,
        "max_seq_len": 896,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
        "distractor_count": 3,
        "max_items": 120,
        "smallest_effect_of_interest": 0.05,
        "alpha": 0.05,
        "mcnemar_test": McNemarTest.MID_P,
    },
    # THE SCALE LADDER. Every field except `model_id` is copied from
    # `gpt2-wiki-qa` deliberately: the 2026-09-07 verdict -- that a cartridge
    # loses to BM25 on both accuracy and latency -- was measured at 124M and
    # generalised to "bigger will not rescue it" on the strength of ANOTHER
    # session's unfinished 7B run. These rungs answer that here instead.
    #
    # WHAT THEY DO NOT ANSWER, because it is the same ladder's blind spot:
    # `max_seq_len` stays at 896 and evidence is still truncated to fit, so a
    # bigger model reads the same short prompt. This tests MODEL SCALE and
    # says nothing about CONTEXT LENGTH -- which is the axis a cartridge is
    # supposed to win on, since it exists to compress a long context. A rung
    # that gains here gains despite that, not because of it.
    "gpt2-medium-wiki-qa": {
        "model_id": "gpt2-medium",
        "window": 256,
        "held_out_stride": 4,
        "num_slots": 128,
        "max_seq_len": 896,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
        "distractor_count": 3,
        "max_items": 120,
        "smallest_effect_of_interest": 0.05,
        "alpha": 0.05,
        "mcnemar_test": McNemarTest.MID_P,
    },
    "gpt2-large-wiki-qa": {
        "model_id": "gpt2-large",
        "window": 256,
        "held_out_stride": 4,
        "num_slots": 128,
        "max_seq_len": 896,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
        "distractor_count": 3,
        "max_items": 120,
        "smallest_effect_of_interest": 0.05,
        "alpha": 0.05,
        "mcnemar_test": McNemarTest.MID_P,
    },
    "gpt2-xl-wiki-qa": {
        "model_id": "gpt2-xl",
        "window": 256,
        "held_out_stride": 4,
        "num_slots": 128,
        "max_seq_len": 896,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
        "distractor_count": 3,
        "max_items": 120,
        "smallest_effect_of_interest": 0.05,
        "alpha": 0.05,
        "mcnemar_test": McNemarTest.MID_P,
    },
    # THE POWERED PLANS, and the reason they exist is arithmetic rather than
    # taste. The me-wiki corpus yields 32 items, and a comparison of two
    # arm-gaps across two corpora at that n has a minimum detectable effect of
    # 0.216 to 0.343 depending on how much the arms trade items -- larger than
    # any difference this program has ever measured. Computed with
    # `platform_core.minimum_detectable_effect` before these were added, not
    # after a disappointing result.
    #
    # Everything except `max_items` is copied from `gpt2-wiki-qa` so a rung
    # here differs from its me-wiki twin in the corpus and the item count
    # alone. `max_items` is 240 because the api-codebase wiki offers 235 and
    # 224 items in its raw and reshaped forms: a cap of 120 would have been
    # the binding constraint rather than the corpus, and would have thrown
    # away half the power the bigger corpus was chosen for.
    #
    # THE TWO SIDES ARE NOT EQUAL n (235 against 224) and that is correct.
    # They are different question sets over different text; equalising them
    # would mean discarding real items to make two incomparable numbers look
    # comparable, which is the confusion this whole design is built to avoid.
    "api-wiki-qa": {
        "model_id": "gpt2",
        "window": 256,
        "held_out_stride": 4,
        "num_slots": 128,
        "max_seq_len": 896,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
        "distractor_count": 3,
        "max_items": 240,
        "smallest_effect_of_interest": 0.05,
        "alpha": 0.05,
        "mcnemar_test": McNemarTest.MID_P,
    },
    "gpt2-large-api-wiki-qa": {
        "model_id": "gpt2-large",
        "window": 256,
        "held_out_stride": 4,
        "num_slots": 128,
        "max_seq_len": 896,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
        "distractor_count": 3,
        "max_items": 240,
        "smallest_effect_of_interest": 0.05,
        "alpha": 0.05,
        "mcnemar_test": McNemarTest.MID_P,
    },
    # THE RUNG ABOVE THE GPT-2 FAMILY, and the reason it did not exist until
    # 2026-09-09 is that nobody wrote it. The ladder stopped at gpt2-xl
    # because the scale ladder it was extended from stopped there, while the
    # cartridge SWEEP ladder has run this base on an A30 since image v36 --
    # `gpu_pinned_because: bf16-7b-weights-13.8GB-plus-training-need-24GB`.
    # The crossing this programme reports happens at ~774M, so a ladder ending
    # at 1.5B stops one rung after its own finding and can say nothing about
    # whether it holds.
    #
    # EVERY FIELD BUT `model_id` IS THE LADDER'S, DELIBERATELY. `max_seq_len`
    # stays 896 even though this base has 2048 positions rather than 1024:
    # the evidence budget is what the retrieval arms spend, so letting it grow
    # with the model would confound scale with how much text the retriever is
    # allowed to carry, and the ladder's question is about scale alone.
    "pythia-6.9b-api-wiki-qa": {
        "model_id": "EleutherAI/pythia-6.9b",
        "window": 256,
        "held_out_stride": 4,
        "num_slots": 128,
        "max_seq_len": 896,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
        "distractor_count": 3,
        "max_items": 240,
        "smallest_effect_of_interest": 0.05,
        "alpha": 0.05,
        "mcnemar_test": McNemarTest.MID_P,
    },
    # THE CAPACITY AXIS, WHICH HAD NEVER BEEN VARIED. `num_slots` was 128 in
    # every plan above, six times over, while the programme's stated mechanism
    # for why a cartridge should lose to a retriever is that "the cartridge
    # has a FIXED slot budget and a retriever's index is unbounded". That is a
    # claim about a CURVE, and it was measured at one point.
    #
    # `max_seq_len` IS 768 IN ALL FOUR, NOT 1024 MINUS THE SLOTS. Sizing each
    # arm to its own prefix would give the 32-slot cell 992 tokens of evidence
    # and the 256-slot cell 768, so the cell with the smallest cartridge would
    # also have the largest retrieval budget -- and the axis would measure the
    # two moving in opposite directions at once. Holding every cell to the
    # tightest common budget is what makes the slot count the only thing that
    # differs.
    "gpt2-large-api-wiki-qa-slots-32": {
        "model_id": "gpt2-large",
        "window": 256,
        "held_out_stride": 4,
        "num_slots": 32,
        "max_seq_len": 768,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
        "distractor_count": 3,
        "max_items": 240,
        "smallest_effect_of_interest": 0.05,
        "alpha": 0.05,
        "mcnemar_test": McNemarTest.MID_P,
    },
    "gpt2-large-api-wiki-qa-slots-64": {
        "model_id": "gpt2-large",
        "window": 256,
        "held_out_stride": 4,
        "num_slots": 64,
        "max_seq_len": 768,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
        "distractor_count": 3,
        "max_items": 240,
        "smallest_effect_of_interest": 0.05,
        "alpha": 0.05,
        "mcnemar_test": McNemarTest.MID_P,
    },
    "gpt2-large-api-wiki-qa-slots-128": {
        "model_id": "gpt2-large",
        "window": 256,
        "held_out_stride": 4,
        "num_slots": 128,
        "max_seq_len": 768,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
        "distractor_count": 3,
        "max_items": 240,
        "smallest_effect_of_interest": 0.05,
        "alpha": 0.05,
        "mcnemar_test": McNemarTest.MID_P,
    },
    "gpt2-large-api-wiki-qa-slots-256": {
        "model_id": "gpt2-large",
        "window": 256,
        "held_out_stride": 4,
        "num_slots": 256,
        "max_seq_len": 768,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
        "distractor_count": 3,
        "max_items": 240,
        "smallest_effect_of_interest": 0.05,
        "alpha": 0.05,
        "mcnemar_test": McNemarTest.MID_P,
    },
}


def qa_plan_label(name: str, plan: QaPlan, *, digest: str) -> str:
    """Build the label that identifies one plan's numbers on one corpus.

    Args:
        name: The plan's name.
        plan: The plan.
        digest: Digest of the corpus, from
            :func:`~cartridge_plans.corpus_digest`.

    Returns:
        The label, e.g.
        ``gpt2-wiki-qa-gpt2-w256-s4-c128-m896-e12-lr0.01-d3-n120-seeds7.8.9-1a2b3c4d``.
    """
    seeds = ".".join(str(seed) for seed in plan["seeds"])
    return (
        f"{name}"
        f"-{plan['model_id']}"
        f"-w{plan['window']}"
        f"-s{plan['held_out_stride']}"
        f"-c{plan['num_slots']}"
        f"-m{plan['max_seq_len']}"
        f"-e{plan['epochs']}"
        f"-lr{plan['learning_rate']}"
        f"-d{plan['distractor_count']}"
        f"-n{plan['max_items']}"
        f"-seeds{seeds}"
        f"-{digest[:12]}"
    )


__all__ = [
    "QA_EXPERIMENT",
    "QA_PLANS",
    "QaPlan",
    "qa_plan_label",
]
