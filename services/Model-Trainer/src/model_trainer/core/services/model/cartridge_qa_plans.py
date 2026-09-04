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
        max_items: Cap on the question set's size.
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


#: Fixed rather than a flag, and distinct from the loss experiment's name.
QA_EXPERIMENT = "cartridge-question-set"


#: The plans. ``gpt2-wiki-qa`` mirrors ``gpt2-wiki``'s corpus, window, split
#: and schedule so the two measurements describe the same cartridge, and adds
#: only what a question set needs.
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
