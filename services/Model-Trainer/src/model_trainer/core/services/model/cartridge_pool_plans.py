"""The pool-family plans: varied, diverse, and base-LoRA sweeps.

Split from :mod:`cartridge_plans` when that module passed the 600-line
ceiling, along the seam the work itself has: that module holds the plans the
ORIGINAL certified records were measured under (capacity, composition
scaling, the companion p-sweep), and this one holds the interventions built
over them -- pools whose members are drawn per step, pools whose members
carry different voices, and the base-side LoRA that trains the other half of
the attention. Every plan here subtracts against a record a plan THERE
produced, which is why the shared fields are pinned by test rather than by
prose.
"""

from __future__ import annotations

from typing import Final

from typing_extensions import TypedDict


class VariedCompanionSweepPlan(TypedDict):
    """One complete, reproducible varied-count companionship measurement.

    The intervention over the intervention: single-companion training's
    retention decays with deployment count (44.6% at four compartments,
    26.5% at eight, both recorded), and the hypothesis this plan tests is
    that a cartridge trained under a VARIED number of simultaneous
    companions learns count-invariance a fixed single companion cannot
    teach. One kind and one probability, because both knobs were already
    swept: content-companionship dominated noise everywhere and p=0.5 was
    the best dose, so this plan varies exactly the new thing.

    Attributes:
        model_id: HuggingFace id of the base to measure against.
        window: Tokens per scored item.
        held_out_stride: One window in this many is held out.
        compartment_counts: How many cartridges to compose, in increasing
            order.
        slots: Prefix positions per cartridge, fixed across the grid.
        probability: Chance per training forward that ANY companions are
            present, in (0, 1]; when they are, the count is drawn uniformly
            from one to ``max_companions``.
        max_companions: Pool size, and the largest count a single forward
            can draw. At least two: a pool of one is the recorded
            single-companion recipe and must be spelled as that plan.
        seeds: Initialisation seeds. Every arm runs once per seed.
        epochs: Passes over each corpus.
        learning_rate: Step size for AdamW.
    """

    model_id: str
    window: int
    held_out_stride: int
    compartment_counts: tuple[int, ...]
    slots: int
    probability: float
    max_companions: int
    seeds: tuple[int, ...]
    epochs: int
    learning_rate: float


#: Fixed for the reason the other experiment names are.
VARIED_COMPANION_SWEEP_EXPERIMENT = "cartridge-varied-companioned-composition"


#: The varied-count plans. ``gpt2-companions-varied`` matches the recorded
#: companion grids on every shared field (window, stride, slots, seeds,
#: schedule, and the trained-p0.5 recipe cell's probability), so its n4 and
#: n8 cells subtract against the single-companion records directly: the n8
#: baseline to beat is +26.5% retention, the n4 regression bar is +44.6%.
VARIED_COMPANION_SWEEP_PLANS: Final[dict[str, VariedCompanionSweepPlan]] = {
    "gpt2-companions-varied": {
        "model_id": "gpt2",
        "window": 256,
        "held_out_stride": 4,
        "compartment_counts": (4, 8),
        "slots": 64,
        "probability": 0.5,
        "max_companions": 3,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
    },
}


#: Fixed for the reason the other experiment names are. The diverse sweep
#: REUSES :class:`VariedCompanionSweepPlan` -- the knobs are identical and
#: only the pool's construction differs (K corpora instead of K seeds of
#: one corpus) -- but it is a different measurement answering a different
#: question, so it records under its own experiment.
DIVERSE_COMPANION_SWEEP_EXPERIMENT = "cartridge-diverse-companioned-composition"


#: The diverse-pool plans. ``gpt2-companions-diverse`` matches
#: ``gpt2-companions-varied`` on every field, so the two records isolate
#: exactly one difference: whether the pool's members carry one voice or
#: three. The varied record (n4 +51.0%, n8 +18.3%) and the single-companion
#: record (n4 +44.6%, n8 +26.5%) are the baselines its cells subtract
#: against.
DIVERSE_COMPANION_SWEEP_PLANS: Final[dict[str, VariedCompanionSweepPlan]] = {
    "gpt2-companions-diverse": {
        "model_id": "gpt2",
        "window": 256,
        "held_out_stride": 4,
        "compartment_counts": (4, 8),
        "slots": 64,
        "probability": 0.5,
        "max_companions": 3,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
    },
    # The scale rung: the recorded recipe on a base three times the size.
    # Every field except the base matches ``gpt2-companions-diverse``, so
    # the two records isolate exactly the parameter count -- including the
    # schedule, deliberately: a retuned schedule would confound scale with
    # tuning. The question is whether the diverse-pool verdicts (n4 +55.5%,
    # n8 +28.0%, content interference fully trained away) survive a base
    # where 448 foreign slots are a smaller fraction of attention.
    "gpt2-medium-companions-diverse": {
        "model_id": "gpt2-medium",
        "window": 256,
        "held_out_stride": 4,
        "compartment_counts": (4, 8),
        "slots": 64,
        "probability": 0.5,
        "max_companions": 3,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
    },
}


class BaseLoraSweepPlan(TypedDict):
    """One complete, reproducible base-side composition-LoRA measurement.

    The lever left standing after the cartridge-side arc: a LoRA on the
    base's attention, trained to do language modeling behind a drawn number
    of frozen composed cartridges, then measured by the recorded grid's own
    arms with the ADAPTED base underneath. Two cell families ask the two
    questions that matter: plain-trained cartridges on the adapted base
    (does base-side training alone rescue composition) and
    diverse-companioned cartridges on the adapted base (do the two sides
    compose).

    Attributes:
        model_id: HuggingFace id of the base to adapt and measure.
        window: Tokens per scored item.
        held_out_stride: One window in this many is held out.
        compartment_counts: How many cartridges to compose, in increasing
            order.
        slots: Prefix positions per cartridge, fixed across the grid.
        probability: Companion-presence probability for the
            diverse-companioned cells, in (0, 1].
        max_companions: Pool size for the diverse-companioned cells' own
            companion pool -- one member per pool corpus.
        lora_rank: LoRA rank on the attention projections.
        lora_alpha: LoRA scaling numerator.
        lora_epochs: Passes over the pool corpora during LoRA training.
        lora_learning_rate: Step size for AdamW over the LoRA parameters.
        max_drawn: Largest cartridge count one LoRA-training forward may
            draw, at least two and at most the crowding pool's size.
        pool_members_per_corpus: Seed-variant cartridges trained per pool
            corpus for the crowding pool.
        seeds: Measurement seeds. Every arm runs once per seed.
        epochs: Passes over each corpus when training a measured cartridge.
        learning_rate: Step size for AdamW over a measured cartridge.
    """

    model_id: str
    window: int
    held_out_stride: int
    compartment_counts: tuple[int, ...]
    slots: int
    probability: float
    max_companions: int
    lora_rank: int
    lora_alpha: int
    lora_epochs: int
    lora_learning_rate: float
    max_drawn: int
    pool_members_per_corpus: int
    seeds: tuple[int, ...]
    epochs: int
    learning_rate: float


#: Fixed for the reason the other experiment names are.
BASE_LORA_SWEEP_EXPERIMENT = "cartridge-base-lora-composition"


#: The base-LoRA plans. ``gpt2-base-lora`` matches the diverse grid on every
#: measurement field, so its cells subtract against the whole recorded
#: ladder: naive (aa61330b), single companion and n8 (9e87e816/6e63dad7),
#: and the diverse pool (c9e1cf4f). The LoRA knobs are deliberately modest:
#: rank 8 on the attention projections is the smallest adapter with a
#: literature track record, and three passes over the three pool corpora is
#: enough for the epoch-loss trail to show convergence or its absence.
BASE_LORA_SWEEP_PLANS: Final[dict[str, BaseLoraSweepPlan]] = {
    "gpt2-base-lora": {
        "model_id": "gpt2",
        "window": 256,
        "held_out_stride": 4,
        "compartment_counts": (4, 8),
        "slots": 64,
        "probability": 0.5,
        "max_companions": 3,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_epochs": 3,
        "lora_learning_rate": 0.0001,
        "max_drawn": 8,
        "pool_members_per_corpus": 3,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
    },
    # The lever aimed at the base that needs it: gpt2-medium COLLAPSED at n8
    # (-86.6%, cross-node bit-identical) because depth compounds crowded-
    # prefix interference, and the gpt2 record proved the LoRA repairs
    # exactly that structural component (-45.4% -> -6.9% at n4 with plain
    # cartridges). Every field but the base matches ``gpt2-base-lora`` --
    # the schedule and LoRA knobs deliberately included, so scale is not
    # confounded with tuning.
    "gpt2-medium-base-lora": {
        "model_id": "gpt2-medium",
        "window": 256,
        "held_out_stride": 4,
        "compartment_counts": (4, 8),
        "slots": 64,
        "probability": 0.5,
        "max_companions": 3,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_epochs": 3,
        "lora_learning_rate": 0.0001,
        "max_drawn": 8,
        "pool_members_per_corpus": 3,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
    },
}


#: Fixed for the reason the other experiment names are. The content-LoRA
#: sweep REUSES :class:`BaseLoraSweepPlan` -- every knob is identical and
#: only the LoRA's TRAINING OBJECTIVE differs (crowd-invariance distillation
#: instead of language modeling) -- but it is a different measurement
#: answering a different question, so it records under its own experiment,
#: exactly as the diverse sweep reuses the varied plan.
CONTENT_LORA_SWEEP_EXPERIMENT = "cartridge-content-lora-composition"


#: The content-LoRA plans. Each row matches its ``*-base-lora`` twin on
#: EVERY field -- pinned by test -- so the two records isolate exactly one
#: difference: what the LoRA was trained to do. The lever aims at the one
#: enemy the base-LoRA record left standing (372cee59, cross-node
#: bit-identical): gpt2-medium's n8 with real content collapses (-79.4%)
#: while its repaired noise control sits +0.42, a 1.04 gap that is CONTENT
#: interference at depth. The LM objective teaches the base to read past a
#: crowd's structure; crowd-invariance distillation teaches it to read past
#: the crowd's CONTENT, by matching, behind a full roster, the predictions
#: the plain base makes behind the target member alone.
CONTENT_LORA_SWEEP_PLANS: Final[dict[str, BaseLoraSweepPlan]] = {
    "gpt2-content-lora": {
        "model_id": "gpt2",
        "window": 256,
        "held_out_stride": 4,
        "compartment_counts": (4, 8),
        "slots": 64,
        "probability": 0.5,
        "max_companions": 3,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_epochs": 3,
        "lora_learning_rate": 0.0001,
        "max_drawn": 8,
        "pool_members_per_corpus": 3,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
    },
    # The verdict-carrier: depth is where content interference lives, so
    # gpt2-medium runs first and gpt2-small is its cross-scale anchor.
    "gpt2-medium-content-lora": {
        "model_id": "gpt2-medium",
        "window": 256,
        "held_out_stride": 4,
        "compartment_counts": (4, 8),
        "slots": 64,
        "probability": 0.5,
        "max_companions": 3,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_epochs": 3,
        "lora_learning_rate": 0.0001,
        "max_drawn": 8,
        "pool_members_per_corpus": 3,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
    },
}


def base_lora_sweep_label(name: str, plan: BaseLoraSweepPlan, *, digest: str) -> str:
    """Build the label identifying one base-LoRA sweep on one primary corpus.

    Args:
        name: The plan's name.
        plan: The plan.
        digest: Digest of the PRIMARY corpus, from :func:`corpus_digest`.

    Returns:
        The label, e.g.
        ``gpt2-base-lora-gpt2-w256-s4-e12-lr0.01-n4.8-c64-p0.5-K3-R8-a16-le3-llr0.0001-D8-m3-seeds7.8.9-1a2b3c4d5e6f``.
    """
    counts = ".".join(str(count) for count in plan["compartment_counts"])
    seeds = ".".join(str(seed) for seed in plan["seeds"])
    return (
        f"{name}"
        f"-{plan['model_id']}"
        f"-w{plan['window']}"
        f"-s{plan['held_out_stride']}"
        f"-e{plan['epochs']}"
        f"-lr{plan['learning_rate']}"
        f"-n{counts}"
        f"-c{plan['slots']}"
        f"-p{plan['probability']}"
        f"-K{plan['max_companions']}"
        f"-R{plan['lora_rank']}"
        f"-a{plan['lora_alpha']}"
        f"-le{plan['lora_epochs']}"
        f"-llr{plan['lora_learning_rate']}"
        f"-D{plan['max_drawn']}"
        f"-m{plan['pool_members_per_corpus']}"
        f"-seeds{seeds}"
        f"-{digest[:12]}"
    )


def varied_companion_sweep_label(name: str, plan: VariedCompanionSweepPlan, *, digest: str) -> str:
    """Build the label identifying one varied-count sweep on one primary corpus.

    Args:
        name: The plan's name.
        plan: The plan.
        digest: Digest of the PRIMARY corpus, from :func:`corpus_digest`.

    Returns:
        The label, e.g.
        ``gpt2-companions-varied-gpt2-w256-s4-e12-lr0.01-n4.8-c64-p0.5-K3-seeds7.8.9-1a2b3c4d5e6f``.
        ``K`` carries the pool size; ``c`` and ``p`` keep the meanings the
        companion-sweep label gave them.
    """
    counts = ".".join(str(count) for count in plan["compartment_counts"])
    seeds = ".".join(str(seed) for seed in plan["seeds"])
    return (
        f"{name}"
        f"-{plan['model_id']}"
        f"-w{plan['window']}"
        f"-s{plan['held_out_stride']}"
        f"-e{plan['epochs']}"
        f"-lr{plan['learning_rate']}"
        f"-n{counts}"
        f"-c{plan['slots']}"
        f"-p{plan['probability']}"
        f"-K{plan['max_companions']}"
        f"-seeds{seeds}"
        f"-{digest[:12]}"
    )


__all__ = [
    "BASE_LORA_SWEEP_EXPERIMENT",
    "BASE_LORA_SWEEP_PLANS",
    "CONTENT_LORA_SWEEP_EXPERIMENT",
    "CONTENT_LORA_SWEEP_PLANS",
    "DIVERSE_COMPANION_SWEEP_EXPERIMENT",
    "DIVERSE_COMPANION_SWEEP_PLANS",
    "VARIED_COMPANION_SWEEP_EXPERIMENT",
    "VARIED_COMPANION_SWEEP_PLANS",
    "BaseLoraSweepPlan",
    "VariedCompanionSweepPlan",
    "base_lora_sweep_label",
    "varied_companion_sweep_label",
]
