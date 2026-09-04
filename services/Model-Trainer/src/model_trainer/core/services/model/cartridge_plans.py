"""Which cartridge measurements exist, and what identifies each one's numbers.

WHY A TABLE OF PLANS AND NOT A SET OF FLAGS. The same argument
:mod:`probe_shapes` makes, and for the same reason: a cartridge gain means
nothing apart from the corpus, window, schedule and seeds that produced it,
and a configuration assembled from nine flags is one nobody can reproduce
without also recovering the command line. So a caller names a PLAN and the
plan is a constant.

The corpus is the deliberate exception and stays a path. It is data, it is
large, and it lives somewhere different on every machine the run might use --
a laptop, a compute node, inside an image. What keeps that from reopening the
hole is :func:`plan_label`, which folds a digest of the corpus text into the
label, so a run against different bytes carries a different name and cannot be
differenced against this one by accident.

WHY THE SEEDS ARE IN THE PLAN. They are not a detail of how the plan was run,
they are part of what it measures: every arm's spread across those seeds is
what its mean is judged against. A run that chose its own seeds would report a
floor from a different draw and compare it to this one's numbers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Final, TypeVar

from typing_extensions import TypedDict

#: One plan row, whatever its shape. :func:`require_cartridge_plan` looks up
#: rows in both plan tables this module declares, and the lookup neither reads
#: nor writes a field, so the row type passes through unchanged.
_PlanT = TypeVar("_PlanT")


class CartridgePlan(TypedDict):
    """One complete, reproducible cartridge measurement.

    Attributes:
        model_id: HuggingFace id of the base to measure against. A cartridge
            is a block of one model's own attention keys and values, so this
            is the single most important field: nothing here transfers
            between bases, which is the finding the plans exist to record.
        window: Tokens per scored item.
        held_out_stride: One window in this many is held out.
        slot_counts: Prefix lengths to sweep, in increasing order.
        composition_slots: Prefix length for EACH cartridge in the
            composition arm; the composed prefix is twice this.
        seeds: Initialisation seeds. Every arm runs once per seed, and the
            spread across them is this plan's noise floor.
        epochs: Passes over the corpus.
        learning_rate: Step size for AdamW.
    """

    model_id: str
    window: int
    held_out_stride: int
    slot_counts: tuple[int, ...]
    composition_slots: int
    seeds: tuple[int, ...]
    epochs: int
    learning_rate: float


#: Fixed rather than a flag, for the reason :data:`PROBE_EXPERIMENT` is:
#: ``experiment`` is what makes two records comparable at all, so a run under
#: a caller-supplied name could not be compared with the record it was meant
#: to check.
CARTRIDGE_EXPERIMENT = "cartridge-capacity-and-composition"


#: The plans. Each is a whole measurement; adding a slot count or moving the
#: schedule makes a NEW plan rather than editing one, because every number
#: already recorded under a plan name was produced by the old one.
#:
#: ``gpt2-wiki`` is the plan the shipped findings were measured under. Its
#: slot counts step by 4x rather than evenly, because the gain they measure
#: turns out to rise with the LOGARITHM of the count -- +0.087, +0.058,
#: +0.030, +0.028 per step -- so an evenly spaced sweep would spend most of
#: its arms resolving a difference smaller than its own noise.
CARTRIDGE_PLANS: Final[dict[str, CartridgePlan]] = {
    "gpt2-wiki": {
        "model_id": "gpt2",
        "window": 256,
        "held_out_stride": 4,
        "slot_counts": (2, 8, 32, 128, 512),
        "composition_slots": 128,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
    },
}


class CompositionSweepPlan(TypedDict):
    """One complete, reproducible composition-scaling measurement.

    Asks the question :data:`CARTRIDGE_PLANS`' composition arm cannot: not
    whether TWO compartments compose, but how retention moves as the count
    grows. Two slot policies are swept because they bound the design space
    from opposite sides -- holding each compartment's size fixed grows the
    prefix with the count, and holding the total budget fixed shrinks each
    compartment as the count grows.

    Attributes:
        model_id: HuggingFace id of the base to measure against.
        window: Tokens per scored item.
        held_out_stride: One window in this many is held out.
        compartment_counts: How many cartridges to compose, in increasing
            order. Each count needs one fewer other-corpus than its value.
        fixed_slots: Prefix positions per cartridge under the fixed-size
            policy; the composed prefix is the count times this.
        total_slot_budget: Total prefix positions under the fixed-budget
            policy, divided evenly across the count. Every count in
            ``compartment_counts`` must divide it, or the plan describes a
            budget nobody can allocate.
        seeds: Initialisation seeds. Every arm runs once per seed.
        epochs: Passes over each corpus.
        learning_rate: Step size for AdamW.
    """

    model_id: str
    window: int
    held_out_stride: int
    compartment_counts: tuple[int, ...]
    fixed_slots: int
    total_slot_budget: int
    seeds: tuple[int, ...]
    epochs: int
    learning_rate: float


#: Fixed for the reason :data:`CARTRIDGE_EXPERIMENT` is: the experiment name
#: is what makes two records comparable at all.
COMPOSITION_SWEEP_EXPERIMENT = "cartridge-composition-scaling"


#: The composition-scaling plans. ``gpt2-compartments`` carries the settings
#: the corrected ``gpt2-wiki`` findings were measured under (window, stride,
#: seeds, schedule), so its numbers sit beside those rather than beside a
#: configuration nobody ran. Its two policies both peak at a 512-slot prefix,
#: which with a 256-token window is 768 positions against gpt2's 1024 -- the
#: window arithmetic is part of the plan because exceeding it raises an
#: IndexError from inside the position embedding that names neither the
#: cartridge nor the limit.
COMPOSITION_SWEEP_PLANS: Final[dict[str, CompositionSweepPlan]] = {
    "gpt2-compartments": {
        "model_id": "gpt2",
        "window": 256,
        "held_out_stride": 4,
        "compartment_counts": (2, 4, 8),
        "fixed_slots": 64,
        "total_slot_budget": 512,
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
    },
}


def composition_sweep_label(name: str, plan: CompositionSweepPlan, *, digest: str) -> str:
    """Build the label identifying one composition sweep on one primary corpus.

    Every field that moves a retention appears, for the reason
    :func:`plan_label` includes them: a record under a colliding label would
    be differenced against numbers it cannot reproduce. The other corpora are
    not in the label for the reason the second corpus is not in
    :func:`plan_label`'s -- the label names the corpus whose retention is the
    finding.

    Args:
        name: The plan's name.
        plan: The plan.
        digest: Digest of the PRIMARY corpus, from :func:`corpus_digest`.

    Returns:
        The label, e.g.
        ``gpt2-compartments-gpt2-w256-s4-e12-lr0.01-n2.4.8-f64-b512-seeds7.8.9-1a2b3c4d5e6f``.
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
        f"-f{plan['fixed_slots']}"
        f"-b{plan['total_slot_budget']}"
        f"-seeds{seeds}"
        f"-{digest[:12]}"
    )


class CompanionSweepPlan(TypedDict):
    """One complete, reproducible composition-aware-training measurement.

    The intervention grid over the composition ceiling: at each compartment
    count, every cartridge is trained with a frozen companion present at
    each swept probability, and the composed retention is compared against
    the plain-trained baseline recorded under
    :data:`COMPOSITION_SWEEP_PLANS`. Probability zero is deliberately not a
    row: p=0 IS the baseline record, and re-running it here would register a
    second copy of those numbers under a different label.

    Attributes:
        model_id: HuggingFace id of the base to measure against.
        window: Tokens per scored item.
        held_out_stride: One window in this many is held out.
        compartment_counts: How many cartridges to compose, in increasing
            order.
        slots: Prefix positions per cartridge, fixed across the grid so the
            probability is the only knob a row varies.
        probabilities: Companion-presence probabilities to sweep, each in
            (0, 1], in increasing order.
        seeds: Initialisation seeds. Every arm runs once per seed.
        epochs: Passes over each corpus.
        learning_rate: Step size for AdamW.
    """

    model_id: str
    window: int
    held_out_stride: int
    compartment_counts: tuple[int, ...]
    slots: int
    probabilities: tuple[float, ...]
    seeds: tuple[int, ...]
    epochs: int
    learning_rate: float


#: Fixed for the reason the other experiment names are.
COMPANION_SWEEP_EXPERIMENT = "cartridge-companioned-composition"


#: The companion-sweep plans. ``gpt2-companions`` matches ``gpt2-compartments``
#: on every shared field (window, stride, seeds, schedule) and fixes the slot
#: count at that plan's fixed policy, so its numbers subtract against the
#: recorded baseline: fixed-64 retention 62.8% at n2 and -45.4% at n4.
COMPANION_SWEEP_PLANS: Final[dict[str, CompanionSweepPlan]] = {
    "gpt2-companions": {
        "model_id": "gpt2",
        "window": 256,
        "held_out_stride": 4,
        "compartment_counts": (2, 4),
        "slots": 64,
        "probabilities": (0.25, 0.5, 1.0),
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
    },
    # ``gpt2-companions-n8`` asks the question the recorded grid cannot: the
    # recipe trains every cartridge beside ONE companion, and at eight
    # compartments each cartridge meets SEVEN strangers at deployment, so
    # retention here measures whether single-companion exposure generalises
    # past the counts it was tuned on. Every shared field matches
    # ``gpt2-companions`` so its cells subtract against that record and
    # against ``gpt2-compartments``' naive n8 baseline; p=1.0 is not a row
    # because the overdose endpoint is already recorded in both kinds and
    # its solo collapse carries no retention to compare.
    "gpt2-companions-n8": {
        "model_id": "gpt2",
        "window": 256,
        "held_out_stride": 4,
        "compartment_counts": (8,),
        "slots": 64,
        "probabilities": (0.25, 0.5),
        "seeds": (7, 8, 9),
        "epochs": 12,
        "learning_rate": 0.01,
    },
}


def companion_sweep_label(name: str, plan: CompanionSweepPlan, *, digest: str) -> str:
    """Build the label identifying one companion sweep on one primary corpus.

    Args:
        name: The plan's name.
        plan: The plan.
        digest: Digest of the PRIMARY corpus, from :func:`corpus_digest`.

    Returns:
        The label, e.g.
        ``gpt2-companions-gpt2-w256-s4-e12-lr0.01-n2.4-c64-p0.25.0.5.1.0-seeds7.8.9-1a2b3c4d5e6f``.
        The per-cartridge slot count takes the ``c`` prefix
        :func:`plan_label` already uses for a slot width, because ``s``
        is taken by the stride.
    """
    counts = ".".join(str(count) for count in plan["compartment_counts"])
    probabilities = ".".join(str(probability) for probability in plan["probabilities"])
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
        f"-p{probabilities}"
        f"-seeds{seeds}"
        f"-{digest[:12]}"
    )


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
}


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


def require_cartridge_plan(plans: Mapping[str, _PlanT], name: str) -> _PlanT:
    """Look up a plan by name in a supplied table.

    The table is a parameter rather than :data:`CARTRIDGE_PLANS` directly, for
    the reason :func:`probe_ladder.ladder_run_record` takes its rungs as one:
    the real table's plans are minutes of GPU each, and a suite that could
    only reach them would either not cover this path or would run them.

    Args:
        plans: The table to look in. Production passes :data:`CARTRIDGE_PLANS`
            or :data:`COMPOSITION_SWEEP_PLANS` through the CLI's hook; the
            lookup is the same for both, which is why the row type is a
            variable rather than one table's shape.
        name: The plan name, a key of that table.

    Returns:
        That plan.

    Raises:
        KeyError: If no such plan exists, naming the ones that do. A bare dict
            index would raise the same class carrying only the missing key,
            and the answer to a mistyped plan is nearly always the list.
    """
    plan = plans.get(name)
    if plan is None:
        raise KeyError(f"unknown cartridge plan {name!r}; known plans: {', '.join(plans)}")
    return plan


def corpus_digest(documents: Sequence[str]) -> str:
    """Digest the exact text a measurement will train on.

    Document boundaries are hashed, not just the concatenation. Two corpora
    holding the same words split into different documents produce different
    windows -- :func:`build_windows` never crosses a boundary -- so they are
    different corpora and must not share a digest.

    Args:
        documents: The document bodies, in the order they will be windowed.

    Returns:
        Hex digest of the corpus.
    """
    accumulator = hashlib.sha256()
    for document in documents:
        accumulator.update(str(len(document)).encode("utf-8"))
        accumulator.update(b"\x00")
        accumulator.update(document.encode("utf-8"))
    return accumulator.hexdigest()


def plan_label(name: str, plan: CartridgePlan, *, digest: str) -> str:
    """Build the label that identifies one plan's numbers on one corpus.

    Every field that moves a gain appears, including the seeds: a plan re-run
    on a different draw would otherwise register under an existing name and be
    differenced against numbers it cannot reproduce.

    Args:
        name: The plan's name.
        plan: The plan.
        digest: Digest of the corpus, from :func:`corpus_digest`. Truncated
            into the label -- the full value is long, and twelve hex
            characters distinguish any two corpora anybody will run.

    Returns:
        The label, e.g.
        ``gpt2-wiki-gpt2-w256-s4-e12-lr0.01-slots2.8.32.128.512-c128-seeds7.8.9-1a2b3c4d5e6f``.
    """
    slots = ".".join(str(count) for count in plan["slot_counts"])
    seeds = ".".join(str(seed) for seed in plan["seeds"])
    return (
        f"{name}"
        f"-{plan['model_id']}"
        f"-w{plan['window']}"
        f"-s{plan['held_out_stride']}"
        f"-e{plan['epochs']}"
        f"-lr{plan['learning_rate']}"
        f"-slots{slots}"
        f"-c{plan['composition_slots']}"
        f"-seeds{seeds}"
        f"-{digest[:12]}"
    )


__all__ = [
    "CARTRIDGE_EXPERIMENT",
    "CARTRIDGE_PLANS",
    "COMPANION_SWEEP_EXPERIMENT",
    "COMPANION_SWEEP_PLANS",
    "COMPOSITION_SWEEP_EXPERIMENT",
    "COMPOSITION_SWEEP_PLANS",
    "DIVERSE_COMPANION_SWEEP_EXPERIMENT",
    "DIVERSE_COMPANION_SWEEP_PLANS",
    "VARIED_COMPANION_SWEEP_EXPERIMENT",
    "VARIED_COMPANION_SWEEP_PLANS",
    "CartridgePlan",
    "CompanionSweepPlan",
    "CompositionSweepPlan",
    "VariedCompanionSweepPlan",
    "companion_sweep_label",
    "composition_sweep_label",
    "corpus_digest",
    "plan_label",
    "require_cartridge_plan",
    "varied_companion_sweep_label",
]
