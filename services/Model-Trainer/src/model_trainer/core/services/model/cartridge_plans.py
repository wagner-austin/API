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
from typing import Final

from typing_extensions import TypedDict


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


def require_cartridge_plan(plans: Mapping[str, CartridgePlan], name: str) -> CartridgePlan:
    """Look up a plan by name in a supplied table.

    The table is a parameter rather than :data:`CARTRIDGE_PLANS` directly, for
    the reason :func:`probe_ladder.ladder_run_record` takes its rungs as one:
    the real table's plans are minutes of GPU each, and a suite that could
    only reach them would either not cover this path or would run them.

    Args:
        plans: The table to look in. Production passes
            :data:`CARTRIDGE_PLANS` through the CLI's hook.
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
    "CartridgePlan",
    "corpus_digest",
    "plan_label",
    "require_cartridge_plan",
]
