"""A build-order policy: decide the next action from one observation.

Deliberately a pure function. ``decide`` takes a world sample and the plan's
progress and returns what to do; it opens no sockets, reads no clock, and
mutates nothing. That is what makes the playing logic testable without a game
running, and it is the same separation the agent holds on its own side —
dispatch there, decision here (wiki: runtime-split-java-agent-python-brain).

The policy is small on purpose. It executes a fixed sequence of structures,
waiting when it cannot afford the next one and stopping when it cannot make
progress at all. It does not fight, expand, or react to an opponent. What it
demonstrates is a loop that observes, chooses, acts, and can be scored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.wire.state import Entity, Sample

BUILDER_TYPE = "builder"

#: World-space offsets from the builder at which successive structures are
#: placed. A ring rather than a line: buildings occupy space, and walking the
#: builder steadily away from its base would put later structures out of reach
#: of anything that has to defend them.
PLACEMENT_RING: tuple[tuple[float, float], ...] = (
    (200.0, 120.0),
    (-200.0, 120.0),
    (200.0, -120.0),
    (-200.0, -120.0),
    (280.0, 0.0),
    (-280.0, 0.0),
    (0.0, 240.0),
    (0.0, -240.0),
)


class Decision(TypedDict):
    """What the policy wants to happen next.

    Attributes:
        action: ``"build"`` to place the next structure, ``"wait"`` to do
            nothing this tick, ``"done"`` when the plan is complete,
            ``"blocked"`` when it cannot proceed at all, and ``"stalled"``
            when an order was accepted but never took effect.
        reason: Human-readable justification, carried for the run log so a
            session can be read back without re-deriving why it stalled.
        type_name: Structure to place. Empty unless ``action`` is ``"build"``.
        unit_id: Builder to order. Zero unless ``action`` is ``"build"``.
        x: Placement world x. Zero unless ``action`` is ``"build"``.
        y: Placement world y. Zero unless ``action`` is ``"build"``.
    """

    action: Literal["build", "wait", "done", "blocked", "stalled"]
    reason: str
    type_name: str
    unit_id: int
    x: float
    y: float


def completed_count(sample: Sample, plan: Sequence[str]) -> int:
    """Count how much of the plan the world already shows.

    Progress is read from the roster rather than tracked in a counter, so a
    structure that was destroyed is no longer counted as built and the policy
    will replace it. Counting from observation is also what makes the policy
    resumable: a planner that reconnects mid-match sees the same progress as
    one that watched the whole thing.

    Only owned entities count. The stream carries every visible entity, so
    without the ownership check an opponent building the same structure in
    view would advance this plan.

    Args:
        sample: One observation of the world.
        plan: Structures to build, in order.

    Returns:
        How many plan entries are satisfied by structures currently owned.
    """
    remaining: list[str] = list(plan)
    built = 0
    for entity in sample["entities"]:
        if not entity["mine"]:
            continue
        if entity["type_name"] in remaining:
            remaining.remove(entity["type_name"])
            built += 1
    return built


def next_unsatisfied_index(sample: Sample, plan: Sequence[str]) -> int:
    """Return the index of the first plan entry the world does not already show.

    This is deliberately not the same question as :func:`completed_count`, and
    conflating them is a real defect rather than a hypothetical one. The count
    answers "how many entries are satisfied"; using it as "which entry is next"
    is only correct when the satisfied entries form a prefix of the plan.

    They diverge the moment a plan names something the player already owns.
    Every game starts with a builder, so the plan ``("landFactory", "builder")``
    counts as one-satisfied, jumps straight to index 1, builds a second builder
    and never builds the factory at all. Scanning for the first unsatisfied
    entry fixes the order while keeping the inventory reading that makes
    progress resumable.

    Args:
        sample: One observation of the world.
        plan: Structures to build, in order.

    Returns:
        The index to build next, or ``len(plan)`` when the plan is satisfied.
    """
    owned: list[str] = [e["type_name"] for e in sample["entities"] if e["mine"]]
    for index, wanted in enumerate(plan):
        if wanted in owned:
            owned.remove(wanted)
            continue
        return index
    return len(plan)


def find_builder(sample: Sample) -> Entity | None:
    """Return a builder from the roster, or None when the player owns none.

    Enemy builders are visible in the stream and are not orderable, so the
    ownership check is what keeps this from returning one.

    Args:
        sample: One observation of the world.

    Returns:
        The first owned builder, or None.
    """
    for entity in sample["entities"]:
        if entity["mine"] and entity["type_name"] == BUILDER_TYPE:
            return entity
    return None


def decide(
    sample: Sample,
    plan: Sequence[str],
    catalogue: Mapping[str, UnitStats],
) -> Decision:
    """Choose the next action from one observation.

    Args:
        sample: One observation of the world.
        plan: Structures to build, in order.
        catalogue: Unit stats by type name, for prices.

    Returns:
        The decision.
    """
    built = completed_count(sample, plan)
    index = next_unsatisfied_index(sample, plan)
    if index >= len(plan):
        return _decision("done", f"all {len(plan)} structures built")

    target = plan[index]
    stats = catalogue.get(target)
    if stats is None:
        return _decision(
            "blocked",
            f"{target!r} is not in the unit catalogue, so its price is unknown",
        )

    builder = find_builder(sample)
    if builder is None:
        return _decision("blocked", "the player owns no builder")

    if sample["credits"] < stats["price"]:
        return _decision(
            "wait",
            f"{target} costs {stats['price']}, holding {sample['credits']}",
        )

    offset = PLACEMENT_RING[index % len(PLACEMENT_RING)]
    return Decision(
        action="build",
        reason=f"building {target} ({built + 1} of {len(plan)})",
        type_name=target,
        unit_id=builder["unit_id"],
        x=builder["x"] + offset[0],
        y=builder["y"] + offset[1],
    )


def _decision(
    action: Literal["build", "wait", "done", "blocked", "stalled"], reason: str
) -> Decision:
    """Build a decision that carries no order.

    Args:
        action: What to do.
        reason: Why.

    Returns:
        The decision, with the order fields zeroed.
    """
    return Decision(action=action, reason=reason, type_name="", unit_id=0, x=0.0, y=0.0)


__all__ = [
    "BUILDER_TYPE",
    "PLACEMENT_RING",
    "Decision",
    "completed_count",
    "decide",
    "find_builder",
    "next_unsatisfied_index",
]
