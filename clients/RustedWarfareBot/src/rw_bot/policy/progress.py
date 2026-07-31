"""Reading the plan's progress off the world, and nothing else.

Split from :mod:`rw_bot.policy.build_order` when that module outgrew the
size guard; the split is on the question rather than the line count. These
two functions answer "how much of the plan does the world already show" and
"which entry is next" -- inventory reads, consumed by the decision policy,
the order tracker and the economy alike, and owning them in one place is
what keeps those three consumers counting the same way
([[policy-loop]]).
"""

from __future__ import annotations

from collections.abc import Sequence

from rw_bot.mechanics.upgrades import satisfies
from rw_bot.wire.state import Sample


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

    Only **finished** ones count, which is a distinction the roster alone
    cannot make. A building joins the roster the moment construction starts,
    so presence is not completion: counting on presence reported a plan done
    while a factory was still a shell, and an unfinished factory produces
    nothing, so the next entry could be ordered against a building that could
    not accept it.

    Args:
        sample: One observation of the world.
        plan: What to make, in order. Entries may be structures or units.

    Returns:
        How many plan entries are satisfied by finished structures the player
        owns.
    """
    remaining: list[str] = list(plan)
    built = 0
    for entity in sample["entities"]:
        if not entity["mine"] or not entity["complete"]:
            continue
        # An upgraded structure still answers the entry that asked for it. This
        # matched the type name exactly, which was right until a structure
        # could convert itself: an extractor that upgraded stopped satisfying
        # the plan that built it, so the plan ordered another and did that
        # forever, and the builder was never free again
        # ([[policy-holding-ground]]).
        match = next((name for name in remaining if satisfies(entity["type_name"], name)), None)
        if match is not None:
            remaining.remove(match)
            built += 1
    return built


def unsatisfied_indices(sample: Sample, plan: Sequence[str]) -> tuple[int, ...]:
    """Return every plan entry the world does not already show, in order.

    All of them rather than only the first, because an entry with nowhere to
    stand defers to the next one: a plan that waited on whichever entry it had
    reached stopped dead when the third extractor had no free pool, and never
    built the factory that funds everything after it
    ([[policy-holding-ground]]).

    Args:
        sample: One observation of the world.
        plan: What to make, in order. Entries may be structures or units.

    Returns:
        The indices still wanted, in plan order. Empty when the plan is
        satisfied.
    """
    owned: list[str] = [e["type_name"] for e in sample["entities"] if e["mine"] and e["complete"]]
    pending: list[int] = []
    for index, wanted in enumerate(plan):
        held = next((name for name in owned if satisfies(name, wanted)), None)
        if held is None:
            pending.append(index)
            continue
        owned.remove(held)
    return tuple(pending)


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

    Unfinished structures do not satisfy an entry, for the same reason they do
    not advance :func:`completed_count`. That keeps the two answers consistent
    -- a half-built factory leaves the plan pointing at the same entry, and the
    caller's once-per-position rule is what stops it being ordered twice while
    it goes up.

    An upgraded structure satisfies the entry that asked for it, by the same
    :func:`~rw_bot.mechanics.upgrades.satisfies` rule the count uses. The two
    answers have to agree about what counts as satisfied or the plan reports
    progress it will not act on ([[policy-holding-ground]]).

    Args:
        sample: One observation of the world.
        plan: What to make, in order. Entries may be structures or units.

    Returns:
        The index to build next, or ``len(plan)`` when the plan is satisfied.
    """
    pending = unsatisfied_indices(sample, plan)
    return pending[0] if pending else len(plan)


__all__ = ["completed_count", "next_unsatisfied_index", "unsatisfied_indices"]
