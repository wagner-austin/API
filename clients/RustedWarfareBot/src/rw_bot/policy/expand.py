"""Turn a list of goals into a plan that can actually be executed.

The policy executes a sequence, and until now a human wrote every entry of it
including the prerequisites. That is the difference between a script and a
plan: ask for a tank and the old planner reported ``blocked``, because nothing
the player owned could make one — correct, and useless. The information needed
to fix it was already there, in the engine's own build tree
([[mechanics-build-tree]]).

Expansion is deliberately separate from :mod:`rw_bot.policy.build_order` and
runs once, before the game loop. It is a pure function of the goals, the tree
and what is already owned, so the plan it produces can be read and argued with
before a single order is sent — which matters more here than in most places,
because an unexecutable plan costs a whole match to discover.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rw_bot import RwBotError
from rw_bot.mechanics.build_tree import producers_of
from rw_bot.mechanics.catalogue import UnitStats

_UNREACHABLE = "RW-EXPAND-001"


class ExpansionError(RwBotError):
    """A goal cannot be reached from what the player owns.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description, naming the goal.
    """


def expand(
    goals: Sequence[str],
    tree: Mapping[str, frozenset[str]],
    owned: Sequence[str],
    catalogue: Mapping[str, UnitStats],
) -> tuple[str, ...]:
    """Insert the prerequisites each goal needs, in build order.

    Every goal is kept, in the order asked for. What changes is what appears
    *before* each one: if nothing available can make it, whatever can make it is
    built first, recursively.

    Availability grows as the plan does. A prerequisite inserted for one goal is
    available to every later goal, so asking for two tanks inserts one factory
    rather than two — which is the whole reason expansion runs over the list
    rather than per entry.

    Cheapest producer wins when several would do. The tie is broken by name so
    the same goals always expand to the same plan; a plan that varied run to run
    would make one run's failure impossible to reproduce with another's.

    Args:
        goals: What the player wants, in order.
        tree: Products by producer, from the engine's registry.
        owned: Type names the player already has finished.
        catalogue: Unit stats by type name, for choosing between producers.

    Returns:
        The executable plan: prerequisites and goals, in build order.

    Raises:
        ExpansionError: ``RW-EXPAND-001`` when a goal cannot be reached from
            what the player owns, naming the goal.
    """
    available: set[str] = set(owned)
    plan: list[str] = []
    for goal in goals:
        _ensure(goal, tree, available, catalogue, plan, ())
    return tuple(plan)


def _ensure(
    target: str,
    tree: Mapping[str, frozenset[str]],
    available: set[str],
    catalogue: Mapping[str, UnitStats],
    plan: list[str],
    pending: tuple[str, ...],
) -> None:
    """Append ``target`` to the plan, preceded by whatever it needs.

    Args:
        target: The type to make.
        tree: Products by producer.
        available: Types owned or already planned. Extended in place.
        catalogue: Unit stats by type name, for choosing between producers.
        plan: The plan being built. Extended in place.
        pending: Targets already being resolved further up the recursion, which
            is what stops the search cycling.

    Raises:
        ExpansionError: ``RW-EXPAND-001`` when nothing reachable makes it.
    """
    producers = producers_of(tree, target)
    if not producers:
        raise ExpansionError(
            _UNREACHABLE,
            f"nothing in the build tree makes {target!r}, so no plan reaches it",
        )
    if producers & available:
        plan.append(target)
        available.add(target)
        return

    # The build tree has cycles -- a factory makes a builder and a builder makes
    # a factory -- so a producer already being resolved higher up cannot be the
    # answer for this one. Excluding the whole pending chain is what makes the
    # search terminate.
    reachable: list[tuple[int, str]] = [
        (_price(name, catalogue), name) for name in producers - set(pending) - {target}
    ]
    candidates = [name for _, name in sorted(reachable)]
    for candidate in candidates:
        try:
            _ensure(candidate, tree, available, catalogue, plan, (*pending, target))
        except ExpansionError:
            continue
        plan.append(target)
        available.add(target)
        return

    raise ExpansionError(
        _UNREACHABLE,
        f"{target!r} needs one of {sorted(producers)}, and none of those can be "
        "reached from what the player owns",
    )


def _price(type_name: str, catalogue: Mapping[str, UnitStats]) -> int:
    """Return a type's price for ranking producers.

    A type the catalogue does not price sorts last rather than failing the
    expansion. The catalogue and the build tree are separate dumps and their
    coverage is not guaranteed to be identical, so an unpriced producer is a
    worse-known option rather than a broken plan.

    Args:
        type_name: The type to price.
        catalogue: Unit stats by type name.

    Returns:
        The price, or a sentinel above any real one when it is unknown.
    """
    stats = catalogue.get(type_name)
    return stats["price"] if stats is not None else 1_000_000


__all__ = ["ExpansionError", "expand"]
