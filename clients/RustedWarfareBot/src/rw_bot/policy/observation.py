"""What the world shows about an order already given.

Two questions, asked by every loop that issues an order and then has to decide
whether to issue it again: is the builder still walking there, and is the thing
it was sent to build already going up. Both are answered from what the engine
reports rather than from a deadline the planner invents, which is the property
that made the build loop's stall detector independent of price, distance and map
size alike ([[policy-loop]]).

They live here because two callers now need them. The build loop asks so it can
tell a slow order from a refused one; the economy asks so it does not re-task a
builder that is already halfway to a resource pool. Answering the same question
in two places is how the two answers start to disagree.

Pure, like the rest of the policy layer: a sample goes in and a fact comes out.
"""

from __future__ import annotations

from rw_bot.wire.state import Entity, Sample

#: World-unit displacement between samples below which a unit counts as
#: stationary. A parked unit reports byte-identical coordinates, so this only
#: has to survive float noise rather than distinguish slow movement.
MOVEMENT_EPSILON = 0.5


def position_of(entity: Entity | None) -> tuple[float, float] | None:
    """Return an entity's position, or None when it is not there.

    The None case is the point. A builder that has died is absent from the
    roster, and the callers track positions across samples -- so "gone" has to
    be representable rather than defaulted to an origin no unit ever stands on.

    Args:
        entity: The entity, or None when the roster does not hold one.

    Returns:
        The position, or None when there is no entity.
    """
    if entity is None:
        return None
    return (entity["x"], entity["y"])


def has_moved(before: tuple[float, float] | None, after: tuple[float, float] | None) -> bool:
    """Report whether a unit moved between two samples.

    A unit that has died, or that was not in the roster to begin with, has not
    moved. Treating a missing unit as movement would keep the caller's stall
    clock permanently reset and turn a lost builder into an infinite wait.

    Args:
        before: Position at the previous sample, if it was there.
        after: Position now, if it is there.

    Returns:
        True when both positions are known and differ by more than float noise.
    """
    if before is None or after is None:
        return False
    return abs(after[0] - before[0]) + abs(after[1] - before[1]) > MOVEMENT_EPSILON


def is_rising(sample: Sample, type_name: str) -> bool:
    """Report whether an unfinished structure of this type is going up.

    Ownership is checked, or an opponent's half-built factory in view would
    answer for ours -- keeping a build loop's clock alive indefinitely in one
    caller, and suppressing our own expansion in the other.

    Args:
        sample: The current observation.
        type_name: The type that was ordered.

    Returns:
        True when the player owns an unfinished entity of that type.
    """
    for entity in sample["entities"]:
        if entity["mine"] and not entity["complete"] and entity["type_name"] == type_name:
            return True
    return False


__all__ = ["MOVEMENT_EPSILON", "has_moved", "is_rising", "position_of"]
