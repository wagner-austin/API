"""Where the bot will be shot, judged from what it can see.

The build policy used to choose a resource pool by distance alone. That is a
complete answer only on an empty map, and the map is not empty: a builder was
sent to a free pool 4,293 world units out and killed crossing two enemy bases
on the way. The pool was reachable, unoccupied, and the nearest one left. It was
also on the far side of the opponent.

So the missing question is not "how far" but "through whose guns", and it has to
be asked about the *route* rather than the destination. A pool can sit in
perfectly safe ground with the only approach to it running down an enemy
frontage, and screening destinations alone would have sent that builder to its
death exactly as before.

Two things make the answer legitimate rather than invented. Hostility comes from
the engine's own alliance comparison, carried per entity on the wire, so an
ally's tank and a neutral map object are not mistaken for threats
([[perception-visibility]]). Reach comes from each unit's declared attack range
in the stat catalogue ([[mechanics-unit-catalogue]]), so no radius here is a
number this module made up.

**What this deliberately does not model.** The route is a straight line, and the
engine's pathfinder does not walk straight lines — it steers around terrain, so
the real path can enter danger this misses and can avoid danger this reports.
Nothing here predicts movement either: a hostile is judged where it currently
stands, so a tank that drives out to meet the builder was never counted. Both
are real gaps, and closing them means measuring the engine's paths rather than
guessing at them. What the straight-line test does catch is the case that
actually killed a builder — a fixed enemy base sitting between the bot and the
pool it wanted.
"""

from __future__ import annotations

from collections.abc import Mapping

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.wire.state import Entity, Sample


def reach_of(entity: Entity, catalogue: Mapping[str, UnitStats]) -> float:
    """Return how far an entity can shoot.

    Read from the catalogue's declared attack range rather than assumed from
    what a unit looks like. A unit with no weapon has no reach, which is the
    catalogue's own way of saying it: the weapon entry is absent for anything
    the engine printed no attack range for.

    A type the catalogue does not describe is treated as harmless. The
    alternative is to invent a range for it, and there is no honest number to
    invent — the catalogue covers every registered type, so a miss means a type
    from a mod that is not loaded rather than a gap in coverage.

    Args:
        entity: The entity to measure.
        catalogue: Unit stats by type name.

    Returns:
        Attack range in world units, comparable with entity positions, or zero
        when the entity cannot shoot.
    """
    stats = catalogue.get(entity["type_name"])
    if stats is None:
        return 0.0
    weapon = stats["weapon"]
    if weapon is None:
        return 0.0
    return weapon["attack_range"]


def route_is_exposed(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    """Report whether a hostile can shoot anything walking from start to end.

    The destination is the segment's endpoint, so a target standing in a
    turret's field of fire is caught by the same test that catches a route
    passing through one. There is no separate destination check and there does
    not need to be.

    Only hostiles count, and hostility is the engine's answer rather than
    "everything that is not mine" — see :attr:`~rw_bot.wire.state.Entity.hostile`
    for why the two differ. Unarmed hostiles are skipped: an enemy builder
    standing on the route is an obstacle, not a threat, and treating it as one
    would rule out ground the bot can safely cross.

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for attack ranges.
        start: Where the walk begins, as world x and y.
        end: Where it ends.

    Returns:
        True when some visible hostile's attack range covers any point of the
        straight line between them.
    """
    for entity in sample["entities"]:
        if not entity["hostile"]:
            continue
        reach = reach_of(entity, catalogue)
        if reach <= 0.0:
            continue
        if _distance_squared_to_segment(entity["x"], entity["y"], start, end) <= reach**2:
            return True
    return False


def _distance_squared_to_segment(
    x: float, y: float, start: tuple[float, float], end: tuple[float, float]
) -> float:
    """Return the squared distance from a point to a line segment.

    Squared and left squared: every caller compares it against a squared reach,
    and a square root would cost precision to answer a question nobody asks.

    The segment is a segment rather than an infinite line, which is the whole
    reason this is not a two-line formula. A hostile far behind the builder is
    not on the route, and projecting onto an unbounded line would place it
    there.

    Args:
        x: The point's world x.
        y: The point's world y.
        start: The segment's first endpoint.
        end: Its second.

    Returns:
        The squared distance to the nearest point of the segment.
    """
    span_x = end[0] - start[0]
    span_y = end[1] - start[1]
    length_squared = span_x**2 + span_y**2
    if length_squared == 0.0:
        # A walk of no distance, which happens: the builder is standing on the
        # pool it is about to build on.
        return (x - start[0]) ** 2 + (y - start[1]) ** 2
    along = ((x - start[0]) * span_x + (y - start[1]) * span_y) / length_squared
    along = min(1.0, max(0.0, along))
    return (x - (start[0] + along * span_x)) ** 2 + (y - (start[1] + along * span_y)) ** 2


__all__ = ["reach_of", "route_is_exposed"]
