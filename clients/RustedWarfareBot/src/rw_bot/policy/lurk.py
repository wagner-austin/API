"""Keep cheap intruders alive at the enemy's door, so its armies stay home.

The decompiled AI recalls its attack groups to "defending base" for 500
ticks whenever an intruder stands in one of its base zones, and the recall
branch runs *before* the attack branch ([[ai-opponent-strategy]]). The raid
verb already exploits this -- and pays for each recall with the raiders'
lives, because a raid party fights until it is confirmed dead. Forty-one
Impossible matches say that trade loses: the leash is re-armed a handful of
times and then the leash-holders are gone ([[policy-raid]]).

A lurker is the same intrusion without the death. It walks to the rim of
the enemy's base zone -- inside the 420-radius circle that counts as
intrusion, outside the ~310 its longest static defence reaches -- stands
there re-arming the recall, and **retreats the moment anything hostile
comes near**, then walks back when the air is clear. The
unit never fights, so the only way the opponent silences the leash is to
chase a scout with something faster than a scout, which nothing it fields
is. One surviving lurker cycling in and out is a standing order for the
enemy's armies to stay home.

Scouts carry the job for the same reasons they carry patrol: the fastest
unit on the roster, cheap enough to lose, and armed with nothing anyone
waits for. Lurkers are stripped from the army exactly as the patrol scout
is -- counted into a wave they would march into the fight and die as one
more trickle ([[policy-combat]]).

Pure: samples in, move orders out. The channel is the campaign's.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.rush import mirror_point
from rw_bot.policy.scouting import SCOUT_TYPE
from rw_bot.policy.siting import find_anchor
from rw_bot.wire.command import MoveOrder, move_order
from rw_bot.wire.state import Entity, Sample

#: How close a hostile may come before a lurker runs.
#:
#: Above every ground unit's attack range in the shipped roster (the heavy
#: tank's 160 is the longest a chaser carries), so the retreat begins before
#: the first shot rather than after it ([[mechanics-combat-profile]]).
RETREAT_RADIUS: Final = 220.0

#: How far apart the loiter posts stand.
#:
#: Several lurkers on one point are one target for one wave; spread, each
#: needs its own chaser and each chase is its own recall.
LOITER_SPREAD: Final = 140.0

#: How far short of the enemy start the posts stand.
#:
#: The AI's home base zone is a 420-radius circle and its longest static
#: defence reaches ~310, so the band between is where a lurker counts as
#: an intruder without standing in anything's reach. The first live probe
#: posted lurkers at the centre of a base ringed by seventeen AA and
#: fourteen gun turrets, and bought eleven replacement scouts for it
#: ([[engine-ai-zones]], [[ai-opponent-strategy]]).
POST_STANDOFF: Final = 380.0

#: How far a threatened lurker falls back before checking again.
#:
#: Far enough that the chaser's reach is fully cleared next sample, near
#: enough that the walk back re-arms the recall quickly.
RETREAT_STEP: Final = 400.0


class Lurker:
    """Runs the loiter-and-retreat cycle for a handful of scouts.

    Attributes are internal: which unit holds which post, and which mode it
    was last ordered into, so an order is sent when the answer changes
    rather than every sample -- the engine runs the newest waypoint, and a
    stream of identical moves is noise in the run log.
    """

    def __init__(self) -> None:
        self._modes: dict[int, str] = {}

    def need(self, sample: Sample, wanted: int) -> tuple[str, ...]:
        """Return the scouts production should add for the lurk line.

        Args:
            sample: One observation of the world.
            wanted: Lurkers the doctrine keeps alive.

        Returns:
            One ``scout`` entry per missing lurker, for the composition.
        """
        alive = sum(
            1
            for entity in sample["entities"]
            if entity["mine"] and entity["complete"] and entity["type_name"] == SCOUT_TYPE
        )
        return (SCOUT_TYPE,) * max(0, wanted - alive)

    def orders(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        wanted: int,
        skip: int = 0,
    ) -> tuple[MoveOrder, ...]:
        """Order every lurker toward its post, or away from its chaser.

        Args:
            sample: One observation of the world.
            catalogue: Unit stats by type name, for the anchor.
            wanted: Lurkers the doctrine keeps alive.
            skip: Scouts allotted to other verbs before this one -- the
                patrol takes the first, the lurk line the next block, so
                the three scout verbs never fight over one unit.

        Returns:
            Move orders for lurkers whose mode changed this sample.
        """
        post = mirror_point(sample, catalogue)
        anchor = find_anchor(sample, catalogue)
        if post is None or anchor is None or wanted <= 0:
            return ()
        rim = _rim_of(post, (anchor["x"], anchor["y"]))
        hostiles = [e for e in sample["entities"] if e["hostile"]]
        lurkers = [
            entity
            for entity in sample["entities"]
            if entity["mine"] and entity["complete"] and entity["type_name"] == SCOUT_TYPE
        ][skip : skip + wanted]
        self._modes = {e["unit_id"]: self._modes.get(e["unit_id"], "") for e in lurkers}
        orders: list[MoveOrder] = []
        for slot, lurker in enumerate(lurkers):
            chaser = nearest_within(lurker, hostiles, RETREAT_RADIUS)
            if chaser is None:
                mode = f"loiter:{slot}"
                target = _post_of(rim, slot)
            else:
                # Straight away from the chaser, not toward home: a retreat
                # through the enemy base's far side is still a retreat, and
                # the shortest safe line is the one the chaser defines.
                mode = f"flee:{chaser['unit_id']}"
                target = away_from(lurker, chaser)
            if self._modes.get(lurker["unit_id"]) == mode:
                continue
            self._modes[lurker["unit_id"]] = mode
            orders.append(move_order(unit_id=lurker["unit_id"], x=target[0], y=target[1]))
        return tuple(orders)


def _rim_of(post: tuple[float, float], home: tuple[float, float]) -> tuple[float, float]:
    """Return the zone-rim point on the line from the enemy start home.

    Inside the 420-radius base zone, outside every static defence's reach:
    the standoff band is where the intrusion is free ([[engine-ai-zones]]).

    Args:
        post: The estimated enemy start.
        home: Our own anchor's position.

    Returns:
        The rim point, or the midpoint when the two bases stand closer
        than the standoff itself.
    """
    dx = home[0] - post[0]
    dy = home[1] - post[1]
    length = math.hypot(dx, dy)
    if length <= POST_STANDOFF:
        return ((post[0] + home[0]) / 2.0, (post[1] + home[1]) / 2.0)
    return (
        post[0] + dx / length * POST_STANDOFF,
        post[1] + dy / length * POST_STANDOFF,
    )


def _post_of(post: tuple[float, float], slot: int) -> tuple[float, float]:
    """Return one lurker's own loiter point.

    Args:
        post: The rim point the line of posts is centred on.
        slot: The lurker's index among its peers.

    Returns:
        The post, offset so peers do not stack.
    """
    offsets = ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.0, 1.0))
    dx, dy = offsets[slot % len(offsets)]
    return (post[0] + dx * LOITER_SPREAD, post[1] + dy * LOITER_SPREAD)


def nearest_within(lurker: Entity, hostiles: list[Entity], radius: float) -> Entity | None:
    """Return the closest hostile inside the radius, if any.

    Args:
        lurker: The unit checking its surroundings.
        hostiles: Every hostile entity visible.
        radius: How close counts as threatening.

    Returns:
        The nearest threatening hostile, or None.
    """
    best: Entity | None = None
    best_range = radius * radius
    for hostile in hostiles:
        dx = hostile["x"] - lurker["x"]
        dy = hostile["y"] - lurker["y"]
        squared = dx * dx + dy * dy
        if squared <= best_range:
            best = hostile
            best_range = squared
    return best


def away_from(lurker: Entity, chaser: Entity) -> tuple[float, float]:
    """Return the point one retreat step directly away from the chaser.

    Args:
        lurker: The unit retreating.
        chaser: The hostile it retreats from.

    Returns:
        The retreat waypoint.
    """
    dx = lurker["x"] - chaser["x"]
    dy = lurker["y"] - chaser["y"]
    length = math.hypot(dx, dy)
    if length == 0.0:
        # Standing on the chaser: any direction beats none.
        return (lurker["x"] + RETREAT_STEP, lurker["y"])
    return (
        lurker["x"] + dx / length * RETREAT_STEP,
        lurker["y"] + dy / length * RETREAT_STEP,
    )


__all__ = [
    "LOITER_SPREAD",
    "POST_STANDOFF",
    "RETREAT_RADIUS",
    "RETREAT_STEP",
    "Lurker",
    "away_from",
    "nearest_within",
]
