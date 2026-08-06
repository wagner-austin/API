"""Cheap units scattered wide, because our placement is the enemy's aim.

The decompiled AI picks each attack group's target uniformly at random over
EVERY unit we own -- no fog term, no distance term, no worth term anywhere
in the eligibility chain ([[ai-opponent-strategy]]). That means the
distribution of its attacks is not something it decides; it is something we
lay out. A handful of scouts posted far from anything that matters are
extra tickets in its lottery, and every wave that draws one walks to an
empty flank, fights a unit that flees it, and spends its five-hundred-tick
commitment on ground we never needed.

The decoy is the lurker's cycle on our own side of the map: hold a post,
run from anything inside reach, come back when the air clears
([[policy-lurk]]). Posts flank and trail our base at fractions of the
base-to-base line, far enough from the economy that a misdirected wave
shoots nothing on arrival, spread enough that each draws its own chase.

Pure: samples in, move orders out. The channel is the campaign's.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.lurk import RETREAT_RADIUS, away_from, nearest_within
from rw_bot.policy.rush import mirror_point
from rw_bot.policy.scouting import SCOUT_TYPE
from rw_bot.policy.siting import find_anchor
from rw_bot.wire.command import MoveOrder, move_order
from rw_bot.wire.state import Sample

#: Where the decoys stand, as fractions of the base-to-base line.
#:
#: First figure is along the axis toward the enemy (negative is behind our
#: base), second is perpendicular to it. Flanks first because a wave drawn
#: sideways crosses the most extra ground; the rear pair catches waves that
#: would otherwise arrive at the economy from behind.
POSTS: Final = (
    (0.30, 0.55),
    (0.30, -0.55),
    (0.05, 0.70),
    (0.05, -0.70),
    (-0.15, 0.45),
    (-0.15, -0.45),
)


class Decoys:
    """Runs the post-and-flee cycle for the scatter line.

    The same shape as the lurker: which unit holds which post and the mode
    it was last ordered into, so an order is sent when the answer changes
    rather than every sample.
    """

    def __init__(self) -> None:
        self._modes: dict[int, str] = {}

    def orders(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        wanted: int,
        skip: int = 0,
    ) -> tuple[MoveOrder, ...]:
        """Order every decoy toward its post, or away from its chaser.

        Args:
            sample: One observation of the world.
            catalogue: Unit stats by type name, for the anchor.
            wanted: Decoys the doctrine keeps alive.
            skip: Scouts allotted to other verbs before this one -- the
                patrol takes the first, the lurk line the next block, and
                the scatter takes what follows.

        Returns:
            Move orders for decoys whose mode changed this sample.
        """
        goal = mirror_point(sample, catalogue)
        anchor = find_anchor(sample, catalogue)
        if goal is None or anchor is None or wanted <= 0:
            return ()
        posts = _posts_of((anchor["x"], anchor["y"]), goal)
        hostiles = [e for e in sample["entities"] if e["hostile"]]
        decoys = [
            entity
            for entity in sample["entities"]
            if entity["mine"] and entity["complete"] and entity["type_name"] == SCOUT_TYPE
        ][skip : skip + wanted]
        self._modes = {e["unit_id"]: self._modes.get(e["unit_id"], "") for e in decoys}
        orders: list[MoveOrder] = []
        for slot, decoy in enumerate(decoys):
            chaser = nearest_within(decoy, hostiles, RETREAT_RADIUS)
            if chaser is None:
                mode = f"post:{slot}"
                target = posts[slot % len(posts)]
            else:
                mode = f"flee:{chaser['unit_id']}"
                target = away_from(decoy, chaser)
            if self._modes.get(decoy["unit_id"]) == mode:
                continue
            self._modes[decoy["unit_id"]] = mode
            orders.append(move_order(unit_id=decoy["unit_id"], x=target[0], y=target[1]))
        return tuple(orders)


def scout_shortfall(sample: Sample, wanted: int) -> tuple[str, ...]:
    """Return the scouts production owes the scout-fed verbs together.

    One count for the patrol, the lurk line and the scatter combined,
    because each verb counting the whole roster against its own figure
    would leave every one of them satisfied by the others' scouts.

    Args:
        sample: One observation of the world.
        wanted: Scouts all three verbs need alive between them.

    Returns:
        One ``scout`` entry per missing scout, for the composition.
    """
    alive = sum(
        1
        for entity in sample["entities"]
        if entity["mine"] and entity["complete"] and entity["type_name"] == SCOUT_TYPE
    )
    return (SCOUT_TYPE,) * max(0, wanted - alive)


def _posts_of(
    anchor: tuple[float, float], goal: tuple[float, float]
) -> tuple[tuple[float, float], ...]:
    """Return the scatter posts for this map's geometry.

    Args:
        anchor: Our own base.
        goal: The estimated enemy start.

    Returns:
        One world point per entry of :data:`POSTS`.
    """
    dx = goal[0] - anchor[0]
    dy = goal[1] - anchor[1]
    length = math.hypot(dx, dy)
    if length == 0.0:
        return (anchor,)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    return tuple(
        (
            anchor[0] + (ux * fa + px * fp) * length,
            anchor[1] + (uy * fa + py * fp) * length,
        )
        for fa, fp in POSTS
    )


__all__ = ["POSTS", "Decoys", "scout_shortfall"]
