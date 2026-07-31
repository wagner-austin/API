"""Keeping one pair of eyes moving, so intel arrives before the fight does.

The community corpus is unambiguous about what this is worth: continuous
scouting is named as the difference between winning and losing, because every
counter is cheap when chosen in advance ([[community-play-strategies]]). The
bot has never scouted; its knowledge of the opponent has been whatever walked
into its own vision, which is to say whatever was already shooting at it.

This module is the smallest scouting that feeds the existing machinery: one
scout, kept alive by the composition exactly as the builder is, walking a
circuit of the map's resource pools farthest-first -- the far pools are the
opponent's side, and pools are where anything worth seeing stands
([[mechanics-resource-pools]]). What the scout sees goes into
:class:`~rw_bot.policy.intel.Intel` like every other sighting; this module
only decides where it walks.

The runner holds the same kind of memory the wave controller does: which leg
of the circuit its scout is on, and whether that leg's order has been sent --
the engine runs a waypoint until it is replaced, so re-sending every sample
would reset the walk ([[issuing-orders]]).
"""

from __future__ import annotations

from collections.abc import Mapping

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.combat import RALLY_RADIUS
from rw_bot.policy.siting import find_anchor
from rw_bot.wire.command import MoveOrder, move_order
from rw_bot.wire.state import Entity, ResourcePool, Sample

#: The unit that does the looking.
#:
#: Produced directly by the Command Center, so keeping one alive never needs
#: a factory the plan did not already build.
SCOUT_TYPE = "scout"

#: Builders that must exist before a scout may compete for production.
#:
#: The same floor the expander applies before diverting a worker to
#: throughput, borrowed for the same reason: with fewer, every scout the
#: Command Center makes is a builder it does not ([[policy-production]]).
_WORKER_FLOOR = 2


def _scout_of(sample: Sample) -> Entity | None:
    """Return the first finished scout the player owns, if any.

    Args:
        sample: One observation of the world.

    Returns:
        The scout, or None when none is alive.
    """
    for entity in sample["entities"]:
        if entity["mine"] and entity["complete"] and entity["type_name"] == SCOUT_TYPE:
            return entity
    return None


class ScoutRunner:
    """Walks one scout around the map's pools, farthest-first.

    Attributes:
        legs_walked: Circuit legs completed, for the run log.
    """

    def __init__(self) -> None:
        """Open a runner with no scout and no route."""
        self.legs_walked = 0
        self._riding = 0
        self._leg = 0
        self._sent = False

    def need(self, sample: Sample, workers: int) -> tuple[str, ...]:
        """Return the scout the composition should carry, if one is wanted.

        The same shape as the builder's rule: a scout joins the composition
        while none is alive, and leaves it entirely once one is
        ([[policy-production]]). One is the number -- the intel memory does
        not get better with a second pair of eyes on the same circuit.

        **The scout yields to the economy, and that was measured rather than
        assumed.** V1 asked for a scout whenever none was alive, and a scout
        on this circuit dies often -- so it was permanently the
        furthest-behind share, and the Command Center, the one producer of
        builders, spent whole matches replacing scouts instead. One match ran
        its economy on a single builder: the disease the worker rules exist
        to prevent, reintroduced by the eyes. Two workers is the same floor
        the expander uses before diverting labour, for the same reason.

        Args:
            sample: One observation of the world.
            workers: Builders owned, as the workforce counts them.

        Returns:
            ``("scout",)`` when none is alive and the economy can spare the
            production, empty otherwise.
        """
        if workers < _WORKER_FLOOR:
            return ()
        return () if _scout_of(sample) is not None else (SCOUT_TYPE,)

    def patrol(self, sample: Sample, catalogue: Mapping[str, UnitStats]) -> tuple[MoveOrder, ...]:
        """Advance the circuit by at most one order.

        The route is the pool list sorted farthest-from-anchor first, ties on
        tile coordinates so two runs of one seed walk identically. A dead
        scout resets the circuit: its replacement starts from the far pools
        again, because the far side is where the intel is.

        Args:
            sample: One observation of the world.
            catalogue: Unit stats by type name, for the anchor.

        Returns:
            The move order to send, or nothing while the current leg is still
            being walked.
        """
        scout = _scout_of(sample)
        if scout is None:
            self._riding = 0
            self._sent = False
            return ()
        anchor = find_anchor(sample, catalogue)
        pools = sample["pools"]
        if anchor is None or not pools:
            return ()

        def farness(pool: ResourcePool) -> tuple[float, int, int]:
            dx = pool["x"] - anchor["x"]
            dy = pool["y"] - anchor["y"]
            return (-(dx * dx + dy * dy), pool["tile_x"], pool["tile_y"])

        route = sorted(pools, key=farness)
        if scout["unit_id"] != self._riding:
            self._riding = scout["unit_id"]
            self._leg = 0
            self._sent = False

        target = route[self._leg % len(route)]
        arrived = (scout["x"] - target["x"]) ** 2 + (
            scout["y"] - target["y"]
        ) ** 2 <= RALLY_RADIUS**2
        if arrived:
            self._leg += 1
            self.legs_walked += 1
            self._sent = False
            target = route[self._leg % len(route)]
        if self._sent:
            return ()
        self._sent = True
        return (move_order(unit_id=scout["unit_id"], x=target["x"], y=target["y"]),)


__all__ = ["SCOUT_TYPE", "ScoutRunner"]
