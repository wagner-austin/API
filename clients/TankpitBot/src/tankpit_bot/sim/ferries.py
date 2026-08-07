"""Law 2c — autonomous ferry drift (wiki [[ferry-mechanics]]).

The sim modelled ferries as furniture: a static list of water tiles
that moved only when a tank rode one (``movement._update_ridden_ferry``).
``scenarios.py`` even documented the truth it did not implement —
"ferries DRIFT — the live map guarantees no boarding tile near a
floating container at any given instant" — so every ferry-doctrine
conclusion a sim run produced was drawn against one stationary ferry
at a hardcoded tile ([[session-state-deglobalisation]]).

The wire law, mined from the archive 2026-08-06 (205 terrain updates
across 407 sessions, chained per ferry by tile identity):

* A ferry move is ONE 0x4A frame carrying exactly two updates: the
  vacated tile set to 0 and the occupied tile set to 5. All 205 pairs
  have this shape; none has any other.  0 is the wire's "nothing
  here" value, so the client falls back to the static map underneath
  — water, which is what a vacated ferry tile is.
* Moves land on the 2 s tick. Of the 121 steps that chain back to a
  tile the same session had seen a ferry occupy, the gap is 1982 ms
  minimum and 2003 ms median — one tick — with a p75 of 8005 ms
  because a ferry idles between moves.
* One tile per move. The modal one-tick step is a single axial tile
  (18 samples) or a single diagonal (7). The larger one-tick steps in
  the corpus are RIDDEN ferries: a tank walking N tiles carries its
  ferry N tiles in one tick, which ``movement`` already models — so
  drift here is unridden ferries only, and never fights the rider for
  the same ferry in the same tick.

Direction is deterministic (tick- and index-derived, the scatter idiom
``bot_policy.reactivate_practice_bot`` uses) so a soak replays. A ferry
whose chosen neighbour is not water simply idles that tick, which is
also where the long observed gaps come from.

What is NOT claimed: the long-run trajectory. The archive bounds one
step, not one voyage — no session is long enough to say whether a real
ferry wanders off or circulates. The heading cycle here walks all eight
neighbours in turn, so an unobstructed ferry orbits its start rather
than escaping; that is a property of the chosen sequence, not a
measured law, and a longer capture is what would settle it.
"""

from __future__ import annotations

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.protocol.types import BinaryMessage, TerrainUpdateDict
from tankpit_bot.sim.world import SimWorldDict

MAP_SPAN = 256
"""Tile coordinates run 0-255; a heading off the edge is not a move."""

WIRE_TERRAIN_CLEARED = 0
"""The 0x4A value a vacated ferry tile reverts to (static map shows through)."""

WIRE_TERRAIN_FERRY = 5
"""The 0x4A value an occupied ferry tile carries (``TerrainType.FERRY``)."""

#: The eight neighbours a drifting ferry chooses between, in the fixed
#: order the tick-derived index selects from.
_DRIFT_HEADINGS: tuple[tuple[int, int], ...] = (
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
)


def _heading(tick: int, index: int) -> tuple[int, int]:
    """Pick one ferry's heading for one tick, deterministically.

    Args:
        tick: The world tick being resolved.
        index: The ferry's position in ``world["ferries"]``.

    Returns:
        The ``(dx, dy)`` step to attempt.
    """
    return _DRIFT_HEADINGS[(tick * 37 + index * 73) % len(_DRIFT_HEADINGS)]


def drift_ferries(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    messages: list[BinaryMessage],
) -> None:
    """Drift every unridden ferry one tile and emit each move's 0x4A.

    Args:
        world: Simulated world (ferries mutated in place).
        terrain: Static terrain of the world's field — a ferry may
            only drift onto WATER, so the shoreline bounds the walk
            without any extra bookkeeping.
        messages: This tick's outgoing batch (appended).
    """
    occupied = {(ferry["x"], ferry["y"]) for ferry in world["ferries"]}
    ridden = {(tank["x"], tank["y"]) for tank in world["tanks"].values() if tank["alive"]}
    for index, ferry in enumerate(world["ferries"]):
        if (ferry["x"], ferry["y"]) in ridden:
            # Ridden: the rider drives it, and movement already moved
            # it this tick. Drifting it again would double the step.
            continue
        dx, dy = _heading(world["tick"], index)
        target_x, target_y = ferry["x"] + dx, ferry["y"] + dy
        if not (0 <= target_x < MAP_SPAN and 0 <= target_y < MAP_SPAN):
            continue
        target = (target_x, target_y)
        if target in occupied or terrain.get_terrain(target_x, target_y) != terrain.WATER:
            continue
        occupied.discard((ferry["x"], ferry["y"]))
        occupied.add(target)
        messages.append(
            TerrainUpdateDict(
                msg_type=0x4A,
                updates=[
                    (ferry["x"], ferry["y"], WIRE_TERRAIN_CLEARED),
                    (target[0], target[1], WIRE_TERRAIN_FERRY),
                ],
            )
        )
        ferry["x"], ferry["y"] = target


__all__ = [
    "MAP_SPAN",
    "WIRE_TERRAIN_CLEARED",
    "WIRE_TERRAIN_FERRY",
    "drift_ferries",
]
