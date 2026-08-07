"""Law 9 — room churn: other players joining and leaving mid-session.

The sim's roster was fixed at boot, so the room never gained or lost a
player and the archived 0x28 TankEntry / 0x29 TankExit families had no
counterpart at all ([[session-state-deglobalisation]]).

The law, measured across all 285 archived sessions:

* Churn is RARE. Sixteen entries and fifteen exits in the whole
  archive, in 16 of the 285 sessions — about one arrival per 1,700
  ticks at the median 95-tick session length. A soak sees one when it
  runs long enough to deserve one, which is the point.
* An entry carries NO live position: 15 of the 16 report ``(0, 0)``.
  The joining tank is not in the client's view, and the server says so
  by reporting nothing rather than a tile.
* Entries arrive at rank 0 or 1 (7 and 9) — new or nearly-new tanks.
* Every one of the ten paired exits is ``was_silent=False,
  was_eliminated=False``: the player LEFT, and the client is told
  plainly. Nobody in the archive was ever kicked.
* Visits ran 3 to 446 ticks, median 59.

What is NOT modelled: the seven exits with no matching entry and the
five entries with no matching exit. Those are players who were already
in the room when we joined, or still there when we left — the sim's
visitor arrives and leaves inside one session because that is the only
span it can observe.
"""

from __future__ import annotations

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.protocol.types import BinaryMessage, TankEntryDict, TankExitDict
from tankpit_bot.sim.spawn import find_open_tile
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank

VISITOR_ARRIVAL_PERIOD_TICKS = 1692
"""Ticks between arrivals — 16 entries over 285 sessions of 95 ticks."""

VISITOR_STAY_TICKS = 59
"""How long a visitor stays: the median of the ten paired visits."""

VISITOR_ENTRY_X = 0
VISITOR_ENTRY_Y = 0
"""The position an entry reports: none. 15 of 16 archived entries
carry ``(0, 0)`` — the joining tank is not in the client's view."""

VISITOR_ID_BASE = 3000
"""Visitor ids start past every seeded roster and scripted opponent."""

_VISITOR_TEAM_CYCLE: tuple[int, ...] = (0, 3, 1, 2)
"""The archived entries' teams — all four colours, no skew to model."""

_VISITOR_RANK_CYCLE: tuple[int, ...] = (1, 0, 1, 0, 1)
"""Ranks 1 and 0, in the archive's 9-to-7 proportion."""


class RoomChurn:
    """One room's arrivals and departures.

    Holds no world of its own: :meth:`advance` is handed the world it
    changes, the same way the ferry drift is.
    """

    def __init__(self) -> None:
        """Start an empty room-churn clock."""
        self.visitor_id: int | None = None
        """The visitor currently in the room, if any."""
        self.arrived_tick = 0
        """The tick the current visitor joined on."""
        self._admitted = 0

    def advance(
        self,
        world: SimWorldDict,
        terrain: TerrainMapProtocol,
        messages: list[BinaryMessage],
    ) -> None:
        """Admit or retire a visitor, if this tick is the one.

        Args:
            world: Simulated world (mutated).
            terrain: Static terrain, for the arrival's landing tile.
            messages: This tick's outgoing batch (appended).
        """
        tick = world["tick"]
        if self.visitor_id is not None:
            if tick - self.arrived_tick >= VISITOR_STAY_TICKS:
                self._retire(world, self.visitor_id, messages)
            return
        if tick == 0 or tick % VISITOR_ARRIVAL_PERIOD_TICKS != 0:
            return
        self._admit(world, terrain, messages)

    def _admit(
        self,
        world: SimWorldDict,
        terrain: TerrainMapProtocol,
        messages: list[BinaryMessage],
    ) -> None:
        """Bring one visitor into the room and announce it.

        Args:
            world: Simulated world (mutated).
            terrain: Static terrain, for the landing tile.
            messages: This tick's outgoing batch (appended).
        """
        landing = find_open_tile(world, terrain, world["tick"])
        if landing is None:
            return
        tank_id = VISITOR_ID_BASE + self._admitted
        team = _VISITOR_TEAM_CYCLE[self._admitted % len(_VISITOR_TEAM_CYCLE)]
        rank = _VISITOR_RANK_CYCLE[self._admitted % len(_VISITOR_RANK_CYCLE)]
        self._admitted += 1
        world["tanks"][tank_id] = make_sim_tank(
            tank_id, team, rank, landing[0], landing[1], fuel_capacity(rank)
        )
        self.visitor_id = tank_id
        self.arrived_tick = world["tick"]
        messages.append(
            TankEntryDict(
                msg_type=0x28,
                team=team,
                tank_id=tank_id,
                rank=rank,
                # Damage 3 (full) on every archived entry — a tank
                # joins at capacity.
                damage_state=3,
                score=0,
                x=VISITOR_ENTRY_X,
                y=VISITOR_ENTRY_Y,
            )
        )

    def _retire(self, world: SimWorldDict, tank_id: int, messages: list[BinaryMessage]) -> None:
        """Send the visitor home and announce it.

        A visitor killed during its stay still leaves this way: the
        corpse was already taken out of view by its 0x58, and the
        player departing is a separate fact the client is told
        ([[session-state-deglobalisation]]).

        Args:
            world: Simulated world (mutated).
            tank_id: The departing visitor.
            messages: This tick's outgoing batch (appended).
        """
        tank = world["tanks"].pop(tank_id)
        self.visitor_id = None
        messages.append(
            TankExitDict(
                msg_type=0x29,
                team=tank["team"],
                tank_id=tank_id,
                # Every archived exit is a plain departure, announced.
                was_silent=False,
                was_eliminated=False,
            )
        )


__all__ = [
    "VISITOR_ARRIVAL_PERIOD_TICKS",
    "VISITOR_ENTRY_X",
    "VISITOR_ENTRY_Y",
    "VISITOR_ID_BASE",
    "VISITOR_STAY_TICKS",
    "RoomChurn",
]
