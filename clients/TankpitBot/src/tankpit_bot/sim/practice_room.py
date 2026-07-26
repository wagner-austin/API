"""Practice-room driver: certified bot minds for sim sessions.

Bridges the certified practice-bot MODEL (``sim/bot_policy`` — the
module the ``bot-return-fire`` / ``bot-reactivation`` shadow laws
price against the archive) into a running sim session, replacing the
deterministic ``sim/opponent`` harness when fidelity matters more
than a scripted kill path. The driver owns the roster's policy
states, notes hits from each tick's emission batch exactly the way
the live wire reveals them (a 0x53 whose target tile holds a tank),
and queues each bot's next-tick decision.

The roster is a REAL practice-room state: the full 36-bot layout
(ids 500-535, 9 per team, ranks 0-1) lifted from an archive capture
by ``analysis_scripts/mine_practice_roster.py`` and shipped in
``sim.world_seed.PRACTICE_LAYOUTS`` — bots at their actually
observed map positions, so a session means finding fights across
the whole field, not a scripted clearing.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.bot_policy import (
    PracticeBotStateDict,
    decide_practice_bot,
    make_practice_bot_state,
    note_hit_for_team_aggro,
    note_hit_on_bot,
)
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.spawn import find_open_tile_near
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank


class PracticeRoomDriver:
    """Owns roster policy states and drives them each tick."""

    def __init__(
        self,
        world: SimWorldDict,
        terrain: TerrainMapProtocol,
        client_id: int,
        roster: tuple[tuple[int, int, int, int, int], ...],
    ) -> None:
        """Seed the roster into the world and initialize states.

        Each bot lands on the nearest open tile to its mined
        ``(x, y)`` position. Mined positions are NOT always passable
        ground: an archive snapshot can catch a bot afloat on a ferry
        (layout bot-20260706-223721 holds tank 511 at (38,1), open
        sea, nearest land 11 tiles away) — the sim seeds no ferry
        there, so the bot lands on the nearest coast instead. Bots
        boot at their rank's full fuel — the reactivation law's
        full-tank state, the only fuel level the archive pins for a
        bot at a known moment.

        Args:
            world: Simulated world (roster tanks added).
            terrain: Static terrain for the placement search.
            client_id: The connected client's tank id (never seeded).
            roster: ``(tank_id, team, rank, x, y)`` rows, absolute
                map positions (see ``sim.world_seed.PRACTICE_LAYOUTS``).

        Raises:
            RuntimeError: If no open tile exists within the search
                radius of a mined position — a layout that far at sea
                is bad data, not a placement problem.
        """
        del client_id
        self.states: dict[int, PracticeBotStateDict] = {}
        for tank_id, team, rank, seed_x, seed_y in roster:
            landing = find_open_tile_near(
                world, terrain, seed_x, seed_y, world["tick"], min_radius=0, max_radius=16
            )
            if landing is None:
                raise RuntimeError(
                    f"practice roster tank {tank_id} at ({seed_x},{seed_y}) has no "
                    "open tile within 16 — bad layout data"
                )
            world["tanks"][tank_id] = make_sim_tank(
                tank_id, team, rank, landing[0], landing[1], fuel_capacity(rank)
            )
            self.states[tank_id] = make_practice_bot_state()

    def roster_ids(self) -> frozenset[int]:
        """Return the roster's tank ids (for the server's corpse hook).

        Returns:
            The ids seeded by this driver.
        """
        return frozenset(self.states)

    def note_batch(self, world: SimWorldDict, batch: list[BinaryMessage]) -> None:
        """Note every hit the tick's emissions reveal.

        Mirrors the wire semantics the shadow law judges: a 0x53
        whose target tile holds a living tank is a hit on that tank —
        the victim queues its return and sighted teammates ignite
        (``note_hit_for_team_aggro``).

        Args:
            world: Simulated world.
            batch: The tick's emitted messages.
        """
        for message in batch:
            if message["msg_type"] != 0x53:
                continue
            shooter = world["tanks"].get(message["shooter_id"])
            if shooter is None:
                continue
            target = (message["target_x"], message["target_y"])
            for tank_id, tank in world["tanks"].items():
                if tank_id == message["shooter_id"] or not tank["alive"]:
                    continue
                if (tank["x"], tank["y"]) != target:
                    continue
                state = self.states.get(tank_id)
                if state is not None:
                    note_hit_on_bot(state, shooter["x"], shooter["y"])
                note_hit_for_team_aggro(world, self.states, tank_id, message["shooter_id"])

    def decide_all(
        self,
        world: SimWorldDict,
        terrain: TerrainMapProtocol,
    ) -> list[tuple[int, ClientCommandDict]]:
        """Collect every roster bot's command for the coming tick.

        Args:
            world: Simulated world.
            terrain: Terrain for escape-tile searches.

        Returns:
            ``(bot_id, command)`` pairs, holds omitted.
        """
        decisions: list[tuple[int, ClientCommandDict]] = []
        for bot_id, state in self.states.items():
            command = decide_practice_bot(state, world, terrain, bot_id)
            if command is not None:
                decisions.append((bot_id, command))
        return decisions


__all__ = [
    "PracticeRoomDriver",
]
