"""Practice-room driver: certified bot minds for sim sessions.

Bridges the certified practice-bot MODEL (``sim/bot_policy`` — the
module the ``bot-return-fire`` / ``bot-reactivation`` shadow laws
price against the archive) into a running sim session, replacing the
deterministic ``sim/opponent`` harness when fidelity matters more
than a scripted kill path. The driver owns the roster's policy
states, notes hits from each tick's emission batch exactly the way
the live wire reveals them (a 0x53 whose target tile holds a tank),
and queues each bot's next-tick decision.

The default roster mirrors a real practice-room encounter from the
client's perspective (client is team 2): three purple bots clustered
within sight of each other — so gang-up fire ignites when the client
engages one — plus one blue ally bot that assists the client's own
fights, the live blue-7 shape.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
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

PRACTICE_ROSTER: tuple[tuple[int, int, int, int, int], ...] = (
    (510, 1, 0, 8, 0),
    (511, 1, 0, 9, 1),
    (512, 1, 0, 8, -2),
    (524, 2, 0, 6, 1),
)
"""Default roster as (tank_id, team, rank, dx, dy) client-relative
offsets: three sighted purple recruits (the gang-up cluster) and one
blue ally (the assist shape). Ids follow the live 36-slot roster."""

_PRACTICE_BOT_FUEL = 800


class PracticeRoomDriver:
    """Owns roster policy states and drives them each tick."""

    def __init__(
        self,
        world: SimWorldDict,
        terrain: TerrainMapProtocol,
        client_id: int,
    ) -> None:
        """Seed the roster into the world and initialize states.

        Each bot lands on the nearest open tile to its offset from
        the client's spawn (a sealed offset leaves the bot exactly on
        it — the seed positions are chosen inside the arena clearing).

        Args:
            world: Simulated world (roster tanks added).
            terrain: Static terrain for the placement search.
            client_id: The connected client's tank id.
        """
        client = world["tanks"][client_id]
        self.states: dict[int, PracticeBotStateDict] = {}
        for tank_id, team, rank, dx, dy in PRACTICE_ROSTER:
            seed_x = client["x"] + dx
            seed_y = client["y"] + dy
            landing = find_open_tile_near(
                world, terrain, seed_x, seed_y, world["tick"], min_radius=0, max_radius=4
            )
            spot = landing if landing is not None else (seed_x, seed_y)
            world["tanks"][tank_id] = make_sim_tank(
                tank_id, team, rank, spot[0], spot[1], _PRACTICE_BOT_FUEL
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
    "PRACTICE_ROSTER",
    "PracticeRoomDriver",
]
