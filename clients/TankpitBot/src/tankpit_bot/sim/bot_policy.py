"""The practice-room game-bot policy, archive-mined 2026-07-24.

Unlike ``sim/opponent.py`` (a deterministic HARNESS aggressor,
explicitly outside the fidelity certification), this module is a
MODEL: every constant and predicate below was fitted on the full
capture archive (246 sessions, 12.5 bot-hours, 2,247 bot shots —
``analysis_scripts/mine_bot_policy.py``, [[enemy-bot-behavior]]
"Corpus-mined policy") and is re-judged against that archive by the
``bot-return-fire`` shadow law on every ``make shadow``.

The mined policy, certified for ranks 0-1 (the only bot ranks the
archive contains):

- bots are stationary (79 walk echoes in 12.5 hours, zero
  unexplained drifts) and never place mines;
- when hit, a bot returns exactly one ``weapon=0`` single
  (2,247/2,247) at the attacker's exact tile (98.7%) on the next
  2 s queue tick (latency mass 1.5-2.5 s);
- after the rank's hit threshold (modal 7 at recruit, 8 at private
  — Sigma's table, wire-corroborated), the bot teleports off.

Sim assumptions documented here, not certified: the teleport-off
DESTINATION is unmined (the sim uses a deterministic ring band
beyond the viewport radius), and the corporal threshold (9) is
guide-sourced only — no corporal+ bot exists in the archive.
"""

from __future__ import annotations

from typing import TypedDict

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.protocol.commands import CMD_MAP_TELEPORT, CMD_SHOOT
from tankpit_bot.sim.actions import VIEWPORT_RADIUS
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.spawn import find_open_tile_near
from tankpit_bot.sim.world import SimWorldDict

BOT_RETURN_WEAPON = 0
"""The only weapon practice bots ever fire (2,247/2,247 singles)."""

BOT_RETURN_WINDOW_MS = 3000
"""Return fire lands within this window of the provoking hit (96.2%
of archive bot shots; the latency mass is one 2 s queue tick)."""

BOT_TELEPORT_OFF_HITS: dict[int, int] = {0: 7, 1: 8, 2: 9}
"""Hits taken before the bot teleports off, by rank. Ranks 0-1 are
archive-corroborated modes (20/49 and 37/82 samples); rank 2 is
guide-sourced only (Sigma v3.4) — no corporal bot exists in the
archive."""

_ESCAPE_MIN_RADIUS = VIEWPORT_RADIUS + 4
_ESCAPE_MAX_RADIUS = VIEWPORT_RADIUS + 16


class PracticeBotStateDict(TypedDict):
    """Per-bot policy memory the harness threads between ticks.

    ``pending_return_*`` hold the attacker tile recorded when the bot
    was hit; the return single fires on the NEXT decision (the mined
    one-tick latency). ``hits_taken`` counts hits since the last
    teleport-off.
    """

    hits_taken: int
    has_pending_return: bool
    pending_return_x: int
    pending_return_y: int


def make_practice_bot_state() -> PracticeBotStateDict:
    """Create the initial (calm, unhit) policy state.

    Returns:
        A fresh state with no hits and no pending return.
    """
    return PracticeBotStateDict(
        hits_taken=0,
        has_pending_return=False,
        pending_return_x=0,
        pending_return_y=0,
    )


def note_hit_on_bot(state: PracticeBotStateDict, attacker_x: int, attacker_y: int) -> None:
    """Record one hit landing on the bot and queue the return single.

    Args:
        state: The bot's policy state (mutated in place).
        attacker_x: The attacker's tile x at hit time.
        attacker_y: The attacker's tile y at hit time.
    """
    state["hits_taken"] += 1
    state["has_pending_return"] = True
    state["pending_return_x"] = attacker_x
    state["pending_return_y"] = attacker_y


def teleport_off_threshold(rank: int) -> int:
    """Return the rank's hits-before-teleport-off threshold.

    Args:
        rank: The bot's rank (0-8).

    Returns:
        The threshold; ranks above the measured/guide table fall back
        to the highest known row (no such bot has ever been observed).
    """
    return BOT_TELEPORT_OFF_HITS.get(rank, BOT_TELEPORT_OFF_HITS[2])


def decide_practice_bot(
    state: PracticeBotStateDict,
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    bot_id: int,
) -> ClientCommandDict | None:
    """Decide the practice bot's command for the coming tick.

    Mined decision order: a bot over its rank's hit threshold
    teleports off (destination = deterministic ring band beyond the
    viewport radius — documented assumption); a bot with a pending
    return fires exactly one single at the recorded attacker tile;
    otherwise the bot holds still (the mined stationary default).

    Args:
        state: The bot's policy state (mutated in place).
        world: Simulated world.
        terrain: Terrain map for the escape-tile search.
        bot_id: The bot's tank id.

    Returns:
        The command to queue, or None (hold — the common case).
    """
    bot = world["tanks"][bot_id]
    if not bot["alive"]:
        return None
    if state["hits_taken"] >= teleport_off_threshold(bot["rank"]):
        landing = find_open_tile_near(
            world,
            terrain,
            bot["x"],
            bot["y"],
            world["tick"],
            min_radius=_ESCAPE_MIN_RADIUS,
            max_radius=_ESCAPE_MAX_RADIUS,
        )
        if landing is not None:
            state["hits_taken"] = 0
            state["has_pending_return"] = False
            return ClientCommandDict(
                kind="teleport",
                command=CMD_MAP_TELEPORT,
                x=landing[0],
                y=landing[1],
                target_id=0,
                slot=0,
            )
    if state["has_pending_return"]:
        state["has_pending_return"] = False
        return ClientCommandDict(
            kind="shoot",
            command=CMD_SHOOT,
            x=state["pending_return_x"],
            y=state["pending_return_y"],
            target_id=0,
            slot=0,
        )
    return None


MIN_RESPAWN_DISPLACEMENT = 24
"""Measured floor of the respawn displacement: every one of the 102
archive death→next-seen pairs sits at least this far (Chebyshev) from
the corpse; 70/102 exceed 96 tiles. Bots respawn far away, never in
place (user contract 2026-07-24 + archive sweep)."""

_MAP_SPAN = 256


def reactivate_practice_bot(world: SimWorldDict, terrain: TerrainMapProtocol, tank_id: int) -> None:
    """Reactivate a dead roster bot: same id, full fuel, FAR away.

    The archive-mined reactivation law (2026-07-24,
    [[enemy-bot-behavior]]): a killed practice bot returns when its
    corpse clears — 27 measured death→full-fuel pairs, gap moded at
    exactly the 22 s corpse window — at a DISTANT map location
    (102/102 measured pairs ≥ 24 tiles from the corpse). Fuel resets
    to the rank's capacity, so every derived emission (tier, map
    shade) is full-health without stored state. Sim assumption: the
    scatter point is tick/id-derived (deterministic), not the real
    server's placement distribution; a fully sealed scatter area
    falls back to reactivating in place.

    Args:
        world: Simulated world (mutated).
        terrain: Static terrain for the respawn-tile search.
        tank_id: The roster bot's fixed id.
    """
    tank = world["tanks"][tank_id]
    scatter_x = (tank_id * 73 + world["tick"] * 37) % _MAP_SPAN
    scatter_y = (tank_id * 151 + world["tick"] * 91) % _MAP_SPAN
    if max(abs(scatter_x - tank["x"]), abs(scatter_y - tank["y"])) < MIN_RESPAWN_DISPLACEMENT:
        scatter_x = (scatter_x + _MAP_SPAN // 2) % _MAP_SPAN
    landing = find_open_tile_near(
        world,
        terrain,
        scatter_x,
        scatter_y,
        world["tick"],
        min_radius=0,
        max_radius=24,
    )
    tank["alive"] = True
    tank["fuel"] = fuel_capacity(tank["rank"])
    if landing is not None:
        tank["x"] = landing[0]
        tank["y"] = landing[1]


__all__ = [
    "BOT_RETURN_WEAPON",
    "BOT_RETURN_WINDOW_MS",
    "BOT_TELEPORT_OFF_HITS",
    "MIN_RESPAWN_DISPLACEMENT",
    "PracticeBotStateDict",
    "decide_practice_bot",
    "make_practice_bot_state",
    "note_hit_on_bot",
    "reactivate_practice_bot",
    "teleport_off_threshold",
]
