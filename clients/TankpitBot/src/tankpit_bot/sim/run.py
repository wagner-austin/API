"""``tankpit-sim-run`` — a full production-bot session against the sim.

The step-(e) entry point, promoted to a CLI: the REAL ``Bot`` and the
REAL ``_tick_once`` play a timed session against :class:`SimServer`
on the REAL field terrain (``field01_r.gif`` — actual mountains and
water shape the router, shot clipping, and teleport displacement),
with the scripted opponent returning fire. Artifacts land where the
standard tooling can read them:

- ``runs/probe/latest.sim.log`` / ``latest.sim.events.jsonl`` — the
  probe-mode runtime logging channel (the live ``runs/bot`` archive
  stays reserved for real-server evidence);
- ``runs/sim/sim-<stamp>.capture_session.json`` — the recorded wire,
  standard ``CaptureSession`` shape (``tankpit-audit --runs-dir``
  can price it);
- ``runs/sim/sim-<stamp>.world.json`` — the sim world's final state.

No live server, no browser, no fuel spent: free soaks.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_loop import _tick_once
from tankpit_bot.runtime_artifacts import make_run_stamp
from tankpit_bot.runtime_logging import configure_probe_runtime_logging
from tankpit_bot.sim.opponent import decide_opponent, maybe_revive_opponent
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.session import SimCDPSession, build_capture_session, deliver_batch
from tankpit_bot.sim.world import (
    SimContainerDict,
    SimEquipmentDict,
    SimWorldDict,
    encode_sim_world,
    make_sim_tank,
    make_sim_world,
)
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.sniffer.xor import build_global_xor_table, get_global_xor_table, reset_xor_state
from tankpit_bot.types import encode_capture_session

log = get_logger(__name__)

SIM_MAGIC = "simmagic"
SIM_CLIENT_ID = 9
SIM_ENEMY_ID = 11
SIM_FIELD = "field01_r.gif"
_SIM_DIR = Path("runs") / "sim"
_DEFAULT_ROUNDS = 150

# The default arena is the most open ground on field01: a fully
# passable 21x21 clearing centered on (216, 108) (verified against
# the real GIF — the naive (100, 100) region is coastal, and six of
# its would-be container tiles sit in water).
_ARENA_X = 216
_ARENA_Y = 108
_DEFAULT_FUEL_CONTAINERS: tuple[tuple[int, int, int], ...] = (
    (219, 108, 300),
    (213, 112, 400),
    (222, 103, 400),
    (210, 105, 400),
    (225, 112, 400),
    (216, 116, 400),
    (208, 111, 300),
    (221, 100, 300),
)
# Enough equipment that the bot can actually rebuild its radar
# buffer: collect hops burn extras on scan-on-landing roughly as fast
# as one container's radar stack refills them, so a sparse seeding
# strands the session below the hunt threshold (first clearing run:
# exit no_productive_collect at round 39 with all three containers
# drained).
_DEFAULT_EQUIPMENT: tuple[tuple[int, int], ...] = (
    (219, 111),
    (214, 106),
    (210, 114),
    (224, 105),
    (212, 100),
    (218, 115),
    (208, 104),
    (223, 113),
)


class SimRunResultDict(TypedDict):
    """One finished sim session, summarized.

    ``exit_reason`` is ``"rounds_exhausted"`` when the session played
    every requested round, else the production ``SessionExitError``
    reason the bot actually raised.
    """

    stamp: str
    rounds_played: int
    exit_reason: str
    exit_detail: str
    commands_sent: int
    client_fuel: int
    client_alive: bool
    enemy_alive: bool
    capture_path: str
    world_path: str
    events_path: str


def make_default_sim_world() -> SimWorldDict:
    """Build the default scenario: a sustainable fight-and-collect map.

    The client spawns rank 1 in the field01 clearing combat-ready
    (the fight runs first; the extras it burns scanning pull the
    equipment-collection paths in afterwards), the armed rank-8
    opponent holds ten tiles east, and enough fuel and equipment is
    on the ground that a long session stays productive. A finite
    world still drains eventually — container respawns are an
    uncertified gap — so the production ``no_productive_collect``
    exit is this scenario's natural old age.

    Returns:
        The seeded world.
    """
    world = make_sim_world(SIM_FIELD)
    world["tanks"][SIM_CLIENT_ID] = make_sim_tank(SIM_CLIENT_ID, 2, 1, _ARENA_X, _ARENA_Y, 1100)
    world["tanks"][SIM_CLIENT_ID]["counts"] = [25, 25, 25, 25, 25]
    # A winnable fight, not an execution. The bot FIGHTS WITH ARMOR
    # OFF by policy (desired_equipment is dual/homing/radar — the
    # scenario-harness baseline pins armor disabled), so its
    # effective HP is its fuel — and an out-of-ammo opponent still
    # lands unlimited 45-fuel singles, so ammo does not cap damage.
    # A 500-fuel opponent (six kill-hits needed) lets the client win
    # the deterministic knife fight while still eating real return
    # fire; heavier seedings killed the bot by round ~20 every run
    # (that ``deactivated`` ending is itself covered by the exit
    # law's tests).
    world["tanks"][SIM_ENEMY_ID] = make_sim_tank(SIM_ENEMY_ID, 1, 8, _ARENA_X + 10, _ARENA_Y, 500)
    world["tanks"][SIM_ENEMY_ID]["counts"] = [0, 4, 0, 2, 3]
    for x, y, volume in _DEFAULT_FUEL_CONTAINERS:
        world["containers"].append(SimContainerDict(x=x, y=y, volume=volume))
    for x, y in _DEFAULT_EQUIPMENT:
        world["equipment"].append(SimEquipmentDict(x=x, y=y))
    return world


def _require_seeds_passable(world: SimWorldDict, terrain: _test_hooks.TerrainMapProtocol) -> None:
    """Reject a scenario whose seeds sit on rock or water — loudly.

    The first real-terrain run (2026-07-22) seeded the coastal
    (100, 100) region and drowned six containers: the bot starved
    among dots it could never reach. Nothing on the real map spawns
    on impassable ground, so a seed there is a harness bug, not a
    world.

    Args:
        world: The seeded world.
        terrain: The loaded field terrain.

    Raises:
        RuntimeError: Listing every impassable seed.
    """
    offenders: list[str] = []
    for tank_id, tank in sorted(world["tanks"].items()):
        if not terrain.is_passable(tank["x"], tank["y"]):
            offenders.append(f"tank {tank_id} at ({tank['x']},{tank['y']})")
    for container in world["containers"]:
        if not terrain.is_passable(container["x"], container["y"]):
            offenders.append(f"fuel container at ({container['x']},{container['y']})")
    for equipment in world["equipment"]:
        if not terrain.is_passable(equipment["x"], equipment["y"]):
            offenders.append(f"equipment at ({equipment['x']},{equipment['y']})")
    if offenders:
        raise RuntimeError(
            "impassable scenario seeds on " + world["field"] + ": " + "; ".join(offenders)
        )


def _boot(world: SimWorldDict) -> tuple[Bot, SimServer, SimCDPSession]:
    """Wire a real Bot to the sim over the CDP seam.

    Args:
        world: The seeded world the server will own.

    Returns:
        The bot, the server, and the seam link, with the join
        handshake already delivered.

    Raises:
        RuntimeError: If the XOR static key or the field terrain GIF
            is unavailable, or a scenario seed sits on impassable
            ground — a sim run needs all three right, loudly.
    """
    reset_world_state()
    reset_xor_state()
    build_global_xor_table(SIM_MAGIC)
    table = get_global_xor_table()
    if table is None:
        raise RuntimeError("XOR static key unavailable — cannot boot the sim session")
    gif_path = Path(world["field"])
    if not _test_hooks.path_exists(gif_path):
        raise RuntimeError(f"terrain GIF {gif_path} not found — run `make download-fields` first")
    terrain = _test_hooks.load_terrain_map(gif_path)
    _require_seeds_passable(world, terrain)
    server = SimServer(world, terrain, client_id=SIM_CLIENT_ID)
    bot = Bot("https://sim.tankpit.local/", headless=True)
    bot._magic = SIM_MAGIC
    bot._on_magic_captured(SIM_MAGIC)
    link = SimCDPSession(server, table)
    bot._cdp = link
    deliver_batch(bot._cdp_message_buffer, server.handshake(), link)
    return bot, server, link


def run_sim_session(
    rounds: int,
    *,
    opponent: bool = True,
    stamp: str | None = None,
) -> SimRunResultDict:
    """Play one production-bot session against the sim and archive it.

    Args:
        rounds: Maximum server ticks to play.
        opponent: Whether the scripted opponent returns fire.
        stamp: Optional archive stamp override for deterministic tests.

    Returns:
        The session summary (also written to the artifacts).

    Raises:
        RuntimeError: If the static key or terrain is unavailable.
    """
    run_stamp = stamp if stamp is not None else make_run_stamp()
    artifacts = configure_probe_runtime_logging("sim", run_stamp)
    world = make_default_sim_world()
    bot, server, link = _boot(world)
    exit_reason = "rounds_exhausted"
    exit_detail = ""
    played = 0
    enemy_id = SIM_ENEMY_ID
    try:
        for _ in range(rounds):
            _tick_once(bot)
            if opponent:
                enemy_id = maybe_revive_opponent(server, enemy_id, SIM_CLIENT_ID)
                opponent_command = decide_opponent(server.world, enemy_id, SIM_CLIENT_ID)
                if opponent_command is not None:
                    server.queue_command(enemy_id, opponent_command)
            deliver_batch(bot._cdp_message_buffer, server.advance_tick(), link)
            played += 1
    except SessionExitError as error:
        exit_reason = error.reason
        exit_detail = error.detail
        log.info(
            "sim session ended by the production exit path: %s (%s)",
            error.reason,
            error.detail,
        )
    capture_path = _SIM_DIR / f"sim-{run_stamp}.capture_session.json"
    world_path = _SIM_DIR / f"sim-{run_stamp}.world.json"
    session = build_capture_session(link, SIM_MAGIC, f"sim-{run_stamp}")
    _test_hooks.write_text(capture_path, dump_json_str(encode_capture_session(session)))
    _test_hooks.write_text(world_path, dump_json_str(encode_sim_world(server.world)))
    client = server.world["tanks"][SIM_CLIENT_ID]
    enemy = server.world["tanks"][enemy_id]
    return SimRunResultDict(
        stamp=run_stamp,
        rounds_played=played,
        exit_reason=exit_reason,
        exit_detail=exit_detail,
        commands_sent=len(link.sent_commands),
        client_fuel=client["fuel"],
        client_alive=client["alive"],
        enemy_alive=enemy["alive"],
        capture_path=str(capture_path),
        world_path=str(world_path),
        events_path=artifacts["latest_events_path"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``make sim-run``.

    Args:
        argv: Command-line arguments (``--rounds N``,
            ``--no-opponent``, ``--stamp S``). Uses ``sys.argv[1:]``
            when None.

    Returns:
        Process exit code (0 — a session that ends via the production
        exit path is still a successful sim run).
    """
    args = list(argv) if argv is not None else list(sys.argv[1:])
    rounds = _DEFAULT_ROUNDS
    opponent = True
    stamp: str | None = None
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--rounds" and index + 1 < len(args):
            rounds = int(args[index + 1])
            index += 2
        elif token == "--no-opponent":
            opponent = False
            index += 1
        elif token == "--stamp" and index + 1 < len(args):
            stamp = args[index + 1]
            index += 2
        else:
            index += 1
    result = run_sim_session(rounds, opponent=opponent, stamp=stamp)
    sys.stdout.write(
        f"sim session {result['stamp']}: {result['rounds_played']}/{rounds} rounds, "
        f"{result['commands_sent']} commands, exit={result['exit_reason']}\n"
        f"  client fuel={result['client_fuel']} alive={result['client_alive']} "
        f"enemy alive={result['enemy_alive']}\n"
        f"  capture: {result['capture_path']}\n"
        f"  world:   {result['world_path']}\n"
        f"  events:  {result['events_path']}\n"
    )
    return 0


__all__ = [
    "SIM_CLIENT_ID",
    "SIM_ENEMY_ID",
    "SIM_FIELD",
    "SIM_MAGIC",
    "SimRunResultDict",
    "main",
    "make_default_sim_world",
    "run_sim_session",
]
