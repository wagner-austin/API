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
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_loop import _tick_once
from tankpit_bot.capture.xor import build_session_xor_table
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.protocol.naming import is_practice_bot_name
from tankpit_bot.runtime_artifacts import make_run_stamp
from tankpit_bot.runtime_logging import configure_probe_runtime_logging
from tankpit_bot.sim.atlas_seed import seed_atlas_population
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.ghost import (
    GhostSpecDict,
    GhostTracker,
    ghost_events_for_tick,
    seed_ghost_world_population,
)
from tankpit_bot.sim.opponent import decide_opponent, maybe_revive_opponent
from tankpit_bot.sim.practice_room import PracticeRoomDriver, seed_practice_roster
from tankpit_bot.sim.scenarios import (
    _FERRY_CLIENT_FUEL,
    _SIM_DIR,
    SIM_CLIENT_ID,
    SIM_ENEMY_ID,
    SIM_FIELD,
    SIM_MAGIC,
    _parse_cli,
    _require_seeds_passable,
    _resolve_session_mode,
    make_default_sim_world,
    make_ferry_sim_world,
)
from tankpit_bot.sim.server import TICK_MS, SimServer
from tankpit_bot.sim.session import SimCDPSession, build_capture_session, deliver_batch
from tankpit_bot.sim.spawn import find_open_tile_near
from tankpit_bot.sim.world import SimWorldDict, encode_sim_world, make_sim_tank
from tankpit_bot.sim.world_seed import (
    seed_field_population,
    seed_practice_client,
    select_practice_layout,
)
from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
from tankpit_bot.types import encode_capture_session

log = get_logger(__name__)


class TickPacedClock:
    """The sim session's decision clock: one real tick per round.

    Wall time is the wrong clock for a sim session — an in-process
    round takes microseconds, so live TTLs (forage coverage 180 s,
    harvest memory 10 min, belief freshness gates) either never age
    or age at the mercy of machine load (the 2026-08-01 flake: the
    same 600-round session exited at round 54 solo and never under
    xdist). Pacing the decision clock at the measured 2 s server tick
    makes sessions deterministic AND live-realistic: a 300-round soak
    now ages exactly like a 10-minute session.
    """

    def __init__(self, start_ms: int) -> None:
        """Anchor the clock at the session's real start time."""
        self._now_ms = start_ms

    def __call__(self) -> int:
        """Return the paced session time."""
        return self._now_ms

    def advance(self, delta_ms: int) -> None:
        """Advance the session by one round's worth of time."""
        self._now_ms += delta_ms


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


def _seed_ghost_world(
    world: SimWorldDict,
    terrain: _test_hooks.TerrainMapProtocol,
    ghost_spec: GhostSpecDict,
    atlas_path: Path | None,
) -> None:
    """Seed a recorded session's world: client, ghosts, containers.

    Args:
        world: Simulated world (mutated).
        terrain: The loaded field terrain.
        ghost_spec: The compiled recording.
        atlas_path: Optional mined-atlas underlay — fills tiles the
            recording never observed; the capture's own reads stay
            per-tile authoritative.
    """
    client = make_sim_tank(
        SIM_CLIENT_ID,
        ghost_spec["client_team"],
        ghost_spec["client_rank"],
        ghost_spec["client_x"],
        ghost_spec["client_y"],
        ghost_spec["client_fuel"],
    )
    client["counts"] = list(ghost_spec["client_counts"])
    world["tanks"][SIM_CLIENT_ID] = client
    placed = 0
    for ghost in ghost_spec["ghosts"]:
        spawn_x, spawn_y = ghost["x"], ghost["y"]
        if not terrain.is_passable(spawn_x, spawn_y):
            spot = find_open_tile_near(
                world, terrain, spawn_x, spawn_y, world["tick"], min_radius=1, max_radius=4
            )
            if spot is None:
                continue
            spawn_x, spawn_y = spot
        world["tanks"][ghost["tank_id"]] = make_sim_tank(
            ghost["tank_id"],
            ghost["team"],
            ghost["rank"],
            spawn_x,
            spawn_y,
            fuel_capacity(ghost["rank"]),
            name=ghost["name"],
        )
        placed += 1
    if atlas_path is not None:
        atlas_tally = seed_atlas_population(
            world, terrain, atlas_path, frozenset(ghost_spec["dot_atlas"])
        )
        ghost_tiles = {(c["x"], c["y"]) for c in ghost_spec["containers"]}
        ghost_tiles.update(ghost_spec["equipment"])
        world["containers"] = [
            c for c in world["containers"] if (c["x"], c["y"]) not in ghost_tiles
        ]
        world["equipment"] = [e for e in world["equipment"] if (e["x"], e["y"]) not in ghost_tiles]
        log.info("ghost atlas underlay: %s", atlas_tally)
    skipped = seed_ghost_world_population(
        world["containers"], world["equipment"], ghost_spec, terrain
    )
    log.info(
        "ghost world: %d ghosts placed (%d identities unsighted), "
        "%d containers/equipment seeded (%d unseedable), %d ticks of timeline",
        placed,
        ghost_spec["unplaced_tanks"],
        len(world["containers"]) + len(world["equipment"]),
        skipped,
        ghost_spec["ticks"],
    )


def _boot(
    world: SimWorldDict,
    *,
    practice: bool = False,
    stamp: str = "",
    atlas_path: Path | None = None,
    ghost_spec: GhostSpecDict | None = None,
) -> tuple[Bot, SimServer, SimCDPSession, PracticeRoomDriver | None]:
    """Wire a real Bot to the sim over the CDP seam.

    Args:
        world: The world the server will own. In practice mode this
            arrives EMPTY of tanks and containers — the stamp-selected
            real layout seeds the client spawn and the full 36-bot
            roster, and ``seed_field_population`` lays down the static
            container field ([[game-economy]] 2026-07-25: the world
            never spawns at runtime).
        practice: When True, build the practice-room world before the
            handshake so the join roster dump includes the bots, and
            hand the server their ids for the corpse-window
            reactivation hook.
        stamp: The run stamp (the layout selector for practice and
            atlas spawns; unused otherwise).
        atlas_path: When set, the mined longitudinal atlas replaces
            the statistical container field ([[game-economy]]
            2026-08-01): in practice mode the roster still seeds, but
            the ground truth is the REAL room; standalone it is a
            pure-forage world on the real field.
        ghost_spec: When set, seed a recorded session's world: the
            client at its recorded spawn state, every sighted
            opponent as a replayable ghost (at its first PASSABLE
            sighting; ferry riders and other water sightings spawn at
            their first dry tile), and the capture's first-observed
            containers.

    Returns:
        The bot, the server, the seam link, and the practice-room
        driver (None outside practice mode), with the join handshake
        already delivered.

    Raises:
        XorStaticKeyUnavailableError: If the XOR static key cannot be
            read — the sim's binary seam needs the real cipher.
        RuntimeError: If the field terrain GIF is unavailable, or a
            scenario seed sits on impassable ground — a sim run needs
            both right, loudly.
    """
    reset_world_state()
    table = build_session_xor_table(SIM_MAGIC)
    gif_path = Path(world["field"])
    if not _test_hooks.path_exists(gif_path):
        raise RuntimeError(f"terrain GIF {gif_path} not found — run `make download-fields` first")
    terrain = _test_hooks.load_terrain_map(gif_path)
    # The lobby TEXT handshake (ROOM_LIST + SELECT) is outside the
    # binary seam, but a session always has a selected room — without
    # it the bot's decision terrain stays None for the whole run ("No
    # selected room is available for terrain-map loading") and every
    # terrain-gated behavior (greeting stand-off landings, larder
    # landing legality, LOS composition) silently self-disables.
    # First surfaced by the 2026-07-31 human-opponent soak: the greet
    # approach could never vouch a landing, so the session exited
    # no_viable_targets two rounds in. Register the sim room exactly
    # as the lobby decoders would have.
    service = get_world_service()
    service.register_room_image("sim", gif_path.name.replace("_r.gif", ".gif"))
    service.set_selected_room("sim")
    driver: PracticeRoomDriver | None = None
    roster_ids: frozenset[int] = frozenset()
    if practice:
        layout = select_practice_layout(stamp)
        log.info(
            "practice layout %s: client spawn %s, %d bots",
            layout["provenance"],
            layout["client_spawn"],
            len(layout["roster"]),
        )
        seed_practice_client(world, terrain, layout, SIM_CLIENT_ID)
        roster_ids = seed_practice_roster(world, terrain, layout["roster"])
        driver = PracticeRoomDriver(roster_ids)
        if atlas_path is not None:
            tally = seed_atlas_population(world, terrain, atlas_path)
            log.info("atlas field %s: %s", atlas_path, tally)
        else:
            seed_field_population(world, terrain, seed=zlib.crc32(stamp.encode("utf-8")))
    elif ghost_spec is not None:
        _seed_ghost_world(world, terrain, ghost_spec, atlas_path)
        # Reactive ghosts (2026-08-03): bot-named ghosts carry the
        # certified roster policy UNDER their recorded timeline — the
        # live bot's shots draw the mined shot-for-shot return fire
        # and team aggro even where the recording has no answer, and
        # a killed bot ghost reactivates by the corpse-window law.
        # Human-named ghosts stay pure recordings.
        roster_ids = frozenset(
            ghost["tank_id"]
            for ghost in ghost_spec["ghosts"]
            if is_practice_bot_name(ghost["name"]) and ghost["tank_id"] in world["tanks"]
        )
        if roster_ids:
            driver = PracticeRoomDriver(roster_ids)
    elif atlas_path is not None:
        layout = select_practice_layout(stamp)
        seed_practice_client(world, terrain, layout, SIM_CLIENT_ID)
        # A pure-forage world has no targets, so a hunt-ready spawn
        # would exit ``no_viable_targets`` on tick 2 (first standalone
        # atlas run did exactly that). Start lean: the session's work
        # IS the real field's larder economics.
        world["tanks"][SIM_CLIENT_ID]["fuel"] = _FERRY_CLIENT_FUEL
        tally = seed_atlas_population(world, terrain, atlas_path)
        log.info("atlas forage world %s: %s", atlas_path, tally)
    _require_seeds_passable(world, terrain)
    server = SimServer(world, terrain, client_id=SIM_CLIENT_ID, roster_ids=roster_ids)
    bot = Bot("https://sim.tankpit.local/", headless=True)
    bot._magic = SIM_MAGIC
    bot._on_magic_captured(SIM_MAGIC)
    link = SimCDPSession(server, table)
    bot._cdp = link
    deliver_batch(bot._cdp_message_buffer, server.handshake(), link)
    return bot, server, link, driver


def _queue_ghost_round(server: SimServer, spec: GhostSpecDict, round_index: int) -> frozenset[int]:
    """Feed one round's recorded ghost actions into the server.

    Dead ghosts skip their remaining timeline (the live fight may
    have killed them earlier than the recording did).

    Args:
        server: The live sim server.
        spec: The compiled ghost spec.
        round_index: The session tick about to be played.

    Returns:
        Ids of ghosts that acted from the recording this tick — the
        reactive-policy layer yields to them (recorded authority).
    """
    acted: set[int] = set()
    for event in ghost_events_for_tick(spec, round_index):
        tank = server.world["tanks"].get(event["tank_id"])
        if tank is None or not tank["alive"]:
            continue
        acted.add(event["tank_id"])
        if event["kind"] == "place":
            server.relocate_tank(event["tank_id"], event["x"], event["y"])
        elif event["kind"] == "shoot":
            server.queue_command(
                event["tank_id"],
                ClientCommandDict(
                    kind="shoot",
                    command=115,
                    x=event["x"],
                    y=event["y"],
                    target_id=0,
                    slot=0,
                    message_id=0,
                    direction=0,
                ),
            )
        else:
            server.queue_command(
                event["tank_id"],
                ClientCommandDict(
                    kind="chat",
                    command=109,
                    x=event["x"],
                    y=event["y"],
                    target_id=0,
                    slot=0,
                    message_id=event["message_id"],
                    direction=0,
                ),
            )
    return frozenset(acted)


def _queue_round_opponents(
    server: SimServer,
    driver: PracticeRoomDriver | None,
    opponent: bool,
    ghost_spec: GhostSpecDict | None,
    enemy_id: int,
    round_index: int,
) -> int:
    """Queue this round's non-client actions for the active mode.

    Args:
        server: The live sim server.
        driver: The roster-policy driver — practice mode's seeded
            roster, or ghost mode's reactive bot-named ghosts.
        opponent: Whether the scripted opponent plays.
        ghost_spec: The ghost timeline, when in ghost mode.
        enemy_id: The scripted opponent's current wire id.
        round_index: The session tick about to be played.

    Returns:
        The (possibly revived) scripted opponent id.
    """
    recorded_actors: frozenset[int] = frozenset()
    if ghost_spec is not None:
        recorded_actors = _queue_ghost_round(server, ghost_spec, round_index)
    if driver is not None:
        decisions = driver.decide_all(server.world, server.terrain, hold_ids=recorded_actors)
        for bot_id, command in decisions:
            server.queue_command(bot_id, command)
    elif opponent:
        enemy_id = maybe_revive_opponent(server, enemy_id, SIM_CLIENT_ID)
        opponent_command = decide_opponent(server.world, enemy_id, SIM_CLIENT_ID)
        if opponent_command is not None:
            server.queue_command(enemy_id, opponent_command)
    return enemy_id


def run_sim_session(
    rounds: int,
    *,
    opponent: bool = True,
    practice: bool = False,
    ferry: bool = False,
    atlas: str | None = None,
    ghost: str | None = None,
    stamp: str | None = None,
    opponent_name: str = "",
) -> SimRunResultDict:
    """Play one production-bot session against the sim and archive it.

    Args:
        rounds: Maximum server ticks to play.
        opponent: Whether the scripted opponent returns fire (ignored
            in practice and ferry modes).
        practice: Face the certified practice-bot roster
            (``sim/practice_room``) instead of the scripted harness.
        ferry: Play the ferry forage scenario
            (:func:`make_ferry_sim_world`) — no opponent, a
            water-locked larder behind one scope pan. Ignored when
            ``practice`` is set.
        atlas: Path to the mined longitudinal atlas
            (``container_atlas.json``). With ``practice`` it replaces
            the statistical container field under the roster; alone
            it is a pure-forage session on the real room. Ignored in
            ferry mode.
        ghost: Path to a recorded ``capture_session.json`` to replay
            as ghosts ([[capture-differ]] stage 4): the production
            bot plays live against the recording's opponents doing
            exactly what they did; the ``ghost_summary`` diagnostic
            reports how long the live run tracked the recorded
            client. Takes precedence over every other scenario flag.
        stamp: Optional archive stamp override for deterministic tests.
        opponent_name: Optional wire name for the scripted opponent.
            A human-shaped name (e.g. ``guest``) runs the session
            under the human-consent gate and the fair-fight contracts
            (2026-07-31) — the opponent shoots first, which consents
            it into acquisition. Ignored in practice mode.

    Returns:
        The session summary (also written to the artifacts).

    Raises:
        RuntimeError: If the static key or terrain is unavailable.
    """
    run_stamp = stamp if stamp is not None else make_run_stamp()
    artifacts = configure_probe_runtime_logging("sim", run_stamp)
    world, opponent, practice, ghost_spec, atlas_path, ferry_mode = _resolve_session_mode(
        opponent=opponent,
        practice=practice,
        ferry=ferry,
        atlas=atlas,
        ghost=ghost,
        opponent_name=opponent_name,
    )
    bot, server, link, driver = _boot(
        world, practice=practice, stamp=run_stamp, atlas_path=atlas_path, ghost_spec=ghost_spec
    )
    exit_reason = "rounds_exhausted"
    exit_detail = ""
    played = 0
    enemy_id = SIM_ENEMY_ID
    clock = TickPacedClock(_test_hooks.get_current_time_ms())
    original_clock = _test_hooks.get_current_time_ms
    _test_hooks.get_current_time_ms = clock
    tracker = GhostTracker(ghost_spec["recorded_path"]) if ghost_spec is not None else None
    if ghost_spec is not None:
        rounds = min(rounds, ghost_spec["ticks"])
    try:
        for round_index in range(rounds):
            _tick_once(bot)
            enemy_id = _queue_round_opponents(
                server, driver, opponent, ghost_spec, enemy_id, round_index
            )
            batch = server.advance_tick()
            if driver is not None:
                driver.note_batch(server.world, batch)
            deliver_batch(bot._cdp_message_buffer, batch, link)
            if tracker is not None:
                live = server.world["tanks"][SIM_CLIENT_ID]
                tracker.note_round(round_index, live["x"], live["y"])
            clock.advance(TICK_MS)
            played += 1
    except SessionExitError as error:
        exit_reason = error.reason
        exit_detail = error.detail
        log.info(
            "sim session ended by the production exit path: %s (%s)",
            error.reason,
            error.detail,
        )
    finally:
        _test_hooks.get_current_time_ms = original_clock
    if tracker is not None:
        tracker.emit_summary()
        log.info(
            "ghost track: %d/%d rounds within reach of the recording; "
            "first divergence at round %d, final drift %d",
            tracker.tracked_ticks,
            tracker.compared_ticks,
            tracker.first_divergence_tick,
            tracker.final_drift,
        )
    capture_path = _SIM_DIR / f"sim-{run_stamp}.capture_session.json"
    world_path = _SIM_DIR / f"sim-{run_stamp}.world.json"
    session = build_capture_session(link, SIM_MAGIC, f"sim-{run_stamp}")
    _test_hooks.write_text(capture_path, dump_json_str(encode_capture_session(session)))
    _test_hooks.write_text(world_path, dump_json_str(encode_sim_world(server.world)))
    client = server.world["tanks"][SIM_CLIENT_ID]
    if practice or ferry_mode or atlas_path is not None or ghost_spec is not None:
        enemy_alive = any(
            tank["alive"] and tank["team"] != client["team"]
            for tank_id, tank in server.world["tanks"].items()
            if tank_id != SIM_CLIENT_ID
        )
    else:
        enemy_alive = server.world["tanks"][enemy_id]["alive"]
    return SimRunResultDict(
        stamp=run_stamp,
        rounds_played=played,
        exit_reason=exit_reason,
        exit_detail=exit_detail,
        commands_sent=len(link.sent_commands),
        client_fuel=client["fuel"],
        client_alive=client["alive"],
        enemy_alive=enemy_alive,
        capture_path=str(capture_path),
        world_path=str(world_path),
        events_path=artifacts["latest_events_path"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``make sim-run``.

    Args:
        argv: Command-line arguments (``--rounds N``,
            ``--no-opponent``, ``--stamp S``, ``--human-opponent
            NAME``, ``--ferry``, ``--from-atlas [PATH]``). Uses
            ``sys.argv[1:]`` when None.

    Returns:
        Process exit code (0 — a session that ends via the production
        exit path is still a successful sim run).
    """
    parsed = _parse_cli(list(argv) if argv is not None else list(sys.argv[1:]))
    result = run_sim_session(
        parsed["rounds"],
        opponent=parsed["opponent"],
        practice=parsed["practice"],
        ferry=parsed["ferry"],
        atlas=parsed["atlas"],
        ghost=parsed["ghost"],
        stamp=parsed["stamp"],
        opponent_name=parsed["opponent_name"],
    )
    rounds = parsed["rounds"]
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
    "make_ferry_sim_world",
    "run_sim_session",
]
