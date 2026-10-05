"""Several production bots on one sim field: multi-tank combat on demand.

Phase 2 of the multiplayer track (board task b008ab91). A one-bot sim
session fights a scripted opponent or the practice roster, which never
produce the wire a fight between two real clients does: the bot's own
combat machinery firing back at it, a second client's receipts NOT
reaching the first, a rival quitting mid-field. Solo practice-room runs
were blind to exactly those patterns ([[recipient-policy]]). This module
seats ``N`` production bots on one :class:`SimServer`, each over its own
connection, each playing the unmodified ``_tick_once``.

Three things keep the bots separate while they share one process:

* **One contextvars context per bot.** The runtime logging's active run
  and the tick's runtime context are ``ContextVar`` slots, so each bot
  ticks inside its own copied context and writes its own event stream
  under ``<runs_root>/tank-<id>``.
* **One connection per bot.** Each bot's link queues under its own tank,
  and each receives only its own batch from ``advance_tick``.
* **A bot that exits LEAVES.** The production exit path raises
  :class:`SessionExitError`; the bot then sends its graceful quit and its
  connection is closed with :meth:`SimServer.disconnect`, so the rivals
  see the 0x29 a departing player draws instead of hunting an abandoned
  tank.

Only the arena (rivals, no scripted opponent) and the practice room
(rivals plus the bot_policy roster) take part: ghost, ferry, larder and
atlas-forage scenarios are written around one client and are refused.
"""

from __future__ import annotations

from contextvars import Context, copy_context
from pathlib import Path
from typing import TypedDict

from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_body import _tick_once
from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.runtime_artifacts import ProbeRunArtifactsDict
from tankpit_bot.runtime_logging import configure_probe_runtime_logging
from tankpit_bot.sim.field_choice import resolve_field
from tankpit_bot.sim.field_clients import FieldSeatError, field_account
from tankpit_bot.sim.lobby import SIM_ACCOUNT
from tankpit_bot.sim.practice_room import PracticeRoomDriver
from tankpit_bot.sim.run_boot import (
    TickPacedClock,
    _attach_bot,
    _queue_round_opponents,
    _seed_world,
    resolve_named_world,
)
from tankpit_bot.sim.scenarios import (
    SIM_CLIENT_ID,
    SIM_ENEMY_ID,
    SIM_FIELD,
    SIM_MAGIC,
    make_default_sim_world,
)
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.session import SimCDPSession, build_capture_session, deliver_batch
from tankpit_bot.sim.world import SimTankDict, SimWorldDict, encode_sim_world, make_sim_world
from tankpit_bot.types import encode_capture_session

log = get_logger(__name__)

#: Where each bot's probe log and events land when no root is named.
DEFAULT_FIELD_RUNS_ROOT = Path("runs") / "probe"


class FieldClientResultDict(TypedDict):
    """One bot's part in a field session.

    Attributes:
        tank_id: The tank it played.
        name: The tank's wire name.
        team: The tank's team.
        rounds_played: Ticks it played before it left or the session ended.
        exit_reason: ``rounds_exhausted``, or the production exit reason.
        exit_detail: The production exit's detail, empty otherwise.
        departed: Whether it quit the field before the session ended.
        alive: Whether its tank was alive at the end (False once departed).
        kills: Tanks it deactivated (its 0x56 ``destroyed``).
        deaths: Times it was deactivated (its 0x56 ``deactivated``).
        commands_sent: Commands its link carried.
        capture_path: Its own connection's wire.
        events_path: Its own event stream.
    """

    tank_id: int
    name: str
    team: int
    rounds_played: int
    exit_reason: str
    exit_detail: str
    departed: bool
    alive: bool
    kills: int
    deaths: int
    commands_sent: int
    capture_path: str
    events_path: str


class FieldRunResultDict(TypedDict):
    """One finished field session.

    Attributes:
        stamp: The run stamp.
        rounds_played: Ticks the field ran.
        world_path: The field's final world.
        clients: Every bot's part, in seating order.
    """

    stamp: str
    rounds_played: int
    world_path: str
    clients: list[FieldClientResultDict]


class _Seat:
    """One bot's place at the field: its bot, link, context and fate."""

    def __init__(
        self,
        tank: SimTankDict,
        bot: Bot,
        link: SimCDPSession,
        context: Context,
        artifacts: ProbeRunArtifactsDict,
    ) -> None:
        """Seat a bot that has already joined.

        Args:
            tank: The tank it plays, as it was seated. Its id, name and
                team are kept here because a bot that quits takes its
                tank off the field.
            bot: The production bot.
            link: Its seam link.
            context: The contextvars context it ticks in.
            artifacts: Its own probe run artifacts.
        """
        self.tank_id = tank["tank_id"]
        self.name = tank["name"]
        self.team = tank["team"]
        self.bot = bot
        self.link = link
        self.context = context
        self.artifacts = artifacts
        self.active = True
        self.rounds_played = 0
        self.exit_reason = "rounds_exhausted"
        self.exit_detail = ""


def make_field_world(practice: bool) -> SimWorldDict:
    """The world a field session starts from, before seeding.

    Args:
        practice: The practice room (empty: the boot seeds the client,
            the roster and the container field) rather than the arena.

    Returns:
        The arena with its scripted opponent removed — the rivals are
        the opposition — or an empty practice field.
    """
    if practice:
        return make_sim_world(SIM_FIELD)
    world = make_default_sim_world()
    del world["tanks"][SIM_ENEMY_ID]
    return world


def _open_seat(server: SimServer, tank_id: int, stamp: str, runs_root: Path) -> _Seat:
    """Join one bot to its connected tank inside a context of its own.

    Args:
        server: The field's server; the tank is already connected.
        tank_id: The tank the bot plays.
        stamp: The field's run stamp.
        runs_root: The root each bot's own run directory sits under.

    Returns:
        The seat.
    """
    context = copy_context()
    artifacts = context.run(
        configure_probe_runtime_logging,
        "sim",
        f"{stamp}-tank-{tank_id}",
        runs_root=str(runs_root / f"tank-{tank_id}"),
    )
    account = SIM_ACCOUNT if tank_id == SIM_CLIENT_ID else field_account(server.world, tank_id)
    bot, link = context.run(_attach_bot, server, tank_id, account)
    return _Seat(server.world["tanks"][tank_id], bot, link, context, artifacts)


def _leave(server: SimServer, seat: _Seat, error: SessionExitError) -> None:
    """Take a bot that exited off the field.

    Args:
        server: The field's server.
        seat: The bot's seat.
        error: The production exit it raised.
    """
    seat.active = False
    seat.exit_reason = error.reason
    seat.exit_detail = error.detail
    seat.context.run(seat.bot._send_graceful_quit)
    server.disconnect(seat.tank_id)
    log.info(
        "field: tank %d left at round %d (%s: %s)",
        seat.tank_id,
        seat.rounds_played,
        error.reason,
        error.detail,
    )


def _play_round(
    server: SimServer,
    seats: list[_Seat],
    driver: PracticeRoomDriver | None,
    round_index: int,
) -> None:
    """Play one round: every seated bot ticks, the roster decides, the field advances.

    The order is the one-bot session's: the bots' commands first, then
    the roster's, then the tick, then each bot's own batch.

    Args:
        server: The field's server.
        seats: Every seat; the departed are skipped.
        driver: The practice roster's driver, or None in the arena.
        round_index: The round being played.
    """
    for seat in seats:
        if not seat.active:
            continue
        try:
            seat.context.run(_tick_once, seat.bot)
        except SessionExitError as error:
            _leave(server, seat, error)
            continue
        seat.rounds_played += 1
    _queue_round_opponents(server, driver, False, None, SIM_ENEMY_ID, round_index)
    batches = server.advance_tick()
    if driver is not None:
        driver.note_field_batches(server.world, list(batches.values()))
    for seat in seats:
        if seat.active:
            seat.context.run(
                deliver_batch, seat.bot._cdp_message_buffer, batches[seat.tank_id], seat.link
            )


def _seat_result(
    server: SimServer, seat: _Seat, archive_dir: Path, stamp: str
) -> FieldClientResultDict:
    """Archive one bot's wire and summarise its part.

    Args:
        server: The field's server.
        seat: The bot's seat.
        archive_dir: Where the capture is written.
        stamp: The field's run stamp.

    Returns:
        The bot's result.
    """
    session_id = f"sim-{stamp}-tank-{seat.tank_id}"
    capture_path = archive_dir / f"{session_id}.capture_session.json"
    session = build_capture_session(seat.link, SIM_MAGIC, session_id)
    _test_hooks.write_text(capture_path, dump_json_str(encode_capture_session(session)))
    tank = server.world["tanks"].get(seat.tank_id)
    return FieldClientResultDict(
        tank_id=seat.tank_id,
        name=seat.name,
        team=seat.team,
        rounds_played=seat.rounds_played,
        exit_reason=seat.exit_reason,
        exit_detail=seat.exit_detail,
        departed=not seat.active,
        alive=tank is not None and tank["alive"],
        kills=server.combat.destroyed_by(seat.tank_id),
        deaths=server.combat.deactivations_of(seat.tank_id),
        commands_sent=len(seat.link.sent_commands),
        capture_path=str(capture_path),
        events_path=seat.artifacts["latest_events_path"],
    )


def run_field_session(
    rounds: int,
    *,
    clients: int,
    archive_dir: Path,
    practice: bool = False,
    stamp: str | None = None,
    layout: str | None = None,
    population_seed: int | None = None,
    runs_root: str | None = None,
    field: str | None = None,
) -> FieldRunResultDict:
    """Play ``clients`` production bots against each other on one field.

    Args:
        rounds: Most ticks the field runs; it also ends when every bot
            has left.
        clients: How many bots to seat, at least two (one bot is
            :func:`~tankpit_bot.sim.run.run_sim_session`'s job).
        archive_dir: Where each bot's capture and the field's world land.
        practice: Seat the bots in the practice room among the
            bot_policy roster, rather than alone in the arena.
        stamp: The run stamp, a label; None for a fresh one.
        layout: The practice layout's provenance; None derives it.
        population_seed: The container seed; None derives it.
        runs_root: The root each bot's own ``tank-<id>`` run directory
            sits under; None is :data:`DEFAULT_FIELD_RUNS_ROOT`.
        field: The shipped field to play (``field05``); None plays field01.

    Returns:
        The session's result.

    Raises:
        FieldSeatError: If fewer than two bots, or more than a field
            seats, are asked for.
        FieldChoiceError: If ``field`` names no shipped minimap.
        RuntimeError: If the static key or the terrain is unavailable.
    """
    if clients < 2:
        raise FieldSeatError(
            f"SIM_FIELD_SEATS: a field session seats 2 or more bots, not {clients}"
        )
    run_stamp, run_layout, run_population_seed = resolve_named_world(stamp, layout, population_seed)
    world = make_field_world(practice)
    world["field"] = SIM_FIELD if field is None else resolve_field(field)
    terrain, roster_ids, driver = _seed_world(
        world,
        practice=practice,
        layout=run_layout,
        population_seed=run_population_seed,
        atlas_path=None,
        ghost_spec=None,
        rivals=clients - 1,
    )
    server = SimServer(world, terrain, roster_ids=roster_ids)
    tank_ids = [SIM_CLIENT_ID + k for k in range(clients)]
    for tank_id in tank_ids:
        server.connect(tank_id)
    root = Path(runs_root) if runs_root is not None else DEFAULT_FIELD_RUNS_ROOT
    seats = [_open_seat(server, tank_id, run_stamp, root) for tank_id in tank_ids]
    clock = TickPacedClock(_test_hooks.get_current_time_ms())
    original_clock = _test_hooks.get_current_time_ms
    _test_hooks.get_current_time_ms = clock
    played = 0
    try:
        while played < rounds and any(seat.active for seat in seats):
            _play_round(server, seats, driver, played)
            clock.advance(TICK_RATE_MS)
            played += 1
    finally:
        _test_hooks.get_current_time_ms = original_clock
        # A bot still at the field when it ends quits the way it does
        # live, through the production teardown.
        for seat in seats:
            if seat.active:
                seat.context.run(seat.bot._send_graceful_quit)
    world_path = archive_dir / f"sim-{run_stamp}.world.json"
    _test_hooks.write_text(world_path, dump_json_str(encode_sim_world(server.world)))
    results = [_seat_result(server, seat, archive_dir, run_stamp) for seat in seats]
    log.info(
        "field %s: %d rounds, %s",
        run_stamp,
        played,
        ", ".join(
            f"tank {r['tank_id']} {r['kills']}k/{r['deaths']}d {r['exit_reason']}" for r in results
        ),
    )
    return FieldRunResultDict(
        stamp=run_stamp, rounds_played=played, world_path=str(world_path), clients=results
    )


__all__ = [
    "DEFAULT_FIELD_RUNS_ROOT",
    "FieldClientResultDict",
    "FieldRunResultDict",
    "make_field_world",
    "run_field_session",
]
