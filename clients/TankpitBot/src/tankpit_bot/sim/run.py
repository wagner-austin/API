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
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_body import _tick_once
from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.runtime_artifacts import make_run_stamp
from tankpit_bot.runtime_logging import configure_probe_runtime_logging
from tankpit_bot.sim.ghost import (
    GhostTracker,
)
from tankpit_bot.sim.run_boot import (
    TickPacedClock,
    _boot,
    _queue_round_opponents,
)
from tankpit_bot.sim.scenarios import (
    SIM_CLIENT_ID,
    SIM_ENEMY_ID,
    SIM_MAGIC,
    _parse_cli,
    _resolve_session_mode,
)
from tankpit_bot.sim.session import build_capture_session, deliver_batch
from tankpit_bot.sim.world import encode_sim_world
from tankpit_bot.types import encode_capture_session

log = get_logger(__name__)


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


def run_sim_session(
    rounds: int,
    *,
    archive_dir: Path,
    opponent: bool = True,
    practice: bool = False,
    ferry: bool = False,
    larder: bool = False,
    atlas: str | None = None,
    ghost: str | None = None,
    stamp: str | None = None,
    opponent_name: str = "",
) -> SimRunResultDict:
    """Play one production-bot session against the sim and archive it.

    Args:
        rounds: Maximum server ticks to play.
        archive_dir: Directory the capture and world artifacts are
            written to. Required rather than defaulted because WHICH
            archive a session lands in decides which corpus the
            response-shape differ later reads it as: ``runs/sim``
            accumulates every generation of the sim ever run, so a
            fidelity verdict has to be taken over a directory holding
            one generation only ([[capture-differ]]).
        opponent: Whether the scripted opponent returns fire (ignored
            in practice and ferry modes).
        practice: Face the certified practice-bot roster
            (``sim/practice_room``) instead of the scripted harness.
        larder: Play the own-tile collection scenario
            (:func:`make_larder_sim_world`) — no opponent, the client
            standing ON equipment with empty slots, so the grant-
            without-a-walk and free-radar branches execute. Ignored
            when ``practice`` or ``ferry`` is set.
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
        larder=larder,
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
            clock.advance(TICK_RATE_MS)
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
        # The PRODUCTION teardown, not a sim-local imitation: the bot
        # already owns a graceful quit (``build_quit_command`` — the
        # plain, un-XOR'd ``-``, sent so the server records a
        # deliberate lobby exit rather than a socket drop). A sim
        # session used to just stop mid-stream, so the archive's quit
        # frames had no counterpart and the teardown path was never
        # exercised ([[session-state-deglobalisation]]).
        bot._send_graceful_quit()
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
    capture_path = archive_dir / f"sim-{run_stamp}.capture_session.json"
    world_path = archive_dir / f"sim-{run_stamp}.world.json"
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
            NAME``, ``--ferry``, ``--larder``, ``--from-atlas [PATH]``, ``--out
            DIR``). Uses ``sys.argv[1:]`` when None.

    Returns:
        Process exit code (0 — a session that ends via the production
        exit path is still a successful sim run).
    """
    parsed = _parse_cli(list(argv) if argv is not None else list(sys.argv[1:]))
    result = run_sim_session(
        parsed["rounds"],
        archive_dir=Path(parsed["out"]),
        opponent=parsed["opponent"],
        practice=parsed["practice"],
        ferry=parsed["ferry"],
        larder=parsed["larder"],
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
    "SimRunResultDict",
    "log",
    "main",
    "run_sim_session",
]
