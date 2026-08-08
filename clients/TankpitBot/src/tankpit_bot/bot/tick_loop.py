"""The bot session loop: run ticks until a bound or a signal ends it.

Owns the loop itself, the exit boundary, the interrupt flag, the
per-tick context and status publication, and the end-of-session
scorecard and runs-index row. One tick's work is
:mod:`tankpit_bot.bot.tick_body`.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger
from playwright._impl._errors import TargetClosedError

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import PageProtocol
from tankpit_bot.bot import tick_body
from tankpit_bot.bot.ai.types import (
    AIStateDict,
)
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.bus.session_status import (
    SessionStatusDict,
    make_live_stats,
    make_session_status,
    manual_to_wire_mode,
    wire_mode_to_manual,
)
from tankpit_bot.diagnostics.runs_index import (
    append_index_row,
    count_stall_timeouts,
    make_index_row,
)
from tankpit_bot.ledger.damage_book import summarize_side, total_fuel
from tankpit_bot.ledger.decision import verify_outcome_invariant
from tankpit_bot.ledger.events import ACTION_KINDS
from tankpit_bot.ledger.ring import outcome_counts
from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.runtime_context import set_runtime_context
from tankpit_bot.runtime_logging import (
    emit_diagnostic,
    get_bot_runtime_artifacts,
)
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state

log = get_logger(__name__)

# Early-wake slice width. The server processes commands in fixed 2 s
# windows (TICK_RATE_MS, fire-spam verified), and a completion message
# that lands just after the loop goes to sleep used to wait out the
# whole window before the next decision -- a phase drift that cost one
# full server tick whenever the wakeup missed a window boundary by
# milliseconds (user observation 2026-07-30: "it seems like we are
# losing a tick after each equipment pickup sometimes"; measured: 294
# of 302 completion->dispatch pairs were same-second, so the pipeline
# is clean and ONLY the sleep is blind). Waking on fresh wire traffic
# while an action is in flight puts the next decision within one slice
# of the completion, like a player clicking the moment the tank
# arrives.
_WAKE_SLICE_MS = 250

#: True when an OS signal (SIGINT / SIGTERM) has requested a graceful
#: shutdown. The tick loop checks this once per iteration so the bot
#: exits at a clean tick boundary, writing the session scorecard +
#: index row before the process dies. Reset to ``False`` by
#: :func:`reset_interrupt_flag` so consecutive sessions start clean.
# Final-stretch length of a bounded session spent disengaging and
# topping off for the clean ``session_complete`` exit.
_WIND_DOWN_SECONDS = 60

_INTERRUPT_REQUESTED: bool = False


def run_tick_loop(
    bot: Bot,
    page: PageProtocol,
    *,
    session_seconds: int,
    session_kills: int = 0,
    stop_file_path: Path,
) -> None:
    """Run the main tick loop.

    A positive ``session_seconds`` bounds the session by requested
    wait time (``seconds * 1000`` ms of between-tick sleeping; the
    early-wake sleep charges only what it actually waited, so busy
    stretches never shorten a bounded run); the loop then returns so
    ``Bot.run`` saves the capture session and shuts the browser down
    cleanly. Zero or negative runs until stopped. Bounded runs
    previously worked by killing the browser, which ended every session
    with an uncaught ``TargetClosedError`` and made crash exits
    indistinguishable from intended stops in the artifacts.

    The stop file is the external graceful-shutdown channel: creating
    it (``make bot-stop``) ends the run at the next tick boundary with
    the same clean teardown as a tick-budget exit. The sentinel is
    consumed so the next run does not stop instantly.

    Args:
        bot: Bot instance.
        page: Playwright page for waiting between ticks.
        session_seconds: Bounded session length in seconds; zero or
            negative runs until externally stopped.
        session_kills: Kill-target bound; when positive, reaching this
            many session kills triggers the wind-down (finish nothing
            new, top off, exit ``session_complete``) — the kill
            boundary is the natural clean-exit point (user request
            2026-07-26: end on kills, not mid-action on the clock).
            Zero disables the kill bound.
        stop_file_path: Sentinel file whose existence requests a
            graceful shutdown.
    """
    budget_ms = session_seconds * 1000 if session_seconds > 0 else 0
    # Wind-down: in the final stretch of a bounded run the AI stops
    # opening engagements, disengages, and tops off so the session
    # ends CLEANLY (``session_complete``) instead of the tick budget
    # cutting it mid-action (user request 2026-07-26) — and the next
    # session boots combat-ready on the leftover stock. Sessions of
    # two windows or less skip it (short diagnostic runs must still
    # exercise the full loop).
    wind_down_at_ms = (
        budget_ms - _WIND_DOWN_SECONDS * 1000 if session_seconds > 2 * _WIND_DOWN_SECONDS else 0
    )
    ticks_done = 0
    # The budget counts REQUESTED wait time, not iterations: the
    # early-wake sleep below can end a wait after a fraction of
    # TICK_RATE_MS, and charging a full window per iteration would
    # shorten bounded sessions in proportion to how busy they were.
    waited_ms = 0
    while True:
        _publish_tick_context(bot, ticks_done + 1)
        _apply_pending_mode_override(bot)
        if wind_down_at_ms > 0 and waited_ms >= wind_down_at_ms and not bot._ai_state["wind_down"]:
            bot._ai_state["wind_down"] = True
            log.info(
                "Session wind-down (final %ds): disengaging and topping off for a clean exit",
                _WIND_DOWN_SECONDS,
            )
        if (
            session_kills > 0
            and not bot._ai_state["wind_down"]
            and bot._ai_state["session_kill_count"] >= session_kills
        ):
            bot._ai_state["wind_down"] = True
            log.info(
                "Kill target reached (%d): winding down for a clean exit",
                session_kills,
            )
        exit_reason = _tick_with_exit_boundary(bot, ticks_done)
        if exit_reason is not None:
            _emit_session_scorecard(bot, ticks_done, exit_reason=exit_reason)
            return
        _publish_session_status(bot)
        ticks_done += 1
        if budget_ms > 0 and waited_ms + TICK_RATE_MS >= budget_ms:
            # Exit when the next full window would not fit: a 4 s
            # session is exactly two ticks and one wait, same as the
            # tick-counted budget this accounting replaced.
            log.info(
                "Session budget reached (%d ticks / %ds), ending run",
                ticks_done,
                session_seconds,
            )
            _emit_session_scorecard(bot, ticks_done, exit_reason="completed")
            return
        if _INTERRUPT_REQUESTED:
            log.info("Interrupt signal received, ending run gracefully")
            _emit_session_scorecard(bot, ticks_done, exit_reason="interrupted")
            return
        if _test_hooks.path_exists(stop_file_path):
            _test_hooks.remove_file(stop_file_path)
            log.info("Stop file %s detected, ending run", stop_file_path)
            _emit_session_scorecard(bot, ticks_done, exit_reason="stop_file")
            return
        try:
            waited_ms += _wait_between_ticks(bot, page)
        except TargetClosedError:
            log.info("Browser closed between ticks, ending run gracefully")
            _emit_session_scorecard(bot, ticks_done, exit_reason="browser_closed")
            return


def _tick_with_exit_boundary(bot: Bot, ticks_done: int) -> str | None:
    """Run one guarded tick and translate its endings into exit reasons.

    The session's single exception boundary ([[bot-behavior-contract]]
    §1.2/§1.3): a closed browser and a self-directed
    :class:`SessionExitError` become graceful exit reasons the caller
    finalizes; any OTHER unhandled exception finalizes the artifacts
    HERE — scorecard, ``latest.summary.txt``, and the ``_index.tsv``
    row as ``exit_reason="crashed"`` — and then RE-RAISES so the
    process still fails loudly. Before 2026-07-31 the contract
    promised "crashed" with no writer anywhere: a crashed session
    simply vanished from the runs index, which is exactly the "which
    runs died?" blindness the index exists to prevent.

    Args:
        bot: Bot instance.
        ticks_done: Completed tick count (for the crash scorecard).

    Returns:
        A graceful exit reason for the caller to finalize, or ``None``
        to continue the loop.
    """
    try:
        tick_body._tick_once(bot)
        tick_body._sync_live_view_demand(bot)
    except TargetClosedError:
        log.info("Browser closed during tick, ending run gracefully")
        return "browser_closed"
    except SessionExitError as exit_request:
        log.info("Session exit: %s -- %s", exit_request.reason, exit_request.detail)
        return exit_request.reason
    except Exception:
        log.exception(
            "Unhandled exception in tick %d - finalizing artifacts as crashed",
            ticks_done + 1,
        )
        _emit_session_scorecard(bot, ticks_done, exit_reason="crashed")
        raise
    return None


def _wait_between_ticks(bot: Bot, page: PageProtocol) -> int:
    """Sleep up to one tick window, waking early on in-flight progress.

    Args:
        bot: Bot instance (in-flight state + CDP buffer).
        page: Playwright page providing the wait primitive.

    Returns:
        Milliseconds of wait actually requested -- the session budget
        charges real waiting, so early wakes never shorten a bounded
        run.
    """
    if not has_in_flight_action(bot):
        page.wait_for_timeout(TICK_RATE_MS)
        return TICK_RATE_MS
    waited = 0
    baseline = len(bot._cdp_message_buffer)
    while waited < TICK_RATE_MS:
        page.wait_for_timeout(_WAKE_SLICE_MS)
        waited += _WAKE_SLICE_MS
        if len(bot._cdp_message_buffer) > baseline:
            break
    return waited


def _emit_session_scorecard(bot: Bot, ticks: int, *, exit_reason: str) -> None:
    """Emit a structured session summary at run end and append to the index.

    Args:
        bot: Bot instance carrying the AI scorecard.
        ticks: Total ticks executed before exit.
        exit_reason: How the session ended -- ``"completed"`` (tick
            budget exhausted), ``"stop_file"`` (graceful shutdown via
            the sentinel file), or ``"interrupted"`` (SIGINT/SIGTERM
            handler, registered by the CLI entry point).
    """
    ai = bot._ai_state
    ws = bot.world
    inv = get_inventory_state(ws)
    self_state = ws.world_state.get("self_state")
    fuel = self_state["fuel"] if self_state is not None else 0
    kills = ai["session_kill_count"]
    hits = ai["session_hit_count"]
    misses = ai["session_miss_count"]
    rejected = ai["session_reject_count"]
    dual = inv["dual_shots"]["count"]
    homing = inv["homing_shots"]["count"]
    radar = inv["extra_radars"]["count"]
    blocked = len(ai["blocked_combat_targets"])
    mode = ai["mode"]
    mode_state = ai["mode_state"]
    unresolved = verify_outcome_invariant(bot.world.ledger)
    for kind in ACTION_KINDS:
        counts = outcome_counts(bot.world.ledger, kind)
        if counts:
            emit_diagnostic(
                diagnostic_kind="session_outcome_counts",
                action_kind=kind,
                **dict(sorted(counts.items())),
            )
    if unresolved:
        emit_diagnostic(
            diagnostic_kind="session_unresolved_decisions",
            **dict(sorted(unresolved.items())),
        )
    emit_diagnostic(
        diagnostic_kind="session_scorecard",
        ticks=ticks,
        kills=kills,
        hits=hits,
        misses=misses,
        rejected=rejected,
        fuel_remaining=fuel,
        dual_shots_remaining=dual,
        homing_shots_remaining=homing,
        extra_radars_remaining=radar,
        targets_blocked=blocked,
        ai_mode=mode,
        ai_mode_state=mode_state,
        exit_reason=exit_reason,
    )
    damage_book = bot.world.damage_book
    fuel_totals = bot.world.fuel_book["totals"]
    emit_diagnostic(
        diagnostic_kind="damage_ledger",
        dealt=summarize_side(damage_book["dealt"]),
        taken=summarize_side(damage_book["taken"]),
        dealt_fuel=total_fuel(damage_book["dealt"]),
        taken_fuel=total_fuel(damage_book["taken"]),
        **{f"{kind}_count": total["count"] for kind, total in sorted(fuel_totals.items())},
        **{f"{kind}_fuel_lo": total["lo_sum"] for kind, total in sorted(fuel_totals.items())},
        **{f"{kind}_fuel_hi": total["hi_sum"] for kind, total in sorted(fuel_totals.items())},
    )
    shots = hits + misses + rejected
    hit_rate = f"{hits * 100 // shots}%" if shots > 0 else "n/a"
    summary = (
        f"TANKPIT SESSION SUMMARY\n"
        f"{'=' * 40}\n"
        f"Ticks:    {ticks}\n"
        f"Exit:     {exit_reason}\n"
        f"Kills:    {kills}\n"
        f"Shots:    {shots} ({hits} hits, {misses} misses, {rejected} rejected)\n"
        f"Hit rate: {hit_rate}\n"
        f"Blocked:  {blocked}\n"
        f"{'=' * 40}\n"
        f"Fuel:     {fuel}\n"
        f"Duals:    {dual}\n"
        f"Homings:  {homing}\n"
        f"Radars:   {radar}\n"
        f"{'=' * 40}\n"
        f"Mode:     {mode}/{mode_state}\n"
    )
    log.info("\n%s", summary)
    _test_hooks.write_text(Path("runs/bot/latest.summary.txt"), summary)
    _append_index_row(ticks, shots, kills, exit_reason)


def _append_index_row(ticks: int, shots: int, kills: int, exit_reason: str) -> None:
    """Append one row to ``runs/bot/_index.tsv`` summarising this run.

    The stamp is extracted from the active bot runtime artifacts. When
    the bot was never configured (test/probe path), the index append
    is skipped.

    Args:
        ticks: Total ticks executed.
        shots: ``hits + misses`` from the AI scorecard.
        kills: ``session_kill_count`` from the AI scorecard.
        exit_reason: Lifecycle outcome string.
    """
    artifacts = get_bot_runtime_artifacts()
    if artifacts is None:
        return
    stamp = _extract_stamp_from_archive_path(artifacts["archive_events_path"])
    duration_s = ticks * TICK_RATE_MS // 1000
    stalls = count_stall_timeouts(Path(artifacts["latest_events_path"]))
    row = make_index_row(
        stamp=stamp,
        duration_s=duration_s,
        exit_reason=exit_reason,
        ticks=ticks,
        stalls=stalls,
        shots_fired=shots,
        kills=kills,
    )
    append_index_row(row)


def _extract_stamp_from_archive_path(archive_events_path: str) -> str:
    """Pull the ``YYYYMMDD-HHMMSS`` stamp from an archive events path.

    The runtime artifacts builder writes archives at
    ``runs/bot/bot-<stamp>.events.jsonl``. This helper extracts the
    ``<stamp>`` segment so the index row matches the archive filenames.

    Args:
        archive_events_path: Archive path string from
            :class:`BotRunArtifactsDict`.

    Returns:
        The embedded run stamp.

    Raises:
        ValueError: If the archive path does not follow the
            ``bot-<stamp>.events.jsonl`` convention.
    """
    name = Path(archive_events_path).name
    if not name.startswith("bot-") or not name.endswith(".events.jsonl"):
        raise ValueError(f"archive events path does not match bot-<stamp>: {name}")
    return name[len("bot-") : -len(".events.jsonl")]


def request_interrupt() -> None:
    """Signal the tick loop to exit at the next tick boundary.

    Idempotent: calling more than once between resets has no extra
    effect. Used as the SIGINT/SIGTERM handler installed by
    :func:`tankpit_bot.bot.entry.main`.
    """
    global _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = True


def reset_interrupt_flag() -> None:
    """Clear the interrupt flag so a new run starts unblocked.

    Tests rely on this to keep state from one session out of the next.
    Production code does NOT need to call it -- a fresh process
    starts with the flag already ``False``.
    """
    global _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = False


def is_interrupt_requested() -> bool:
    """Return True when an interrupt has been requested.

    Returns:
        Current value of the module-level flag.
    """
    return _INTERRUPT_REQUESTED


def _publish_tick_context(bot: Bot, tick_n: int) -> None:
    """Update the runtime logger's context for the upcoming tick.

    The context is auto-attached to every ``emit_*`` event the bot
    produces during the tick, so a single JSONL line is enough to
    reconstruct what mode the bot was in and which action it was
    waiting on. Called once per tick from :func:`run_tick_loop`
    immediately before :func:`_tick_once`.

    Args:
        bot: Bot instance.
        tick_n: 1-based index of the tick about to execute.
    """
    ai = bot._ai_state
    action = bot._state_data["in_flight_action"]
    set_runtime_context(
        tick_n=tick_n,
        bot_state=f"{ai['mode']}/{ai['mode_state']}",
        in_flight_action_kind=action["kind"],
    )


def _apply_pending_mode_override(bot: Bot) -> None:
    """Drain the SPA mode bridge and stamp the override onto ``bot._ai_state``.

    Runs at the top of every tick iteration in :func:`run_tick_loop`.
    A drained :data:`WireMode` is translated through
    :func:`wire_mode_to_manual` and written to
    ``bot._ai_state["manual_mode"]`` so the tick's :func:`decide`
    respects the pin. When the bridge is empty (no SPA input since the
    last tick), ``bot._ai_state["manual_mode"]`` is left untouched —
    manual overrides are sticky across ticks by design.

    Args:
        bot: Bot instance whose ``_mode_bridge`` and ``_ai_state`` are
            mutated in place.
    """
    pending = bot._mode_bridge.drain()
    if pending is None:
        return
    manual = wire_mode_to_manual(pending)
    bot._ai_state = AIStateDict(**{**bot._ai_state, "manual_mode": manual})


def _publish_session_status(bot: Bot) -> None:
    """Build a :class:`SessionStatusDict` from ``bot._ai_state`` and publish it.

    Runs at the bottom of every completed tick iteration in
    :func:`run_tick_loop`. The published frame reflects the AI state
    the tick just finalised — durable mode ownership, live counters,
    the SPA-selected pin. Consumers (SSE subscribers) receive it
    through the shared :class:`StatusBusProtocol`.

    Args:
        bot: Bot instance whose ``_status_bus`` receives the frame.
    """
    ai = bot._ai_state
    stats = make_live_stats(
        kills=ai["session_kill_count"],
        hits=ai["session_hit_count"],
        misses=ai["session_miss_count"],
        radars_used=ai["live_radars_used"],
        teleports=ai["live_teleports"],
    )
    status: SessionStatusDict = make_session_status(
        running=True,
        manual_mode=manual_to_wire_mode(ai["manual_mode"]),
        active_mode=ai["mode"],
        active_mode_state=ai["mode_state"],
        session_started_ms=bot._start_timestamp_ms,
        tick_timestamp_ms=get_current_time_ms(),
        stats=stats,
    )
    bot._status_bus.publish(status)


__all__ = [
    "is_interrupt_requested",
    "log",
    "request_interrupt",
    "reset_interrupt_flag",
    "run_tick_loop",
]
