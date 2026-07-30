"""Tick loop orchestrator for the bot.

Runs the sync-decide-execute cycle on each server tick. The game uses
a command queue — move/pickup commands are queued and the tank walks
there automatically. No need to wait for arrival.

Cycle: SYNC → DECIDE → EXECUTE → WAIT
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger
from playwright._impl._errors import TargetClosedError

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import PageProtocol
from tankpit_bot.action_lab.client_structure import maybe_emit_client_structure_survey
from tankpit_bot.action_lab.page_client_snapshot import (
    PageClientSnapshotDict,
    capture_page_client_snapshot,
)
from tankpit_bot.bot import ai_strategy, executor, world_sync
from tankpit_bot.bot.ai.types import AIStateDict, make_initial_ai_state, render_reason
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.states import make_initial_state_data
from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.browser.overlay import OverlayStateDict
from tankpit_bot.browser.overlay_hud import update_bot_overlay
from tankpit_bot.diagnostics.entity_alignment import maybe_emit_entity_alignment_sample
from tankpit_bot.diagnostics.runs_index import (
    append_index_row,
    count_stall_timeouts,
    make_index_row,
)
from tankpit_bot.diagnostics.self_alignment import maybe_emit_self_alignment_sample
from tankpit_bot.ledger.damage_book import resolve_dealt, summarize_side
from tankpit_bot.ledger.decision import latest_decision_event_id, verify_outcome_invariant
from tankpit_bot.ledger.events import ACTION_KINDS
from tankpit_bot.ledger.mode_transition import emit_mode_transition
from tankpit_bot.ledger.outcome.shoot import (
    emit_shoot_command_rejected,
    emit_shoot_hit,
    emit_shoot_miss,
)
from tankpit_bot.ledger.ring import outcome_counts
from tankpit_bot.physics.capacity import fuel_capacity, inventory_capacity
from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.runtime_logging import (
    emit_ai,
    emit_diagnostic,
    emit_sync,
    get_bot_runtime_artifacts,
    set_runtime_context,
)
from tankpit_bot.service.types import (
    SessionStatusDict,
    make_live_stats,
    make_session_status,
    manual_to_wire_mode,
    wire_mode_to_manual,
)
from tankpit_bot.sniffer.world_state import (
    get_terrain_map,
    get_world_service,
)
from tankpit_bot.sniffer.world_state_combat import (
    check_and_clear_ammo_delta_hit,
    check_and_clear_combat_hit,
    check_and_clear_command_error,
    check_and_clear_last_shot_victim_id,
    check_and_clear_our_shot_response,
    drain_killed_tank_ids,
    peek_combat_hit,
    peek_command_error,
    peek_our_shot_response,
)
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state
from tankpit_bot.state import SelfStateDict

# 0x52 Supervisor codes a shoot dispatch can draw. Any of these while a
# shot is pending is the server's authoritative refusal of THAT shot --
# no 0x53 ShootEvent and no ammo delta will ever arrive for it (live
# run 2026-07-03 20:34: five code-0 rejections at an off-viewport aim
# produced zero wire feedback and each burned the full 4 s feedback
# window before an identical redispatch).
_SHOT_REJECTING_COMMAND_ERRORS = frozenset(
    {
        0,  # "You can't do this" -- aim outside the viewport
        3,  # "Friendly fire!"
        8,  # "Insufficient fuel"
    }
)

log = get_logger(__name__)


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
        try:
            _tick_once(bot)
            _sync_live_view_demand(bot)
        except TargetClosedError:
            log.info("Browser closed during tick, ending run gracefully")
            _emit_session_scorecard(bot, ticks_done, exit_reason="browser_closed")
            return
        except SessionExitError as exit_request:
            log.info("Session exit: %s -- %s", exit_request.reason, exit_request.detail)
            _emit_session_scorecard(bot, ticks_done, exit_reason=exit_request.reason)
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
    ws = get_world_service()
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
    unresolved = verify_outcome_invariant()
    for kind in ACTION_KINDS:
        counts = outcome_counts(kind)
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
    damage_book = get_world_service().damage_book
    fuel_totals = get_world_service().fuel_book["totals"]
    emit_diagnostic(
        diagnostic_kind="damage_ledger",
        dealt=summarize_side(damage_book["dealt"]),
        taken=summarize_side(damage_book["taken"]),
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


_WS_READY_STATE_OPEN = 1

#: True when an OS signal (SIGINT / SIGTERM) has requested a graceful
#: shutdown. The tick loop checks this once per iteration so the bot
#: exits at a clean tick boundary, writing the session scorecard +
#: index row before the process dies. Reset to ``False`` by
#: :func:`reset_interrupt_flag` so consecutive sessions start clean.
# Final-stretch length of a bounded session spent disengaging and
# topping off for the clean ``session_complete`` exit.
_WIND_DOWN_SECONDS = 60

_INTERRUPT_REQUESTED: bool = False


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


def _sync_live_view_demand(bot: Bot) -> None:
    """Keep the in-page caster matched to viewer demand.

    Runs once per tick, directly after ``_tick_once`` inside the same
    ``TargetClosedError`` guard (a closed browser during the toggle
    ends the run as ``browser_closed``). Demand is the frame bus's
    subscriber count: a ``/video`` (or ``/frame``) connection on the
    service creates it; the last disconnect removes it. While demand
    holds, :meth:`LiveViewService.ensure` re-evaluates the idempotent
    caster snippet EVERY tick — that repetition is the self-heal for
    page navigations, which wipe injected JS. Sessions nobody watches
    never run the caster, and ``make run`` / replay sessions (inert
    default bus, zero subscribers) never start it at all.

    Skipped silently before the CDP session is attached — the tick
    loop's readiness gates run this only in ticks where the browser is
    already up, but the very first iterations of a session can land
    here pre-attach.

    Args:
        bot: Bot instance whose ``_live_view`` and ``_frame_bus``
            drive the decision.
    """
    cdp = bot._cdp
    if cdp is None:
        return
    if bot._frame_bus.subscriber_count() > 0:
        bot._live_view.ensure(cdp)
    elif bot._live_view.active:
        bot._live_view.stop(cdp)


def _enforce_autoscroll_once(bot: Bot) -> None:
    """Run the autoscroll-off dance on the FIRST spawned tick only.

    The toggle only acks in-game (user ruling 2026-07-29: "you cant
    enable or disable autoscroll til the bot is in the game"), and the
    caller invokes this after confirming ``self_state`` -- fed by THIS
    tick's drain -- which is the proof of being in-game. Wiring it
    before the tick loop was the 23:08/23:16 double failure: the world
    service is pull-fed, so a pre-loop wait starved forever on a state
    nothing was draining yet.

    Args:
        bot: Bot instance carrying the one-shot latch and live page.
    """
    if bot._autoscroll_enforced or bot._page is None:
        return
    _test_hooks.ensure_autoscroll_off(bot._page, bot._messages)
    bot._autoscroll_enforced = True


# How long a dead tank waits for its respawn sync before the session
# exits ``deactivated`` anyway. The real server respawns promptly; a
# world that never respawns (the sim has no respawn law) must not
# wait forever on a sync that cannot come.
_RESPAWN_WAIT_MS = 60_000


def _handle_own_deactivation(bot: Bot, self_state: SelfStateDict) -> None:
    """Reset the corpse's beliefs and start the respawn wait.

    User contract 2026-07-30: "if the tank dies, it should just wait
    for respawn and then go into collecting mode." Every tactical
    belief the dead tank carried — combat lock, escape latch,
    resource locks, in-flight action — describes a tank that no
    longer exists, so the AI state rebuilds from initial values with
    only the session-scoped facts carried over (kill count, wind-down
    flag, greeting latch, config). The dead self record is dropped
    outright (the post-mortem fuel field reads garbage, e.g. 65482 in
    run bot-20260730-004144), which turns the loop's self-None early
    exit into the wait itself.

    Args:
        bot: Bot instance.
        self_state: The corpse's final self record, for the receipt.
    """
    service = get_world_service()
    service.self_deactivated = False
    emit_diagnostic(
        diagnostic_kind="self_respawn_wait",
        died_x=self_state["x"],
        died_y=self_state["y"],
        session_kills=bot._ai_state["session_kill_count"],
    )
    log.info(
        "Deactivated at (%d,%d) - waiting for respawn, then collecting",
        self_state["x"],
        self_state["y"],
    )
    fresh = make_initial_ai_state(bot._ai_state["config"])
    fresh["session_kill_count"] = bot._ai_state["session_kill_count"]
    fresh["wind_down"] = bot._ai_state["wind_down"]
    fresh["greeted_target_id"] = bot._ai_state["greeted_target_id"]
    bot._ai_state = fresh
    bot._state_data = make_initial_state_data()
    service.world_state["self_state"] = None
    bot._respawn_deadline_ms = get_current_time_ms() + _RESPAWN_WAIT_MS


def _note_respawn(bot: Bot, self_state: SelfStateDict) -> None:
    """Clear the respawn wait once fresh self state proves the respawn.

    The tank is back on the field with empty stocks, so the normal
    arbitration hands the next tick to COLLECT exactly as the
    contract asks.

    Args:
        bot: Bot instance.
        self_state: The freshly synced self record.
    """
    if bot._respawn_deadline_ms <= 0:
        return
    log.info(
        "Respawned at (%d,%d) fuel=%d - resuming with collection",
        self_state["x"],
        self_state["y"],
        self_state["fuel"],
    )
    bot._respawn_deadline_ms = 0


def _check_respawn_deadline(bot: Bot) -> None:
    """Fail the respawn wait loudly once the deadline passes.

    Args:
        bot: Bot instance.

    Raises:
        SessionExitError: When the tank died and no respawn sync
            arrived within :data:`_RESPAWN_WAIT_MS`.
    """
    if bot._respawn_deadline_ms > 0 and get_current_time_ms() >= bot._respawn_deadline_ms:
        raise SessionExitError(
            "deactivated",
            f"no respawn sync within {_RESPAWN_WAIT_MS // 1000}s of the own 0x41",
        )


def _tick_once(bot: Bot) -> None:
    """Execute one sync-decide-execute cycle.

    Args:
        bot: Bot instance.
    """
    # 1. SYNC — drain CDP message buffer
    world_sync.drain_messages(bot)

    # 1b. Record the in-game text log as a capture witness. The DOM log
    # is the client's rendering of wire messages the bot already decodes
    # (0x41 Deactivation for kills, 0x52 error codes for rejections --
    # capture replay 2026-07-19 falsified the June claim that the wire
    # was silent on own kills and failed pickups: the messages were
    # 0x2E-tunneled and the June decoder could not unwrap them). The
    # entries act on nothing; they land in the capture artifact so the
    # analyzer can diff the client's rendering against the wire.
    bot._record_game_log_witness(bot._poll_game_log())

    # 2. Read state
    world = bot.get_world_state()
    self_state = world["self_state"]
    if self_state is None:
        _check_respawn_deadline(bot)
        return

    _note_respawn(bot, self_state)
    _enforce_autoscroll_once(bot)

    # 2a. The wire announced OUR OWN death (0x41, victim == self).
    # User contract 2026-07-30 ("if the tank dies, it should just
    # wait for respawn and then go into collecting mode"): instead of
    # ending the session, reset every tactical belief the corpse
    # carried and wait for the respawn sync. The Artax death receipt
    # (run bot-20260730-004144, fuel read 65482 post-mortem) shows
    # the dead self_state is garbage, so it is dropped outright and
    # the loop's self-None early exit becomes the wait.
    if get_world_service().self_deactivated:
        _handle_own_deactivation(bot, self_state)
        return

    bot._update_state_from_world()
    if not _is_ready_for_decision(bot):
        return
    if has_in_flight_action(bot):
        return

    # Re-read state after sync/action resolution. In-flight handlers may
    # mutate world state (for example marking failed pickups) before
    # allowing replanning in the same tick.
    world = bot.get_world_state()
    self_state = world["self_state"]
    if self_state is None:
        return

    # 3. Merge kills from protocol
    bot._ai_state = _merge_protocol_kills(bot._ai_state)
    now = get_current_time_ms()
    if _has_pending_shot_feedback(bot, now):
        return

    # 4. Authoritative live-client read, shared by the registry truth
    # ingest, the decision, and the dispatch boundary gates (map already
    # open, WS down, JS hung). Done after every early-exit gate so a
    # tick that decides nothing pays no CDP cost.
    snapshot = capture_page_client_snapshot(bot._require_cdp())
    if not _is_page_client_healthy(snapshot):
        return

    world = bot.get_world_state()

    # 5. Combat feedback (counters incremented inside _get_combat_feedback)
    combat_feedback = _get_combat_feedback(bot)

    # 6. DECIDE
    inventory = get_inventory_state(get_world_service())
    terrain = get_terrain_map()

    decision = ai_strategy.decide(
        world,
        self_state,
        bot._ai_state,
        inventory,
        now,
        terrain,
        combat_feedback,
        get_world_service().map_fuel_dots,
    )

    maybe_emit_self_alignment_sample(self_state, snapshot)
    maybe_emit_entity_alignment_sample(
        world,
        snapshot,
        in_combat=bot._ai_state["mode"] == "HUNT",
    )
    maybe_emit_client_structure_survey(bot._require_cdp())
    # Account-wide ground truth (lifetime kills, play time, promotion
    # points) baselined on the first healthy tick; the loading screen
    # ignores the C hotkey at bootstrap.
    bot.maybe_capture_account_stats_once()

    # 7. EXECUTE — game queues commands
    command_sent = executor.execute(bot, decision, snapshot)

    # 8. Persist AI state only after the command actually dispatches.
    # This prevents speculative shot feedback state from leaking across
    # executor-side validation failures.
    if command_sent:
        previous_mode = bot._ai_state["mode"]
        bot._ai_state = decision["updated_ai_state"]
        if bot._ai_state["mode"] != previous_mode:
            emit_mode_transition(
                from_mode=previous_mode,
                to_mode=bot._ai_state["mode"],
                reason_kind=decision["behavior"]["reason_kind"],
                caused_by=(
                    0 if decision["command"]["cmd_type"] == "hold" else latest_decision_event_id()
                ),
            )

    # 9. Update the in-page HUD so a human watching the browser sees what
    # the bot decided this tick without tailing artifacts, keep the flag
    # binding armed, and ring-buffer the payload so a flag click can
    # snapshot the ticks that led up to it.
    overlay = OverlayStateDict(
        hfsm_state=bot.get_state(),
        ai_mode=bot._ai_state["mode"],
        ai_mode_state=bot._ai_state["mode_state"],
        behavior_mode=decision["behavior"]["mode"],
        behavior_reason=render_reason(decision["behavior"]),
        command_type=decision["command"]["cmd_type"],
        target_x=decision["behavior"]["target_x"],
        target_y=decision["behavior"]["target_y"],
        command_sent=command_sent,
        in_flight_kind=bot._state_data["in_flight_action"]["kind"],
        fuel=self_state["fuel"],
        fuel_cap=fuel_capacity(self_state["rank"]),
        self_x=self_state["x"],
        self_y=self_state["y"],
        armor=inventory["armor_shields"]["count"],
        duals=inventory["dual_shots"]["count"],
        missiles=inventory["missile_shots"]["count"],
        homings=inventory["homing_shots"]["count"],
        radars=inventory["extra_radars"]["count"],
        inv_cap=inventory_capacity(self_state["rank"]),
        kills=bot._ai_state["session_kill_count"],
        hits=bot._ai_state["session_hit_count"],
        misses=bot._ai_state["session_miss_count"],
        rejects=bot._ai_state["session_reject_count"],
        target_id=bot._ai_state["combat_target_id"],
        target_name=bot._ai_state["last_shot_target_name"],
    )
    hud_cdp = bot._require_cdp()
    bot._flag_capture.ensure(hud_cdp)
    bot._flag_capture.record_tick(overlay)
    update_bot_overlay(hud_cdp, overlay)


def _is_page_client_healthy(snapshot: PageClientSnapshotDict) -> bool:
    """Return True when the live JS client is ready to receive commands.

    Reads the authoritative live signals from the captured snapshot rather
    than guessing from local send-side state. Two failure modes block
    the tick:

    1. ``client_present`` is False -- the inject script hasn't captured
       ``window.__tankpitActiveGame`` yet, so the game isn't initialized.
    2. ``ws_ready_state`` is anything other than ``OPEN`` (1) -- sends
       would land in a dead socket. ``None`` means the socket has not
       been captured yet; the tick waits rather than dispatching
       against unknown state.

    ``heartbeat_age_ms`` (``activeGame.va.j``) is deliberately NOT a
    health signal: live-run measurement (run 20260609-233736) showed it
    refreshes only about every 30 seconds while the wire is demonstrably
    alive, so gating on it froze the bot ~25 of every 30 seconds. The
    browser WebSocket ``readyState`` is the authoritative liveness
    signal.
    """
    if not snapshot["client_present"]:
        emit_sync("page client not present; skipping tick")
        return False
    ws_state = snapshot["ws_ready_state"]
    if ws_state != _WS_READY_STATE_OPEN:
        emit_sync("page websocket not OPEN (ws_ready_state=%s); skipping tick", str(ws_state))
        return False
    return True


def _is_ready_for_decision(bot: Bot) -> bool:
    """Return True when the bot state machine is ready to execute AI plans.

    The planner may only dispatch commands from executable states. During
    startup and disconnect handling, world state may already exist while the
    bot state machine is still converging to ``IDLE``. Replanning in those
    transitional states can produce invalid command transitions.

    Args:
        bot: Bot instance.

    Returns:
        True if the bot can safely plan and execute a command this tick.
    """
    state = bot.get_state()
    if state in ("INITIALIZING", "WAITING_FOR_POSITION", "DISCONNECTED"):
        emit_sync("deferring decisions while state=%s", state)
        return False
    return True


def _merge_protocol_kills(ai_state: AIStateDict) -> AIStateDict:
    """Merge Deactivation kills from protocol into AI killed_tank_ids.

    Args:
        ai_state: Current AI state.

    Returns:
        Updated AI state with new kills merged.
    """
    new_kills = drain_killed_tank_ids(get_world_service())
    if not new_kills:
        return ai_state
    now = get_current_time_ms()
    merged = dict(ai_state["killed_tank_ids"])
    for tank_id in new_kills:
        merged[str(tank_id)] = now
        emit_ai("kill registered (tank_id=%d)", tank_id)
    # The shot-target fields are NOT cleared here: when the killed tank
    # is the pending shot's target, ``_get_combat_feedback`` must still
    # see the target id to resolve the shot as ``kill_confirmed`` (a
    # kill produces no damage-change feedback, so this is the kill
    # shot's only resolution path). The classifier clears the fields
    # itself after emitting the outcome.
    clear_combat_target = ai_state["combat_target_id"] in new_kills
    return AIStateDict(
        **{
            **ai_state,
            "killed_tank_ids": merged,
            "session_kill_count": ai_state["session_kill_count"] + len(new_kills),
            "combat_target_id": -1 if clear_combat_target else ai_state["combat_target_id"],
            "combat_target_x": 0 if clear_combat_target else ai_state["combat_target_x"],
            "combat_target_y": 0 if clear_combat_target else ai_state["combat_target_y"],
        }
    )


def _has_pending_shot_feedback(bot: Bot, timestamp_ms: int) -> bool:
    """Return True while a fired shot is still inside its feedback window.

    Args:
        bot: Bot instance.
        timestamp_ms: Current timestamp in milliseconds.

    Returns:
        True if a shot outcome is still pending and the tick should wait.
    """
    target_id = bot._ai_state["last_shot_target_id"]
    if target_id == -1:
        return False
    if peek_combat_hit(get_world_service()):
        return False
    if peek_our_shot_response(get_world_service()):
        return False
    if peek_command_error(get_world_service()) in _SHOT_REJECTING_COMMAND_ERRORS:
        # The server refused the shot outright -- no ShootEvent or
        # ammo delta will ever arrive, so waiting out the feedback
        # window is pure dead time. The classifier consumes the error.
        return False
    if str(target_id) in bot._ai_state["killed_tank_ids"]:
        return False
    elapsed_ms = timestamp_ms - bot._ai_state["last_shoot_ms"]
    if elapsed_ms >= bot._ai_state["config"]["shot_feedback_timeout_ms"]:
        return False
    emit_sync(
        "waiting for shot feedback for %s (id=%d)",
        bot._ai_state["last_shot_target_name"],
        target_id,
    )
    return True


def _get_combat_feedback(bot: Bot) -> CombatFeedback:
    """Get combat feedback from the per-shot ammo consumption ledger.

    **Consumption = hit** (user contract 2026-07-02): the server only
    spends dual / missile / homing ammo on a shot that lands, and the
    0x53 ShootEvent ``weapon`` field records the spend per shot --
    the same per-shot inventory delta the page client renders.
    ``weapon=0`` (free single, resolved against empty ground) spends
    nothing and is a genuine miss. The earlier tile-occupancy signal
    (``victim_id``) classified off-viewport pursuit hits as misses
    because the impact tile is outside the local registry's view (run
    2026-07-02 01:21: five ``weapon=3`` debits killed orange-3 while
    ``victim_id`` was -1 on every shot).

    Outcomes:
      - ammo debited (weapon > 0)   -> "hit"
      - target already in killed    -> "hit" (kill confirmed)
      - 0x49 sync shows a debit the shot event missed -> "hit"
      - shot response, no debit     -> "miss"
      - 0x52 shot-rejecting error   -> "rejected" (server refused the
        dispatch; neither hit nor miss -- no ammo moved)
      - no response yet             -> ""  (keep waiting)

    Ammo count is decremented in ``mark_combat_hit`` per the weapon
    byte. Combat feedback never rewrites inventory counts -- wire
    messages 0x49, 0x67, 0x74 remain the absolute authority the
    shadow count reconciles against.

    Args:
        bot: Bot instance.

    Returns:
        "hit", "miss", or "" when feedback is indeterminate.
    """
    target_id = bot._ai_state["last_shot_target_id"]
    target_name = bot._ai_state["last_shot_target_name"]
    if target_id == -1:
        return ""
    got_hit = check_and_clear_combat_hit(get_world_service())
    victim_id = check_and_clear_last_shot_victim_id(get_world_service())
    got_response = check_and_clear_our_shot_response(get_world_service())
    ammo_hit = check_and_clear_ammo_delta_hit(get_world_service())

    def _inc_hit() -> None:
        bot._ai_state = AIStateDict(
            **{**bot._ai_state, "session_hit_count": bot._ai_state["session_hit_count"] + 1}
        )

    def _inc_miss() -> None:
        bot._ai_state = AIStateDict(
            **{**bot._ai_state, "session_miss_count": bot._ai_state["session_miss_count"] + 1}
        )

    duration_ms = get_current_time_ms() - bot._ai_state["last_shoot_ms"]
    if got_hit:
        # Distinguish intended-target hit from incidental hit (e.g.
        # homing seeker landed on a closer enemy than commanded).
        on_intended = victim_id == target_id
        emit_shoot_hit(
            duration_ms=duration_ms,
            target_id=target_id,
            target_name=target_name,
            victim_id=victim_id,
            on_intended_target=on_intended,
            hit_signal="tile_occupied",
        )
        resolve_dealt(get_world_service().damage_book, victim_id, target_name, target_id)
        _inc_hit()
        return "hit"
    if str(target_id) in bot._ai_state["killed_tank_ids"]:
        emit_shoot_hit(
            duration_ms=duration_ms,
            target_id=target_id,
            target_name=target_name,
            victim_id=target_id,
            on_intended_target=True,
            hit_signal="kill_confirmed",
        )
        resolve_dealt(get_world_service().damage_book, target_id, target_name, target_id)
        _inc_hit()
        # Clear the shot-target fields directly: the trigger is
        # ``killed_tank_ids`` membership, which is not a consumable
        # wire flag -- without the clear, a tick that dispatches no
        # command (so ``updated_ai_state`` never persists) would
        # re-emit this outcome every tick until one does.
        bot._ai_state = AIStateDict(
            **{
                **bot._ai_state,
                "last_shot_target_id": -1,
                "last_shot_target_name": "",
            }
        )
        return "hit"
    if ammo_hit:
        # Reconciliation channel: the per-shot ``weapon`` byte is the
        # primary consumption signal (handled above via got_hit), but
        # if the 0x53 echo is lost, the server's 0x49 absolute
        # inventory sync still reveals the debit against the pre-shot
        # snapshot. A debit is a landed shot regardless of which wire
        # channel reported it.
        emit_shoot_hit(
            duration_ms=duration_ms,
            target_id=target_id,
            target_name=target_name,
            victim_id=victim_id,
            on_intended_target=True,
            hit_signal="ammo_delta",
        )
        resolve_dealt(get_world_service().damage_book, victim_id, target_name, target_id)
        _inc_hit()
        return "hit"
    if got_response:
        # No tile-occupied hit, no ammo debit, and a wire response did
        # arrive -- the shot genuinely missed.
        emit_shoot_miss(
            duration_ms=duration_ms,
            target_id=target_id,
            target_name=target_name,
        )
        _inc_miss()
        return "miss"
    if peek_command_error(get_world_service()) in _SHOT_REJECTING_COMMAND_ERRORS:
        error_code = check_and_clear_command_error(get_world_service())
        emit_shoot_command_rejected(
            duration_ms=duration_ms,
            target_id=target_id,
            target_name=target_name,
            error_code=error_code,
        )
        bot._ai_state = AIStateDict(
            **{
                **bot._ai_state,
                "session_reject_count": bot._ai_state["session_reject_count"] + 1,
            }
        )
        return "rejected"
    return ""


__all__ = [
    "run_tick_loop",
]
