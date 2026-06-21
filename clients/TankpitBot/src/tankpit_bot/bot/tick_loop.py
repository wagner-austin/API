"""Tick loop orchestrator for the bot.

Runs the sync-decide-execute cycle on each server tick. The game uses
a command queue — move/pickup commands are queued and the tank walks
there automatically. No need to wait for arrival.

Cycle: SYNC → DECIDE → EXECUTE → WAIT
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import PageProtocol
from tankpit_bot.action_lab.client_structure import maybe_emit_client_structure_survey
from tankpit_bot.action_lab.page_client_snapshot import (
    PageClientSnapshotDict,
    capture_page_client_snapshot,
)
from tankpit_bot.bot import ai_strategy, executor, world_sync
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.browser.overlay import OverlayStateDict, update_bot_overlay
from tankpit_bot.diagnostics.entity_alignment import maybe_emit_entity_alignment_sample
from tankpit_bot.diagnostics.game_log_feedback import register_world_feedback_from_game_log
from tankpit_bot.diagnostics.game_log_kills import register_kills_from_game_log
from tankpit_bot.diagnostics.runs_index import (
    append_index_row,
    count_stall_timeouts,
    make_index_row,
)
from tankpit_bot.diagnostics.self_alignment import maybe_emit_self_alignment_sample
from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.runtime_logging import (
    emit_ai,
    emit_diagnostic,
    emit_sync,
    get_bot_runtime_artifacts,
    set_runtime_context,
)
from tankpit_bot.sniffer.world_state import (
    get_terrain_map,
    get_world_service,
)
from tankpit_bot.sniffer.world_state_combat import (
    check_and_clear_combat_hit,
    check_and_clear_last_shot_victim_id,
    check_and_clear_our_shot_response,
    drain_killed_tank_ids,
    peek_combat_hit,
    peek_our_shot_response,
)
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state

log = get_logger(__name__)


def run_tick_loop(
    bot: Bot,
    page: PageProtocol,
    *,
    session_seconds: int,
    stop_file_path: Path,
) -> None:
    """Run the main tick loop.

    A positive ``session_seconds`` bounds the session at
    ``seconds * 1000 // TICK_RATE_MS`` ticks; the loop then returns so
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
        stop_file_path: Sentinel file whose existence requests a
            graceful shutdown.
    """
    max_ticks = session_seconds * 1000 // TICK_RATE_MS if session_seconds > 0 else 0
    ticks_done = 0
    while True:
        _publish_tick_context(bot, ticks_done + 1)
        _tick_once(bot)
        ticks_done += 1
        if max_ticks > 0 and ticks_done >= max_ticks:
            log.info(
                "Session tick budget reached (%d ticks / %ds), ending run",
                max_ticks,
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
        page.wait_for_timeout(TICK_RATE_MS)


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
    dual = inv["dual_shots"]["count"]
    homing = inv["homing_shots"]["count"]
    radar = inv["extra_radars"]["count"]
    blocked = len(ai["blocked_combat_targets"])
    mode = ai["mode"]
    mode_state = ai["mode_state"]
    emit_diagnostic(
        diagnostic_kind="session_scorecard",
        ticks=ticks,
        kills=kills,
        hits=hits,
        misses=misses,
        fuel_remaining=fuel,
        dual_shots_remaining=dual,
        homing_shots_remaining=homing,
        extra_radars_remaining=radar,
        targets_blocked=blocked,
        ai_mode=mode,
        ai_mode_state=mode_state,
        exit_reason=exit_reason,
    )
    shots = hits + misses
    hit_rate = f"{hits * 100 // shots}%" if shots > 0 else "n/a"
    summary = (
        f"TANKPIT SESSION SUMMARY\n"
        f"{'=' * 40}\n"
        f"Ticks:    {ticks}\n"
        f"Exit:     {exit_reason}\n"
        f"Kills:    {kills}\n"
        f"Shots:    {shots} ({hits} hits, {misses} misses)\n"
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


def _tick_once(bot: Bot) -> None:
    """Execute one sync-decide-execute cycle.

    Args:
        bot: Bot instance.
    """
    # 1. SYNC — drain CDP message buffer
    world_sync.drain_messages(bot)

    # 1b. Consume the in-game text log. The wire 0x41 Deactivation never
    # arrives for own kills (proven across two live runs) and the wire is
    # silent on failed pickups, full-tank pickups, and rejected moves --
    # the rendered log lines are the only truth channel for all four.
    log_entries = bot._poll_game_log()
    log_world = bot.get_world_state()
    register_kills_from_game_log(log_entries, log_world)
    register_world_feedback_from_game_log(log_entries, log_world)

    # 2. Read state
    world = bot.get_world_state()
    self_state = world["self_state"]
    if self_state is None:
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
        bot._ai_state = decision["updated_ai_state"]

    # 9. Update the in-page HUD so a human watching the browser sees what
    # the bot decided this tick without tailing artifacts.
    update_bot_overlay(
        bot._require_cdp(),
        OverlayStateDict(
            hfsm_state=bot.get_state(),
            ai_mode=bot._ai_state["mode"],
            ai_mode_state=bot._ai_state["mode_state"],
            behavior_mode=decision["behavior"]["mode"],
            behavior_reason=decision["behavior"]["reason"],
            command_type=decision["command"]["cmd_type"],
            target_x=decision["behavior"]["target_x"],
            target_y=decision["behavior"]["target_y"],
            command_sent=command_sent,
            in_flight_kind=bot._state_data["in_flight_action"]["kind"],
            fuel=self_state["fuel"],
            self_x=self_state["x"],
            self_y=self_state["y"],
        ),
    )


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
    clear_shot_target = ai_state["last_shot_target_id"] in new_kills
    clear_combat_target = ai_state["combat_target_id"] in new_kills
    return AIStateDict(
        **{
            **ai_state,
            "killed_tank_ids": merged,
            "session_kill_count": ai_state["session_kill_count"] + len(new_kills),
            "last_shot_target_id": -1 if clear_shot_target else ai_state["last_shot_target_id"],
            "last_shot_target_name": "" if clear_shot_target else ai_state["last_shot_target_name"],
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
    """Get combat feedback from the wire tile-occupancy signal.

    The 0x53 ShootEvent decoder extracts ``target_x``, ``target_y``
    from the wire and the dispatcher looks up which tank (if any) was
    on that tile. That tile-occupancy result is the authoritative hit
    signal per tpclient.js ``Gg.prototype.h`` (case 18 -> "You hit X"),
    which the old weapon_byte heuristic could not reliably reproduce --
    weapon=0 is indistinguishable from miss per the wiki.

    Outcomes:
      - tile had any tank        -> "hit"
      - target already in killed -> "hit" (kill confirmed)
      - shot response, empty tile -> "miss"
      - no response yet           -> ""  (keep waiting)

    Ammo count is decremented in ``mark_combat_hit`` (legacy function
    name predating the 2026-06-19 decoder unification; the function
    now consumes ShootEvent fields, not the deleted CombatHit
    container). Combat feedback never rewrites inventory counts --
    wire messages 0x49, 0x67, 0x74 are the sole authority for item
    counts.

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

    def _inc_hit() -> None:
        bot._ai_state = AIStateDict(
            **{**bot._ai_state, "session_hit_count": bot._ai_state["session_hit_count"] + 1}
        )

    def _inc_miss() -> None:
        bot._ai_state = AIStateDict(
            **{**bot._ai_state, "session_miss_count": bot._ai_state["session_miss_count"] + 1}
        )

    if got_hit:
        # Distinguish intended-target hit from incidental hit (e.g.
        # homing seeker landed on a closer enemy than commanded).
        on_intended = victim_id == target_id
        emit_diagnostic(
            diagnostic_kind="combat_feedback",
            result="hit",
            reason="tile_occupied",
            target_name=target_name,
            target_id=target_id,
            actual_victim_id=victim_id,
            on_intended_target=on_intended,
        )
        _inc_hit()
        return "hit"
    if str(target_id) in bot._ai_state["killed_tank_ids"]:
        emit_diagnostic(
            diagnostic_kind="combat_feedback",
            result="kill",
            target_name=target_name,
            target_id=target_id,
        )
        _inc_hit()
        return "hit"
    if got_response:
        emit_diagnostic(
            diagnostic_kind="combat_feedback",
            result="miss",
            reason="tile_empty",
            target_name=target_name,
            target_id=target_id,
        )
        _inc_miss()
        return "miss"
    return ""


__all__ = [
    "run_tick_loop",
]
