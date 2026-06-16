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
from tankpit_bot.diagnostics.registry_truth import register_tank_truth_from_page_snapshot
from tankpit_bot.diagnostics.self_alignment import maybe_emit_self_alignment_sample
from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.runtime_logging import emit_ai, emit_sync
from tankpit_bot.sniffer.world_state import (
    get_terrain_map,
    get_world_service,
)
from tankpit_bot.sniffer.world_state_combat import (
    check_and_clear_combat_hit,
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
        _tick_once(bot)
        ticks_done += 1
        if max_ticks > 0 and ticks_done >= max_ticks:
            log.info(
                "Session tick budget reached (%d ticks / %ds), ending run",
                max_ticks,
                session_seconds,
            )
            return
        if _test_hooks.path_exists(stop_file_path):
            _test_hooks.remove_file(stop_file_path)
            log.info("Stop file %s detected, ending run", stop_file_path)
            return
        page.wait_for_timeout(TICK_RATE_MS)


_WS_READY_STATE_OPEN = 1


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

    # 4b. Re-anchor rendered tanks from the client registry. The wire is
    # silent on enemy positions between movement messages; the registry
    # gives every visible enemy's current tile each tick, so HUNT
    # engages from live positions instead of stale map intel. World
    # state is re-read because ingestion replaces the world dict.
    register_tank_truth_from_page_snapshot(snapshot, world)
    world = bot.get_world_state()

    # 5. Combat feedback
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
    """Get combat feedback from protocol weapon byte.

    Hit detection relies on the weapon byte in CombatHit responses:
    - weapon_byte > 0: special ammo used = confirmed hit
    - weapon_byte == 0 with dual enabled + stocked: miss (target empty)
    - weapon_byte == 0 without dual: normal single shot, can't tell
    - no response (timeout) with dual enabled + stocked: miss
    - no response without dual: can't tell

    Ammo count is decremented on each confirmed hit by mark_combat_hit.
    Combat feedback never rewrites inventory counts — protocol messages
    (0x49, 0x67, 0x74) are the sole authority for item counts.

    Args:
        bot: Bot instance.

    Returns:
        "hit" if weapon byte > 0 or kill confirmed, "miss" if dual was
        available but no hit detected, "" if feedback is indeterminate.
    """
    if bot._ai_state["last_shot_target_id"] == -1:
        log.info("FEEDBACK: no shot pending (last_shot_target_id=-1)")
        return ""
    got_hit = check_and_clear_combat_hit(get_world_service())
    got_response = check_and_clear_our_shot_response(get_world_service())
    if got_hit:
        log.info("FEEDBACK: hit confirmed")
        return "hit"
    if str(bot._ai_state["last_shot_target_id"]) in bot._ai_state["killed_tank_ids"]:
        log.info("FEEDBACK: kill confirmed")
        return "hit"
    inventory = get_inventory_state(get_world_service())
    dual_available = inventory["dual_shots"]["enabled"] and inventory["dual_shots"]["count"] > 0
    if got_response:
        if dual_available:
            log.info("FEEDBACK: miss (dual active, server used single)")
            return "miss"
        dual_count = inventory["dual_shots"]["count"]
        log.info(
            "FEEDBACK: single shot (no dual, count=%d)",
            dual_count,
        )
        return ""
    # No CombatHit response at all (timeout).
    if dual_available:
        log.info("FEEDBACK: miss (dual active, no combat hit)")
        return "miss"
    log.info(
        "FEEDBACK: no dual, ignoring (count=%d)",
        inventory["dual_shots"]["count"],
    )
    return ""


__all__ = [
    "run_tick_loop",
]
