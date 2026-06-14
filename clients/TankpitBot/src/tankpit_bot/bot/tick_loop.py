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
from tankpit_bot.bot.ai.reachability import (
    is_collection_reachable_in_viewport,
    is_move_reachable_in_viewport,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.bot.states import InFlightActionDict, make_no_action, transition_to
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.browser.overlay import OverlayStateDict, update_bot_overlay
from tankpit_bot.diagnostics.entity_alignment import maybe_emit_entity_alignment_sample
from tankpit_bot.diagnostics.game_log_feedback import register_world_feedback_from_game_log
from tankpit_bot.diagnostics.game_log_kills import register_kills_from_game_log
from tankpit_bot.diagnostics.registry_truth import register_tank_truth_from_page_snapshot
from tankpit_bot.diagnostics.self_alignment import maybe_emit_self_alignment_sample
from tankpit_bot.diagnostics.teleport_attempts import emit_teleport_attempt_outcome
from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.runtime_logging import emit_ai, emit_sync, emit_wire_complete
from tankpit_bot.sniffer.world_state import (
    check_and_clear_map_data_processed,
    get_terrain_map,
    is_move_target_failed,
    mark_move_target_failed,
    mark_scan_viewport_failed,
)
from tankpit_bot.sniffer.world_state_combat import (
    check_and_clear_combat_hit,
    check_and_clear_our_shot_response,
    drain_killed_tank_ids,
    peek_combat_hit,
    peek_our_shot_response,
)
from tankpit_bot.sniffer.world_state_containers import increment_container_failed_pickups
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
    if _has_in_flight_action(bot):
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
    inventory = get_inventory_state()
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


def _has_in_flight_action(bot: Bot) -> bool:
    """Return True when a previously issued action is still resolving.

    Reads the authoritative InFlightActionDict from bot state. Every
    action kind is handled explicitly — move, collect, teleport, scan,
    shoot, and map_open. There is no implicit fallback; the action
    record is the single source of truth for command lifecycle.

    Args:
        bot: Bot instance.

    Returns:
        True if the bot should wait for an in-flight action to resolve.
    """
    action = bot._state_data["in_flight_action"]
    if action["kind"] == "none" or action["outcome"] != "pending":
        return False

    kind = action["kind"]
    if kind in ("move", "collect", "teleport"):
        return _wait_for_movement_action(bot, action)

    # Scan: waits for radar completion signal, subject to stall timeout
    if kind == "scan":
        return _wait_for_scan_action(bot, action)

    # map_open: wait for at least one fresh server sync before replanning.
    # The game does not expose a reliable map-open flag, so we only use the
    # recorded action timestamp plus fresh world sync as the sequencing guard.
    if kind == "map_open":
        return _wait_for_map_open_action(bot, action)

    # Shoot: fire-and-forget, do not block replanning.
    # The action record is authoritative (records what was sent) but
    # this action resolves immediately — no server confirmation needed.
    return False


def _wait_for_movement_action(bot: Bot, action: InFlightActionDict) -> bool:
    """Return True while a move/collect/teleport action is still resolving."""
    kind = action["kind"]
    tx, ty = action["target_x"], action["target_y"]
    if _clear_rejected_movement(bot, action):
        return False
    if _clear_stalled_action(bot, action):
        return False
    if kind == "move" and _clear_blocked_walk(bot, action):
        return False
    if kind == "collect" and _clear_blocked_collection(bot, action):
        return False
    if kind == "move":
        emit_sync("waiting for movement to (%d,%d)", tx, ty)
    elif kind == "teleport":
        emit_sync("waiting for teleport to (%d,%d)", tx, ty)
    else:
        emit_sync("waiting for collection at (%d,%d)", tx, ty)
    return True


def _wait_for_scan_action(bot: Bot, action: InFlightActionDict) -> bool:
    """Return True while a radar scan is still pending."""
    if _clear_stalled_action(bot, action):
        return False
    emit_sync("waiting for radar results")
    return True


def _wait_for_map_open_action(bot: Bot, action: InFlightActionDict) -> bool:
    """Return True while a map-open action is waiting on fresh server sync."""
    if _clear_stalled_action(bot, action):
        return False
    if _clear_completed_map_open(bot, action):
        return False
    emit_sync("waiting for map open sync")
    return True


def _clear_rejected_movement(
    bot: Bot,
    action: InFlightActionDict,
) -> bool:
    """Clear a move/collect whose target the server rejected.

    The "You can't go there!" game-log feedback marks the dispatch
    target as a failed move target the moment the line renders;
    waiting out the stall timer after that point is pure dead time
    (live run 20260611-000x: two 10s stalls whose rejections had
    arrived within 2s of dispatch).

    Args:
        bot: Bot instance.
        action: The in-flight action record to check.

    Returns:
        True if the rejected action was cleared and the tick should
        replan.
    """
    kind = action["kind"]
    if kind not in ("move", "collect"):
        return False
    tx, ty = action["target_x"], action["target_y"]
    now = get_current_time_ms()
    if not is_move_target_failed(tx, ty, now):
        return False
    started_ms = action["started_ms"]
    elapsed_ms = now - started_ms if started_ms > 0 else -1
    emit_sync("%s to (%d,%d) rejected by server, replanning", kind, tx, ty)
    emit_wire_complete(
        action_kind=kind,
        duration_ms=elapsed_ms,
        signal="movement_rejected",
        target_x=tx,
        target_y=ty,
    )
    if kind == "collect":
        increment_container_failed_pickups(tx, ty)
        emit_sync("marked container at (%d,%d) as failed pickup", tx, ty)
    bot._transition("IDLE", in_flight_action=make_no_action())
    return True


def _clear_stalled_action(
    bot: Bot,
    action: InFlightActionDict,
) -> bool:
    """Clear a stalled action using its authoritative timing.

    Args:
        bot: Bot instance.
        action: The in-flight action record to check.

    Returns:
        True if the stalled action was cleared and the tick should replan.
    """
    started_ms = action["started_ms"]
    if started_ms <= 0:
        return False
    elapsed_ms = get_current_time_ms() - started_ms
    timeout_ms = bot._ai_state["config"]["action_stall_timeout_ms"]
    if elapsed_ms < timeout_ms:
        return False
    tx, ty = action["target_x"], action["target_y"]
    emit_sync(
        "%s to (%d,%d) stalled for %d ms, replanning",
        action["kind"],
        tx,
        ty,
        elapsed_ms,
    )
    emit_wire_complete(
        action_kind=action["kind"],
        duration_ms=elapsed_ms,
        signal="stall_timeout",
        target_x=tx,
        target_y=ty,
        timeout_ms=timeout_ms,
    )
    if action["kind"] == "collect":
        increment_container_failed_pickups(tx, ty)
        emit_sync("marked container at (%d,%d) as failed pickup", tx, ty)
    if action["kind"] == "scan":
        _mark_current_viewport_scan_failed(bot, get_current_time_ms())
    if action["kind"] in ("move", "teleport"):
        now = get_current_time_ms()
        mark_move_target_failed(tx, ty, now)
        emit_sync("marked (%d,%d) as failed %s target", tx, ty, action["kind"])
    if action["kind"] == "teleport":
        emit_teleport_attempt_outcome(status="stall_timeout", messages=bot._messages)
    bot._transition("IDLE", in_flight_action=make_no_action())
    return True


def _mark_current_viewport_scan_failed(bot: Bot, timestamp_ms: int) -> None:
    """Record the current viewport as a failed radar target.

    Args:
        bot: Bot instance.
        timestamp_ms: Failure timestamp in milliseconds.
    """
    viewport = bot.get_world_state()["viewport"]
    mark_scan_viewport_failed(viewport["left"], viewport["top"], timestamp_ms)
    emit_sync(
        "marked viewport (%d,%d) as failed scan target",
        viewport["left"],
        viewport["top"],
    )


def _clear_completed_map_open(
    bot: Bot,
    action: InFlightActionDict,
) -> bool:
    """Clear a pending map_open once the authoritative MAP_DATA was processed.

    The wire ``CMD_MAP_OPEN`` triggers a MAP_DATA response carrying every
    tank's position; the sniffer dispatcher calls
    :func:`~tankpit_bot.sniffer.world_state.mark_map_data_processed` after
    the blob is decoded into ``world_state["tanks"]``. This gate consumes
    that signal so replanning resumes ONLY when the bot is looking at
    refreshed map intelligence -- not after an incidental ``TankStatus``
    or ``ViewportUpdate`` happens to land first.

    Args:
        bot: Bot instance.
        action: The pending map_open action record.

    Returns:
        True if MAP_DATA was processed since the dispatch and the action
        was cleared.
    """
    if not check_and_clear_map_data_processed():
        return False
    started_ms = action["started_ms"]
    duration_ms = get_current_time_ms() - started_ms if started_ms > 0 else -1
    emit_wire_complete(
        action_kind="map_open",
        duration_ms=duration_ms,
        signal="map_data_processed",
    )
    bot._state_data = transition_to(
        bot._state_data,
        bot.get_state(),
        in_flight_action=make_no_action(),
    )
    return True


def _clear_blocked_walk(
    bot: Bot,
    action: InFlightActionDict,
) -> bool:
    """Clear a walk when terrain shows the destination is unreachable.

    Args:
        bot: Bot instance.
        action: The in-flight move action record.

    Returns:
        True if the blocked walk was cleared and the tick should replan.
    """
    world = bot.get_world_state()
    self_state = world["self_state"]
    terrain = get_terrain_map()
    if self_state is None or terrain is None:
        return False
    tx, ty = action["target_x"], action["target_y"]
    if is_move_reachable_in_viewport(
        world,
        terrain,
        self_state["x"],
        self_state["y"],
        tx,
        ty,
        world["mines"],
    ):
        return False
    emit_sync("movement to (%d,%d) is terrain-blocked, replanning", tx, ty)
    bot._transition("IDLE", in_flight_action=make_no_action())
    return True


def _clear_blocked_collection(
    bot: Bot,
    action: InFlightActionDict,
) -> bool:
    """Clear a collection when the target is terrain-blocked.

    pickup_move can legitimately complete from an adjacent tile, so
    adjacency is treated as still viable. Otherwise, if terrain says
    there is no path, the queued collection is abandoned.

    Args:
        bot: Bot instance.
        action: The in-flight collect action record.

    Returns:
        True if the blocked collection was cleared and the tick should replan.
    """
    world = bot.get_world_state()
    self_state = world["self_state"]
    terrain = get_terrain_map()
    if self_state is None or terrain is None:
        return False
    tx, ty = action["target_x"], action["target_y"]
    if abs(self_state["x"] - tx) <= 1 and abs(self_state["y"] - ty) <= 1:
        return False
    if is_collection_reachable_in_viewport(
        world,
        terrain,
        self_state["x"],
        self_state["y"],
        tx,
        ty,
        world["mines"],
    ):
        return False
    emit_sync("collection target (%d,%d) is terrain-blocked, replanning", tx, ty)
    bot._transition("IDLE", in_flight_action=make_no_action())
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
    new_kills = drain_killed_tank_ids()
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
    if peek_combat_hit():
        return False
    if peek_our_shot_response():
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
    got_hit = check_and_clear_combat_hit()
    got_response = check_and_clear_our_shot_response()
    if got_hit:
        log.info("FEEDBACK: hit confirmed")
        return "hit"
    if str(bot._ai_state["last_shot_target_id"]) in bot._ai_state["killed_tank_ids"]:
        log.info("FEEDBACK: kill confirmed")
        return "hit"
    inventory = get_inventory_state()
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
