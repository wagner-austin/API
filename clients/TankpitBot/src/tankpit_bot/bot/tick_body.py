"""One tick: the decide-dispatch pass and its readiness guards.

``_tick_once`` plus the health, respawn, wire-silence, and autoscroll
checks it consults. The session loop that drives it is
:mod:`tankpit_bot.bot.tick_loop`; the combat feedback it reads is
:mod:`tankpit_bot.bot.tick_combat_feedback`.
"""

from __future__ import annotations

from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.bot import ai_strategy, executor, world_sync
from tankpit_bot.bot.ai.scoring_types import render_reason
from tankpit_bot.bot.ai.types import (
    make_respawn_ai_state,
)
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.states import make_initial_state_data
from tankpit_bot.bot.tick_combat_feedback import (
    _get_combat_feedback,
    _has_pending_shot_feedback,
    _merge_protocol_kills,
    _resolve_pending_ground_shot,
)
from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
from tankpit_bot.browser import _test_hooks as browser_hooks
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.browser.overlay import OverlayStateDict, render_overlay_payload
from tankpit_bot.browser.overlay_hud import update_bot_overlay
from tankpit_bot.browser.page_client_snapshot import (
    PageClientSnapshotDict,
    capture_page_client_snapshot,
)
from tankpit_bot.fleetshare import (
    build_fleet_report,
    merge_fleet_reports,
    read_team_reports,
    write_fleet_report,
)
from tankpit_bot.ledger.decision import latest_decision_event_id
from tankpit_bot.ledger.mode_transition import emit_mode_transition
from tankpit_bot.physics.capacity import fuel_capacity, inventory_capacity
from tankpit_bot.runtime_artifacts import bot_run_dir, resolve_bot_instance
from tankpit_bot.runtime_logging import (
    emit_diagnostic,
    emit_sync,
)
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state
from tankpit_bot.state import SelfStateDict

log = get_logger(__name__)

_WS_READY_STATE_OPEN = 1

# How long a dead tank waits for its respawn sync before the session
# exits ``deactivated`` anyway. The real server respawns promptly; a
# world that never respawns (the sim has no respawn law) must not
# wait forever on a sync that cannot come.
_RESPAWN_WAIT_MS = 60_000

# Wire-silence limit for the connection-lost watchdog. Live game
# traffic is near-continuous (fuel ticks, tank movement, viewport
# churn arrive many times a minute), and the longest sanctioned quiet
# stretch is the 60 s respawn wait (_RESPAWN_WAIT_MS) -- during which
# the corpse's viewport still streams other tanks. 90 s sits safely
# above both while turning session 3's 43-minute zombie into a
# 90-second clean exit.
_WIRE_SILENCE_LIMIT_MS = 90_000


def _tick_once(bot: Bot) -> None:
    """Execute one sync-decide-execute cycle.

    Args:
        bot: Bot instance.
    """
    # 1. SYNC — drain CDP message buffer
    world_sync.drain_messages(bot, bot.world)
    _check_wire_silence(bot)

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
    if bot.world.self_deactivated:
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
    bot._ai_state = _merge_protocol_kills(bot.world, bot._ai_state)
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

    # 5. Combat feedback (counters incremented inside _get_combat_feedback).
    # Ground-aimed shots resolve first: their echo receipt is not
    # combat feedback and must not linger as stale flags for the
    # id-keyed classifier below.
    _resolve_pending_ground_shot(bot)
    combat_feedback = _get_combat_feedback(bot)

    # 6. DECIDE
    inventory = get_inventory_state(bot.world)
    terrain = bot.world.get_terrain_map()

    decision = ai_strategy.decide(
        world,
        self_state,
        bot._ai_state,
        inventory,
        now,
        terrain,
        combat_feedback,
        ws=bot.world,
    )

    bot._self_alignment.maybe_emit(self_state, snapshot)
    bot._entity_alignment.maybe_emit(
        world,
        snapshot,
        in_combat=bot._ai_state["mode"] == "HUNT",
    )
    bot._client_structure.maybe_emit(bot._require_cdp())
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
                bot.world.ledger,
                from_mode=previous_mode,
                to_mode=bot._ai_state["mode"],
                reason_kind=decision["behavior"]["reason_kind"],
                caused_by=(
                    0
                    if decision["command"]["cmd_type"] == "hold"
                    else latest_decision_event_id(bot.world.ledger)
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
    # The same payload the in-page HUD renders, mirrored to disk each
    # tick so the fleet page can show the identical card per instance.
    _test_hooks.write_text(
        bot_run_dir(resolve_bot_instance()) / "hud.json",
        dump_json_str(render_overlay_payload(overlay)),
    )
    _exchange_fleet_knowledge(bot)


def _exchange_fleet_knowledge(bot: Bot) -> None:
    """Run one fleet knowledge exchange: publish, then merge siblings.

    The shared knowledge layer ([[fleet-coordination]], fleet ruling
    2026-08-14): each tick the bot atomically replaces its own
    ``knowledge.json`` and merges the fresh SAME-TEAM reports of its
    siblings. Rides the run-directory channel the HUD mirror above
    already uses, so a single tank exchanges with an empty fleet at
    the cost of one file write, and a fleet coordinates with no
    manager process required. Before the session has an established
    self there is nothing attributable to offer and no team to merge
    for, so the exchange starts with the first entered tick.

    Args:
        bot: Bot instance.
    """
    now_ms = get_current_time_ms()
    instance = resolve_bot_instance()
    claimed = bot._ai_state["resource_target_kind"] != ""
    report = build_fleet_report(
        bot.world,
        instance=instance,
        role=bot._ai_state["config"]["role"],
        engaged_target_id=bot._ai_state["combat_target_id"],
        forage_goal_x=bot._ai_state["forage_goal_x"],
        forage_goal_y=bot._ai_state["forage_goal_y"],
        collect_claim_x=bot._ai_state["resource_target_x"] if claimed else -1,
        collect_claim_y=bot._ai_state["resource_target_y"] if claimed else -1,
        now_ms=now_ms,
    )
    if report is None:
        return
    write_fleet_report(report)
    reports = read_team_reports(instance, report["team"], report["room"], now_ms)
    summary = merge_fleet_reports(
        bot.world,
        reports,
        own_tank_id=report["tank_id"],
        own_team=report["team"],
    )
    if summary["reports"] > 0:
        emit_diagnostic(
            diagnostic_kind="fleet_knowledge_merged",
            reports=summary["reports"],
            enemies=summary["enemies"],
            containers=summary["containers"],
            removed=summary["removed"],
            scanned=summary["scanned"],
        )


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


def _check_wire_silence(bot: Bot) -> None:
    """End the session when the game wire has gone silent.

    The ws-ready page-health gate cannot catch this failure: session 3
    of run 20260730 lost its game socket mid-move at 11:58:32, the
    page auto-reconnected to the LOBBY -- a perfectly OPEN socket the
    server no longer associates with an in-game tank -- and every
    injected map_open vanished for 43 minutes (243 consecutive stalls,
    zero inbound world messages). Inbound game traffic is the only
    truthful liveness signal, so its absence past the limit is
    terminal. A zero stamp means no game message has EVER arrived
    (boot, lobby) and the watchdog stays disarmed.

    Args:
        bot: Bot instance (unused state anchor; the stamp lives on the
            world service singleton the sniffer writes).

    Raises:
        SessionExitError: When the last dispatched game message is
            older than :data:`_WIRE_SILENCE_LIMIT_MS`.
    """
    last_ms = bot.world.last_game_message_ms
    if last_ms <= 0:
        return
    silence_ms = get_current_time_ms() - last_ms
    if silence_ms < _WIRE_SILENCE_LIMIT_MS:
        return
    raise SessionExitError(
        "connection_lost",
        f"no game wire message for {silence_ms // 1000}s "
        f"(limit {_WIRE_SILENCE_LIMIT_MS // 1000}s) - game session is dead",
    )


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
    browser_hooks.ensure_autoscroll_off(bot._page, bot._require_cdp(), bot._messages, bot.world)
    bot._autoscroll_enforced = True


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
    service = bot.world
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
    bot._ai_state = make_respawn_ai_state(bot._ai_state)
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


__all__ = [
    "log",
]
