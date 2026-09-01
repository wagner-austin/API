"""Combat feedback for one tick: kills, pending shots, and hit resolution.

Whether a dispatched shot has resolved, what it resolved to, and the
friendly-fire disproof that clears a wrong target.

This module used to concentrate the largest cluster of
``get_world_service()`` call sites -- 17 of them at its peak. All but
one now read ``bot.world``; the survivor is ``_merge_protocol_kills``,
whose only argument is the AI state
([[session-state-deglobalisation]] step 8).
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.bot.ai.combat_target import clear_combat_target
from tankpit_bot.bot.ai.types import (
    AIStateDict,
)
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.ledger.damage_book import resolve_dealt
from tankpit_bot.ledger.outcome.shoot import (
    emit_shoot_command_rejected,
    emit_shoot_fired,
    emit_shoot_hit,
    emit_shoot_miss,
)
from tankpit_bot.protocol.constants import (
    SUPERVISOR_ERROR_CANT_DO,
    SUPERVISOR_ERROR_FRIENDLY_FIRE,
    SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
)
from tankpit_bot.runtime_logging import (
    emit_ai,
    emit_diagnostic,
    emit_sync,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_combat import (
    check_and_clear_ammo_delta_hit,
    check_and_clear_combat_hit,
    check_and_clear_command_error,
    check_and_clear_last_shot_victim_id,
    check_and_clear_our_shot_response,
    clear_pending_ground_shot,
    drain_killed_tank_ids,
    peek_combat_hit,
    peek_command_error,
    peek_our_shot_response,
)

# 0x52 Supervisor codes a shoot dispatch can draw. Any of these while a
# shot is pending is the server's authoritative refusal of THAT shot --
# no 0x53 ShootEvent and no ammo delta will ever arrive for it (live
# run 2026-07-03 20:34: five code-0 rejections at an off-viewport aim
# produced zero wire feedback and each burned the full 4 s feedback
# window before an identical redispatch).
_SHOT_REJECTING_COMMAND_ERRORS = frozenset(
    {
        SUPERVISOR_ERROR_CANT_DO,  # aim outside the viewport
        SUPERVISOR_ERROR_FRIENDLY_FIRE,
        SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
    }
)

log = get_logger(__name__)


def _merge_protocol_kills(ws: WorldService, ai_state: AIStateDict) -> AIStateDict:
    """Merge Deactivation kills from protocol into AI killed_tank_ids.

    Every victim enters ``killed_tank_ids`` — the dead-tank registry
    keeps the bot off corpses no matter who made them — but
    ``session_kill_count`` (the scorecard and the ``session_kills``
    wind-down trigger) advances only for victims whose 0x41 names THIS
    tank as the killer. Solo sessions made the two indistinguishable;
    the 2026-08-14 fleet run falsified that: arterial's scorecard
    banked artax's two kills (both ``killed by 1301``) on zero shots
    fired.

    Args:
        ws: The session's world service, holding the kill queue.
        ai_state: Current AI state.

    Returns:
        Updated AI state with new kills merged.
    """
    new_kills = drain_killed_tank_ids(ws)
    if not new_kills:
        return ai_state
    self_state = ws.world_state["self_state"]
    # An unattributable kill (no established identity yet — e.g. a
    # spectated 0x41 during entry) still lands in the dead registry;
    # it can never be OUR kill, -1 is not a wire tank id.
    own_tank_id = -1 if self_state is None else self_state["tank_id"]
    now = get_current_time_ms()
    merged = dict(ai_state["killed_tank_ids"])
    own_kill_count = 0
    for tank_id, killer_id in new_kills.items():
        merged[str(tank_id)] = now
        if killer_id == own_tank_id:
            own_kill_count += 1
            # Rank rides along for the points-floor survey (operator
            # flags 5/6/8/12/13, 2026-09-01): the wire delivers victim
            # rank via 0x3D but nothing ever wrote it to the ledger, so
            # "Enemy's rank was too low" could not be correlated with
            # the rank it judged. A victim already 0x58-removed from
            # the registry reports rank -1, never a guess.
            victim = ws.world_state["tanks"].get(str(tank_id))
            victim_name = "" if victim is None else victim["name"]
            victim_rank = -1 if victim is None else victim["rank"]
            emit_ai(
                "kill registered (tank_id=%d name=%s rank=%d)",
                tank_id,
                victim_name,
                victim_rank,
            )
            emit_diagnostic(
                diagnostic_kind="kill_registered",
                victim_id=tank_id,
                victim_name=victim_name,
                victim_rank=victim_rank,
            )
    # The shot-target fields are NOT cleared here: when the killed tank
    # is the pending shot's target, ``_get_combat_feedback`` must still
    # see the target id to resolve the shot as ``kill_confirmed`` (a
    # kill produces no damage-change feedback, so this is the kill
    # shot's only resolution path). The classifier clears the fields
    # itself after emitting the outcome.
    target_was_killed = ai_state["combat_target_id"] in new_kills
    return AIStateDict(
        **{
            **ai_state,
            "killed_tank_ids": merged,
            "session_kill_count": ai_state["session_kill_count"] + own_kill_count,
            "combat_target_id": -1 if target_was_killed else ai_state["combat_target_id"],
            "combat_target_x": 0 if target_was_killed else ai_state["combat_target_x"],
            "combat_target_y": 0 if target_was_killed else ai_state["combat_target_y"],
        }
    )


def _resolve_pending_ground_shot(bot: Bot) -> None:
    """Resolve a ground-aimed shot's ledger outcome from its receipts.

    A clearance shot targets a tile, not a tank, so the id-keyed combat
    classifier never sees it — before this resolver every one of its
    decisions rotted into ``superseded`` (the 2026-08-21 false
    liveness alarm: 13 wire dispatches, 12/12 superseded). Two
    receipts can arrive:

    * the own 0x53 echo — the server accepted, billed, and fired the
      shot: resolve ``fired`` and consume the per-shot flags the echo
      set, so they cannot leak into the NEXT tank-targeted shot's
      classification as a stale instant response;
    * a shot-rejecting 0x52 — no echo will ever come ([[shot-range]]:
      five code-0 rejections, zero echoes): resolve
      ``command_rejected``.

    Neither receipt yet: keep waiting; a superseding shot closes the
    decision through the ledger's dispatched-supersede path, alarm-free.

    Args:
        bot: Bot instance.
    """
    ws = bot.world
    dispatch_ms = ws.pending_ground_shot_dispatch_ms
    if dispatch_ms == 0:
        return
    aim_x = ws.pending_ground_shot_aim_x
    aim_y = ws.pending_ground_shot_aim_y
    duration_ms = get_current_time_ms() - dispatch_ms
    if peek_command_error(ws) in _SHOT_REJECTING_COMMAND_ERRORS:
        error_code = check_and_clear_command_error(ws)
        emit_shoot_command_rejected(
            ws.ledger,
            duration_ms=duration_ms,
            target_id=0,
            target_name="",
            error_code=error_code,
        )
        clear_pending_ground_shot(ws)
        return
    if not check_and_clear_our_shot_response(ws):
        return
    check_and_clear_combat_hit(ws)
    check_and_clear_last_shot_victim_id(ws)
    ws.pending_shot_inventory_snapshot = None
    emit_shoot_fired(ws.ledger, duration_ms=duration_ms, aim_x=aim_x, aim_y=aim_y)
    clear_pending_ground_shot(ws)


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
    if peek_combat_hit(bot.world):
        return False
    if peek_our_shot_response(bot.world):
        return False
    if peek_command_error(bot.world) in _SHOT_REJECTING_COMMAND_ERRORS:
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
    got_hit = check_and_clear_combat_hit(bot.world)
    victim_id = check_and_clear_last_shot_victim_id(bot.world)
    got_response = check_and_clear_our_shot_response(bot.world)
    ammo_hit = check_and_clear_ammo_delta_hit(bot.world)

    def _inc_hit() -> None:
        bot._ai_state = AIStateDict(
            **{**bot._ai_state, "session_hit_count": bot._ai_state["session_hit_count"] + 1}
        )

    duration_ms = get_current_time_ms() - bot._ai_state["last_shoot_ms"]
    if got_hit:
        # Distinguish intended-target hit from incidental hit (e.g.
        # homing seeker landed on a closer enemy than commanded).
        on_intended = victim_id == target_id
        emit_shoot_hit(
            bot.world.ledger,
            duration_ms=duration_ms,
            target_id=target_id,
            target_name=target_name,
            victim_id=victim_id,
            on_intended_target=on_intended,
            hit_signal="tile_occupied",
        )
        resolve_dealt(bot.world.damage_book, victim_id, target_name, target_id)
        _inc_hit()
        return "hit"
    if str(target_id) in bot._ai_state["killed_tank_ids"]:
        emit_shoot_hit(
            bot.world.ledger,
            duration_ms=duration_ms,
            target_id=target_id,
            target_name=target_name,
            victim_id=target_id,
            on_intended_target=True,
            hit_signal="kill_confirmed",
        )
        resolve_dealt(bot.world.damage_book, target_id, target_name, target_id)
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
            bot.world.ledger,
            duration_ms=duration_ms,
            target_id=target_id,
            target_name=target_name,
            victim_id=victim_id,
            on_intended_target=True,
            hit_signal="ammo_delta",
        )
        resolve_dealt(bot.world.damage_book, victim_id, target_name, target_id)
        _inc_hit()
        return "hit"
    if got_response:
        return _classify_confirmed_miss(bot, target_id, target_name, duration_ms)
    if peek_command_error(bot.world) in _SHOT_REJECTING_COMMAND_ERRORS:
        error_code = check_and_clear_command_error(bot.world)
        emit_shoot_command_rejected(
            bot.world.ledger,
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
        if error_code == SUPERVISOR_ERROR_FRIENDLY_FIRE:
            _disprove_target_by_friendly_fire(bot, target_id, target_name)
        return "rejected"
    return ""


def _classify_confirmed_miss(
    bot: Bot, target_id: int, target_name: str, duration_ms: int
) -> CombatFeedback:
    """Book a wire-confirmed miss: ledger, session count, divert block.

    No tile-occupied hit, no ammo debit, and a wire response did
    arrive -- the shot genuinely missed. A missed OPPORTUNITY divert
    (firefight doctrine 2026-08-14) additionally blocks the divert
    id: the engage path's stationary-miss handling only ever judges
    the main lock, so without the block a shielded or afterimage
    divert target would be re-selected every tick -- an
    uncorrectable free-shot livelock that also starves the main
    fight.

    Args:
        bot: Bot instance.
        target_id: The missed shot's target id.
        target_name: The missed shot's target name.
        duration_ms: Dispatch-to-feedback wall-clock ms.

    Returns:
        The literal ``"miss"``.
    """
    emit_shoot_miss(
        bot.world.ledger,
        duration_ms=duration_ms,
        target_id=target_id,
        target_name=target_name,
    )
    bot._ai_state = AIStateDict(
        **{**bot._ai_state, "session_miss_count": bot._ai_state["session_miss_count"] + 1}
    )
    if target_id != bot._ai_state["combat_target_id"]:
        _block_missed_divert_target(bot, target_id, target_name)
    return "miss"


def _block_missed_divert_target(bot: Bot, target_id: int, target_name: str) -> None:
    """Block an opportunity-divert target whose shot came back a miss.

    A consumption-miss against a visible, position-confirmed enemy
    means the tank is not killable right now (corpse from an
    unwitnessed kill, or shields) -- the same verdict the engage
    path's stationary-miss block renders for the main lock. Diverted
    shots bypass that path (their feedback is scoped away from the
    lock by design), so the block lands here in the feedback layer.
    The held combat lock is untouched.

    Args:
        bot: Bot instance.
        target_id: The missed divert's target id.
        target_name: The missed divert's target name (log receipt).
    """
    blocked = dict(bot._ai_state["blocked_combat_targets"])
    blocked[str(target_id)] = get_current_time_ms()
    bot._ai_state = AIStateDict(**{**bot._ai_state, "blocked_combat_targets": blocked})
    log.info(
        "AI: opportunity shot at %s (id=%d) missed - divert target blocked, lock held",
        target_name,
        target_id,
    )
    emit_diagnostic(
        diagnostic_kind="divert_target_blocked",
        target_id=target_id,
        target_name=target_name,
    )


def _disprove_target_by_friendly_fire(bot: Bot, target_id: int, target_name: str) -> None:
    """Consume a friendly-fire rejection as proof the target is not engageable.

    The server's err=3 on an id-targeted shot is the only unfakeable
    receipt that the id no longer resolves to an enemy. Session 4 of
    run 20260730 (20:36): Yuppler left the game, the 0x58 grace kept
    his registry entry (by design -- it powers the pursuit volley),
    and every subsequent map open re-stamped the ghost's map
    freshness, so acquisition re-selected him and the bot fired 43
    consecutive rejected shots ("Friendly fire!" client spam). One
    rejection now blocklists the id for the block TTL and releases the
    combat lock, so the next tick re-acquires from live truth. The
    registry entry itself is deliberately NOT deleted -- 0x58
    semantics (tracking churn, reroute grace) stay intact.

    Args:
        bot: Bot instance.
        target_id: The shot's intended target id.
        target_name: The shot's intended target name (log receipt).
    """
    blocked = dict(bot._ai_state["blocked_combat_targets"])
    blocked[str(target_id)] = get_current_time_ms()
    updated = AIStateDict(**{**bot._ai_state, "blocked_combat_targets": blocked})
    if updated["combat_target_id"] == target_id:
        updated = clear_combat_target(updated)
    bot._ai_state = updated
    log.info(
        "AI: shot at %s (id=%d) rejected as friendly fire - "
        "target disproved, blocked and lock released",
        target_name,
        target_id,
    )
    emit_diagnostic(
        diagnostic_kind="target_disproved_by_friendly_fire",
        target_id=target_id,
        target_name=target_name,
    )


__all__ = [
    "log",
]
