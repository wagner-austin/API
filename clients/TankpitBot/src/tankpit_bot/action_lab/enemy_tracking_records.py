"""Record building for the enemy-tracking probe.

Shot-feedback waiting, identity resolution, and the tracked/observation
records the run summary is rendered from. The probe that drives them is
:mod:`tankpit_bot.action_lab.enemy_tracking`.
"""

from __future__ import annotations

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.enemy_tracking_types import (
    EnemyTrackingProbeSessionDict,
    TrackedEnemyDict,
    TrackingObservationDict,
)
from tankpit_bot.action_lab.probe_base import ProbeBase
from tankpit_bot.action_lab.tracking_observation import (
    build_tracking_observation,
    find_js_entry_by_position,
    select_js_identity_key,
)
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.browser.page_client_snapshot import (
    PageClientSnapshotDict,
)
from tankpit_bot.sniffer.world_state import (
    get_world_service,
)
from tankpit_bot.sniffer.world_state_combat import (
    check_and_clear_combat_hit,
    check_and_clear_our_shot_response,
)
from tankpit_bot.state.types import WorldStateDict

#: Polling interval used by the shot-feedback wait. Short enough not
#: to add noticeable latency, long enough to keep CPU use reasonable.
_SHOT_POLL_INTERVAL_MS = 100.0


def _wait_for_shot_feedback(
    page: action_session.WaitPageProtocol,
    probe: ProbeBase,
    *,
    timeout_ms: int,
) -> tuple[bool, bool]:
    """Wait for the server's response to our shot.

    Mirrors the combat-probe shot wait so both probes interpret hit
    / miss / timeout the same way.

    Args:
        page: Active page handle used for cadence sleeps.
        probe: Probe instance whose buffered messages to drain.
        timeout_ms: Maximum wait before giving up.

    Returns:
        ``(got_response, was_hit)`` -- ``got_response`` is False when
        the wait timed out.
    """
    ws = get_world_service()
    started = action_hooks.get_current_time_ms()
    while action_hooks.get_current_time_ms() - started < timeout_ms:
        action_hooks.drain_buffered_messages(probe)
        if ws.got_our_shot_response:
            was_hit = check_and_clear_combat_hit(ws)
            check_and_clear_our_shot_response(ws)
            return (True, was_hit)
        page.wait_for_timeout(_SHOT_POLL_INTERVAL_MS)
    return (False, False)


def _build_tracked_records(
    threats: list[EnemyThreatDict],
    snapshot: PageClientSnapshotDict,
) -> list[TrackedEnemyDict]:
    """Build :class:`TrackedEnemyDict` records for every visible enemy.

    Resolves the JS-side identity for each tank by position-matching
    a ``P.j`` entry against the wire-derived ``(x, y)`` at
    acquisition time. The selected JS field becomes the cross-tick
    join key for the sampling loop.

    Args:
        threats: Visible enemies returned by ``analyze_threats``.
        snapshot: Page-client snapshot captured at acquisition time.

    Returns:
        One record per visible enemy. Records carry empty
        ``tracked_js_key`` and ``tracked_js_value`` when no JS entry
        could be paired -- the row still records the wire-side view
        so divergence stays visible.
    """
    records: list[TrackedEnemyDict] = []
    for threat in threats:
        js_entry = find_js_entry_by_position(
            snapshot["world_collections"],
            threat["x"],
            threat["y"],
        )
        if js_entry is None:
            tracked_key = ""
            tracked_value = ""
        else:
            tracked_key, tracked_value = _resolve_identity(js_entry, threat["tank_id"])
        records.append(
            TrackedEnemyDict(
                tank_id=threat["tank_id"],
                name=threat["name"],
                team=threat["team"],
                rank=threat["rank"],
                acquired_x=threat["x"],
                acquired_y=threat["y"],
                tracked_js_key=tracked_key,
                tracked_js_value=tracked_value,
            ),
        )
    return records


def _resolve_identity(
    js_entry: dict[str, int | float | bool | str | None],
    tank_id: int,
) -> tuple[str, str]:
    """Pair our tank id with a stable JS-side identity field.

    The JS registry hands us minified field names whose semantics
    we do not statically know. We pair against ``tank_id`` because
    we just confirmed this entry by position -- whichever field
    holds an integer equal to our ``tank_id`` is the JS-side tank
    id.

    Args:
        js_entry: Registry entry that matched our tank by position.
        tank_id: Tank id from our world state.

    Returns:
        ``(key, str(value))`` -- empty strings when no field equals
        the tank id.
    """
    from tankpit_bot.state.types.tank import make_tank_state

    surrogate = make_tank_state(
        tank_id=tank_id,
        x=0,
        y=0,
        team=0,
        rank=0,
        damage_state=0,
        name="",
        is_bot=False,
        is_self=False,
    )
    return select_js_identity_key(js_entry, surrogate)


def _build_sample_observations(
    *,
    sample_index: int,
    sample_timestamp_ms: int,
    tracked: list[TrackedEnemyDict],
    world: WorldStateDict,
    threats: list[EnemyThreatDict],
    snapshot: PageClientSnapshotDict,
    bot_combat_target_id: int,
    bot_mode_state: str,
) -> list[TrackingObservationDict]:
    """Build one observation row per tracked tank for one sample.

    Args:
        sample_index: Zero-based sample number.
        sample_timestamp_ms: Wall-clock time of this sample.
        tracked: Enemies the probe locked on to at acquisition.
        world: World state captured at sample time.
        threats: ``analyze_threats`` output at sample time.
        snapshot: Page-client snapshot at sample time.
        bot_combat_target_id: ``ai_state.combat_target_id`` at sample time.
        bot_mode_state: ``ai_state.mode_state`` at sample time.

    Returns:
        One observation row per tracked tank.
    """
    return [
        build_tracking_observation(
            sample_index=sample_index,
            sample_timestamp_ms=sample_timestamp_ms,
            tank_id=record["tank_id"],
            tracked_label=record["name"],
            tracked_js_key=record["tracked_js_key"],
            tracked_js_value=record["tracked_js_value"],
            world=world,
            threats=threats,
            world_collections=snapshot["world_collections"],
            bot_combat_target_id=bot_combat_target_id,
            bot_mode_state=bot_mode_state,
        )
        for record in tracked
    ]


def format_enemy_tracking_probe_summary(session: EnemyTrackingProbeSessionDict) -> str:
    """Format a compact human-readable summary for the tracking session.

    Highlights the divergence count -- the rows where our wire-side
    belief disagreed with the JS-side belief about a tank's
    presence. A non-zero divergence count is what the user wants
    to read in the terminal after the run.

    Args:
        session: Completed session payload.

    Returns:
        One-line summary string.
    """
    diverged = 0
    our_present_js_absent = 0
    js_present_our_absent = 0
    for observation in session["observations"]:
        our_present = observation["our_belief"]["would_locked_target_return"]
        js_present = observation["js_belief"]["present"]
        if our_present == js_present:
            continue
        diverged += 1
        if our_present:
            our_present_js_absent += 1
        else:
            js_present_our_absent += 1
    return (
        "Enemy tracking probe complete: "
        f"tracked={len(session['tracked'])} "
        f"samples={len(session['observations'])} "
        f"divergence={diverged} "
        f"our_present_js_absent={our_present_js_absent} "
        f"js_present_our_absent={js_present_our_absent}"
    )


__all__ = [
    "format_enemy_tracking_probe_summary",
]
