"""Live combat accuracy probe — fires at enemies and records hit/miss per shot.

Teleports adjacent to an enemy, then fires repeatedly. Records each
shot's distance, target position, and server feedback (hit/miss/weapon
byte). When the enemy flees, keeps firing at updated positions to
measure effective homing range empirically.
"""

from __future__ import annotations

from typing import Literal

from platform_core.logging import get_logger

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.combat_probe_types import (
    CombatEngagementDict,
    CombatProbeSessionDict,
    CombatShotResultDict,
    encode_combat_probe_session,
)
from tankpit_bot.action_lab.probe_base import ProbeBase
from tankpit_bot.action_lab.probe_entrypoint import (
    run_and_save_standard_probe_session,
)
from tankpit_bot.action_lab.probe_runtime import (
    ProbeCommandReadyContextDict,
    execute_live_probe_bootstrap,
)
from tankpit_bot.action_lab.probe_session import build_probe_session_envelope
from tankpit_bot.action_lab.teleport_acquisition import run_tracked_acquisition_phase
from tankpit_bot.action_lab.teleport_helpers import TeleportProbeError
from tankpit_bot.action_lab.teleport_phase import run_tracked_teleport_command
from tankpit_bot.action_lab.types import TeleportTargetDict
from tankpit_bot.bot.ai.combat_landing import (
    choose_combat_landing_tile,
    has_cardinal_enemy_adjacency,
)
from tankpit_bot.bot.ai.threats import analyze_threats, find_closest_threat
from tankpit_bot.bot.ai.types import EnemyThreatDict
from tankpit_bot.sniffer.world_state import get_terrain_map, get_world_service
from tankpit_bot.sniffer.world_state_combat import (
    check_and_clear_combat_hit,
    check_and_clear_our_shot_response,
)
from tankpit_bot.state.types import has_known_position

log = get_logger(__name__)

_SHOT_FEEDBACK_TIMEOUT_MS = 4000
_SHOT_POLL_INTERVAL_MS = 100.0


def _find_fresh_enemy(
    probe: ProbeBase,
    started_ms: int,
    excluded_ids: frozenset[int],
) -> EnemyThreatDict | None:
    """Return closest enemy confirmed after a probe action."""
    self_state = probe.get_self_state()
    if self_state is None:
        return None
    threats = analyze_threats(
        probe.get_world_state(),
        self_state,
        action_hooks.get_current_time_ms(),
    )
    fresh = [
        t for t in threats if t["timestamp_ms"] > started_ms and t["tank_id"] not in excluded_ids
    ]
    return find_closest_threat(fresh)


def _current_enemy_by_id(
    probe: ProbeBase,
    tank_id: int,
) -> EnemyThreatDict | None:
    """Return threat snapshot for a specific tank id."""
    self_state = probe.get_self_state()
    if self_state is None:
        return None
    for t in analyze_threats(
        probe.get_world_state(),
        self_state,
        action_hooks.get_current_time_ms(),
    ):
        if t["tank_id"] == tank_id:
            return t
    return None


def _enemy_from_world_state(
    probe: ProbeBase,
    tank_id: int,
) -> tuple[int, int] | None:
    """Return (x, y) for a tank from the world state dict (map-known)."""
    tank = probe.get_world_state()["tanks"].get(str(tank_id))
    if tank is None or not has_known_position(tank):
        return None
    return (tank["x"], tank["y"])


def _wait_for_shot_feedback(
    page: action_session.WaitPageProtocol,
    probe: ProbeBase,
) -> tuple[bool, bool]:
    """Wait for the server to respond to our shot.

    Returns:
        (got_response, was_hit): whether any response came,
        and whether it was a confirmed hit.
    """
    ws = get_world_service()
    started = action_hooks.get_current_time_ms()
    while action_hooks.get_current_time_ms() - started < _SHOT_FEEDBACK_TIMEOUT_MS:
        action_hooks.drain_buffered_messages(probe)
        if ws.got_our_shot_response:
            was_hit = check_and_clear_combat_hit(ws)
            check_and_clear_our_shot_response(ws)
            return (True, was_hit)
        page.wait_for_timeout(_SHOT_POLL_INTERVAL_MS)
    return (False, False)


def _manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)


class CombatProbe(ProbeBase):
    """Live combat accuracy probe — fire at enemies, record per-shot data."""

    def _engage_single_target(
        self,
        enemy: EnemyThreatDict,
        max_shots: int,
    ) -> CombatEngagementDict:
        """Fire repeatedly at one enemy and record per-shot results."""
        page = self._require_page()
        self_state = self._require_self_state()
        landed_x, landed_y = self_state["x"], self_state["y"]
        target_id = enemy["tank_id"]
        target_name = enemy["name"]
        initial_x, initial_y = enemy["x"], enemy["y"]
        initial_distance = _manhattan(
            landed_x,
            landed_y,
            initial_x,
            initial_y,
        )

        shots: list[CombatShotResultDict] = []
        kill_confirmed = False
        target_fled = False
        last_x, last_y = initial_x, initial_y

        for shot_num in range(1, max_shots + 1):
            action_hooks.drain_buffered_messages(self)
            self_state = self._require_self_state()
            sx, sy = self_state["x"], self_state["y"]

            current_enemy = _current_enemy_by_id(self, target_id)
            if current_enemy is not None:
                tx, ty = current_enemy["x"], current_enemy["y"]
            else:
                ws_pos = _enemy_from_world_state(self, target_id)
                if ws_pos is None:
                    kill_confirmed = True
                    break
                tx, ty = ws_pos

            dist = _manhattan(sx, sy, tx, ty)
            if (tx, ty) != (last_x, last_y):
                target_fled = True
                log.info(
                    "COMBAT: target %s moved (%d,%d)->(%d,%d) dist=%d",
                    target_name,
                    last_x,
                    last_y,
                    tx,
                    ty,
                    dist,
                )

            log.info(
                "COMBAT: shot %d at %s (%d,%d) dist=%d self=(%d,%d)",
                shot_num,
                target_name,
                tx,
                ty,
                dist,
                sx,
                sy,
            )
            self.shoot(tx, ty, target_id)
            got_response, was_hit = _wait_for_shot_feedback(page, self)

            result: Literal["hit", "miss", "timeout"]
            if not got_response:
                result = "timeout"
            elif was_hit:
                result = "hit"
            else:
                result = "miss"

            ws = get_world_service()
            weapon_byte: int | None = 1 if was_hit else None

            shot_result = CombatShotResultDict(
                shot_number=shot_num,
                self_x=sx,
                self_y=sy,
                target_x=tx,
                target_y=ty,
                distance=dist,
                result=result,
                weapon_byte=weapon_byte,
                target_name=target_name,
                target_id=target_id,
                timestamp_ms=action_hooks.get_current_time_ms(),
            )
            shots.append(shot_result)
            log.info("COMBAT: shot %d result=%s dist=%d", shot_num, result, dist)

            if result == "timeout":
                break

            last_x, last_y = tx, ty

            action_hooks.drain_buffered_messages(self)
            killed_ids = ws.killed_tank_ids
            if target_id in killed_ids:
                kill_confirmed = True
                log.info("COMBAT: KILL confirmed on %s", target_name)
                break

        total_hits = sum(1 for s in shots if s["result"] == "hit")
        total_misses = sum(1 for s in shots if s["result"] == "miss")
        total_timeouts = sum(1 for s in shots if s["result"] == "timeout")
        final_dist = _manhattan(
            self._require_self_state()["x"],
            self._require_self_state()["y"],
            last_x,
            last_y,
        )

        log.info(
            "COMBAT: engagement complete target=%s hits=%d misses=%d "
            "timeouts=%d kill=%s fled=%s final_dist=%d",
            target_name,
            total_hits,
            total_misses,
            total_timeouts,
            kill_confirmed,
            target_fled,
            final_dist,
        )

        return CombatEngagementDict(
            target_id=target_id,
            target_name=target_name,
            initial_target_x=initial_x,
            initial_target_y=initial_y,
            initial_distance=initial_distance,
            landed_x=landed_x,
            landed_y=landed_y,
            shots=shots,
            total_hits=total_hits,
            total_misses=total_misses,
            total_timeouts=total_timeouts,
            kill_confirmed=kill_confirmed,
            target_fled=target_fled,
            final_target_x=last_x,
            final_target_y=last_y,
            final_distance=final_dist,
        )

    def _acquire_and_engage(
        self,
        *,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        max_shots: int,
        excluded_ids: frozenset[int],
    ) -> CombatEngagementDict | None:
        """Open map, teleport to enemy, then engage."""
        page = self._require_page()
        cdp = self._cdp
        if cdp is None:
            raise TeleportProbeError("cdp session is unavailable")

        self._reset_probe_state_to_idle()
        message_start = len(self.messages)
        world_before = self.get_world_state()
        self_before = self._require_self_state()

        (
            acq_started_ms,
            acq_sync_ms,
            page_snapshots,
            capture_snapshot,
        ) = run_tracked_acquisition_phase(
            page,
            self,
            cdp=cdp,
            send_command=lambda: self.open_map(),
            command_name="combat_acquisition",
            capture_before_map_open=True,
            wait_for_sync=True,
            sync_timeout_ms=acquisition_timeout_ms,
            dispatch_failure_error=TeleportProbeError,
            dispatch_failure_message="combat acquisition failed",
            unavailable_error=TeleportProbeError,
            unavailable_message="cdp unavailable",
        )
        if acq_sync_ms is None:
            log.warning("COMBAT: acquisition timeout")
            return None

        enemy = _find_fresh_enemy(self, acq_started_ms, excluded_ids)
        if enemy is None:
            log.warning("COMBAT: no enemy found after acquisition")
            return None

        landing_x, landing_y = choose_combat_landing_tile(
            self.get_world_state(),
            self._require_self_state(),
            enemy,
            get_terrain_map(),
            action_hooks.get_current_time_ms(),
        )
        if landing_x == -1 and landing_y == -1:
            log.warning("COMBAT: no landing tile for %s", enemy["name"])
            return None

        landing_target = TeleportTargetDict(
            label=f"combat_{enemy['tank_id']}_{enemy['x']}_{enemy['y']}",
            x=landing_x,
            y=landing_y,
        )
        teleport_cycle = self._start_action_phase(
            "teleport",
            attempt_label=landing_target["label"],
        )

        from tankpit_bot.action_lab.teleport_helpers import (
            _wait_for_teleport_outcome,
        )

        teleport_result, _ = run_tracked_teleport_command(
            page,
            self,
            landing_target,
            teleport_cycle=teleport_cycle,
            message_start_index=message_start,
            map_open_started_ms=acq_started_ms,
            map_sync_timestamp_ms=acq_sync_ms,
            fuel_before=self_before["fuel"],
            world_timestamp_before=world_before["timestamp_ms"],
            timeout_ms=teleport_timeout_ms,
            page_snapshots=page_snapshots,
            capture_page_snapshot=capture_snapshot,
            wait_for_outcome=_wait_for_teleport_outcome,
            dispatch_failure_error=TeleportProbeError,
        )

        if teleport_result["status"] == "teleport_timeout":
            log.warning("COMBAT: teleport timeout for %s", enemy["name"])
            return None

        current = _current_enemy_by_id(self, enemy["tank_id"])
        if current is None:
            log.warning(
                "COMBAT: enemy %s not visible after teleport",
                enemy["name"],
            )
            return None

        self_after = self._require_self_state()
        if not has_cardinal_enemy_adjacency(self_after, current):
            log.warning(
                "COMBAT: not adjacent to %s after teleport (self=(%d,%d) target=(%d,%d) dist=%d)",
                current["name"],
                self_after["x"],
                self_after["y"],
                current["x"],
                current["y"],
                _manhattan(
                    self_after["x"],
                    self_after["y"],
                    current["x"],
                    current["y"],
                ),
            )

        return self._engage_single_target(current, max_shots)

    def execute_probe(
        self,
        *,
        max_engagements: int,
        max_shots_per_engagement: int,
        initial_sync_timeout_ms: int,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
    ) -> CombatProbeSessionDict:
        """Run the live combat accuracy probe session."""
        if max_engagements <= 0:
            raise ValueError("max_engagements must be positive")

        def _run_ready_session(
            context: ProbeCommandReadyContextDict,
        ) -> CombatProbeSessionDict:
            engagements: list[CombatEngagementDict] = []
            targeted_ids: set[int] = set()
            for i in range(max_engagements):
                log.info(
                    "COMBAT: starting engagement %d/%d",
                    i + 1,
                    max_engagements,
                )
                result = self._acquire_and_engage(
                    acquisition_timeout_ms=acquisition_timeout_ms,
                    teleport_timeout_ms=teleport_timeout_ms,
                    max_shots=max_shots_per_engagement,
                    excluded_ids=frozenset(targeted_ids),
                )
                if result is not None:
                    engagements.append(result)
                    targeted_ids.add(result["target_id"])
            first_started = None
            if engagements and engagements[0]["shots"]:
                first_started = engagements[0]["shots"][0]["timestamp_ms"]
            envelope = build_probe_session_envelope(
                self,
                context=context,
                first_attempt_started_ms=first_started,
            )
            return CombatProbeSessionDict(
                session_id=envelope.session_id,
                start_timestamp_ms=envelope.start_timestamp_ms,
                end_timestamp_ms=envelope.end_timestamp_ms,
                base_url=envelope.base_url,
                spawn_x=envelope.spawn_x,
                spawn_y=envelope.spawn_y,
                max_engagements=max_engagements,
                max_shots_per_engagement=max_shots_per_engagement,
                capture_session_path="",
                initial_sync_timeout_ms=initial_sync_timeout_ms,
                startup_timing=envelope.startup_timing,
                engagements=engagements,
            )

        return execute_live_probe_bootstrap(
            self,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            run_ready_session=_run_ready_session,
        )


def format_combat_probe_summary(session: CombatProbeSessionDict) -> str:
    """Format a compact human-readable summary for the combat session."""
    total_hits = 0
    total_misses = 0
    total_kills = 0
    total_fled = 0
    dist_hit: dict[int, int] = {}
    dist_miss: dict[int, int] = {}

    for eng in session["engagements"]:
        total_hits += eng["total_hits"]
        total_misses += eng["total_misses"]
        if eng["kill_confirmed"]:
            total_kills += 1
        if eng["target_fled"]:
            total_fled += 1
        for shot in eng["shots"]:
            d = shot["distance"]
            if shot["result"] == "hit":
                dist_hit[d] = dist_hit.get(d, 0) + 1
            elif shot["result"] == "miss":
                dist_miss[d] = dist_miss.get(d, 0) + 1

    all_dists = sorted(set(dist_hit) | set(dist_miss))
    dist_lines = " | ".join(
        f"d={d}:{dist_hit.get(d, 0)}h/{dist_miss.get(d, 0)}m" for d in all_dists
    )

    return (
        f"Combat probe: engagements={len(session['engagements'])} "
        f"hits={total_hits} misses={total_misses} "
        f"kills={total_kills} fled={total_fled} | {dist_lines}"
    )


def _create_combat_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> CombatProbe:
    """Factory for CombatProbe with injected services."""
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        CombatProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, CombatProbe)
    return probe


def run_combat_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = False,
    max_engagements: int = 3,
    max_shots_per_engagement: int = 20,
    initial_sync_timeout_ms: int = 10000,
    acquisition_timeout_ms: int = 5000,
    teleport_timeout_ms: int = 10000,
) -> CombatProbeSessionDict:
    """Run a live combat probe and save the session JSON."""

    def _run_session(probe: CombatProbe) -> CombatProbeSessionDict:
        return probe.execute_probe(
            max_engagements=max_engagements,
            max_shots_per_engagement=max_shots_per_engagement,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            acquisition_timeout_ms=acquisition_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
        )

    return run_and_save_standard_probe_session(
        probe_factory=_create_combat_probe,
        run_session=_run_session,
        encoder=encode_combat_probe_session,
        summary_formatter=format_combat_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "CombatProbe",
    "format_combat_probe_summary",
    "run_combat_probe",
]
