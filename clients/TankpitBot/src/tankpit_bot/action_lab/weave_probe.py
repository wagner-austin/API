"""Live shoot+move weave probe — does a move cost the queued shot?

The dodge doctrine hangs on one server rule: when a tank dispatches
a shot AND a 1-tile walk inside the same 2 s serve window, does the
shot still serve on the beat? The cadence probe (2026-08-26) proved
the serve grid is global at one shot per 2 s with queuing; this probe
alternates shoot-only beats with shoot+move beats at one adjacent
enemy and compares served counts. Served ~= all beats -> the move is
free (weave while trading). Served ~= only the shoot-only beats ->
one action per beat (weave only on escape/refuel ticks, where the
shot slot is idle anyway).

Acquisition, ammo-ledger counting, and the exposure floor are the
cadence probe's, inherited — only the burst pattern differs.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.cadence_probe import (
    _ACQUISITION_ATTEMPTS,
    _MIN_BURST_FUEL,
    CadenceProbe,
    _read_fresh_ammo,
)
from tankpit_bot.action_lab.combat_probe import _enemy_from_world_state
from tankpit_bot.action_lab.probe_entrypoint import (
    run_and_save_standard_probe_session,
)
from tankpit_bot.action_lab.probe_runtime import (
    ProbeCommandReadyContextDict,
    execute_live_probe_bootstrap,
)
from tankpit_bot.action_lab.probe_session import build_probe_session_envelope
from tankpit_bot.action_lab.weave_probe_types import (
    WeaveBeatDict,
    WeaveBurstDict,
    WeaveProbeSessionDict,
    encode_weave_probe_session,
)
from tankpit_bot.bot.ai.world_types import EnemyThreatDict

log = get_logger(__name__)

_BEAT_MS = 2000.0
"""One serve-grid beat (measured 2026-08-26: serves land 2.0 s apart)."""

_MOVE_LAG_MS = 200.0
"""Gap between the shot dispatch and the move dispatch on move beats —
inside the same serve window, mirroring a human clicking then dodging."""

_SETTLE_MS = 4000.0


class WeaveProbe(CadenceProbe):
    """Alternate shoot-only and shoot+move beats; count what serves."""

    def _pick_weave_tile(self) -> tuple[int, int] | None:
        """Return a walkable cardinal neighbor of the current tile.

        Walkability is :meth:`TerrainMapProtocol.is_passable` — the
        full walk question including visible mines and tank bodies
        (arterial's third death was a mine walkover; the weave must
        never step on one).

        Returns:
            The neighbor tile, or ``None`` when boxed in.
        """
        self_state = self._require_self_state()
        terrain = self.world.get_terrain_map()
        if terrain is None:
            return None
        sx, sy = self_state["x"], self_state["y"]
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if terrain.is_passable(sx + dx, sy + dy):
                return (sx + dx, sy + dy)
        return None

    def _weave_burst(
        self,
        enemy: EnemyThreatDict,
        beats_per_burst: int,
    ) -> WeaveBurstDict | None:
        """Run one alternating burst and book the ammo ledger.

        Args:
            enemy: Target acquired adjacent by the combat acquisition.
            beats_per_burst: Total beats (even beats also move).

        Returns:
            The burst record, or ``None`` when the tank is boxed in
            with no walkable neighbor to weave to.
        """
        page = self._require_page()
        target_id = enemy["tank_id"]
        home = (self._require_self_state()["x"], self._require_self_state()["y"])
        away = self._pick_weave_tile()
        if away is None:
            log.warning("WEAVE: boxed in at (%d,%d) - no walkable neighbor", home[0], home[1])
            return None

        dual_before, homing_before = _read_fresh_ammo(self)
        fuel_before = self._require_self_state()["fuel"]

        beats: list[WeaveBeatDict] = []
        target_killed = False
        moves = 0
        at_home = True
        for beat_number in range(1, beats_per_burst + 1):
            action_hooks.drain_buffered_messages(self, self.world)
            position = _enemy_from_world_state(self, target_id)
            if position is None:
                target_killed = True
                log.info("WEAVE: target %s gone after %d beats", enemy["name"], len(beats))
                break
            tx, ty = position
            self.shoot(tx, ty, target_id)
            move_beat = beat_number % 2 == 0
            move_x, move_y = -1, -1
            if move_beat:
                page.wait_for_timeout(_MOVE_LAG_MS)
                move_x, move_y = away if at_home else home
                self.move_to(move_x, move_y)
                at_home = not at_home
                moves += 1
            beats.append(
                WeaveBeatDict(
                    beat_number=beat_number,
                    dispatched_ms=action_hooks.get_current_time_ms(),
                    target_x=tx,
                    target_y=ty,
                    moved=move_beat,
                    move_x=move_x,
                    move_y=move_y,
                )
            )
            page.wait_for_timeout(_BEAT_MS - (_MOVE_LAG_MS if move_beat else 0.0))

        page.wait_for_timeout(_SETTLE_MS)
        action_hooks.drain_buffered_messages(self, self.world)
        dual_after, homing_after = _read_fresh_ammo(self)
        fuel_after = self._require_self_state()["fuel"]
        served = (dual_before - dual_after) + (homing_before - homing_after)
        log.info(
            "WEAVE: beats=%d (moves=%d) served=%d (dual %d->%d homing %d->%d) "
            "fuel %d->%d killed=%s",
            len(beats),
            moves,
            served,
            dual_before,
            dual_after,
            homing_before,
            homing_after,
            fuel_before,
            fuel_after,
            target_killed,
        )
        return WeaveBurstDict(
            target_id=target_id,
            target_name=enemy["name"],
            beats=beats,
            shots_dispatched=len(beats),
            moves_dispatched=moves,
            dual_before=dual_before,
            dual_after=dual_after,
            homing_before=homing_before,
            homing_after=homing_after,
            fuel_before=fuel_before,
            fuel_after=fuel_after,
            served_hits=served,
            target_killed=target_killed,
        )

    def execute_weave_probe(
        self,
        *,
        beats_per_burst: int,
        burst_count: int,
        initial_sync_timeout_ms: int,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
    ) -> WeaveProbeSessionDict:
        """Run the weave bursts against freshly acquired targets.

        Args:
            beats_per_burst: Beats per burst (even beats also move).
            burst_count: How many bursts to attempt.
            initial_sync_timeout_ms: Bootstrap sync bound.
            acquisition_timeout_ms: Map-sync bound per acquisition.
            teleport_timeout_ms: Teleport bound per acquisition.

        Returns:
            The session record.

        Raises:
            ValueError: On non-positive bounds.
        """
        if beats_per_burst <= 0 or burst_count <= 0:
            raise ValueError("beats_per_burst and burst_count must be positive")

        def _run_ready_session(
            context: ProbeCommandReadyContextDict,
        ) -> WeaveProbeSessionDict:
            bursts: list[WeaveBurstDict] = []
            for _ in range(burst_count):
                fuel = self._require_self_state()["fuel"]
                if fuel < _MIN_BURST_FUEL:
                    log.warning(
                        "WEAVE: fuel %d below burst floor %d - stopping",
                        fuel,
                        _MIN_BURST_FUEL,
                    )
                    break
                enemy = None
                for attempt in range(_ACQUISITION_ATTEMPTS):
                    enemy = self._acquire_adjacent_enemy(
                        acquisition_timeout_ms=acquisition_timeout_ms,
                        teleport_timeout_ms=teleport_timeout_ms,
                        excluded_ids=frozenset(),
                    )
                    if enemy is not None:
                        break
                    log.warning(
                        "WEAVE: acquisition attempt %d/%d failed",
                        attempt + 1,
                        _ACQUISITION_ATTEMPTS,
                    )
                if enemy is None:
                    log.warning("WEAVE: no target for this burst")
                    continue
                burst = self._weave_burst(enemy, beats_per_burst)
                if burst is not None:
                    bursts.append(burst)
            first_started = None
            if bursts and bursts[0]["beats"]:
                first_started = bursts[0]["beats"][0]["dispatched_ms"]
            envelope = build_probe_session_envelope(
                self,
                context=context,
                first_attempt_started_ms=first_started,
            )
            return WeaveProbeSessionDict(
                session_id=envelope.session_id,
                start_timestamp_ms=envelope.start_timestamp_ms,
                end_timestamp_ms=envelope.end_timestamp_ms,
                base_url=envelope.base_url,
                spawn_x=envelope.spawn_x,
                spawn_y=envelope.spawn_y,
                beats_per_burst=beats_per_burst,
                capture_session_path="",
                initial_sync_timeout_ms=initial_sync_timeout_ms,
                startup_timing=envelope.startup_timing,
                bursts=bursts,
            )

        return execute_live_probe_bootstrap(
            self,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            run_ready_session=_run_ready_session,
        )


def format_weave_probe_summary(session: WeaveProbeSessionDict) -> str:
    """Format the verdict line for the session.

    Args:
        session: Completed weave session.

    Returns:
        Compact human-readable summary.
    """
    parts = []
    for burst in session["bursts"]:
        shoot_only = burst["shots_dispatched"] - burst["moves_dispatched"]
        flag = " KILLED" if burst["target_killed"] else ""
        parts.append(
            f"{burst['target_name']}: {burst['served_hits']} served of "
            f"{burst['shots_dispatched']} shots ({shoot_only} shoot-only + "
            f"{burst['moves_dispatched']} shoot+move){flag}"
        )
    body = " | ".join(parts) if parts else "no bursts completed"
    return f"Weave probe: {body}"


def _create_weave_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> WeaveProbe:
    """Factory for WeaveProbe with injected services.

    Args:
        target_url: URL to navigate to.
        headless: Whether to run the browser headless.
        prefer_account: Whether to prefer account login.

    Returns:
        The constructed probe.
    """
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        WeaveProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, WeaveProbe)
    return probe


def run_weave_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = False,
    beats_per_burst: int = 8,
    burst_count: int = 1,
    initial_sync_timeout_ms: int = 10000,
    acquisition_timeout_ms: int = 5000,
    teleport_timeout_ms: int = 10000,
) -> WeaveProbeSessionDict:
    """Run a live weave probe and save the session JSON.

    Args:
        target_url: Game URL.
        output_path: Where the session JSON lands.
        headless: Whether to run the browser headless.
        prefer_account: Whether to prefer account login.
        beats_per_burst: Beats per burst (even beats also move).
        burst_count: How many bursts to attempt.
        initial_sync_timeout_ms: Bootstrap sync bound.
        acquisition_timeout_ms: Map-sync bound per acquisition.
        teleport_timeout_ms: Teleport bound per acquisition.

    Returns:
        The session record.
    """

    def _run_session(probe: WeaveProbe) -> WeaveProbeSessionDict:
        return probe.execute_weave_probe(
            beats_per_burst=beats_per_burst,
            burst_count=burst_count,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            acquisition_timeout_ms=acquisition_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
        )

    return run_and_save_standard_probe_session(
        probe_factory=_create_weave_probe,
        run_session=_run_session,
        encoder=encode_weave_probe_session,
        summary_formatter=format_weave_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "WeaveProbe",
    "format_weave_probe_summary",
    "run_weave_probe",
]
