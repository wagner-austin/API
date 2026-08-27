"""Live fire-cadence probe — how fast will the server serve our shots?

The bot's combat loop fires once per 2 s tick; humans demonstrably
land two clicks inside that window (brrruh's paired-hit syncs,
2026-08-26). This probe measures the SERVER's serve rate directly:
teleport adjacent to an enemy (the combat probe's acquisition,
lifted), then fire bursts at fixed spacings and count served shots
from server-refreshed 0x49 ammo snapshots — one dual/homing debit
per landed shot, the per-shot ammo ledger of [[weapon-selection]].
A burst whose served count trails its dispatch count (with the
target still alive) has found the cap.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.cadence_probe_types import (
    CadenceBurstDict,
    CadenceProbeSessionDict,
    CadenceShotDict,
    encode_cadence_probe_session,
)
from tankpit_bot.action_lab.combat_probe import CombatProbe, _enemy_from_world_state
from tankpit_bot.action_lab.probe_entrypoint import (
    run_and_save_standard_probe_session,
)
from tankpit_bot.action_lab.probe_runtime import (
    ProbeCommandReadyContextDict,
    execute_live_probe_bootstrap,
)
from tankpit_bot.action_lab.probe_session import build_probe_session_envelope
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state

log = get_logger(__name__)

_INVENTORY_WAIT_MS = 800.0
_SETTLE_MS = 4000.0

#: 2026-07-25 contract ("never leave the tank exposed"): stop opening
#: new bursts once fuel cannot absorb a full burst of return fire.
_MIN_BURST_FUEL = 400

#: Acquisition is flaky by nature (fresh-enemy filter, teleport
#: drift): the first live run burned all four spacings on transient
#: misses. Each spacing gets this many attempts before it is skipped.
_ACQUISITION_ATTEMPTS = 3


def _read_fresh_ammo(probe: CombatProbe) -> tuple[int, int]:
    """Return server-refreshed (dual, homing) counts.

    Requests a 0x49 inventory snapshot and drains it in, so the counts
    are the server's ledger — not the local per-echo decrement.

    Args:
        probe: The live probe.

    Returns:
        (dual count, homing count).
    """
    page = probe._require_page()
    probe.request_inventory()
    page.wait_for_timeout(_INVENTORY_WAIT_MS)
    action_hooks.drain_buffered_messages(probe, probe.world)
    inventory = get_inventory_state(probe.world)
    return (
        inventory["dual_shots"]["count"],
        inventory["homing_shots"]["count"],
    )


class CadenceProbe(CombatProbe):
    """Fire fixed-spacing bursts and count what the server serves."""

    def _fire_burst(
        self,
        enemy: EnemyThreatDict,
        spacing_ms: int,
        shots_per_burst: int,
    ) -> CadenceBurstDict:
        """Fire one burst at ``spacing_ms`` and book the ammo ledger.

        The target position is re-read from the world registry before
        every shot; a vanished registry entry means the target died
        mid-burst and the burst ends there (its served count still
        reads true — served shots debited before the death).

        Args:
            enemy: Target acquired adjacent by the combat acquisition.
            spacing_ms: Wait between shot dispatches.
            shots_per_burst: Dispatch budget for this burst.

        Returns:
            The burst record, ammo-before/after included.
        """
        page = self._require_page()
        target_id = enemy["tank_id"]
        dual_before, homing_before = _read_fresh_ammo(self)
        fuel_before = self._require_self_state()["fuel"]

        shots: list[CadenceShotDict] = []
        target_killed = False
        for shot_number in range(1, shots_per_burst + 1):
            action_hooks.drain_buffered_messages(self, self.world)
            position = _enemy_from_world_state(self, target_id)
            if position is None:
                target_killed = True
                log.info("CADENCE: target %s gone after %d shots", enemy["name"], len(shots))
                break
            tx, ty = position
            self.shoot(tx, ty, target_id)
            shots.append(
                CadenceShotDict(
                    shot_number=shot_number,
                    dispatched_ms=action_hooks.get_current_time_ms(),
                    target_x=tx,
                    target_y=ty,
                )
            )
            page.wait_for_timeout(float(spacing_ms))

        page.wait_for_timeout(_SETTLE_MS)
        action_hooks.drain_buffered_messages(self, self.world)
        dual_after, homing_after = _read_fresh_ammo(self)
        fuel_after = self._require_self_state()["fuel"]
        served = (dual_before - dual_after) + (homing_before - homing_after)
        log.info(
            "CADENCE: spacing=%dms dispatched=%d served=%d (dual %d->%d homing %d->%d) "
            "fuel %d->%d killed=%s",
            spacing_ms,
            len(shots),
            served,
            dual_before,
            dual_after,
            homing_before,
            homing_after,
            fuel_before,
            fuel_after,
            target_killed,
        )
        return CadenceBurstDict(
            spacing_ms=spacing_ms,
            target_id=target_id,
            target_name=enemy["name"],
            shots=shots,
            dispatched=len(shots),
            dual_before=dual_before,
            dual_after=dual_after,
            homing_before=homing_before,
            homing_after=homing_after,
            fuel_before=fuel_before,
            fuel_after=fuel_after,
            served_hits=served,
            target_killed=target_killed,
        )

    def execute_cadence_probe(
        self,
        *,
        spacings_ms: tuple[int, ...],
        shots_per_burst: int,
        initial_sync_timeout_ms: int,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
    ) -> CadenceProbeSessionDict:
        """Run one burst per spacing, each against a fresh enemy.

        Args:
            spacings_ms: Dispatch spacings to test, run in order.
            shots_per_burst: Dispatch budget per burst.
            initial_sync_timeout_ms: Bootstrap sync bound.
            acquisition_timeout_ms: Map-sync bound per acquisition.
            teleport_timeout_ms: Teleport bound per acquisition.

        Returns:
            The session record.

        Raises:
            ValueError: On an empty spacing list or non-positive budget.
        """
        if not spacings_ms or shots_per_burst <= 0:
            raise ValueError("spacings_ms must be non-empty and shots_per_burst positive")

        def _run_ready_session(
            context: ProbeCommandReadyContextDict,
        ) -> CadenceProbeSessionDict:
            bursts: list[CadenceBurstDict] = []
            for spacing_ms in spacings_ms:
                fuel = self._require_self_state()["fuel"]
                if fuel < _MIN_BURST_FUEL:
                    log.warning(
                        "CADENCE: fuel %d below burst floor %d - stopping",
                        fuel,
                        _MIN_BURST_FUEL,
                    )
                    break
                enemy = None
                for attempt in range(_ACQUISITION_ATTEMPTS):
                    # No used-target exclusion (unlike the accuracy
                    # probe): re-bursting the same enemy keeps the
                    # spacings comparable, and the first live run
                    # starved on a one-NPC room because the exclusion
                    # was inherited (2026-08-26 21:29 — spacing 2000
                    # fired 6/6 at red-8, then every later spacing
                    # found "no enemy").
                    enemy = self._acquire_adjacent_enemy(
                        acquisition_timeout_ms=acquisition_timeout_ms,
                        teleport_timeout_ms=teleport_timeout_ms,
                        excluded_ids=frozenset(),
                    )
                    if enemy is not None:
                        break
                    log.warning(
                        "CADENCE: acquisition attempt %d/%d failed for spacing %dms",
                        attempt + 1,
                        _ACQUISITION_ATTEMPTS,
                        spacing_ms,
                    )
                if enemy is None:
                    log.warning("CADENCE: no target for spacing %dms", spacing_ms)
                    continue
                bursts.append(self._fire_burst(enemy, spacing_ms, shots_per_burst))
            first_started = None
            if bursts and bursts[0]["shots"]:
                first_started = bursts[0]["shots"][0]["dispatched_ms"]
            envelope = build_probe_session_envelope(
                self,
                context=context,
                first_attempt_started_ms=first_started,
            )
            return CadenceProbeSessionDict(
                session_id=envelope.session_id,
                start_timestamp_ms=envelope.start_timestamp_ms,
                end_timestamp_ms=envelope.end_timestamp_ms,
                base_url=envelope.base_url,
                spawn_x=envelope.spawn_x,
                spawn_y=envelope.spawn_y,
                shots_per_burst=shots_per_burst,
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


def format_cadence_probe_summary(session: CadenceProbeSessionDict) -> str:
    """Format the one-line serve-rate table for the session.

    Args:
        session: Completed cadence session.

    Returns:
        Compact human-readable summary.
    """
    parts = []
    for burst in session["bursts"]:
        flag = " KILLED" if burst["target_killed"] else ""
        parts.append(
            f"{burst['spacing_ms']}ms: {burst['served_hits']}/{burst['dispatched']} served{flag}"
        )
    body = " | ".join(parts) if parts else "no bursts completed"
    return f"Cadence probe: {body}"


def _create_cadence_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> CadenceProbe:
    """Factory for CadenceProbe with injected services.

    Args:
        target_url: URL to navigate to.
        headless: Whether to run the browser headless.
        prefer_account: Whether to prefer account login.

    Returns:
        The constructed probe.
    """
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        CadenceProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, CadenceProbe)
    return probe


def run_cadence_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = False,
    spacings_ms: tuple[int, ...] = (2000, 1000, 500, 250),
    shots_per_burst: int = 6,
    initial_sync_timeout_ms: int = 10000,
    acquisition_timeout_ms: int = 5000,
    teleport_timeout_ms: int = 10000,
) -> CadenceProbeSessionDict:
    """Run a live cadence probe and save the session JSON.

    Args:
        target_url: Game URL.
        output_path: Where the session JSON lands.
        headless: Whether to run the browser headless.
        prefer_account: Whether to prefer account login.
        spacings_ms: Dispatch spacings to test.
        shots_per_burst: Dispatch budget per burst.
        initial_sync_timeout_ms: Bootstrap sync bound.
        acquisition_timeout_ms: Map-sync bound per acquisition.
        teleport_timeout_ms: Teleport bound per acquisition.

    Returns:
        The session record.
    """

    def _run_session(probe: CadenceProbe) -> CadenceProbeSessionDict:
        return probe.execute_cadence_probe(
            spacings_ms=spacings_ms,
            shots_per_burst=shots_per_burst,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            acquisition_timeout_ms=acquisition_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
        )

    return run_and_save_standard_probe_session(
        probe_factory=_create_cadence_probe,
        run_session=_run_session,
        encoder=encode_cadence_probe_session,
        summary_formatter=format_cadence_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "CadenceProbe",
    "format_cadence_probe_summary",
    "run_cadence_probe",
]
