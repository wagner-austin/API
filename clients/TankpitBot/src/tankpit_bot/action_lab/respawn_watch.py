"""Live respawn-watch probe: engage an adjacent practice bot, then map-poll.

The reactivation law (wiki ``enemy-bot-behavior``) rests on 102
archive-mined death-to-next-seen pairs; this probe produces the live
witness. It reuses the enemy-teleport machinery to land adjacent to a
bot, then replaces the watch dwell with two phases:

1. ENGAGE — fire a single at the target's registry position every
   ``shot_interval_ms`` until it leaves the registry (killed or
   teleported off; the capture distinguishes 0x41 from 0x58 offline)
   or ``engage_ms`` elapses. Kept short by design: an adjacent bot
   returns fire at ~45 fuel per hit, and a full-fuel recruit flees at
   7 hits anyway — kills come from already-damaged targets.
2. MAP-POLL — send a map open every ``poll_interval_ms`` for
   ``poll_ms`` so the 0x4C snapshots pin the same-id reappearance
   tick and tile. Map polls are request-response and therefore immune
   to the push-stream mute (wiki ``server-push-gating``).

All measurement happens offline from the capture, exactly like the
bot-watch runs — the probe is choreography, the capture is the
instrument.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.enemy_teleport import (
    EnemyTeleportProbe,
    _enemy_by_id,
    format_enemy_teleport_probe_summary,
)
from tankpit_bot.action_lab.enemy_teleport_types import (
    EnemyTeleportProbeSessionDict,
    encode_enemy_teleport_probe_session,
)
from tankpit_bot.action_lab.probe_entrypoint import run_and_save_standard_probe_session
from tankpit_bot.bot.ai.world_types import EnemyThreatDict

log = get_logger(__name__)


class RespawnWatchProbe(EnemyTeleportProbe):
    """Enemy-teleport probe whose landed phase engages and then map-polls."""

    engage_ms: int = 30000
    shot_interval_ms: int = 2000
    poll_ms: int = 60000
    poll_interval_ms: int = 2000

    def _post_landing_phase(
        self,
        page: action_session.WaitPageProtocol,
        enemy: EnemyThreatDict,
        settle_delay_ms: int,
        heartbeat_interval_ms: int,
    ) -> None:
        """Engage the landed-adjacent enemy, then poll the map.

        Args:
            page: Playwright page driving the waits.
            enemy: The enemy this attempt teleported to.
            settle_delay_ms: Unused (the dwell is replaced).
            heartbeat_interval_ms: Unused (the dwell is replaced).
        """
        del settle_delay_ms, heartbeat_interval_ms
        vanished = self._engage_phase(page, enemy)
        log.info(
            "Respawn watch: target %d %s; starting map poll",
            enemy["tank_id"],
            "left the registry" if vanished else "survived the engage window",
        )
        self._map_poll_phase(page)

    def _engage_phase(
        self,
        page: action_session.WaitPageProtocol,
        enemy: EnemyThreatDict,
    ) -> bool:
        """Fire at the target's current registry position until it vanishes.

        Args:
            page: Playwright page driving the waits.
            enemy: The enemy to engage.

        Returns:
            True when the target left the registry inside the window
            (killed or fled), False when the window expired first.
        """
        started_ms = action_hooks.get_current_time_ms()
        while action_hooks.get_current_time_ms() - started_ms < self.engage_ms:
            action_hooks.drain_buffered_messages(self)
            current = _enemy_by_id(self, enemy["tank_id"])
            if current is None:
                return True
            self.shoot(current["x"], current["y"], enemy["tank_id"])
            page.wait_for_timeout(float(self.shot_interval_ms))
        return False

    def _map_poll_phase(self, page: action_session.WaitPageProtocol) -> None:
        """Send periodic map opens so 0x4C snapshots pin the respawn.

        Args:
            page: Playwright page driving the waits.
        """
        started_ms = action_hooks.get_current_time_ms()
        while action_hooks.get_current_time_ms() - started_ms < self.poll_ms:
            self.open_map()
            page.wait_for_timeout(float(self.poll_interval_ms))
            action_hooks.drain_buffered_messages(self)


def _create_respawn_watch_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> RespawnWatchProbe:
    """Factory for RespawnWatchProbe with injected services."""
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        RespawnWatchProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, RespawnWatchProbe)
    return probe


def run_respawn_watch_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = False,
    max_attempts: int = 4,
    initial_sync_timeout_ms: int = 10000,
    acquisition_timeout_ms: int = 3000,
    teleport_timeout_ms: int = 10000,
    engage_ms: int = 30000,
    shot_interval_ms: int = 2000,
    poll_ms: int = 60000,
    poll_interval_ms: int = 2000,
) -> EnemyTeleportProbeSessionDict:
    """Run a live respawn-watch probe and save the session JSON."""

    def _run_session(probe: RespawnWatchProbe) -> EnemyTeleportProbeSessionDict:
        probe.engage_ms = engage_ms
        probe.shot_interval_ms = shot_interval_ms
        probe.poll_ms = poll_ms
        probe.poll_interval_ms = poll_interval_ms
        return probe.execute_probe(
            acquisition_strategy="map_open",
            max_attempts=max_attempts,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            acquisition_timeout_ms=acquisition_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            settle_delay_ms=0,
            heartbeat_interval_ms=0,
        )

    return run_and_save_standard_probe_session(
        probe_factory=_create_respawn_watch_probe,
        run_session=_run_session,
        encoder=encode_enemy_teleport_probe_session,
        summary_formatter=format_enemy_teleport_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "RespawnWatchProbe",
    "run_respawn_watch_probe",
]
