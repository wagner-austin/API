"""Equipment probe session entrypoint with save."""

from __future__ import annotations

from tankpit_bot.action_lab.equipment_probe_runner import execute_equipment_probe_session
from tankpit_bot.action_lab.equipment_probe_types import EquipmentProbeSessionDict
from tankpit_bot.action_lab.teleport import TeleportProbe


def run_and_save_equipment_probe_session(
    probe: TeleportProbe,
    *,
    max_targets: int,
    initial_sync_timeout_ms: int,
    map_sync_timeout_ms: int,
    teleport_timeout_ms: int,
    radar_timeout_ms: int,
    pickup_timeout_ms: int,
    settle_delay_ms: int,
) -> EquipmentProbeSessionDict:
    """Run an equipment probe session and save results.

    Args:
        probe: Configured equipment probe instance.
        max_targets: Maximum equipment targets to attempt.
        initial_sync_timeout_ms: Timeout for initial world sync.
        map_sync_timeout_ms: Timeout for map sync per attempt.
        teleport_timeout_ms: Timeout for teleport per attempt.
        radar_timeout_ms: Timeout for radar per attempt.
        pickup_timeout_ms: Timeout for pickup per attempt.
        settle_delay_ms: Post-action settle delay.

    Returns:
        Complete session result.
    """
    return execute_equipment_probe_session(
        probe,
        max_targets=max_targets,
        initial_sync_timeout_ms=initial_sync_timeout_ms,
        map_sync_timeout_ms=map_sync_timeout_ms,
        teleport_timeout_ms=teleport_timeout_ms,
        radar_timeout_ms=radar_timeout_ms,
        pickup_timeout_ms=pickup_timeout_ms,
        settle_delay_ms=settle_delay_ms,
    )


__all__ = [
    "run_and_save_equipment_probe_session",
]
