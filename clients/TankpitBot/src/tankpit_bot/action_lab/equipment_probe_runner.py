"""Equipment probe session runner."""

from __future__ import annotations

from tankpit_bot.action_lab.equipment_probe_types import EquipmentProbeSessionDict
from tankpit_bot.action_lab.teleport import TeleportProbe


def execute_equipment_probe_session(
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
    """Execute an equipment probe session through the probe harness.

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

    Raises:
        NotImplementedError: Session runner not yet implemented.
    """
    raise NotImplementedError("equipment probe session runner not yet implemented")


__all__ = [
    "execute_equipment_probe_session",
]
