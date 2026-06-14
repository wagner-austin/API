"""Summary formatter for equipment probe sessions."""

from __future__ import annotations

from tankpit_bot.action_lab.equipment_probe_types import EquipmentProbeSessionDict


def format_equipment_probe_summary(session: EquipmentProbeSessionDict) -> str:
    """Format a compact summary for an equipment probe session.

    Args:
        session: Completed equipment-probe session payload.

    Returns:
        Human-readable one-line summary.
    """
    picked_up = 0
    no_equipment = 0
    radar_timeout = 0
    teleport_timeout = 0
    for attempt in session["attempts"]:
        status = attempt["status"]
        if status == "picked_up_equipment":
            picked_up += 1
        elif status == "no_equipment_visible":
            no_equipment += 1
        elif status == "radar_timeout":
            radar_timeout += 1
        elif status in ("teleport_timeout", "map_sync_timeout"):
            teleport_timeout += 1
    parts = [f"{len(session['attempts'])} attempts"]
    if picked_up:
        parts.append(f"{picked_up} picked up")
    if no_equipment:
        parts.append(f"{no_equipment} no equipment")
    if radar_timeout:
        parts.append(f"{radar_timeout} radar timeout")
    if teleport_timeout:
        parts.append(f"{teleport_timeout} teleport timeout")
    return " | ".join(parts)


__all__ = [
    "format_equipment_probe_summary",
]
