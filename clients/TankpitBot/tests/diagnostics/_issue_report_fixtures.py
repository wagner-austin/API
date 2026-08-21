"""Shared record builders for the issue-report tests."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from tankpit_bot.runtime_logging import (
    emit_diagnostic,
)


def _emit_session_room(room_id: str, field_image: str) -> None:
    """Emit a ``session_room_joined`` diagnostic through the real pipeline."""
    emit_diagnostic(
        diagnostic_kind="session_room_joined",
        room_id=room_id,
        field_image=field_image,
    )


def _emit_teleport_attempt(
    *,
    target_x: int,
    target_y: int,
    cycle_id: int,
    status: str,
    sent_window: str = "(none)",
    received_window: str = "(none)",
    page_snapshot_count: int = 0,
) -> None:
    """Emit one ``teleport_attempt`` diagnostic through the real pipeline."""
    emit_diagnostic(
        diagnostic_kind="teleport_attempt",
        target_x=target_x,
        target_y=target_y,
        teleport_cycle_id=cycle_id,
        status=status,
        sent_window=sent_window,
        received_window=received_window,
        page_snapshots="(none)",
        page_snapshot_count=page_snapshot_count,
    )


def _emit_fuel_target_selection(
    *,
    cycle_id: int,
    target_present: bool,
    target_x: int = -1,
    target_y: int = -1,
    summary: str = "fuel: total=0",
    decision_basis: str = "world_ts=0",
) -> None:
    """Emit one ``fuel_target_selection`` diagnostic through the real pipeline."""
    emit_diagnostic(
        diagnostic_kind="fuel_target_selection",
        radar_cycle_id=cycle_id,
        target_present=target_present,
        target_x=target_x,
        target_y=target_y,
        summary=summary,
        decision_basis=decision_basis,
        terrain_available=True,
        self_state_available=True,
    )


def _round_trip(encoded: JSONObject) -> JSONObject:
    """Round-trip a dict through ``dump_json_str`` / ``load_json_str``."""
    return narrow_json_to_dict(load_json_str(dump_json_str(encoded)))
