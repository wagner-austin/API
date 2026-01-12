"""Session summary building utilities.

This module provides functions for building processed session summaries
from raw capture sessions.
"""

from __future__ import annotations

from tankpit_bot.capture.stats import build_message_stats
from tankpit_bot.types import (
    CaptureSession,
    CombatEvent,
    SessionSummary,
)


def build_session_summary(session: CaptureSession) -> SessionSummary:
    """Build session summary from capture session.

    Args:
        session: The raw capture session.

    Returns:
        Processed SessionSummary with combat events extracted.
    """
    combat: list[CombatEvent] = []
    combat_log = [e for e in session["game_log"] if e["category"] == "combat"]

    for entry in combat_log:
        text = entry["text"]
        event_type = "unknown"
        target = ""

        if text.startswith("You hit "):
            event_type = "hit"
            target = text[8:]
        elif text.startswith("You killed "):
            event_type = "kill"
            target = text[11:]
        elif " hit you" in text:
            event_type = "hit_by"
            target = text.split(" hit you")[0]
        elif " killed you" in text:
            event_type = "killed_by"
            target = text.split(" killed you")[0]

        if event_type != "unknown":
            combat.append(
                CombatEvent(
                    timestamp_ms=entry["timestamp_ms"],
                    event_type=event_type,
                    target=target,
                    tank_id=None,
                )
            )

    return SessionSummary(
        session_id=session["session_id"],
        start_timestamp_ms=session["start_timestamp_ms"],
        end_timestamp_ms=session["end_timestamp_ms"],
        magic=session["magic"],
        tanks=session["tank_names"],
        combat=combat,
        equipment_gains=[],
        game_log=combat_log,
        message_stats=build_message_stats(session),
    )


__all__ = [
    "build_session_summary",
]
