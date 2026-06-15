"""Coverage test for action_lab/teleport.py line 206: _emit_teleport_attempt_diagnostic.

Exercises the emit_diagnostic call inside _emit_teleport_attempt_diagnostic
by providing a minimal provider with an empty message list.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import FakeFileSystem

from tankpit_bot.action_lab.teleport_helpers import _emit_teleport_attempt_diagnostic
from tankpit_bot.action_lab.types import TeleportTargetDict
from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.runtime_logging import (
    RuntimeEventRecordDict,
    configure_bot_runtime_logging,
)
from tankpit_bot.state import WorldStateDict, make_empty_world_state
from tankpit_bot.types import CapturedMessage


class _MinimalProvider:
    """Minimal provider stub for _emit_teleport_attempt_diagnostic.

    Provides empty messages and no magic key, which is sufficient for
    the format helpers to produce ``"none"`` windows. Implements the
    full ``BufferedWorldStateProviderProtocol``.
    """

    def __init__(self) -> None:
        """Initialize with empty state."""
        self._cdp_message_buffer: list[str] = []

    def get_world_state(self) -> WorldStateDict:
        """Return empty world state."""
        return make_empty_world_state()

    @property
    def messages(self) -> list[CapturedMessage]:
        """Return empty message list."""
        return []

    @property
    def magic(self) -> str | None:
        """Return None (no session magic key)."""
        return None


def _teleport_records(latest_events_path: str) -> list[RuntimeEventRecordDict]:
    """Return every ``teleport_attempt`` record from the artifact."""
    return [
        record
        for record in load_event_records(Path(latest_events_path))
        if record["fields"].get("diagnostic_kind") == "teleport_attempt"
    ]


def test_emit_teleport_attempt_diagnostic(fake_fs: FakeFileSystem) -> None:
    """_emit_teleport_attempt_diagnostic emits one diagnostic event."""
    artifacts = configure_bot_runtime_logging("20260610-120000")

    provider = _MinimalProvider()
    target = TeleportTargetDict(label="test", x=120, y=130)

    _emit_teleport_attempt_diagnostic(
        provider,
        target=target,
        teleport_cycle_id=1,
        status="landed_exact",
        message_start_index=0,
        page_snapshots=[],
    )

    records = _teleport_records(artifacts["latest_events_path"])
    assert len(records) == 1

    fields = records[0]["fields"]
    assert fields["diagnostic_kind"] == "teleport_attempt"
    assert fields["target_x"] == 120
    assert fields["target_y"] == 130
    assert fields["teleport_cycle_id"] == 1
    assert fields["status"] == "landed_exact"
    assert fields["sent_window"] == "none"
    assert fields["received_window"] == "none"
    assert fields["page_snapshots"] == "none"
    assert fields["page_snapshot_count"] == 0
