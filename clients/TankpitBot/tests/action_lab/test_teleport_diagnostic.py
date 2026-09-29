"""Contract test: the live ``teleport_attempt`` emitter feeds the issue report.

``_log_teleport_attempt_diagnostic`` is the only production writer of the
``teleport_attempt`` diagnostic, and ``diagnostics.issue_report`` is its
reader. Before board task 6bfd15c1 the two disagreed on field names
(``cycle`` / ``sent`` / ``received`` / ``page`` written, ``teleport_cycle_id``
/ ``sent_window`` / ``received_window`` / ``page_snapshot_count`` read), and
the issue-report tests passed only because their fixture hand-copied the
reader's schema. This test emits through the real writer and builds the
report through the real reader, so the two cannot drift apart again.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import FakeFileSystem

from tankpit_bot.action_lab.teleport_helpers import _log_teleport_attempt_diagnostic
from tankpit_bot.action_lab.types import TeleportAttemptStatus, TeleportTargetDict
from tankpit_bot.diagnostics.issue_report import build_issue_report
from tankpit_bot.diagnostics.issue_report_types import TeleportAttemptRecordDict
from tankpit_bot.runtime_logging import configure_probe_runtime_logging
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import WorldStateDict, make_empty_world_state
from tankpit_bot.types import CapturedMessage


class _MinimalProvider:
    """Buffered provider with an empty message log.

    Empty messages are enough for the window formatters to render
    ``"none"``. Implements the full ``BufferedWorldStateProviderProtocol``.
    """

    def __init__(self) -> None:
        """Initialize with empty state."""
        self.world = WorldService()
        self._cdp_message_buffer: list[str] = []
        self.xor_table: bytes | None = None

    def get_world_state(self) -> WorldStateDict:
        """Return an empty world state."""
        return make_empty_world_state()

    @property
    def messages(self) -> list[CapturedMessage]:
        """Return the empty message list."""
        return []

    @property
    def magic(self) -> str | None:
        """Return None (no session magic key)."""
        return None


def test_live_teleport_attempt_diagnostic_is_read_by_the_issue_report(
    fake_fs: FakeFileSystem,
) -> None:
    """The report reads every field the live emitter writes, unchanged."""
    artifacts = configure_probe_runtime_logging("fuel", "20260929-040000")

    _log_teleport_attempt_diagnostic(
        _MinimalProvider(),
        target=TeleportTargetDict(label="contract", x=120, y=130),
        teleport_cycle_id=3,
        status=TeleportAttemptStatus.TELEPORT_TIMEOUT,
        message_start_index=0,
        page_snapshots=[],
    )

    report = build_issue_report(Path(artifacts["latest_events_path"]))

    attempts = report["teleport_attempts"]
    assert len(attempts) == 1
    assert attempts[0] == TeleportAttemptRecordDict(
        target_x=120,
        target_y=130,
        teleport_cycle_id=3,
        status="teleport_timeout",
        timestamp=attempts[0]["timestamp"],
        sent_window="none",
        received_window="none",
        page_snapshot_count=0,
    )
    assert report["teleport_success_count"] == 0
    assert report["teleport_failure_count"] == 1
