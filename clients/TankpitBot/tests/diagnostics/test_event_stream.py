"""Tests for the shared JSONL event-stream loading helpers.

Loading itself is exercised end-to-end by the issue-report and self-map
analyzer tests (real emit pipeline -> JSONL -> ``load_event_records``);
this module covers the CLI source-path resolution contract directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import FakeFileSystem

from tankpit_bot.diagnostics.event_stream import load_event_records, resolve_source_path
from tankpit_bot.runtime_logging import (
    configure_bot_runtime_logging,
    emit_diagnostic,
)


def test_resolve_source_path_raises_when_file_missing(
    fake_fs: FakeFileSystem,
) -> None:
    """``resolve_source_path`` raises ``FileNotFoundError`` for a missing path."""
    with pytest.raises(FileNotFoundError, match="events artifact not found"):
        resolve_source_path(["does_not_exist.jsonl"])


def test_resolve_source_path_raises_with_default_when_no_bot_artifact(
    fake_fs: FakeFileSystem,
) -> None:
    """When argv is empty and the default bot path is absent, raise."""
    with pytest.raises(FileNotFoundError, match=r"latest\.events\.jsonl"):
        resolve_source_path([])


def test_resolve_source_path_returns_existing_explicit_path(
    fake_fs: FakeFileSystem,
) -> None:
    """An explicitly supplied existing artifact path is returned unchanged."""
    artifacts = configure_bot_runtime_logging("20260609-120000")
    emit_diagnostic(diagnostic_kind="session_room_joined", room_id="1", field_image="field01.gif")
    latest = Path(artifacts["latest_events_path"])

    resolved = resolve_source_path([str(latest)])

    assert resolved == latest


def test_load_event_records_skips_blank_lines(fake_fs: FakeFileSystem) -> None:
    """Blank lines in an artifact are skipped; real events all decode."""
    artifacts = configure_bot_runtime_logging("20260609-120000")
    emit_diagnostic(diagnostic_kind="session_room_joined", room_id="1", field_image="field01.gif")
    latest = Path(artifacts["latest_events_path"])
    fake_fs.append_text(latest, "\n\n")

    records = load_event_records(latest)

    # The stamp opens the artifact; the emitted record follows it.
    assert len(records) == 2
    assert records[0]["fields"]["diagnostic_kind"] == "session_build"
    assert records[1]["channel"] == "DIAGNOSTIC"
    assert records[1]["fields"]["diagnostic_kind"] == "session_room_joined"
