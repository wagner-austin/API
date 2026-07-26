"""Tests for shared probe entrypoint helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, load_json_str, narrow_json_to_dict
from tests.conftest import FakeFileSystem

from tankpit_bot.action_lab.probe_entrypoint import (
    ProbeArtifactsProtocol,
    ProbeCaptureMetadataDict,
    StandardProbeSessionDict,
    encode_standard_probe_session,
    extract_standard_capture_metadata,
    run_and_save_standard_probe_session,
)
from tankpit_bot.types import CapturedMessage, decode_capture_session


class _ProbeHarness(ProbeArtifactsProtocol):
    @property
    def session_id(self) -> str:
        return "session-1"

    @property
    def messages(self) -> list[CapturedMessage]:
        return [
            CapturedMessage(
                timestamp_ms=100,
                direction="received",
                payload="abc",
                ws_url="wss://tankpit.com/ws/",
            ),
            CapturedMessage(
                timestamp_ms=250,
                direction="sent",
                payload="def",
                ws_url="wss://tankpit.com/ws/",
            ),
        ]

    @property
    def magic(self) -> str | None:
        return "magic"


class _EmptyProbeHarness(_ProbeHarness):
    @property
    def messages(self) -> list[CapturedMessage]:
        return []


def _make_session() -> StandardProbeSessionDict:
    return StandardProbeSessionDict(
        session_id="session-1",
        start_timestamp_ms=100,
        end_timestamp_ms=250,
        base_url="https://tankpit.com/play",
        capture_session_path="",
    )


def _encode(current_session: StandardProbeSessionDict) -> JSONObject:
    return {
        "capture_session_path": current_session["capture_session_path"],
        "session_id": current_session["session_id"],
        "start_timestamp_ms": current_session["start_timestamp_ms"],
        "end_timestamp_ms": current_session["end_timestamp_ms"],
        "base_url": current_session["base_url"],
    }


def test_run_and_save_standard_probe_session_persists_session_and_capture(
    fake_fs: FakeFileSystem,
) -> None:
    session = run_and_save_standard_probe_session(
        probe_factory=lambda target_url, *, headless, prefer_account: _ProbeHarness(),
        run_session=lambda probe: _make_session(),
        encoder=_encode,
        summary_formatter=lambda current_session: current_session["session_id"],
        target_url="https://tankpit.com/play",
        output_path="probe_session.json",
        headless=False,
        prefer_account=False,
    )

    written = fake_fs.read_text(Path("probe_session.json"))
    decoded = narrow_json_to_dict(load_json_str(written))
    capture_written = fake_fs.read_text(Path("probe_session.capture_session.json"))
    capture_decoded = decode_capture_session(narrow_json_to_dict(load_json_str(capture_written)))

    assert session["capture_session_path"] == "probe_session.capture_session.json"
    assert decoded["capture_session_path"] == "probe_session.capture_session.json"
    assert capture_decoded["session_id"] == "session-1"
    assert capture_decoded["magic"] == "magic"


def test_extract_standard_capture_metadata_reads_common_session_fields() -> None:
    session = _make_session()

    metadata = extract_standard_capture_metadata(session)

    assert metadata == ProbeCaptureMetadataDict(
        session_id="session-1",
        start_timestamp_ms=100,
        end_timestamp_ms=250,
        base_url="https://tankpit.com/play",
    )


def test_encode_standard_probe_session_sets_capture_path_before_encoding() -> None:
    session = _make_session()

    encoded = encode_standard_probe_session(
        session,
        "probe.capture_session.json",
        encoder=_encode,
    )

    assert session["capture_session_path"] == "probe.capture_session.json"
    assert encoded["capture_session_path"] == "probe.capture_session.json"
    assert encoded["session_id"] == "session-1"


def _raise_probe_failure(probe: ProbeArtifactsProtocol) -> StandardProbeSessionDict:
    del probe
    raise RuntimeError("probe aborted mid-session")


def test_aborted_session_still_saves_the_capture_evidence(fake_fs: FakeFileSystem) -> None:
    with pytest.raises(RuntimeError, match="probe aborted mid-session"):
        run_and_save_standard_probe_session(
            probe_factory=lambda target_url, *, headless, prefer_account: _ProbeHarness(),
            run_session=_raise_probe_failure,
            encoder=_encode,
            summary_formatter=lambda current_session: current_session["session_id"],
            target_url="https://tankpit.com/play",
            output_path="probe_session.json",
            headless=False,
            prefer_account=False,
        )

    capture_written = fake_fs.read_text(Path("probe_session.capture_session.json"))
    capture_decoded = decode_capture_session(narrow_json_to_dict(load_json_str(capture_written)))
    assert capture_decoded["session_id"] == "session-1"
    assert capture_decoded["magic"] == "magic"
    assert capture_decoded["start_timestamp_ms"] == 100
    assert capture_decoded["end_timestamp_ms"] == 250
    assert len(capture_decoded["messages"]) == 2
    with pytest.raises(FileNotFoundError):
        fake_fs.read_text(Path("probe_session.json"))


def test_aborted_session_with_no_frames_saves_an_empty_capture(fake_fs: FakeFileSystem) -> None:
    with pytest.raises(RuntimeError, match="probe aborted mid-session"):
        run_and_save_standard_probe_session(
            probe_factory=lambda target_url, *, headless, prefer_account: _EmptyProbeHarness(),
            run_session=_raise_probe_failure,
            encoder=_encode,
            summary_formatter=lambda current_session: current_session["session_id"],
            target_url="https://tankpit.com/play",
            output_path="probe_session.json",
            headless=False,
            prefer_account=False,
        )

    capture_written = fake_fs.read_text(Path("probe_session.capture_session.json"))
    capture_decoded = decode_capture_session(narrow_json_to_dict(load_json_str(capture_written)))
    assert capture_decoded["start_timestamp_ms"] == 0
    assert capture_decoded["end_timestamp_ms"] == 0
    assert capture_decoded["messages"] == []
