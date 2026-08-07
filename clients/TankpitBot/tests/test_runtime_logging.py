"""Tests for runtime logging: artifact handlers and emitters.

``test_runtime_logging.py`` was 693 lines; the record codec and context
tests are now a sibling, mirroring the source split.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
)

from tankpit_bot.runtime_logging import (
    configure_bot_runtime_logging,
    configure_probe_runtime_logging,
    configure_sniff_runtime_logging,
    emit_ai,
    emit_diagnostic,
    emit_state,
    emit_sync,
    emit_wire,
    emit_world,
    get_bot_runtime_artifacts,
    get_probe_runtime_artifacts,
    get_sniff_runtime_artifacts,
)
from tankpit_bot.runtime_records import (
    RuntimeEventRecordDict,
    decode_runtime_event_record,
    encode_runtime_event_record,
)
from tests.conftest import FakeFileSystem


def test_configure_bot_runtime_logging_writes_text_and_event_artifacts(
    fake_fs: FakeFileSystem,
) -> None:
    """Bot runtime logging mirrors high-signal events into canonical artifacts."""
    artifacts = configure_bot_runtime_logging("20260331-230405")

    emit_state("IDLE")
    emit_ai("collect fuel at (%d,%d)", 120, 137)
    emit_wire("pickup_move")
    emit_sync("waiting for collection at (%d,%d)", 120, 137)
    emit_world("Fuel: %d -> %d (%+d)", 499, 355, -144)

    files = fake_fs.get_written_files()
    latest_log = files[artifacts["latest_log_path"]]
    archive_log = files[artifacts["archive_log_path"]]
    latest_events = files[artifacts["latest_events_path"]].strip().splitlines()
    archive_events = files[artifacts["archive_events_path"]].strip().splitlines()

    assert "STATE: IDLE" in latest_log
    assert "AI: collect fuel at (120,137)" in latest_log
    assert "WIRE: pickup_move" in latest_log
    assert "SYNC: waiting for collection at (120,137)" in latest_log
    assert "WORLD: Fuel: 499 -> 355 (-144)" in latest_log
    assert archive_log == latest_log
    assert len(latest_events) == 5
    assert len(archive_events) == 5

    decoded_first = decode_runtime_event_record(
        narrow_json_to_dict(load_json_str(latest_events[0]))
    )
    decoded_last = decode_runtime_event_record(
        narrow_json_to_dict(load_json_str(latest_events[-1]))
    )

    assert decoded_first["mode"] == "bot"
    assert decoded_first["channel"] == "STATE"
    assert decoded_first["message"] == "IDLE"
    assert decoded_last["channel"] == "WORLD"
    assert decoded_last["message"] == "Fuel: 499 -> 355 (-144)"


def test_configure_sniff_runtime_logging_resets_latest_files(
    fake_fs: FakeFileSystem,
) -> None:
    """Sniffer runtime logging resets latest files and uses sniff mode in events."""
    fake_fs.write_text(Path("runs\\sniff\\latest.log"), "stale")
    fake_fs.write_text(Path("runs\\sniff\\latest.events.jsonl"), "stale")
    fake_fs.write_text(Path("runs\\sniff\\latest.capture_session.json"), "stale")
    fake_fs.write_text(Path("runs\\sniff\\latest.raw_capture.json"), "stale")
    fake_fs.write_text(Path("runs\\sniff\\latest.session_summary.json"), "stale")

    artifacts = configure_sniff_runtime_logging("20260331-230405")
    emit_world("Captured %d WebSocket messages in %.1fs", 88, 37.3)

    files = fake_fs.get_written_files()
    assert "stale" not in files[artifacts["latest_log_path"]]
    assert files[artifacts["latest_capture_path"]] == ""
    assert files[artifacts["latest_raw_capture_path"]] == ""
    assert files[artifacts["latest_summary_path"]] == ""
    assert "WORLD: Captured 88 WebSocket messages in 37.3s" in files[artifacts["latest_log_path"]]

    event_line = files[artifacts["latest_events_path"]].strip()
    decoded = decode_runtime_event_record(narrow_json_to_dict(load_json_str(event_line)))

    assert decoded["mode"] == "sniff"
    assert decoded["channel"] == "WORLD"
    assert decoded["message"] == "Captured 88 WebSocket messages in 37.3s"


def test_runtime_logging_accessors_track_active_mode(
    fake_fs: FakeFileSystem,
) -> None:
    """Runtime artifact accessors expose only the currently configured mode."""
    bot_artifacts = configure_bot_runtime_logging("20260331-230405")

    assert get_bot_runtime_artifacts() == bot_artifacts
    assert get_sniff_runtime_artifacts() is None
    assert get_probe_runtime_artifacts() is None

    sniff_artifacts = configure_sniff_runtime_logging("20260331-230406")

    assert get_bot_runtime_artifacts() is None
    assert get_sniff_runtime_artifacts() == sniff_artifacts
    assert get_probe_runtime_artifacts() is None

    probe_artifacts = configure_probe_runtime_logging("fuel", "20260331-230407")

    assert get_bot_runtime_artifacts() is None
    assert get_sniff_runtime_artifacts() is None
    assert get_probe_runtime_artifacts() == probe_artifacts


def test_configure_probe_runtime_logging_writes_probe_jsonl(
    fake_fs: FakeFileSystem,
) -> None:
    """Probe runtime logging mirrors structured events into runs/probe artifacts."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")

    emit_diagnostic(diagnostic_kind="movement_probe_map_already_showing")

    files = fake_fs.get_written_files()
    latest_events = files[artifacts["latest_events_path"]].strip()
    decoded_raw = narrow_json_to_dict(load_json_str(latest_events))

    assert decoded_raw["mode"] == "probe:fuel"
    assert decoded_raw["channel"] == "DIAGNOSTIC"
    assert decoded_raw["diagnostic_kind"] == "movement_probe_map_already_showing"


def test_configure_probe_runtime_logging_resets_latest_files(
    fake_fs: FakeFileSystem,
) -> None:
    """Reconfiguring probe logging truncates stale latest files first."""
    from pathlib import Path

    fake_fs.write_text(Path("runs\\probe\\latest.fuel.log"), "stale")
    fake_fs.write_text(Path("runs\\probe\\latest.fuel.events.jsonl"), "stale")

    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    emit_diagnostic(diagnostic_kind="fuel_target_selection", radar_cycle=1)

    files = fake_fs.get_written_files()
    assert "stale" not in files[artifacts["latest_log_path"]]
    assert "stale" not in files[artifacts["latest_events_path"]]


def test_emit_diagnostic_writes_structured_fields_to_jsonl(
    fake_fs: FakeFileSystem,
) -> None:
    """``emit_diagnostic`` spreads ``diagnostic_kind`` plus caller fields into JSONL."""
    artifacts = configure_bot_runtime_logging("20260331-230405")

    emit_diagnostic(
        diagnostic_kind="action_phase_overlap",
        attempt="fuel_ground_163_101",
        active_phase="move",
        active_cycle_id=3,
        active_started_ms=1780821408812,
        next_phase="pickup",
        next_cycle_id=3,
        next_started_ms=1780821408812,
    )

    files = fake_fs.get_written_files()
    event_line = files[artifacts["latest_events_path"]].strip()
    decoded_raw = narrow_json_to_dict(load_json_str(event_line))

    assert decoded_raw["channel"] == "DIAGNOSTIC"
    assert decoded_raw["diagnostic_kind"] == "action_phase_overlap"
    assert decoded_raw["attempt"] == "fuel_ground_163_101"
    assert decoded_raw["active_phase"] == "move"
    assert decoded_raw["active_cycle_id"] == 3

    decoded = decode_runtime_event_record(decoded_raw)
    assert decoded["fields"]["diagnostic_kind"] == "action_phase_overlap"
    assert decoded["fields"]["next_cycle_id"] == 3


def test_runtime_logging_ignores_non_string_runtime_extras(
    fake_fs: FakeFileSystem,
) -> None:
    """Structured event handler ignores malformed runtime extras."""
    artifacts = configure_bot_runtime_logging("20260331-230405")

    from platform_core.logging import stdlib_logging

    logger = stdlib_logging.getLogger("tankpit_bot.runtime.invalid")
    logger.info(
        "plain malformed event",
        extra={"runtime_channel": 1, "runtime_message": "bad"},
    )

    files = fake_fs.get_written_files()
    assert files[artifacts["latest_events_path"]] == ""


def test_runtime_logging_reconfigures_without_duplicate_artifact_handlers(
    fake_fs: FakeFileSystem,
) -> None:
    """Reconfiguration replaces artifact handlers instead of stacking them."""
    first = configure_bot_runtime_logging("20260331-230405")
    emit_ai("first")
    second = configure_bot_runtime_logging("20260331-230406")
    emit_ai("second")

    files = fake_fs.get_written_files()
    assert "first" not in files[second["latest_log_path"]]
    assert files[second["latest_log_path"]].count("AI: second") == 1
    assert files[first["archive_log_path"]].count("AI: first") == 1


def test_action_outcome_emission_writes_structured_fields_to_jsonl(
    fake_fs: FakeFileSystem,
) -> None:
    """Ledger outcome emitters spread their payload into the JSONL stream."""
    from platform_core.json_utils import load_json_str, narrow_json_to_dict

    from tankpit_bot.ledger.outcome.map_open import emit_map_open_data_processed
    from tankpit_bot.sniffer.world_state import get_world_service

    artifacts = configure_bot_runtime_logging("20260331-230405")

    emit_map_open_data_processed(get_world_service().ledger, duration_ms=850)

    files = fake_fs.get_written_files()
    event_line = files[artifacts["latest_events_path"]].strip()

    decoded_raw = narrow_json_to_dict(load_json_str(event_line))
    assert decoded_raw["channel"] == "DIAGNOSTIC"
    assert decoded_raw["diagnostic_kind"] == "action_outcome"
    assert decoded_raw["action_kind"] == "map_open"
    assert decoded_raw["outcome"] == "map_data_processed"
    assert decoded_raw["duration_ms"] == 850
    assert decoded_raw["attempt_id"] == 1
    assert decoded_raw["event_id"] == 1


def test_encode_runtime_event_record_rejects_reserved_key_in_fields() -> None:
    """Field collisions with reserved top-level keys are surfaced, not silenced."""
    import pytest

    record = RuntimeEventRecordDict(
        timestamp="2026-01-01T00:00:00",
        level="INFO",
        logger="t",
        mode="bot",
        channel="WIRE_COMPLETE",
        message="x",
        fields={"timestamp": "shadow"},
    )
    with pytest.raises(ValueError, match="collides with reserved record key"):
        encode_runtime_event_record(record)


def test_decode_runtime_event_record_rejects_non_primitive_field_value() -> None:
    """A non-primitive field value at the top level raises during decode."""
    import pytest
    from platform_core.json_utils import JSONObject, JSONTypeError

    raw: JSONObject = {
        "timestamp": "2026-01-01T00:00:00",
        "level": "INFO",
        "logger": "t",
        "mode": "bot",
        "channel": "WIRE_COMPLETE",
        "message": "x",
        "nested": {"unexpected": "object"},
    }
    with pytest.raises(JSONTypeError, match="non-primitive type"):
        decode_runtime_event_record(raw)


def test_decode_runtime_event_record_handles_record_with_no_extra_fields() -> None:
    """A legacy record (no spread fields) decodes to an empty ``fields`` dict."""
    from platform_core.json_utils import JSONObject

    raw: JSONObject = {
        "timestamp": "2026-01-01T00:00:00",
        "level": "INFO",
        "logger": "t",
        "mode": "bot",
        "channel": "AI",
        "message": "decision",
    }

    decoded = decode_runtime_event_record(raw)

    assert decoded["fields"] == {}


def test_event_handler_skips_record_without_runtime_channel_or_message(
    fake_fs: FakeFileSystem,
) -> None:
    """A record carrying neither runtime_channel nor runtime_message is dropped.

    Covers the earliest guard in ``_HookEventArtifactHandler.emit``: when
    a stdlib LogRecord arrives without any runtime metadata at all, the
    JSONL handler must silently skip it -- writing to the events file
    would corrupt the JSONL with mode-less rows.
    """
    artifacts = configure_bot_runtime_logging("20260331-230405")

    from platform_core.logging import stdlib_logging

    logger = stdlib_logging.getLogger("tankpit_bot.runtime.no_runtime_extras")
    logger.info("plain log line with no runtime extras")

    files = fake_fs.get_written_files()
    assert files[artifacts["latest_events_path"]] == ""


def test_emit_without_runtime_configured_uses_unconfigured_mode(
    fake_fs: FakeFileSystem,
) -> None:
    """``_runtime_mode_name`` returns ``"unconfigured"`` when no mode is set.

    The autouse runtime-logging-state fixture resets every artifact
    holder to ``None`` at test start, so an emit_* before any
    ``configure_*_runtime_logging`` call exercises the unconfigured
    fallback. We assert against the LogRecord extra rather than the
    artifact file because no handler is attached yet.
    """
    from platform_core.logging import stdlib_logging

    records: list[stdlib_logging.LogRecord] = []

    class _RecordCapture(stdlib_logging.Handler):
        def emit(self, record: stdlib_logging.LogRecord) -> None:
            records.append(record)

    capture = _RecordCapture()
    capture.setLevel(stdlib_logging.INFO)
    root = stdlib_logging.getLogger()
    root.addHandler(capture)
    try:
        emit_ai("emitted before configure_bot_runtime_logging")
    finally:
        root.removeHandler(capture)

    from tankpit_bot.runtime_records import _RuntimeRecordMapping

    matching: list[_RuntimeRecordMapping] = []
    for record in records:
        rec_dict: _RuntimeRecordMapping = record.__dict__
        if "runtime_mode" in rec_dict:
            matching.append(rec_dict)
    if not matching:
        raise AssertionError("expected at least one record with runtime_mode extra")
    assert matching[0]["runtime_mode"] == "unconfigured"


def test_event_handler_skips_record_with_missing_runtime_fields_extra(
    fake_fs: FakeFileSystem,
) -> None:
    """A record carrying channel/message but no ``runtime_fields`` is dropped.

    Documents the strict contract on the JSONL artifact: every event line
    has spread fields (possibly empty). A LogRecord missing the
    ``runtime_fields`` extra signals a malformed call site, not a
    "default to empty" case.
    """
    artifacts = configure_bot_runtime_logging("20260331-230405")

    from platform_core.logging import stdlib_logging

    logger = stdlib_logging.getLogger("tankpit_bot.runtime.invalid_fields")
    logger.info(
        "missing runtime_fields",
        extra={"runtime_channel": "AI", "runtime_message": "no fields"},
    )

    files = fake_fs.get_written_files()
    assert files[artifacts["latest_events_path"]] == ""
