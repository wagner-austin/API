"""Tests for runtime logging artifact mirroring and structured events."""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot.runtime_logging import (
    _ARTIFACT_HANDLER_NAME_PREFIX,
    _remove_artifact_handlers,
    configure_bot_runtime_logging,
    configure_sniff_runtime_logging,
    decode_runtime_event_record,
    emit_ai,
    emit_state,
    emit_sync,
    emit_wire,
    emit_world,
    get_bot_runtime_artifacts,
    get_sniff_runtime_artifacts,
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

    artifacts = configure_sniff_runtime_logging("20260331-230405")
    emit_world("Captured %d WebSocket messages in %.1fs", 88, 37.3)

    files = fake_fs.get_written_files()
    assert "stale" not in files[artifacts["latest_log_path"]]
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

    sniff_artifacts = configure_sniff_runtime_logging("20260331-230406")

    assert get_bot_runtime_artifacts() is None
    assert get_sniff_runtime_artifacts() == sniff_artifacts


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


def test_remove_artifact_handlers_keeps_non_artifact_handlers() -> None:
    """Artifact handler cleanup removes only handlers owned by runtime logging."""
    from platform_core.logging import stdlib_logging

    root = stdlib_logging.getLogger()
    original_handlers = list(root.handlers)
    runtime_handler = stdlib_logging.NullHandler()
    runtime_handler.set_name(_ARTIFACT_HANDLER_NAME_PREFIX + "test")
    normal_handler = stdlib_logging.NullHandler()
    normal_handler.set_name("normal")
    root.handlers = [runtime_handler, normal_handler]

    _remove_artifact_handlers(root)

    assert root.handlers == [normal_handler]
    root.handlers = original_handlers
