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
from tankpit_bot.sniffer.world_service import WorldService
from tests._runtime_logging_support import run_child_logger
from tests.conftest import FakeFileSystem


def test_event_handler_skips_record_whose_channel_is_not_a_string(
    fake_fs: FakeFileSystem,
) -> None:
    """A ``runtime_channel`` that is present but not a string is dropped.

    The sibling checks either side of this one are already pinned: a
    record missing the keys entirely, and a record whose
    ``runtime_fields`` extra is absent. Neither reaches this arm, which
    fires only when both keys EXIST and one of them is the wrong type --
    the shape a mis-typed call site produces.

    ``runtime_fields`` is supplied here deliberately. Without it the
    record would be caught by the later fields check instead, and the
    test would pass whether or not this arm existed.

    The artifact must stay empty. Nothing downstream re-validates the
    channel: :func:`encode_runtime_event_record` copies it straight into
    the JSON object, so a non-string channel would be written verbatim
    and every reader that groups by channel would see a phantom stream.
    """
    artifacts = configure_bot_runtime_logging("20260331-230405")

    logger = run_child_logger("20260331-230405", "non_string_channel")
    logger.info(
        "channel arrived as an int",
        extra={"runtime_channel": 7, "runtime_message": "m", "runtime_fields": {}},
    )

    files = fake_fs.get_written_files()
    assert files[artifacts["latest_events_path"]] == ""


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

    logger = run_child_logger("20260331-230405", "invalid")
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

    ws = WorldService()
    artifacts = configure_bot_runtime_logging("20260331-230405")

    emit_map_open_data_processed(ws.ledger, duration_ms=850)

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


def test_emit_rejects_reserved_field_names_at_the_call() -> None:
    """A reserved kwarg raises at emit time, handler or no handler.

    The encoder-level check only runs with the JSONL handler attached,
    which unit tests never do — so a fully covered ``level=`` emit
    shipped and crashed BOTH fleet bots on the first live 0x4E
    decoration announcement (2026-08-26 05:11:17). Validation now
    happens at the call, so coverage of an emit line proves its field
    names are legal.
    """
    import pytest

    with pytest.raises(ValueError, match="'level' collides with reserved record key"):
        emit_diagnostic(diagnostic_kind="test_kind", level=3)


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

    logger = run_child_logger("20260331-230405", "no_runtime_extras")
    logger.info("plain log line with no runtime extras")

    files = fake_fs.get_written_files()
    assert files[artifacts["latest_events_path"]] == ""


def test_emit_without_a_configured_run_still_reaches_the_console(
    fake_fs: FakeFileSystem,
) -> None:
    """An unconfigured process logs the event but writes no artifact.

    Before any ``configure_*_runtime_logging`` call there is no ambient
    run, so ``emit_ai`` resolves the base emitter logger — which carries
    no event handler. The record must still propagate to the root logger
    (console and, once configured, the process text log); what it must
    NOT do is invent an events.jsonl row for a run that does not exist.
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

    emitted: list[_RuntimeRecordMapping] = []
    for record in records:
        rec_dict: _RuntimeRecordMapping = record.__dict__
        if "runtime_channel" in rec_dict:
            emitted.append(rec_dict)
    if not emitted:
        raise AssertionError("expected the emit to reach the root logger")
    assert emitted[0]["runtime_channel"] == "AI"
    assert emitted[0]["runtime_message"] == "emitted before configure_bot_runtime_logging"
    event_artifacts = [
        path for path in fake_fs.get_written_files() if path.endswith(".events.jsonl")
    ]
    assert event_artifacts == []


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

    logger = run_child_logger("20260331-230405", "invalid_fields")
    logger.info(
        "missing runtime_fields",
        extra={"runtime_channel": "AI", "runtime_message": "no fields"},
    )

    files = fake_fs.get_written_files()
    assert files[artifacts["latest_events_path"]] == ""


def test_two_concurrent_runs_keep_their_own_event_streams(
    fake_fs: FakeFileSystem,
) -> None:
    """A second session must not steal the first session's event stream.

    This is the defect [[session-state-deglobalisation]] step 10 names:
    ``_install_artifact_handlers`` used to mount on the ROOT logger and
    remove any prior artifact handlers first, so configuring a second run
    in one process silently detached the first run's event handler and
    its ``events.jsonl`` stopped growing mid-session.

    The threads are stepped through a barrier so the ordering is the one
    that used to break: run A configures, run B configures, and only THEN
    does run A emit. Each thread gets its own context, which is what
    makes the ambient run per-session rather than per-process.
    """
    import threading

    a_configured = threading.Event()
    b_configured = threading.Event()
    a_emitted = threading.Event()
    paths: dict[str, str] = {}
    failures: list[str] = []

    def run_a() -> None:
        artifacts = configure_bot_runtime_logging("20260331-100000")
        paths["a"] = artifacts["latest_events_path"]
        a_configured.set()
        if not b_configured.wait(timeout=5):
            failures.append("run B never configured")
            return
        emit_ai("from run A")
        a_emitted.set()

    def run_b() -> None:
        if not a_configured.wait(timeout=5):
            failures.append("run A never configured")
            return
        artifacts = configure_probe_runtime_logging("fuel", "20260331-200000")
        paths["b"] = artifacts["latest_events_path"]
        b_configured.set()
        if not a_emitted.wait(timeout=5):
            failures.append("run A never emitted")
            return
        emit_ai("from run B")

    thread_a = threading.Thread(target=run_a)
    thread_b = threading.Thread(target=run_b)
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=10)
    thread_b.join(timeout=10)

    assert failures == []
    files = fake_fs.get_written_files()
    a_events = files[paths["a"]]
    b_events = files[paths["b"]]
    assert "from run A" in a_events
    assert "from run B" not in a_events
    assert "from run B" in b_events
    assert "from run A" not in b_events
    # Each stream carries its own mode, taken from its own handler.
    assert '"mode":"bot"' in a_events
    assert '"mode":"probe:fuel"' in b_events


def test_reconfiguring_the_same_run_does_not_double_its_events(
    fake_fs: FakeFileSystem,
) -> None:
    """Two configures on one stamp leave one event handler, not two.

    The run's logger is keyed by mode and stamp, so a deterministic
    stamp -- every test in this file, and a service session restarting
    within the same second -- resolves the same logger twice. Without
    clearing that logger's handlers first, the second configure would
    stack a second handler and write every subsequent event twice.
    """
    configure_bot_runtime_logging("20260331-230405")
    artifacts = configure_bot_runtime_logging("20260331-230405")

    emit_ai("emitted once")

    files = fake_fs.get_written_files()
    lines = files[artifacts["latest_events_path"]].strip().splitlines()
    assert len(lines) == 1


def test_clearing_runtime_logging_state_forgets_a_probe_run(
    fake_fs: FakeFileSystem,
) -> None:
    """Clearing forgets the probe run, which the old reset silently skipped.

    Until step 10 the autouse fixture reset the bot and sniff globals and
    never touched ``_PROBE_ARTIFACTS``, so a probe test leaked its
    artifacts into every test that followed it on the same xdist worker.
    """
    from tankpit_bot.runtime_logging import clear_runtime_logging_state

    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    assert get_probe_runtime_artifacts() == artifacts

    clear_runtime_logging_state()

    assert get_probe_runtime_artifacts() is None
    assert get_bot_runtime_artifacts() is None
    assert get_sniff_runtime_artifacts() is None
