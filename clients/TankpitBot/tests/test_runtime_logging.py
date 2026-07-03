"""Tests for runtime logging artifact mirroring and structured events."""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot.runtime_logging import (
    _ARTIFACT_HANDLER_NAME_PREFIX,
    RuntimeEventRecordDict,
    _remove_artifact_handlers,
    configure_bot_runtime_logging,
    configure_probe_runtime_logging,
    configure_sniff_runtime_logging,
    decode_runtime_event_record,
    emit_ai,
    emit_diagnostic,
    emit_state,
    emit_sync,
    emit_wire,
    emit_wire_complete,
    emit_world,
    encode_runtime_event_record,
    get_bot_runtime_artifacts,
    get_probe_runtime_artifacts,
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


def test_emit_wire_complete_writes_structured_fields_to_jsonl(
    fake_fs: FakeFileSystem,
) -> None:
    """``emit_wire_complete`` spreads action_kind/duration_ms/signal into JSONL."""
    from platform_core.json_utils import load_json_str, narrow_json_to_dict

    artifacts = configure_bot_runtime_logging("20260331-230405")

    emit_wire_complete(
        action_kind="map_open",
        duration_ms=850,
        signal="map_data_processed",
        target_x=131,
        target_y=110,
    )

    files = fake_fs.get_written_files()
    latest_log = files[artifacts["latest_log_path"]]
    event_line = files[artifacts["latest_events_path"]].strip()

    assert "WIRE_COMPLETE: map_open completed in 850ms via map_data_processed" in latest_log

    decoded_raw = narrow_json_to_dict(load_json_str(event_line))
    assert decoded_raw["channel"] == "WIRE_COMPLETE"
    assert decoded_raw["action_kind"] == "map_open"
    assert decoded_raw["duration_ms"] == 850
    assert decoded_raw["signal"] == "map_data_processed"
    assert decoded_raw["target_x"] == 131
    assert decoded_raw["target_y"] == 110

    decoded = decode_runtime_event_record(decoded_raw)
    assert decoded["fields"] == {
        "action_kind": "map_open",
        "duration_ms": 850,
        "signal": "map_data_processed",
        "target_x": 131,
        "target_y": 110,
    }


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

    from tankpit_bot.runtime_logging import _RuntimeRecordMapping

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


def test_event_handler_skips_record_with_malformed_runtime_fields_value(
    fake_fs: FakeFileSystem,
) -> None:
    """A record whose ``runtime_fields`` contains a non-primitive is dropped.

    Same robustness contract as the channel/message validation: malformed
    extras silently skip rather than producing a corrupt JSONL line.
    """
    artifacts = configure_bot_runtime_logging("20260331-230405")

    from platform_core.logging import stdlib_logging

    logger = stdlib_logging.getLogger("tankpit_bot.runtime.invalid_fields_value")
    logger.info(
        "bad value",
        extra={
            "runtime_channel": "AI",
            "runtime_message": "bad",
            "runtime_fields": {"target": [1, 2, 3]},
        },
    )

    files = fake_fs.get_written_files()
    assert files[artifacts["latest_events_path"]] == ""


def test_event_handler_skips_record_with_non_string_key_in_runtime_fields(
    fake_fs: FakeFileSystem,
) -> None:
    """A non-string field key (e.g. int) is rejected at the handler boundary."""
    artifacts = configure_bot_runtime_logging("20260331-230405")

    from platform_core.logging import stdlib_logging

    logger = stdlib_logging.getLogger("tankpit_bot.runtime.invalid_fields_key")
    logger.info(
        "bad key",
        extra={
            "runtime_channel": "AI",
            "runtime_message": "bad",
            "runtime_fields": {42: "answer"},
        },
    )

    files = fake_fs.get_written_files()
    assert files[artifacts["latest_events_path"]] == ""


class TestRequireFieldAccessors:
    """Strict-typed extractors for ``RuntimeEventRecordDict.fields`` values."""

    def test_require_int_field_returns_int_value(self) -> None:
        """A present int field is returned unchanged."""
        from tankpit_bot.runtime_logging import require_int_field

        fields: dict[str, str | int | float | bool] = {"duration_ms": 850}
        assert require_int_field(fields, "duration_ms") == 850

    def test_require_int_field_raises_when_key_missing(self) -> None:
        """Missing key raises KeyError with the field name."""
        import pytest

        from tankpit_bot.runtime_logging import require_int_field

        fields: dict[str, str | int | float | bool] = {}
        with pytest.raises(KeyError, match="duration_ms"):
            require_int_field(fields, "duration_ms")

    def test_require_int_field_rejects_string_value(self) -> None:
        """A string-valued field raises TypeError."""
        import pytest

        from tankpit_bot.runtime_logging import require_int_field

        fields: dict[str, str | int | float | bool] = {"duration_ms": "850"}
        with pytest.raises(TypeError, match="must be int"):
            require_int_field(fields, "duration_ms")

    def test_require_int_field_rejects_bool_value(self) -> None:
        """A bool-valued field raises TypeError even though Python treats bool as int."""
        import pytest

        from tankpit_bot.runtime_logging import require_int_field

        fields: dict[str, str | int | float | bool] = {"duration_ms": True}
        with pytest.raises(TypeError, match="must be int"):
            require_int_field(fields, "duration_ms")

    def test_require_str_field_returns_str_value(self) -> None:
        """A present str field is returned unchanged."""
        from tankpit_bot.runtime_logging import require_str_field

        fields: dict[str, str | int | float | bool] = {"signal": "map_data_processed"}
        assert require_str_field(fields, "signal") == "map_data_processed"

    def test_require_str_field_raises_when_key_missing(self) -> None:
        """Missing key raises KeyError with the field name."""
        import pytest

        from tankpit_bot.runtime_logging import require_str_field

        fields: dict[str, str | int | float | bool] = {}
        with pytest.raises(KeyError, match="signal"):
            require_str_field(fields, "signal")

    def test_require_str_field_rejects_int_value(self) -> None:
        """An int-valued field raises TypeError."""
        import pytest

        from tankpit_bot.runtime_logging import require_str_field

        fields: dict[str, str | int | float | bool] = {"signal": 42}
        with pytest.raises(TypeError, match="must be str"):
            require_str_field(fields, "signal")

    def test_require_bool_field_returns_bool_value(self) -> None:
        """A present bool field is returned unchanged."""
        from tankpit_bot.runtime_logging import require_bool_field

        fields: dict[str, str | int | float | bool] = {"uses_extra": True}
        assert require_bool_field(fields, "uses_extra") is True

    def test_require_bool_field_raises_when_key_missing(self) -> None:
        """Missing key raises KeyError with the field name."""
        import pytest

        from tankpit_bot.runtime_logging import require_bool_field

        fields: dict[str, str | int | float | bool] = {}
        with pytest.raises(KeyError, match="uses_extra"):
            require_bool_field(fields, "uses_extra")

    def test_require_bool_field_rejects_int_value(self) -> None:
        """An int-valued field raises TypeError -- ints are not booleans."""
        import pytest

        from tankpit_bot.runtime_logging import require_bool_field

        fields: dict[str, str | int | float | bool] = {"uses_extra": 1}
        with pytest.raises(TypeError, match="must be bool"):
            require_bool_field(fields, "uses_extra")


class TestRuntimeContext:
    """Tests for the per-tick context auto-attached to emit_* events."""

    def test_set_and_get_context_round_trips(self) -> None:
        """``set_runtime_context`` stores; ``get_runtime_context`` returns a copy."""
        from tankpit_bot.runtime_logging import (
            get_runtime_context,
            set_runtime_context,
        )

        set_runtime_context(tick_n=42, bot_state="HUNT/engaging", in_flight_action_kind="shoot")
        ctx = get_runtime_context()
        assert ctx == {
            "tick_n": 42,
            "bot_state": "HUNT/engaging",
            "in_flight_action_kind": "shoot",
        }

    def test_get_returns_independent_copy(self) -> None:
        """Mutating the returned dict must not affect the module cache."""
        from tankpit_bot.runtime_logging import (
            get_runtime_context,
            set_runtime_context,
        )

        set_runtime_context(tick_n=1, bot_state="IDLE/", in_flight_action_kind="none")
        snapshot = get_runtime_context()
        snapshot["tick_n"] = 99
        assert get_runtime_context()["tick_n"] == 1

    def test_set_with_none_leaves_previous_value_intact(self) -> None:
        """Passing ``None`` for a field preserves the prior value."""
        from tankpit_bot.runtime_logging import (
            get_runtime_context,
            set_runtime_context,
        )

        set_runtime_context(tick_n=5, bot_state="HUNT/searching", in_flight_action_kind="scan")
        set_runtime_context(tick_n=6)  # only update tick_n
        ctx = get_runtime_context()
        assert ctx["tick_n"] == 6
        assert ctx["bot_state"] == "HUNT/searching"
        assert ctx["in_flight_action_kind"] == "scan"

    def test_clear_removes_every_field(self) -> None:
        """``clear_runtime_context`` empties the cache."""
        from tankpit_bot.runtime_logging import (
            clear_runtime_context,
            get_runtime_context,
            set_runtime_context,
        )

        set_runtime_context(tick_n=10, bot_state="IDLE/", in_flight_action_kind="none")
        clear_runtime_context()
        assert get_runtime_context() == {}

    def test_set_with_all_nones_is_a_noop(self) -> None:
        """``set_runtime_context()`` with every arg ``None`` changes nothing."""
        from tankpit_bot.runtime_logging import get_runtime_context, set_runtime_context

        set_runtime_context(tick_n=7)
        set_runtime_context()
        assert get_runtime_context() == {"tick_n": 7}

    def test_context_keys_constant_matches_typeddict(self) -> None:
        """``RUNTIME_CONTEXT_KEYS`` lists exactly the context field names."""
        from tankpit_bot.runtime_logging import RUNTIME_CONTEXT_KEYS

        assert frozenset({"tick_n", "bot_state", "in_flight_action_kind"}) == RUNTIME_CONTEXT_KEYS

    def test_context_fields_attached_to_emit_ai(self, fake_fs: FakeFileSystem) -> None:
        """``emit_ai`` events carry the active context fields."""
        from tankpit_bot.runtime_logging import set_runtime_context

        artifacts = configure_bot_runtime_logging("20260620-150138")
        set_runtime_context(
            tick_n=12,
            bot_state="HUNT/engaging",
            in_flight_action_kind="shoot",
        )
        emit_ai("HUNT score=0.5")

        event_line = fake_fs.get_written_files()[artifacts["latest_events_path"]].strip()
        decoded = narrow_json_to_dict(load_json_str(event_line))
        assert decoded["channel"] == "AI"
        assert decoded["message"] == "HUNT score=0.5"
        assert decoded["tick_n"] == 12
        assert decoded["bot_state"] == "HUNT/engaging"
        assert decoded["in_flight_action_kind"] == "shoot"

    def test_explicit_fields_override_context_fields_on_collision(
        self, fake_fs: FakeFileSystem
    ) -> None:
        """An explicit ``tick_n=`` arg wins over the context's ``tick_n``."""
        from tankpit_bot.runtime_logging import set_runtime_context

        artifacts = configure_bot_runtime_logging("20260620-150138")
        set_runtime_context(tick_n=12)
        emit_diagnostic(diagnostic_kind="explicit_override", tick_n=999)

        event_line = fake_fs.get_written_files()[artifacts["latest_events_path"]].strip()
        decoded = narrow_json_to_dict(load_json_str(event_line))
        assert decoded["tick_n"] == 999

    def test_unset_context_fields_do_not_appear(self, fake_fs: FakeFileSystem) -> None:
        """Fields never set in the context are absent from emitted events."""
        from tankpit_bot.runtime_logging import set_runtime_context

        artifacts = configure_bot_runtime_logging("20260620-150138")
        set_runtime_context(tick_n=3)  # bot_state / in_flight_action_kind not set
        emit_state("IDLE")

        event_line = fake_fs.get_written_files()[artifacts["latest_events_path"]].strip()
        decoded = narrow_json_to_dict(load_json_str(event_line))
        assert decoded["tick_n"] == 3
        assert "bot_state" not in decoded
        assert "in_flight_action_kind" not in decoded

    def test_context_is_attached_to_wire_complete_events(self, fake_fs: FakeFileSystem) -> None:
        """``emit_wire_complete`` also picks up the context.

        This is the highest-value attachment: the post-mortem JSONL
        query "which tick fired the stall_timeout?" works because every
        WIRE_COMPLETE event carries ``tick_n``.
        """
        from tankpit_bot.runtime_logging import set_runtime_context

        artifacts = configure_bot_runtime_logging("20260620-150138")
        set_runtime_context(
            tick_n=42,
            bot_state="HUNT/engaging",
            in_flight_action_kind="shoot",
        )
        emit_wire_complete(
            action_kind="shoot",
            duration_ms=80,
            signal="our_shot_response",
        )

        event_line = fake_fs.get_written_files()[artifacts["latest_events_path"]].strip()
        decoded = narrow_json_to_dict(load_json_str(event_line))
        assert decoded["channel"] == "WIRE_COMPLETE"
        assert decoded["tick_n"] == 42
        assert decoded["bot_state"] == "HUNT/engaging"
        assert decoded["in_flight_action_kind"] == "shoot"


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
