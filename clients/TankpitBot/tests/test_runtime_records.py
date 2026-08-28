"""Tests for the runtime record codec and per-tick context."""

from __future__ import annotations

from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
)

from tankpit_bot.runtime_logging import (
    configure_bot_runtime_logging,
    emit_ai,
    emit_diagnostic,
    emit_state,
)
from tankpit_bot.runtime_logging_handlers import (
    ARTIFACT_HANDLER_NAME_PREFIX,
    remove_artifact_handlers,
)
from tankpit_bot.sniffer.world_service import WorldService
from tests._runtime_logging_support import run_child_logger
from tests.conftest import FakeFileSystem


def test_event_handler_skips_record_with_malformed_runtime_fields_value(
    fake_fs: FakeFileSystem,
) -> None:
    """A record whose ``runtime_fields`` contains a non-primitive is dropped.

    Same robustness contract as the channel/message validation: malformed
    extras silently skip rather than producing a corrupt JSONL line.
    """
    artifacts = configure_bot_runtime_logging("20260331-230405")

    logger = run_child_logger("20260331-230405", "invalid_fields_value")
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

    logger = run_child_logger("20260331-230405", "invalid_fields_key")
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
        from tankpit_bot.runtime_records import require_int_field

        fields: dict[str, str | int | float | bool] = {"duration_ms": 850}
        assert require_int_field(fields, "duration_ms") == 850

    def test_require_int_field_raises_when_key_missing(self) -> None:
        """Missing key raises KeyError with the field name."""
        import pytest

        from tankpit_bot.runtime_records import require_int_field

        fields: dict[str, str | int | float | bool] = {}
        with pytest.raises(KeyError, match="duration_ms"):
            require_int_field(fields, "duration_ms")

    def test_require_int_field_rejects_string_value(self) -> None:
        """A string-valued field raises TypeError."""
        import pytest

        from tankpit_bot.runtime_records import require_int_field

        fields: dict[str, str | int | float | bool] = {"duration_ms": "850"}
        with pytest.raises(TypeError, match="must be int"):
            require_int_field(fields, "duration_ms")

    def test_require_int_field_rejects_bool_value(self) -> None:
        """A bool-valued field raises TypeError even though Python treats bool as int."""
        import pytest

        from tankpit_bot.runtime_records import require_int_field

        fields: dict[str, str | int | float | bool] = {"duration_ms": True}
        with pytest.raises(TypeError, match="must be int"):
            require_int_field(fields, "duration_ms")

    def test_require_str_field_returns_str_value(self) -> None:
        """A present str field is returned unchanged."""
        from tankpit_bot.runtime_records import require_str_field

        fields: dict[str, str | int | float | bool] = {"signal": "map_data_processed"}
        assert require_str_field(fields, "signal") == "map_data_processed"

    def test_require_str_field_raises_when_key_missing(self) -> None:
        """Missing key raises KeyError with the field name."""
        import pytest

        from tankpit_bot.runtime_records import require_str_field

        fields: dict[str, str | int | float | bool] = {}
        with pytest.raises(KeyError, match="signal"):
            require_str_field(fields, "signal")

    def test_require_str_field_rejects_int_value(self) -> None:
        """An int-valued field raises TypeError."""
        import pytest

        from tankpit_bot.runtime_records import require_str_field

        fields: dict[str, str | int | float | bool] = {"signal": 42}
        with pytest.raises(TypeError, match="must be str"):
            require_str_field(fields, "signal")

    def test_require_bool_field_returns_bool_value(self) -> None:
        """A present bool field is returned unchanged."""
        from tankpit_bot.runtime_records import require_bool_field

        fields: dict[str, str | int | float | bool] = {"uses_extra": True}
        assert require_bool_field(fields, "uses_extra") is True

    def test_require_bool_field_raises_when_key_missing(self) -> None:
        """Missing key raises KeyError with the field name."""
        import pytest

        from tankpit_bot.runtime_records import require_bool_field

        fields: dict[str, str | int | float | bool] = {}
        with pytest.raises(KeyError, match="uses_extra"):
            require_bool_field(fields, "uses_extra")

    def test_require_bool_field_rejects_int_value(self) -> None:
        """An int-valued field raises TypeError -- ints are not booleans."""
        import pytest

        from tankpit_bot.runtime_records import require_bool_field

        fields: dict[str, str | int | float | bool] = {"uses_extra": 1}
        with pytest.raises(TypeError, match="must be bool"):
            require_bool_field(fields, "uses_extra")


class TestRuntimeContext:
    """Tests for the per-tick context auto-attached to emit_* events."""

    def test_context_does_not_leak_into_another_thread(self) -> None:
        """A thread that never set the context reads an empty one.

        This is the whole reason the three fields are
        ``ContextVar`` slots rather than module globals
        ([[session-state-deglobalisation]] step 10). With globals, any
        thread emitting an event while a tick was in flight would stamp
        that tick's ``tick_n`` and ``bot_state`` onto its own records.
        """
        import threading

        from tankpit_bot.runtime_context import (
            RuntimeContextDict,
            get_runtime_context,
            set_runtime_context,
        )

        set_runtime_context(tick_n=7, bot_state="HUNT/engaging", in_flight_action_kind="shoot")

        seen: list[RuntimeContextDict] = []

        def _read_in_other_thread() -> None:
            seen.append(get_runtime_context())

        worker = threading.Thread(target=_read_in_other_thread)
        worker.start()
        worker.join()

        assert seen == [{}]
        # The setting thread still sees its own context untouched.
        assert get_runtime_context()["tick_n"] == 7

    def test_set_and_get_context_round_trips(self) -> None:
        """``set_runtime_context`` stores; ``get_runtime_context`` returns a copy."""
        from tankpit_bot.runtime_context import (
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
        from tankpit_bot.runtime_context import (
            get_runtime_context,
            set_runtime_context,
        )

        set_runtime_context(tick_n=1, bot_state="IDLE/", in_flight_action_kind="none")
        snapshot = get_runtime_context()
        snapshot["tick_n"] = 99
        assert get_runtime_context()["tick_n"] == 1

    def test_set_with_none_leaves_previous_value_intact(self) -> None:
        """Passing ``None`` for a field preserves the prior value."""
        from tankpit_bot.runtime_context import (
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
        from tankpit_bot.runtime_context import (
            clear_runtime_context,
            get_runtime_context,
            set_runtime_context,
        )

        set_runtime_context(tick_n=10, bot_state="IDLE/", in_flight_action_kind="none")
        clear_runtime_context()
        assert get_runtime_context() == {}

    def test_set_with_all_nones_is_a_noop(self) -> None:
        """``set_runtime_context()`` with every arg ``None`` changes nothing."""
        from tankpit_bot.runtime_context import (
            get_runtime_context,
            set_runtime_context,
        )

        set_runtime_context(tick_n=7)
        set_runtime_context()
        assert get_runtime_context() == {"tick_n": 7}

    def test_context_keys_constant_matches_typeddict(self) -> None:
        """``RUNTIME_CONTEXT_KEYS`` lists exactly the context field names."""
        from tankpit_bot.runtime_context import RUNTIME_CONTEXT_KEYS

        assert frozenset({"tick_n", "bot_state", "in_flight_action_kind"}) == RUNTIME_CONTEXT_KEYS

    def test_context_fields_attached_to_emit_ai(self, fake_fs: FakeFileSystem) -> None:
        """``emit_ai`` events carry the active context fields."""
        from tankpit_bot.runtime_context import set_runtime_context

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
        from tankpit_bot.runtime_context import set_runtime_context

        artifacts = configure_bot_runtime_logging("20260620-150138")
        set_runtime_context(tick_n=12)
        emit_diagnostic(diagnostic_kind="fake_override", tick_n=999)

        event_line = fake_fs.get_written_files()[artifacts["latest_events_path"]].strip()
        decoded = narrow_json_to_dict(load_json_str(event_line))
        assert decoded["tick_n"] == 999

    def test_unset_context_fields_do_not_appear(self, fake_fs: FakeFileSystem) -> None:
        """Fields never set in the context are absent from emitted events."""
        from tankpit_bot.runtime_context import set_runtime_context

        artifacts = configure_bot_runtime_logging("20260620-150138")
        set_runtime_context(tick_n=3)  # bot_state / in_flight_action_kind not set
        emit_state("IDLE")

        event_line = fake_fs.get_written_files()[artifacts["latest_events_path"]].strip()
        decoded = narrow_json_to_dict(load_json_str(event_line))
        assert decoded["tick_n"] == 3
        assert "bot_state" not in decoded
        assert "in_flight_action_kind" not in decoded

    def test_context_is_attached_to_action_outcome_events(self, fake_fs: FakeFileSystem) -> None:
        """Ledger outcome events also pick up the runtime context.

        This is the highest-value attachment: the post-mortem JSONL
        query "which tick fired the stall_timeout?" works because every
        ``action_outcome`` event carries ``tick_n``.
        """
        from tankpit_bot.ledger.outcome.shoot import emit_shoot_miss
        from tankpit_bot.runtime_context import set_runtime_context

        ws = WorldService()
        artifacts = configure_bot_runtime_logging("20260620-150138")
        set_runtime_context(
            tick_n=42,
            bot_state="HUNT/engaging",
            in_flight_action_kind="shoot",
        )
        emit_shoot_miss(ws.ledger, duration_ms=80, target_id=530, target_name="orange-3")

        event_line = fake_fs.get_written_files()[artifacts["latest_events_path"]].strip()
        decoded = narrow_json_to_dict(load_json_str(event_line))
        assert decoded["channel"] == "DIAGNOSTIC"
        assert decoded["diagnostic_kind"] == "action_outcome"
        assert decoded["tick_n"] == 42
        assert decoded["bot_state"] == "HUNT/engaging"
        assert decoded["in_flight_action_kind"] == "shoot"


def test_remove_artifact_handlers_keeps_non_artifact_handlers() -> None:
    """Artifact handler cleanup removes only handlers owned by runtime logging."""
    from platform_core.logging import stdlib_logging

    root = stdlib_logging.getLogger()
    original_handlers = list(root.handlers)
    runtime_handler = stdlib_logging.NullHandler()
    runtime_handler.set_name(ARTIFACT_HANDLER_NAME_PREFIX + "test")
    normal_handler = stdlib_logging.NullHandler()
    normal_handler.set_name("normal")
    root.handlers = [runtime_handler, normal_handler]

    remove_artifact_handlers(root)

    assert root.handlers == [normal_handler]
    root.handlers = original_handlers
