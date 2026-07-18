"""Tests for the bot-query CLI.

File I/O goes through ``_test_hooks`` with save-and-restore. Each
query is exercised on a deterministic synthetic JSONL corpus.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import (
    AppendTextProtocol,
    PathExistsProtocol,
    ReadTextProtocol,
)
from tankpit_bot.diagnostics import bot_query
from tankpit_bot.diagnostics.bot_query import (
    DEFAULT_EVENTS_PATH,
    BotQueryRecord,
    _decode_record,
    load_records,
    query_action_spans,
    query_stalls,
    query_target_decisions,
    query_timeline,
    run,
)


class _FakeFileSystem:
    """Save-and-restore fake for the bot_query file hooks."""

    def __init__(self) -> None:
        """Initialise with no virtual files registered."""
        self._files: dict[str, str] = {}

    def write(self, path: Path, content: str) -> None:
        """Register a virtual file's contents."""
        self._files[str(path)] = content

    def path_exists(self, path: Path) -> bool:
        """Return True when ``path`` was registered."""
        return str(path) in self._files

    def read_text(self, path: Path) -> str:
        """Return the contents of ``path``."""
        return self._files[str(path)]

    def append_text(self, path: Path, content: str) -> None:
        """Append ``content`` to the virtual file at ``path``."""
        existing = self._files.get(str(path), "")
        self._files[str(path)] = existing + content


def _install_fake_filesystem() -> tuple[
    _FakeFileSystem, PathExistsProtocol, ReadTextProtocol, AppendTextProtocol
]:
    """Swap the script hooks for a fake; return originals for restore."""
    fake = _FakeFileSystem()
    original_path_exists: PathExistsProtocol = _test_hooks.path_exists
    original_read_text: ReadTextProtocol = _test_hooks.read_text
    original_append_text: AppendTextProtocol = _test_hooks.append_text
    _test_hooks.path_exists = fake.path_exists
    _test_hooks.read_text = fake.read_text
    _test_hooks.append_text = fake.append_text
    return (fake, original_path_exists, original_read_text, original_append_text)


def _event(
    channel: str,
    message: str,
    *,
    timestamp: str = "2026-06-20T15:00:00",
    **fields: JSONValue,
) -> str:
    """Build one JSONL line for a record with the given channel/message/fields."""
    payload: dict[str, JSONValue] = {
        "timestamp": timestamp,
        "level": "INFO",
        "logger": "tankpit_bot.runtime.events",
        "mode": "bot",
        "channel": channel,
        "message": message,
    }
    payload.update(fields)
    return dump_json_str(payload) + "\n"


class _RecordingWriter:
    """Collects written text into a buffer; mirrors ``sys.stdout.write``."""

    def __init__(self) -> None:
        """Initialise with an empty buffer."""
        self._buffer: list[str] = []

    def __call__(self, text: str) -> int:
        """Append ``text`` to the buffer; return its length."""
        self._buffer.append(text)
        return len(text)

    @property
    def text(self) -> str:
        """Return the concatenated written text."""
        return "".join(self._buffer)


def _record(
    channel: str,
    message: str,
    *,
    timestamp: str = "2026-06-20T15:00:00",
    **fields: JSONValue,
) -> BotQueryRecord:
    """Build a BotQueryRecord directly (without round-tripping through JSON)."""
    return BotQueryRecord(
        timestamp=timestamp,
        channel=channel,
        message=message,
        fields=dict(fields),
    )


class TestDecodeRecord:
    """Tests for the per-record decoder boundary."""

    def test_strips_reserved_keys_from_fields(self) -> None:
        """Reserved record-level keys are stripped from the fields view."""
        parsed: dict[str, JSONValue] = {
            "timestamp": "2026-06-20T15:00:00",
            "level": "INFO",
            "logger": "tankpit_bot.runtime.events",
            "mode": "bot",
            "channel": "AI",
            "message": "HUNT score=0.5",
            "tick_n": 7,
            "combat_target_x": 131,
        }
        rec = _decode_record(parsed)
        assert rec.channel == "AI"
        assert rec.fields == {"tick_n": 7, "combat_target_x": 131}

    def test_raises_when_channel_missing(self) -> None:
        """A record without a channel raises ``JSONTypeError``."""
        parsed: dict[str, JSONValue] = {
            "timestamp": "2026-06-20T15:00:00",
            "level": "INFO",
            "logger": "tankpit_bot.runtime.events",
            "mode": "bot",
            "message": "no channel",
        }
        with pytest.raises(JSONTypeError):
            _decode_record(parsed)


class TestQueryTimeline:
    """Tests for the timeline query."""

    def test_emits_one_line_per_target_channel(self) -> None:
        """STATE / WIRE / DIAGNOSTIC events surface; AI is skipped."""
        writer = _RecordingWriter()
        records = [
            _record("STATE", "IDLE", tick_n=1),
            _record("AI", "HUNT score=0", tick_n=2),  # not in timeline channels
            _record("WIRE", "WIRE: shoot_at", tick_n=3),
            _record("DIAGNOSTIC", "diagnostic_kind=map_data_snapshot", tick_n=5),
        ]
        query_timeline(records, writer)
        lines = writer.text.splitlines()
        assert len(lines) == 3
        assert "STATE\tIDLE" in lines[0]
        assert "WIRE\tWIRE: shoot_at" in lines[1]
        assert "DIAGNOSTIC\tdiagnostic_kind=map_data_snapshot" in lines[2]

    def test_renders_dash_when_no_tick_n(self) -> None:
        """Records without ``tick_n`` print ``tick=-``."""
        writer = _RecordingWriter()
        query_timeline([_record("STATE", "IDLE")], writer)
        assert "tick=-" in writer.text


class TestQueryStalls:
    """Tests for the stall_timeout query."""

    def test_lists_only_stall_timeout_events(self) -> None:
        """Other outcomes (and other channels) are skipped."""
        writer = _RecordingWriter()
        records = [
            _record(
                "DIAGNOSTIC",
                "map_open resolved",
                tick_n=1,
                diagnostic_kind="action_outcome",
                outcome="map_data_processed",
                action_kind="map_open",
                duration_ms=250,
                bot_state="HUNT/searching",
            ),
            _record(
                "DIAGNOSTIC",
                "move stalled",
                tick_n=2,
                diagnostic_kind="action_outcome",
                outcome="stall_timeout",
                action_kind="move",
                duration_ms=10000,
                bot_state="HUNT/engaging",
            ),
            _record(
                "AI",
                "stall_timeout",  # wrong channel
                outcome="stall_timeout",
            ),
        ]
        query_stalls(records, writer)
        lines = writer.text.splitlines()
        assert len(lines) == 1
        assert "action=move" in lines[0]
        assert "duration_ms=10000" in lines[0]
        assert "state=HUNT/engaging" in lines[0]

    def test_renders_dashes_when_fields_missing(self) -> None:
        """Missing tick_n / action_kind / duration_ms / bot_state print ``-``."""
        writer = _RecordingWriter()
        records = [
            _record(
                "DIAGNOSTIC",
                "minimal stall",
                diagnostic_kind="action_outcome",
                outcome="stall_timeout",
            )
        ]
        query_stalls(records, writer)
        out = writer.text
        assert "tick=-" in out
        assert "action=-" in out
        assert "duration_ms=-" in out
        assert "state=-" in out


class TestQueryActionSpans:
    """Tests for the action-spans pair-up query."""

    def test_pairs_dispatch_with_completion(self) -> None:
        """A WIRE dispatch followed by a matching outcome prints one line."""
        writer = _RecordingWriter()
        records = [
            _record(
                "WIRE",
                "WIRE: shoot_at (131,124)",
                timestamp="2026-06-20T15:00:00",
                action_kind="shoot",
            ),
            _record(
                "DIAGNOSTIC",
                "shoot resolved",
                timestamp="2026-06-20T15:00:00",
                diagnostic_kind="action_outcome",
                action_kind="shoot",
                duration_ms=80,
                outcome="miss",
            ),
        ]
        query_action_spans(records, writer)
        lines = writer.text.splitlines()
        assert len(lines) == 1
        assert "action=shoot" in lines[0]
        assert "outcome=miss" in lines[0]
        assert "duration_ms=80" in lines[0]

    def test_orphan_complete_without_matching_dispatch(self) -> None:
        """An outcome without a prior WIRE prints an orphan line."""
        writer = _RecordingWriter()
        records = [
            _record(
                "DIAGNOSTIC",
                "shoot resolved",
                diagnostic_kind="action_outcome",
                action_kind="shoot",
                duration_ms=80,
                outcome="miss",
            ),
        ]
        query_action_spans(records, writer)
        out = writer.text
        assert "(orphan)" in out

    def test_skips_wire_without_action_kind(self) -> None:
        """WIRE events without ``action_kind`` do not open a span.

        Documents the contract: outcome events without
        ``action_kind`` also have no opening span and are skipped.
        """
        writer = _RecordingWriter()
        records = [
            _record("WIRE", "wire ping"),
            _record("DIAGNOSTIC", "outcome ping", diagnostic_kind="action_outcome"),
        ]
        query_action_spans(records, writer)
        # Both lacked action_kind so no output.
        assert writer.text == ""

    def test_skips_records_of_unrelated_channels(self) -> None:
        """STATE / AI / DIAGNOSTIC records are skipped (do not affect spans).

        Documents the contract: action-span pairing only sees WIRE
        dispatches and ``action_outcome`` resolutions; any other
        record interleaved between a dispatch and its resolution is
        invisible to the pairing logic.
        """
        writer = _RecordingWriter()
        records = [
            _record(
                "WIRE",
                "dispatch",
                timestamp="2026-06-20T15:00:00",
                action_kind="move",
            ),
            _record("STATE", "IDLE", timestamp="2026-06-20T15:00:01"),
            _record("AI", "decided IDLE", timestamp="2026-06-20T15:00:02"),
            _record(
                "DIAGNOSTIC",
                "move resolved",
                timestamp="2026-06-20T15:00:03",
                diagnostic_kind="action_outcome",
                action_kind="move",
                duration_ms=3000,
                outcome="position_reached",
            ),
        ]
        query_action_spans(records, writer)
        lines = writer.text.splitlines()
        assert len(lines) == 1
        assert "action=move" in lines[0]
        assert "STATE" not in writer.text
        assert "IDLE" not in writer.text

    def test_second_dispatch_replaces_open_span(self) -> None:
        """Two WIRE dispatches of the same kind keep only the second.

        Documents the contract: a duplicate dispatch (mid-action
        re-issuing the same command) overrides the prior open span. The
        first dispatch's pairing is lost rather than producing
        duplicate orphan rows.
        """
        writer = _RecordingWriter()
        records = [
            _record(
                "WIRE",
                "first dispatch",
                timestamp="2026-06-20T15:00:00",
                action_kind="move",
            ),
            _record(
                "WIRE",
                "second dispatch",
                timestamp="2026-06-20T15:00:01",
                action_kind="move",
            ),
            _record(
                "DIAGNOSTIC",
                "move resolved",
                timestamp="2026-06-20T15:00:02",
                diagnostic_kind="action_outcome",
                action_kind="move",
                duration_ms=2000,
                outcome="position_reached",
            ),
        ]
        query_action_spans(records, writer)
        # One paired line, opener = second dispatch (15:00:01).
        lines = writer.text.splitlines()
        assert len(lines) == 1
        assert "2026-06-20T15:00:01\t->\t2026-06-20T15:00:02" in lines[0]


class TestQueryTargetDecisions:
    """Tests for the HUNT-target-decisions query."""

    def test_lists_hunt_score_events(self) -> None:
        """HUNT score events with a non-zero target print a coord line."""
        writer = _RecordingWriter()
        records = [
            _record(
                "AI",
                "HUNT score=0.8 target=(131,124)",
                tick_n=7,
                combat_target_x=131,
                combat_target_y=124,
            ),
            _record(
                "AI",
                "decided IDLE",  # skipped
            ),
        ]
        query_target_decisions(records, writer)
        lines = writer.text.splitlines()
        assert len(lines) == 1
        assert "tick=7" in lines[0]
        assert "target=(131,124)" in lines[0]

    def test_renders_dash_when_both_coords_missing(self) -> None:
        """A HUNT event with no target coords prints ``target=-``."""
        writer = _RecordingWriter()
        query_target_decisions(
            [_record("AI", "HUNT score=0")],
            writer,
        )
        assert "target=-" in writer.text

    def test_skips_non_ai_channels(self) -> None:
        """Non-AI events (STATE / WIRE / DIAGNOSTIC) are skipped.

        Documents the contract: target-decisions only summarises AI
        HUNT scores. STATE/WIRE/DIAGNOSTIC events appear in other
        queries and are ignored here so the output stays focused.
        """
        writer = _RecordingWriter()
        records = [
            _record("STATE", "IDLE"),
            _record("WIRE", "WIRE: shoot_at", action_kind="shoot"),
            _record("DIAGNOSTIC", "diagnostic_kind=map_data_snapshot"),
            _record(
                "AI",
                "HUNT score=0.8 target=(131,124)",
                combat_target_x=131,
                combat_target_y=124,
            ),
        ]
        query_target_decisions(records, writer)
        lines = writer.text.splitlines()
        assert len(lines) == 1
        assert "target=(131,124)" in lines[0]

    def test_renders_question_mark_for_partial_coords(self) -> None:
        """One coord present + one missing prints ``(?,Y)``-style row."""
        writer = _RecordingWriter()
        query_target_decisions(
            [
                _record(
                    "AI",
                    "HUNT score=0.4",
                    combat_target_y=124,
                ),
            ],
            writer,
        )
        assert "target=(?,124)" in writer.text


class _LoadRecordsBase:
    """Base setup / teardown for tests that read the JSONL via load_records."""

    def setup_method(self) -> None:
        """Install the fake filesystem."""
        (
            self._fake,
            self._original_path_exists,
            self._original_read_text,
            self._original_append_text,
        ) = _install_fake_filesystem()

    def teardown_method(self) -> None:
        """Restore the real ``_test_hooks`` bindings."""
        _test_hooks.path_exists = self._original_path_exists
        _test_hooks.read_text = self._original_read_text
        _test_hooks.append_text = self._original_append_text


class TestLoadRecords(_LoadRecordsBase):
    """Tests for the JSONL loader."""

    def test_raises_missing_events_file_when_path_absent(self, tmp_path: Path) -> None:
        """Missing path raises ``_MissingEventsFileError`` (exit code 1)."""
        target = tmp_path / "missing.jsonl"
        with pytest.raises(bot_query._MissingEventsFileError) as exc:
            load_records(target)
        assert exc.value.path == target
        assert exc.value.code == 1
        assert "does not exist" in exc.value.message

    def test_skips_blank_lines(self) -> None:
        """Blank lines between records are skipped silently."""
        path = Path("runs/bot/latest.events.jsonl")
        body = _event("STATE", "alpha") + "\n" + _event("STATE", "beta")
        self._fake.write(path, body)
        records = load_records(path)
        assert [r.message for r in records] == ["alpha", "beta"]


class TestRunDispatcher(_LoadRecordsBase):
    """Tests for the ``run(argv)`` argv dispatcher."""

    def test_empty_argv_prints_usage_and_returns_one(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Empty argv -> usage block to stderr and exit code 1."""
        assert run([]) == 1
        err = capsys.readouterr().err
        assert "usage: bot-query" in err

    def test_too_many_argv_prints_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        """More than two arguments -> usage error."""
        assert run(["timeline", "a", "b"]) == 1
        err = capsys.readouterr().err
        assert "usage: bot-query" in err

    def test_unknown_query_prints_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Unknown query name -> usage error."""
        assert run(["bogus"]) == 1
        err = capsys.readouterr().err
        assert "usage: bot-query" in err

    def test_timeline_against_default_path(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``run(["timeline"])`` reads ``DEFAULT_EVENTS_PATH`` and writes to stdout."""
        self._fake.write(
            DEFAULT_EVENTS_PATH,
            _event("STATE", "IDLE", tick_n=1),
        )
        assert run(["timeline"]) == 0
        out = capsys.readouterr().out
        assert "STATE\tIDLE" in out

    def test_query_with_explicit_path(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """The optional second argument overrides the default path."""
        explicit = tmp_path / "events.jsonl"
        self._fake.write(
            explicit,
            _event(
                "AI",
                "HUNT score=0.5",
                combat_target_x=131,
                combat_target_y=124,
            ),
        )
        assert run(["target-decisions", str(explicit)]) == 0
        out = capsys.readouterr().out
        assert "target=(131,124)" in out


class TestMain(_LoadRecordsBase):
    """Tests for ``main()`` and the ``__main__`` runpy entrypoint."""

    def test_main_exits_with_run_code(self) -> None:
        """``main()`` propagates ``run()``'s exit code via ``SystemExit``."""
        self._fake.write(DEFAULT_EVENTS_PATH, _event("STATE", "IDLE", tick_n=1))
        old_argv = sys.argv
        sys.argv = ["bot-query", "timeline"]
        try:
            with pytest.raises(SystemExit) as exc:
                bot_query.main()
            assert exc.value.code == 0
        finally:
            sys.argv = old_argv

    def test_module_entrypoint_runs_main(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``python -m tankpit_bot.diagnostics.bot_query`` executes ``main``."""
        self._fake.write(
            DEFAULT_EVENTS_PATH,
            _event("STATE", "IDLE", tick_n=1),
        )
        old_argv = sys.argv
        sys.argv = ["tankpit_bot.diagnostics.bot_query", "timeline"]
        try:
            sys.modules.pop("tankpit_bot.diagnostics.bot_query", None)
            with pytest.raises(SystemExit) as exc:
                runpy.run_module("tankpit_bot.diagnostics.bot_query", run_name="__main__")
            assert exc.value.code == 0
        finally:
            sys.argv = old_argv
        out = capsys.readouterr().out
        assert "STATE\tIDLE" in out
