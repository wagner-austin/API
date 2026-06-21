"""Tests for the smoke health-gate script.

Drives the scripts.smoke CLI end-to-end and exercises every assertion
branch. File reads go through scripts._test_hooks (save-and-restore)
so no monkeypatch is needed and the production hook surface is the
sole I/O boundary.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
)
from scripts._test_hooks import PathExistsProtocol, ReadTextProtocol

from scripts import _test_hooks, smoke


def _record_object(
    channel: str,
    message: str,
    timestamp: str = "2026-06-20T15:00:00",
    **fields: JSONValue,
) -> JSONObject:
    """Build a parsed-record JSON object directly (no serialisation)."""
    payload: JSONObject = {
        "timestamp": timestamp,
        "level": "INFO",
        "logger": "tankpit_bot.runtime.events",
        "mode": "bot",
        "channel": channel,
        "message": message,
    }
    payload.update(fields)
    return payload


def _record_raw(
    channel: str,
    message: str,
    timestamp: str = "2026-06-20T15:00:00",
    **fields: JSONValue,
) -> str:
    """Build the serialised JSONL line for a record."""
    return dump_json_str(_record_object(channel, message, timestamp, **fields))


def _smoke_record(
    line_no: int,
    channel: str,
    message: str,
    timestamp: str = "2026-06-20T15:00:00",
    **fields: JSONValue,
) -> smoke.SmokeRecord:
    """Build a SmokeRecord through the production decoder.

    Using ``_decode_smoke_record`` instead of the constructor keeps
    the helper aligned with how :func:`smoke.load_records` produces
    records in production.
    """
    parsed = _record_object(channel, message, timestamp, **fields)
    raw = dump_json_str(parsed)
    return smoke._decode_smoke_record(line_no=line_no, raw=raw, parsed=parsed)


def _login_records() -> list[smoke.SmokeRecord]:
    """A typical login sequence: INITIALIZING -> WAITING -> IDLE."""
    return [
        _smoke_record(1, "STATE", "INITIALIZING"),
        _smoke_record(2, "STATE", "INITIALIZING -> WAITING_FOR_POSITION"),
        _smoke_record(3, "STATE", "WAITING_FOR_POSITION -> IDLE"),
    ]


def _map_data_processed_record(line_no: int, timestamp: str) -> smoke.SmokeRecord:
    """A successful map_open WIRE_COMPLETE event."""
    return _smoke_record(
        line_no,
        "WIRE_COMPLETE",
        "map_open completed in 250ms via map_data_processed",
        timestamp=timestamp,
        action_kind="map_open",
        duration_ms=250,
        signal="map_data_processed",
    )


def _full_success_records(
    start: str = "2026-06-20T15:00:00",
) -> list[smoke.SmokeRecord]:
    """Build a record list that passes every assertion."""
    return [
        _smoke_record(1, "STATE", "INITIALIZING", timestamp=start),
        _smoke_record(2, "STATE", "INITIALIZING -> WAITING_FOR_POSITION", timestamp=start),
        _smoke_record(3, "STATE", "WAITING_FOR_POSITION -> IDLE", timestamp=start),
        _map_data_processed_record(4, start),
        _smoke_record(
            5,
            "AI",
            "HUNT score=0.8 target=(131,124)",
            timestamp=start,
            combat_target_x=131,
            combat_target_y=124,
        ),
        _smoke_record(
            6,
            "WIRE",
            "WIRE: shoot_at (131,124)",
            timestamp=start,
            action_kind="shoot",
        ),
    ]


def _full_success_jsonl(start: str = "2026-06-20T15:00:00") -> str:
    """Serialised JSONL string for a fully-passing run."""
    raws = [
        _record_raw("STATE", "INITIALIZING", timestamp=start),
        _record_raw("STATE", "INITIALIZING -> WAITING_FOR_POSITION", timestamp=start),
        _record_raw("STATE", "WAITING_FOR_POSITION -> IDLE", timestamp=start),
        _record_raw(
            "WIRE_COMPLETE",
            "map_open completed in 250ms via map_data_processed",
            timestamp=start,
            action_kind="map_open",
            duration_ms=250,
            signal="map_data_processed",
        ),
        _record_raw(
            "AI",
            "HUNT score=0.8 target=(131,124)",
            timestamp=start,
            combat_target_x=131,
            combat_target_y=124,
        ),
        _record_raw(
            "WIRE",
            "WIRE: shoot_at (131,124)",
            timestamp=start,
            action_kind="shoot",
        ),
    ]
    return "\n".join(raws) + "\n"


class _FakeFileSystem:
    """Save-and-restore stand-in for scripts._test_hooks file ops.

    Tests instantiate this fake, assign its bound methods to
    ``_test_hooks.path_exists`` and ``_test_hooks.read_text``, and
    restore the originals on teardown. No monkeypatch, no module
    constants overridden.
    """

    def __init__(self) -> None:
        """Initialise with no virtual files registered."""
        self._files: dict[str, str] = {}

    def write(self, path: Path, content: str) -> None:
        """Register a virtual file at ``path`` containing ``content``.

        Args:
            path: Virtual path used by the loader.
            content: File contents to return on read.
        """
        self._files[str(path)] = content

    def path_exists(self, path: Path) -> bool:
        """Return True when the virtual filesystem knows ``path``.

        Args:
            path: Path to check.

        Returns:
            True if ``write`` previously registered the path.
        """
        return str(path) in self._files

    def read_text(self, path: Path) -> str:
        """Return the registered contents for ``path``.

        Args:
            path: Path to read.

        Returns:
            Contents previously registered via :meth:`write`.

        Raises:
            KeyError: If ``path`` was never registered.
        """
        return self._files[str(path)]


def _install_fake_filesystem() -> tuple[_FakeFileSystem, PathExistsProtocol, ReadTextProtocol]:
    """Swap the real script hooks for a fake; return originals for restore.

    Returns:
        Tuple of ``(fake, original_path_exists, original_read_text)``.
        Callers MUST restore the originals in teardown.
    """
    fake = _FakeFileSystem()
    original_path_exists: PathExistsProtocol = _test_hooks.path_exists
    original_read_text: ReadTextProtocol = _test_hooks.read_text
    _test_hooks.path_exists = fake.path_exists
    _test_hooks.read_text = fake.read_text
    return (fake, original_path_exists, original_read_text)


class TestDecodeSmokeRecord:
    """Tests for ``_decode_smoke_record`` (record decoder boundary)."""

    def test_returns_record_with_fields_minus_reserved_keys(self) -> None:
        """Reserved record-level keys are stripped from the fields view."""
        rec = _smoke_record(
            1,
            "AI",
            "HUNT score=0.5",
            combat_target_x=131,
            combat_target_y=124,
        )
        assert rec.fields == {"combat_target_x": 131, "combat_target_y": 124}
        assert "channel" not in rec.fields
        assert "message" not in rec.fields
        assert rec.channel == "AI"
        assert rec.line_no == 1

    def test_raises_jsontypeerror_on_missing_channel(self) -> None:
        """JSONTypeError propagates unchanged when ``channel`` is absent."""
        parsed: JSONObject = {
            "timestamp": "2026-06-20T15:00:00",
            "level": "INFO",
            "logger": "tankpit_bot.runtime.events",
            "mode": "bot",
            "message": "noop",
        }
        with pytest.raises(JSONTypeError):
            smoke._decode_smoke_record(line_no=1, raw="{}", parsed=parsed)

    def test_raises_jsontypeerror_on_missing_message(self) -> None:
        """JSONTypeError propagates when ``message`` is absent."""
        parsed: JSONObject = {
            "timestamp": "2026-06-20T15:00:00",
            "level": "INFO",
            "logger": "tankpit_bot.runtime.events",
            "mode": "bot",
            "channel": "STATE",
        }
        with pytest.raises(JSONTypeError):
            smoke._decode_smoke_record(line_no=2, raw="{}", parsed=parsed)

    def test_raises_jsontypeerror_on_missing_timestamp(self) -> None:
        """JSONTypeError propagates when ``timestamp`` is absent."""
        parsed: JSONObject = {
            "level": "INFO",
            "logger": "tankpit_bot.runtime.events",
            "mode": "bot",
            "channel": "STATE",
            "message": "noop",
        }
        with pytest.raises(JSONTypeError):
            smoke._decode_smoke_record(line_no=3, raw="{}", parsed=parsed)


class TestParseIsoTimestampSeconds:
    """Tests for the ISO-ish timestamp parser."""

    def test_returns_seconds_for_full_timestamp(self) -> None:
        """ISO timestamps parse to total seconds-of-day."""
        assert smoke.parse_iso_timestamp_seconds("2026-06-20T01:02:03") == pytest.approx(3723.0)

    def test_accepts_fractional_seconds(self) -> None:
        """Fractional seconds carry through to the float result."""
        assert smoke.parse_iso_timestamp_seconds("2026-06-20T00:00:01.5") == pytest.approx(1.5)

    def test_raises_without_t_separator(self) -> None:
        """Timestamps without 'T' are rejected at parse time."""
        with pytest.raises(ValueError, match="T"):
            smoke.parse_iso_timestamp_seconds("2026-06-20 01:02:03")

    def test_raises_without_seconds_component(self) -> None:
        """Timestamps with only HH:MM are rejected (need seconds)."""
        with pytest.raises(ValueError, match="seconds"):
            smoke.parse_iso_timestamp_seconds("2026-06-20T01:02")


class TestAssertLoginCompleted:
    """Tests for assertion 1 (login ladder)."""

    def test_passes_on_full_ladder(self) -> None:
        """The full ladder produces ``None``."""
        assert smoke.assert_login_completed(_login_records()) is None

    def test_passes_when_extra_state_events_appear_before(self) -> None:
        """Extra STATE events before the ladder do not break the check."""
        records = [
            _smoke_record(0, "STATE", "BOOTING"),
            *_login_records(),
        ]
        assert smoke.assert_login_completed(records) is None

    def test_fails_when_first_transition_missing(self) -> None:
        """Missing the first transition fails with a clear message."""
        records = [
            _smoke_record(1, "STATE", "INITIALIZING"),
            _smoke_record(2, "STATE", "WAITING_FOR_POSITION -> IDLE"),
        ]
        failure = smoke.assert_login_completed(records)
        if failure is None:
            raise AssertionError("missing first transition must fail the gate")
        assert "login ladder" in failure["message"]
        assert failure["pivot"] == 1

    def test_fails_when_second_transition_missing(self) -> None:
        """Missing the second transition fails with a clear message."""
        records = [_smoke_record(1, "STATE", "INITIALIZING -> WAITING_FOR_POSITION")]
        failure = smoke.assert_login_completed(records)
        if failure is None:
            raise AssertionError("missing second transition must fail the gate")
        assert "login ladder" in failure["message"]
        assert failure["pivot"] == 0

    def test_fails_when_no_state_events(self) -> None:
        """No STATE events at all fails the check with pivot=0."""
        records = [_smoke_record(1, "AI", "HUNT score=0", combat_target_x=0, combat_target_y=0)]
        failure = smoke.assert_login_completed(records)
        if failure is None:
            raise AssertionError("no STATE events must fail the login ladder gate")
        assert failure["pivot"] == 0


class TestAssertMapOpenClearedViaMapData:
    """Tests for assertion 2 (map_open cleared via map_data_processed)."""

    def test_passes_when_one_map_open_clears_via_map_data(self) -> None:
        """One matching WIRE_COMPLETE event satisfies the gate."""
        records = [_map_data_processed_record(1, "2026-06-20T15:00:05")]
        assert smoke.assert_map_open_cleared_via_map_data(records) is None

    def test_fails_when_no_map_open_events_at_all(self) -> None:
        """No map_open WIRE_COMPLETE events fails the gate."""
        failure = smoke.assert_map_open_cleared_via_map_data([])
        if failure is None:
            raise AssertionError("empty records must fail the map_open gate")
        assert "map_open" in failure["message"]

    def test_fails_when_map_open_clears_via_stall_timeout(self) -> None:
        """A map_open that cleared via stall_timeout fails the gate.

        Regression: this is exactly the failure mode the 2026-06-20 fix
        cured -- the dispatcher wasn't marking map_data_processed, so
        every map_open cleared via the 10s stall_timeout instead.
        """
        records = [
            _smoke_record(
                1,
                "WIRE_COMPLETE",
                "map_open completed in 10000ms via stall_timeout",
                action_kind="map_open",
                duration_ms=10000,
                signal="stall_timeout",
            )
        ]
        failure = smoke.assert_map_open_cleared_via_map_data(records)
        if failure is None:
            raise AssertionError("stall-cleared map_open must fail the gate")
        assert "map_data_processed" in failure["message"]

    def test_ignores_wire_complete_for_other_action_kinds(self) -> None:
        """Non-map_open WIRE_COMPLETE events do not satisfy the gate."""
        records = [
            _smoke_record(
                1,
                "WIRE_COMPLETE",
                "teleport completed in 250ms via teleport_landed",
                action_kind="teleport",
                duration_ms=250,
                signal="teleport_landed",
            )
        ]
        failure = smoke.assert_map_open_cleared_via_map_data(records)
        if failure is None:
            raise AssertionError("teleport-only completion must fail the gate")


class TestAssertHuntScoredTarget:
    """Tests for assertion 3 (HUNT scored a non-zero target)."""

    def test_passes_when_combat_target_x_is_non_zero(self) -> None:
        """Any HUNT score event with target_x != 0 satisfies the gate."""
        records = [
            _smoke_record(
                1,
                "AI",
                "HUNT score=0.8 target=(131,124)",
                combat_target_x=131,
                combat_target_y=124,
            )
        ]
        assert smoke.assert_hunt_scored_target(records) is None

    def test_passes_when_only_combat_target_y_is_non_zero(self) -> None:
        """target_y alone is enough -- guards against axis-bias misses."""
        records = [
            _smoke_record(
                1,
                "AI",
                "HUNT score=0.4",
                combat_target_x=0,
                combat_target_y=124,
            )
        ]
        assert smoke.assert_hunt_scored_target(records) is None

    def test_fails_when_no_hunt_events(self) -> None:
        """No HUNT events at all fails the gate (pivot=0)."""
        records = [_smoke_record(1, "AI", "decided IDLE")]
        failure = smoke.assert_hunt_scored_target(records)
        if failure is None:
            raise AssertionError("no HUNT events must fail the gate")
        assert failure["pivot"] == 0

    def test_fails_when_every_hunt_score_has_zero_target(self) -> None:
        """Every HUNT event with both coords 0 fails the gate."""
        records = [
            _smoke_record(
                1,
                "AI",
                "HUNT score=0",
                combat_target_x=0,
                combat_target_y=0,
            ),
            _smoke_record(
                2,
                "AI",
                "HUNT score=0",
                combat_target_x=0,
                combat_target_y=0,
            ),
        ]
        failure = smoke.assert_hunt_scored_target(records)
        if failure is None:
            raise AssertionError("all-zero HUNT events must fail the gate")
        assert failure["pivot"] == 1


class TestAssertActionAttempted:
    """Tests for assertion 4 (at least one bot action)."""

    @pytest.mark.parametrize("kind", sorted(smoke.ACTION_KINDS))
    def test_passes_for_each_known_action_kind(self, kind: str) -> None:
        """Every known action_kind satisfies the gate."""
        records = [
            _smoke_record(
                1,
                "WIRE",
                f"WIRE: {kind}_at",
                action_kind=kind,
            )
        ]
        assert smoke.assert_action_attempted(records) is None

    def test_fails_when_no_wire_events(self) -> None:
        """No WIRE events fails the gate (pivot=0)."""
        failure = smoke.assert_action_attempted([])
        if failure is None:
            raise AssertionError("no WIRE events must fail the gate")
        assert failure["pivot"] == 0

    def test_fails_when_wire_event_has_unknown_kind(self) -> None:
        """WIRE events with unknown action_kind fail the gate."""
        records = [
            _smoke_record(
                1,
                "WIRE",
                "WIRE: do_something",
                action_kind="garbage",
            )
        ]
        failure = smoke.assert_action_attempted(records)
        if failure is None:
            raise AssertionError("unknown action_kind must fail the gate")
        assert failure["pivot"] == 0

    def test_fails_when_wire_event_has_no_action_kind(self) -> None:
        """WIRE events lacking action_kind fail the gate."""
        records = [_smoke_record(1, "WIRE", "WIRE: bare ping")]
        failure = smoke.assert_action_attempted(records)
        if failure is None:
            raise AssertionError("missing action_kind must fail the gate")


class TestAssertNoEarlyStall:
    """Tests for assertion 5 (zero stall_timeout in first 10s)."""

    def test_passes_when_no_stall_events(self) -> None:
        """No stall_timeout WIRE_COMPLETE events satisfy the gate."""
        records = [
            _smoke_record(1, "STATE", "INITIALIZING", timestamp="2026-06-20T15:00:00"),
            _smoke_record(2, "STATE", "IDLE", timestamp="2026-06-20T15:00:05"),
        ]
        assert smoke.assert_no_early_stall(records) is None

    def test_passes_when_stall_fires_after_10s(self) -> None:
        """A stall_timeout at t+11s does NOT trip the early-stall gate."""
        records = [
            _smoke_record(1, "STATE", "INITIALIZING", timestamp="2026-06-20T15:00:00"),
            _smoke_record(
                2,
                "WIRE_COMPLETE",
                "map_open completed in 10000ms via stall_timeout",
                timestamp="2026-06-20T15:00:11",
                action_kind="map_open",
                duration_ms=10000,
                signal="stall_timeout",
            ),
        ]
        assert smoke.assert_no_early_stall(records) is None

    def test_fails_when_stall_fires_in_first_10s(self) -> None:
        """A stall_timeout at t+5s trips the gate."""
        records = [
            _smoke_record(1, "STATE", "INITIALIZING", timestamp="2026-06-20T15:00:00"),
            _smoke_record(
                2,
                "WIRE_COMPLETE",
                "move completed in 5000ms via stall_timeout",
                timestamp="2026-06-20T15:00:05",
                action_kind="move",
                duration_ms=5000,
                signal="stall_timeout",
            ),
        ]
        failure = smoke.assert_no_early_stall(records)
        if failure is None:
            raise AssertionError("stall at t+5s must fail the gate")
        assert "stall_timeout fired at t+5.0s" in failure["message"]
        assert "action_kind='move'" in failure["message"]
        assert failure["pivot"] == 1

    def test_fails_when_no_records(self) -> None:
        """An empty JSONL fails the no-stall gate with a clear message."""
        failure = smoke.assert_no_early_stall([])
        if failure is None:
            raise AssertionError("empty records must fail the gate")
        assert "empty JSONL" in failure["message"]

    def test_ignores_non_stall_wire_complete_events(self) -> None:
        """WIRE_COMPLETE events with other signals are ignored."""
        records = [
            _smoke_record(1, "STATE", "INITIALIZING", timestamp="2026-06-20T15:00:00"),
            _smoke_record(
                2,
                "WIRE_COMPLETE",
                "map_open completed in 250ms via map_data_processed",
                timestamp="2026-06-20T15:00:03",
                action_kind="map_open",
                duration_ms=250,
                signal="map_data_processed",
            ),
        ]
        assert smoke.assert_no_early_stall(records) is None


class TestLoadRecords:
    """Tests for load_records using save-and-restore _test_hooks DI."""

    def setup_method(self) -> None:
        """Install fake filesystem; save originals for restore."""
        (
            self._fake,
            self._original_path_exists,
            self._original_read_text,
        ) = _install_fake_filesystem()

    def teardown_method(self) -> None:
        """Restore real _test_hooks bindings."""
        _test_hooks.path_exists = self._original_path_exists
        _test_hooks.read_text = self._original_read_text

    def test_raises_missing_events_file_when_path_absent(self, tmp_path: Path) -> None:
        """Missing path raises ``_MissingEventsFileError`` (exit code 1)."""
        target = tmp_path / "missing.jsonl"
        with pytest.raises(smoke._MissingEventsFileError) as exc:
            smoke.load_records(target)
        assert exc.value.path == target
        assert exc.value.code == 1
        assert "does not exist" in exc.value.message

    def test_returns_records_in_order(self, tmp_path: Path) -> None:
        """Each non-empty line becomes a SmokeRecord in file order."""
        path = tmp_path / "ok.jsonl"
        text = "\n".join(
            _record_raw("STATE", f"step {i}", timestamp=f"2026-06-20T15:00:0{i}") for i in (1, 2, 3)
        )
        self._fake.write(path, text)
        records = smoke.load_records(path)
        assert len(records) == 3
        assert records[0].message == "step 1"
        assert records[-1].message == "step 3"

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        """Blank lines between records are skipped silently."""
        path = tmp_path / "ok.jsonl"
        record_text = _record_raw("STATE", "alpha")
        self._fake.write(path, f"{record_text}\n\n{record_text}\n")
        records = smoke.load_records(path)
        assert len(records) == 2

    def test_raises_jsontypeerror_when_top_level_is_not_object(self, tmp_path: Path) -> None:
        """Non-object top-level values are rejected by ``narrow_json_to_dict``."""
        path = tmp_path / "bad.jsonl"
        self._fake.write(path, '["not an object"]\n')
        with pytest.raises(JSONTypeError):
            smoke.load_records(path)


class TestContextWindow:
    """Tests for the diagnostic context_window helper."""

    def test_returns_pivot_plus_minus_radius(self) -> None:
        """Default radius=5 produces an 11-line window when records allow."""
        records = [_smoke_record(i, "STATE", f"step {i}") for i in range(20)]
        window = smoke.context_window(records, pivot=10, radius=5)
        lines = window.splitlines()
        assert len(lines) == 11
        assert '"message":"step 5"' in lines[0]
        assert '"message":"step 15"' in lines[-1]

    def test_clips_to_record_bounds(self) -> None:
        """Pivot near the start clips at index 0 rather than overflowing."""
        records = [_smoke_record(i, "STATE", f"step {i}") for i in range(3)]
        window = smoke.context_window(records, pivot=0, radius=5)
        # Only 3 records exist, so the window contains all of them.
        assert len(window.splitlines()) == 3


class TestEvaluate:
    """Tests for the ``evaluate`` aggregate."""

    def test_returns_none_on_full_pass(self) -> None:
        """Every assertion passing -> ``evaluate`` returns ``None``."""
        assert smoke.evaluate(_full_success_records()) is None

    def test_returns_first_failure_only(self) -> None:
        """The first failing assertion short-circuits later ones."""
        records = [
            _smoke_record(1, "WIRE", "WIRE: shoot_at (131,124)", action_kind="shoot"),
        ]
        failure = smoke.evaluate(records)
        if failure is None:
            raise AssertionError("missing login ladder must report a failure")
        assert "login ladder" in failure["message"]


class TestRun:
    """End-to-end tests for the script's run() entrypoint."""

    def setup_method(self) -> None:
        """Install fake filesystem; save originals for restore."""
        (
            self._fake,
            self._original_path_exists,
            self._original_read_text,
        ) = _install_fake_filesystem()

    def teardown_method(self) -> None:
        """Restore real _test_hooks bindings."""
        _test_hooks.path_exists = self._original_path_exists
        _test_hooks.read_text = self._original_read_text

    def test_returns_zero_on_full_pass(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A JSONL with every signal present returns exit code 0."""
        path = tmp_path / "latest.events.jsonl"
        self._fake.write(path, _full_success_jsonl())
        assert smoke.run(path) == 0
        out = capsys.readouterr().out
        assert "SMOKE PASSED" in out
        assert "5/5 assertions green" in out

    def test_returns_one_on_first_assertion_failure(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Missing the login ladder returns exit code 1 with a clear message."""
        path = tmp_path / "latest.events.jsonl"
        # Strip the STATE login records.
        raws = [
            _record_raw(
                "WIRE_COMPLETE",
                "map_open completed in 250ms via map_data_processed",
                action_kind="map_open",
                duration_ms=250,
                signal="map_data_processed",
            ),
            _record_raw(
                "AI",
                "HUNT score=0.8 target=(131,124)",
                combat_target_x=131,
                combat_target_y=124,
            ),
            _record_raw(
                "WIRE",
                "WIRE: shoot_at (131,124)",
                action_kind="shoot",
            ),
        ]
        self._fake.write(path, "\n".join(raws) + "\n")
        assert smoke.run(path) == 1
        err = capsys.readouterr().err
        assert "SMOKE FAILED" in err
        assert "login ladder" in err
        assert "Surrounding context" in err

    def test_run_uses_default_path_when_called_without_argument(
        self,
        tmp_path: Path,
    ) -> None:
        """The default ``path`` argument routes through ``LATEST_EVENTS_PATH``.

        We assert by registering the default path in the fake FS with
        a fully-passing payload; ``run()`` must read it and return 0.
        """
        self._fake.write(smoke.LATEST_EVENTS_PATH, _full_success_jsonl())
        assert smoke.run() == 0

    def test_main_exits_with_run_code(self, tmp_path: Path) -> None:
        """``main()`` propagates ``run()``'s exit code via SystemExit."""
        self._fake.write(smoke.LATEST_EVENTS_PATH, _full_success_jsonl())
        with pytest.raises(SystemExit) as exc:
            smoke.main()
        assert exc.value.code == 0

    def test_run_raises_missing_events_file_when_jsonl_absent(
        self,
        tmp_path: Path,
    ) -> None:
        """Missing JSONL -> ``_MissingEventsFileError`` propagates to main()."""
        target = tmp_path / "missing.jsonl"
        with pytest.raises(smoke._MissingEventsFileError):
            smoke.run(target)

    def test_module_entrypoint_runs_main(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``python -m scripts.smoke`` executes the ``if __name__`` block.

        This guards against regressions where ``main()`` exists but
        ``runpy``-style execution falls back to library-only behavior.
        """
        self._fake.write(smoke.LATEST_EVENTS_PATH, _full_success_jsonl())
        old_argv = sys.argv
        sys.argv = ["scripts.smoke"]
        try:
            sys.modules.pop("scripts.smoke", None)
            with pytest.raises(SystemExit) as exc:
                runpy.run_module("scripts.smoke", run_name="__main__")
            assert exc.value.code == 0
        finally:
            sys.argv = old_argv
        out = capsys.readouterr().out
        assert "SMOKE PASSED" in out
