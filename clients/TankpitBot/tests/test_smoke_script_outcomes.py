"""Tests for the smoke script's failure and partial outcomes."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
)

from scripts import (
    _test_hooks,
    smoke,
)
from tests._smoke_records import (
    _emitted_outcome_object,
    _emitted_outcome_record,
    _full_success_jsonl,
    _full_success_records,
    _install_fake_filesystem,
    _map_open_resolved,
    _map_open_stalled,
    _move_stalled,
    _record_raw,
    _smoke_record,
)
from tests.conftest import FakeFileSystem


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

    def test_passes_when_the_early_outcomes_hold_no_stall(self, fake_fs: FakeFileSystem) -> None:
        """Resolved outcomes in the window, none a stall, satisfy the gate."""
        records = [
            _smoke_record(1, "STATE", "INITIALIZING", timestamp="2026-06-20T15:00:00"),
            _emitted_outcome_record(fake_fs, 2, _map_open_resolved, "2026-06-20T15:00:03"),
        ]
        assert smoke.assert_no_early_stall(records) is None

    def test_passes_when_stall_fires_after_10s(self, fake_fs: FakeFileSystem) -> None:
        """A stall_timeout at t+11s does NOT trip the early-stall gate."""
        records = [
            _smoke_record(1, "STATE", "INITIALIZING", timestamp="2026-06-20T15:00:00"),
            _emitted_outcome_record(fake_fs, 2, _map_open_stalled, "2026-06-20T15:00:11"),
        ]
        assert smoke.assert_no_early_stall(records) is None

    def test_fails_when_stall_fires_in_first_10s(self, fake_fs: FakeFileSystem) -> None:
        """The ledger's own stall line at t+5s trips the gate, naming it."""
        records = [
            _smoke_record(1, "STATE", "INITIALIZING", timestamp="2026-06-20T15:00:00"),
            _emitted_outcome_record(fake_fs, 2, _move_stalled, "2026-06-20T15:00:05"),
        ]
        failure = smoke.assert_no_early_stall(records)
        assert failure == {
            "message": (
                "5) stall_timeout fired at t+5.0s (action_kind='move') within the 10s window"
            ),
            "pivot": 1,
        }

    def test_fails_when_no_records(self) -> None:
        """An empty JSONL fails the no-stall gate with a clear message."""
        failure = smoke.assert_no_early_stall([])
        if failure is None:
            raise AssertionError("empty records must fail the gate")
        assert "empty JSONL" in failure["message"]

    def test_fails_when_no_action_resolved(self) -> None:
        """Events with no outcome among them are no verdict on stalls.

        Until 2026-09-29 this gate filtered on a channel nothing wrote,
        so it passed every run while examining zero events (board task
        14b91fb5). A run with nothing resolved now fails, saying so.
        """
        records = [
            _smoke_record(1, "STATE", "INITIALIZING", timestamp="2026-06-20T15:00:00"),
            _smoke_record(2, "STATE", "IDLE", timestamp="2026-06-20T15:00:05"),
        ]
        failure = smoke.assert_no_early_stall(records)
        assert failure == {
            "message": (
                "5) 0 action outcomes among 2 events -- "
                "no action resolved, so stalls cannot be judged"
            ),
            "pivot": 1,
        }

    def test_the_retired_wire_complete_stall_is_not_an_outcome(self) -> None:
        """A stall in the shape the gate read until 2026-09-29 is not counted.

        The record is no outcome, so the run has none, and the gate
        says so instead of either passing or blaming a stall.
        """
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
            raise AssertionError("the retired shape must not read as a clean run")
        assert failure["message"].startswith("5) 0 action outcomes among 2 events")


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
        self._fake.write_text(path, text)
        records = smoke.load_records(path)
        assert len(records) == 3
        assert records[0].message == "step 1"
        assert records[-1].message == "step 3"

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        """Blank lines between records are skipped silently."""
        path = tmp_path / "ok.jsonl"
        record_text = _record_raw("STATE", "alpha")
        self._fake.write_text(path, f"{record_text}\n\n{record_text}\n")
        records = smoke.load_records(path)
        assert len(records) == 2

    def test_raises_jsontypeerror_when_top_level_is_not_object(self, tmp_path: Path) -> None:
        """Non-object top-level values are rejected by ``narrow_json_to_dict``."""
        path = tmp_path / "bad.jsonl"
        self._fake.write_text(path, '["not an object"]\n')
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

    def test_returns_none_on_full_pass(self, fake_fs: FakeFileSystem) -> None:
        """Every assertion passing -> ``evaluate`` returns ``None``."""
        assert smoke.evaluate(_full_success_records(fake_fs)) is None

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
        fake_fs: FakeFileSystem,
    ) -> None:
        """A JSONL with every signal present returns exit code 0 and counts its outcomes."""
        path = tmp_path / "latest.events.jsonl"
        self._fake.write_text(path, _full_success_jsonl(fake_fs))
        # The emitter building the payload logs to the console; only
        # the run's own output is under test.
        capsys.readouterr()
        assert smoke.run(path) == 0
        out = capsys.readouterr().out
        assert out == (
            "SMOKE PASSED: 6 events, 1 action outcomes, 5/5 assertions green "
            "(login, map_data_processed, HUNT target, action attempted, no early stall).\n"
        )

    def test_returns_one_on_first_assertion_failure(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        fake_fs: FakeFileSystem,
    ) -> None:
        """Missing the login ladder returns exit code 1 with a clear message."""
        path = tmp_path / "latest.events.jsonl"
        # Strip the STATE login records.
        raws = [
            dump_json_str(
                _emitted_outcome_object(fake_fs, _map_open_resolved, "2026-06-20T15:00:00")
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
        self._fake.write_text(path, "\n".join(raws) + "\n")
        assert smoke.run(path) == 1
        err = capsys.readouterr().err
        assert "SMOKE FAILED" in err
        assert "login ladder" in err
        assert "Surrounding context" in err

    def test_run_uses_default_path_when_called_without_argument(
        self,
        tmp_path: Path,
        fake_fs: FakeFileSystem,
    ) -> None:
        """The default ``path`` argument routes through ``LATEST_EVENTS_PATH``.

        We assert by registering the default path in the fake FS with
        a fully-passing payload; ``run()`` must read it and return 0.
        """
        self._fake.write_text(smoke.LATEST_EVENTS_PATH, _full_success_jsonl(fake_fs))
        assert smoke.run() == 0

    def test_main_exits_with_run_code(self, tmp_path: Path, fake_fs: FakeFileSystem) -> None:
        """``main()`` propagates ``run()``'s exit code via SystemExit."""
        self._fake.write_text(smoke.LATEST_EVENTS_PATH, _full_success_jsonl(fake_fs))
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
        fake_fs: FakeFileSystem,
    ) -> None:
        """``python -m scripts.smoke`` executes the ``if __name__`` block.

        This guards against regressions where ``main()`` exists but
        ``runpy``-style execution falls back to library-only behavior.
        """
        self._fake.write_text(smoke.LATEST_EVENTS_PATH, _full_success_jsonl(fake_fs))
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
