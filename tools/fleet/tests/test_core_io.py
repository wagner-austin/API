"""The append-only records, the node probe, the ssh seam, and the hooks.

EVERY FAKE HERE IMPLEMENTS THE REAL PROTOCOL. `FakeRun` is given to
``_test_hooks.run`` and satisfies ``RunProtocol``. The FILE hooks are not
faked at all: the autouse reset in ``conftest`` leaves them on their real
implementations, so these tests read and write exactly the way production
does, against a real temporary directory. Nothing is patched.

The default hook implementations are exercised directly rather than left to
coverage's mercy: they are the code that actually touches the disk and the
subprocess table, and a package whose only tested path is the fake one has
tested its test double.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.budget import NodeBudget
from fleet.contracts.feed import FeedEvent, decode_feed_event
from fleet.contracts.ledger import LedgerEntry, decode_ledger_entry
from fleet.contracts.node import NodeConfig
from fleet.core import _test_hooks, probe, records, remote
from tests.conftest import FakeRun, failed, ok


def _row(*, run_id: str = "run-1", node: str = "lavender", outcome: str = "running") -> LedgerEntry:
    """Build a ledger row through its own decoder.

    Args:
        run_id: The dispatch.
        node: Its node.
        outcome: How it ended, or ``running``.

    Returns:
        The row.
    """
    return decode_ledger_entry(
        {
            "run_id": run_id,
            "node": node,
            "host": node,
            "project": "services/Model-Trainer",
            "agent": "opus-fleet-0904",
            "session_id": "acc774c0-3bc3-4cce-9dda-c7a12fb99519",
            "started_unix": 100,
            "ended_unix": 100,
            "outcome": outcome,
            "exit_code": -1,
            "workers": 6,
            "detail": "",
        }
    )


def _event(*, run_id: str = "run-1", kind: str = "started") -> FeedEvent:
    """Build a feed event.

    Args:
        run_id: The dispatch it belongs to.
        kind: What happened.

    Returns:
        The event, typed through the feed's own decoder.
    """
    return decode_feed_event(
        {
            "at_unix": 1,
            "run_id": run_id,
            "node": "lavender",
            "project": "services/Model-Trainer",
            "kind": kind,
            "detail": "",
        }
    )


class TestDefaultHooks:
    def test_run_executes_a_real_command_and_captures_output(self) -> None:
        result = _test_hooks._default_run([sys.executable, "-c", "print('hello')"])

        assert result["returncode"] == 0
        assert result["stdout"].strip() == "hello"

    def test_run_reports_a_non_zero_status_rather_than_raising(self) -> None:
        """check=False, so the caller decides what a failure means."""
        result = _test_hooks._default_run([sys.executable, "-c", "raise SystemExit(3)"])

        assert result["returncode"] == 3

    def test_run_feeds_stdin_through(self) -> None:
        result = _test_hooks._default_run(
            [sys.executable, "-c", "import sys; print(sys.stdin.read().strip())"],
            stdin_bytes=b"piped",
        )

        assert result["stdout"].strip() == "piped"

    def test_now_reads_whole_seconds_from_the_real_clock(self) -> None:
        """Whole rather than fractional, and moving forwards.

        A float would invite comparisons that differ in their last bit
        between two readers of one lease file, so the truncation is the
        contract rather than an implementation detail.
        """
        seconds = _test_hooks._default_now()

        assert seconds == int(seconds)
        assert seconds > 1_756_000_000
        assert _test_hooks._default_now() >= seconds

    def test_append_creates_the_parent_directory(self, tmp_path: pathlib.Path) -> None:
        """A workspace pointing at a fresh directory is the first-run case."""
        path = tmp_path / "nested" / "ledger.jsonl"

        _test_hooks._default_append_text(path, "first")
        _test_hooks._default_append_text(path, "second")

        assert path.read_text(encoding="utf-8") == "first\nsecond\n"

    def test_write_replaces_and_creates_the_parent(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "nested" / "leases.json"

        _test_hooks._default_write_text(path, "[]")
        _test_hooks._default_write_text(path, "[1]")

        assert path.read_text(encoding="utf-8") == "[1]"

    def test_read_text_returns_what_was_written(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "feed.jsonl"
        path.write_text("line", encoding="utf-8")

        assert _test_hooks._default_read_text(path) == "line"

    def test_file_exists_distinguishes_a_file_from_absence(self, tmp_path: pathlib.Path) -> None:
        """The record files are created by their first write.

        So "absent" is the ordinary first-run state and has to be
        distinguishable from "present and empty" without reading anything.
        """
        present = tmp_path / "ledger.jsonl"
        present.write_text("", encoding="utf-8")

        assert _test_hooks._default_file_exists(present)
        assert not _test_hooks._default_file_exists(tmp_path / "absent.jsonl")

    def test_a_directory_is_not_a_file(self, tmp_path: pathlib.Path) -> None:
        """A workspace pointing its ledger at a directory reads as absent here.

        The failure then comes from the write, with its own message, rather
        than from an invented one at the read.
        """
        assert not _test_hooks._default_file_exists(tmp_path)


class TestLedgerRecords:
    def test_an_absent_ledger_is_empty(self, tmp_path: pathlib.Path) -> None:

        assert records.read_ledger(tmp_path / "ledger.jsonl") == ()

    def test_rows_round_trip_in_append_order(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "ledger.jsonl"
        records.append_ledger(path, _row(run_id="a"))
        records.append_ledger(path, _row(run_id="b"))

        assert [row["run_id"] for row in records.read_ledger(path)] == ["a", "b"]

    def test_a_blank_line_is_skipped(self, tmp_path: pathlib.Path) -> None:
        """A trailing newline is normal; nothing else is."""
        path = tmp_path / "ledger.jsonl"
        records.append_ledger(path, _row())
        path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

        assert len(records.read_ledger(path)) == 1

    def test_a_line_that_is_not_an_object_is_fatal(self, tmp_path: pathlib.Path) -> None:
        """Skipping it would make a running dispatch invisible.

        The next capacity check would then admit work onto a node that is
        already full, which is the failure the package exists to prevent.
        """
        path = tmp_path / "ledger.jsonl"
        path.write_text("[1, 2]\n", encoding="utf-8")

        with pytest.raises(AppError) as excinfo:
            records.read_ledger(path)

        assert excinfo.value.code is FleetErrorCode.LEDGER_ROW_UNPARSABLE
        assert "line 1" in excinfo.value.message

    def test_live_runs_counts_only_running_rows_on_that_node(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "ledger.jsonl"
        records.append_ledger(path, _row(run_id="a", node="lavender", outcome="running"))
        records.append_ledger(path, _row(run_id="b", node="lavender", outcome="passed"))
        records.append_ledger(path, _row(run_id="c", node="loki", outcome="running"))

        assert records.live_runs(path, node="lavender") == 1
        assert records.live_runs(path, node="loki") == 1
        assert records.live_runs(path, node="sedona") == 0


class TestFeedRecords:
    def test_an_absent_feed_is_empty(self, tmp_path: pathlib.Path) -> None:

        assert records.read_feed(tmp_path / "feed.jsonl") == ()

    def test_events_round_trip_in_append_order(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "feed.jsonl"
        records.append_feed(path, _event(run_id="a"))
        records.append_feed(path, _event(run_id="b", kind="passed"))

        read = records.read_feed(path)
        assert [event["run_id"] for event in read] == ["a", "b"]
        assert read[1]["kind"] == "passed"

    def test_a_line_that_is_not_an_object_is_fatal(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "feed.jsonl"
        path.write_text('"started"\n', encoding="utf-8")

        with pytest.raises(AppError) as excinfo:
            records.read_feed(path)

        assert excinfo.value.code is FleetErrorCode.FEED_EVENT_UNPARSABLE


class TestRemote:
    def test_a_command_returns_its_stdout(self) -> None:
        runner = FakeRun([ok("done")])
        _test_hooks.run = runner

        assert remote.run_ssh("lavender", ("echo", "hi")) == "done"
        assert runner.calls[0][0] == "ssh"
        assert "BatchMode=yes" in runner.calls[0]
        assert runner.calls[0][-2:] == ("echo", "hi")

    def test_ssh_failing_to_reach_the_node_is_its_own_code(self) -> None:
        """255 is ssh's own status, and the fix is the tailnet not the work."""
        _test_hooks.run = FakeRun([failed(255, "Connection timed out")])

        with pytest.raises(AppError) as excinfo:
            remote.run_ssh("pendragon", ("echo", "hi"))

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert "Connection timed out" in excinfo.value.message

    def test_a_remote_command_failing_is_a_dispatch_failure(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "make: *** [check] Error 2")])

        with pytest.raises(AppError) as excinfo:
            remote.run_ssh("lavender", ("make", "check"))

        assert excinfo.value.code is FleetErrorCode.DISPATCH_FAILED
        assert "Error 2" in excinfo.value.message

    def test_a_failure_with_no_stderr_still_says_so(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "")])

        with pytest.raises(AppError, match="<no stderr>"):
            remote.run_ssh("lavender", ("make", "check"))

    def test_a_script_body_is_streamed_over_stdin(self) -> None:
        """Never an argument, so no shell between here and the disk sees it."""
        runner = FakeRun([ok("")])
        _test_hooks.run = runner

        remote.send_script("lavender", "C:/tmp/probe.ps1", "Write-Host 'hi'")

        assert runner.stdin[0] == b"Write-Host 'hi'"
        assert "Set-Content" in runner.calls[0][-1]
        assert "C:/tmp/probe.ps1" in runner.calls[0][-1]

    def test_an_unreachable_node_during_send_says_so(self) -> None:
        _test_hooks.run = FakeRun([failed(255, "no route to host")])

        with pytest.raises(AppError) as excinfo:
            remote.send_script("pendragon", "C:/tmp/probe.ps1", "x")

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE

    def test_a_write_that_fails_is_a_dispatch_failure(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "access denied")])

        with pytest.raises(AppError) as excinfo:
            remote.send_script("lavender", "C:/tmp/probe.ps1", "x")

        assert excinfo.value.code is FleetErrorCode.DISPATCH_FAILED
        assert "access denied" in excinfo.value.message

    def test_run_script_sends_then_runs_by_path(self) -> None:
        """The bytes that run are the bytes that were sent."""
        runner = FakeRun([ok(""), ok("output")])
        _test_hooks.run = runner

        assert remote.run_script("lavender", "C:/tmp/p.ps1", "body") == "output"
        assert runner.stdin[0] == b"body"
        assert runner.calls[1][-6:-1] == remote.POWERSHELL_INVOCATION
        assert runner.calls[1][-1] == "C:/tmp/p.ps1"


def _node() -> NodeConfig:
    """Build a node for the probe tests.

    Returns:
        The node.
    """
    return NodeConfig(
        host="lavender",
        stage_root="C:/fleet/stage",
        logical_cores=16,
        ram_gb=32.0,
        gpu=None,
        budget=NodeBudget(
            reserved_cores=2,
            reserved_ram_gb=4.0,
            worker_ram_gb=1.1,
            max_concurrent_runs=2,
            max_disk_gb=20.0,
        ),
    )


class TestProbe:
    def test_it_reads_the_fields_the_script_emits(self) -> None:
        output = "free_ram_gb=27.395\nfree_disk_gb=860.123\nlogical_cores=16\n"

        state = probe.parse_probe("lavender", output, live_runs=2)

        assert state == {
            "host": "lavender",
            "free_ram_gb": 27.395,
            "free_disk_gb": 860.123,
            "live_runs": 2,
        }

    def test_a_thousands_separator_is_read(self) -> None:
        """PowerShell's N3 format writes them; the value is still a number."""
        output = "free_ram_gb=1,027.395\nfree_disk_gb=860.000\n"

        assert probe.parse_probe("lavender", output, live_runs=0)["free_ram_gb"] == 1027.395

    def test_a_line_without_an_equals_is_ignored(self) -> None:
        """PowerShell writes warnings to the same stream."""
        output = "WARNING: something\nfree_ram_gb=1.0\nfree_disk_gb=2.0\n"

        assert probe.parse_probe("lavender", output, live_runs=0)["free_ram_gb"] == 1.0

    def test_a_missing_field_names_itself(self) -> None:
        with pytest.raises(AppError) as excinfo:
            probe.parse_probe("lavender", "free_ram_gb=1.0\n", live_runs=0)

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert "free_disk_gb" in excinfo.value.message

    def test_a_non_numeric_field_shows_what_the_node_said(self) -> None:
        """The usual cause is a PowerShell error printed where a number goes."""
        output = "free_ram_gb=Cannot find drive\nfree_disk_gb=1.0\n"

        with pytest.raises(AppError) as excinfo:
            probe.parse_probe("lavender", output, live_runs=0)

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert "Cannot find drive" in excinfo.value.message

    def test_a_field_with_two_decimal_points_is_not_a_number(self) -> None:
        with pytest.raises(AppError, match="not a number"):
            probe.parse_probe("lavender", "free_ram_gb=1.2.3\nfree_disk_gb=1.0\n", live_runs=0)

    def test_a_signed_value_is_a_number(self) -> None:
        state = probe.parse_probe("lavender", "free_ram_gb=-1.0\nfree_disk_gb=+2.0\n", live_runs=0)

        assert state["free_ram_gb"] == -1.0
        assert state["free_disk_gb"] == 2.0

    def test_a_bare_sign_is_not_a_number(self) -> None:
        with pytest.raises(AppError, match="not a number"):
            probe.parse_probe("lavender", "free_ram_gb=-\nfree_disk_gb=1.0\n", live_runs=0)

    def test_probing_a_node_sends_the_script_and_parses_the_answer(self) -> None:
        runner = FakeRun([ok(""), ok("free_ram_gb=27.0\nfree_disk_gb=860.0\n")])
        _test_hooks.run = runner

        state = probe.probe_node(_node(), live_runs=1)

        assert state["free_ram_gb"] == 27.0
        assert state["live_runs"] == 1
        assert runner.stdin[0] == probe.PROBE_SCRIPT.encode("utf-8")

    def test_the_script_arrives_byte_identical_to_the_constant(self) -> None:
        """THE RENDER-AND-SEND RULE, asserted rather than described.

        The bytes on the wire are the constant itself, so no value this
        package holds can carry a quote into a shell. The braces inside it
        are PowerShell's own format operator and are evaluated on the far
        side -- Python never touches them, which is exactly what this
        equality proves.
        """
        runner = FakeRun([ok(""), ok("free_ram_gb=1.0\nfree_disk_gb=1.0\n")])
        _test_hooks.run = runner

        probe.probe_node(_node(), live_runs=0)

        assert runner.stdin[0] == probe.PROBE_SCRIPT.encode("utf-8")
        assert "{0:N3}" in probe.PROBE_SCRIPT
