"""The append-only records, the ssh seam, and the hooks.

The node probe's tests are in ``test_probe.py``, split from here at the
600-line ceiling.

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
from platform_core.config import config_test_hooks
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.feed import FeedEvent, decode_feed_event
from fleet.contracts.ledger import LedgerEntry, decode_ledger_entry
from fleet.contracts.node import LiveLoad, NodePlatform
from fleet.contracts.project import ProjectConfig
from fleet.core import _test_hooks, dialect_windows, records, remote
from tests.conftest import FakeRun, failed, ok, timed_out

#: The one project the ledger rows below run, at a worker cost that sums exactly.
_PROJECTS = {
    "services/Model-Trainer": ProjectConfig(
        worker_ram_gb=1.5,
        minimum_workers=2,
        expected_minutes=5,
        exclusive_resources=(),
        external_paths=(),
        required_tags=(),
        source=None,
    )
}


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


#: A deadline no test command here comes near, so a real command's result
#: is about the command and never about the clock.
GENEROUS_SECONDS = 60


class TestDefaultHooks:
    def test_run_executes_a_real_command_and_captures_output(self) -> None:
        result = _test_hooks._default_run(
            [sys.executable, "-c", "print('hello')"], timeout_seconds=GENEROUS_SECONDS
        )

        assert result["returncode"] == 0
        assert result["stdout"].strip() == "hello"
        assert result["timed_out"] is False

    def test_run_reports_a_non_zero_status_rather_than_raising(self) -> None:
        """check=False, so the caller decides what a failure means."""
        result = _test_hooks._default_run(
            [sys.executable, "-c", "raise SystemExit(3)"], timeout_seconds=GENEROUS_SECONDS
        )

        assert result["returncode"] == 3
        assert result["timed_out"] is False

    def test_run_feeds_stdin_through(self) -> None:
        result = _test_hooks._default_run(
            [sys.executable, "-c", "import sys; print(sys.stdin.read().strip())"],
            timeout_seconds=GENEROUS_SECONDS,
            stdin_bytes=b"piped",
        )

        assert result["stdout"].strip() == "piped"

    def test_run_gives_a_child_no_bytes_a_closed_stdin(self) -> None:
        """A child that reads stdin sees EOF at once, never this process's
        own handle: under a scheduled task that handle is what a remote shell
        waited on forever (board task 35940277, A2)."""
        result = _test_hooks._default_run(
            [sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"],
            timeout_seconds=GENEROUS_SECONDS,
        )

        assert result["returncode"] == 0
        assert result["stdout"].strip() == "''"

    def test_run_ends_a_command_at_its_deadline_and_says_so(self) -> None:
        """A real child that would sleep a minute is ended after one second
        and reported as timed out, its stderr carrying the elapsed bound so
        a caller that prints only stderr still says why it stopped."""
        result = _test_hooks._default_run(
            [
                sys.executable,
                "-c",
                "import sys, time; sys.stderr.write('still working'); "
                "sys.stderr.flush(); time.sleep(60)",
            ],
            timeout_seconds=1,
        )

        assert result["timed_out"] is True
        assert result["returncode"] == _test_hooks.TIMED_OUT_RETURNCODE == -1
        assert result["stderr"].endswith("timed out after 1 s")
        assert result["stdout"] == ""

    def test_run_at_its_deadline_with_nothing_on_stderr_carries_only_the_bound(self) -> None:
        result = _test_hooks._default_run(
            [sys.executable, "-c", "import time; time.sleep(60)"], timeout_seconds=1
        )

        assert result["timed_out"] is True
        assert result["stderr"] == "timed out after 1 s"

    def test_run_withholds_named_variables_and_passes_every_other_one(self) -> None:
        """A real child, asked which of two variables it can see.

        PATH is the probe because every parent has it, so the test needs no
        environment of its own: withheld, the child reports it absent;
        not withheld, present. The second value proves the rest of the
        environment still arrives rather than the child starting empty.
        """
        probe = [
            sys.executable,
            "-c",
            "import os; print('PATH' in os.environ, len(os.environ) > 1)",
        ]
        withheld = _test_hooks._default_run(
            probe, timeout_seconds=GENEROUS_SECONDS, unset_env=("PATH",)
        )
        inherited = _test_hooks._default_run(probe, timeout_seconds=GENEROUS_SECONDS)

        assert withheld["returncode"] == 0
        assert withheld["stdout"].split() == ["False", "True"]
        assert inherited["stdout"].split() == ["True", "True"]

    def test_a_python_child_logs_text_outside_its_code_page_and_it_arrives_whole(self) -> None:
        """A real child writes the queue's em dash through a real logging handler.

        The shape that failed on 2026-09-29 (MCPs board task 88b8fe61): a
        Python child on Windows wrote its pipes in cp1252, so the dash
        arrived here as U+FFFD, and a child logging a character its stream
        could not encode printed ``--- Logging error ---`` instead of the
        line. Both streams must come back as the child meant them.
        """
        dash = "—"
        probe = [
            sys.executable,
            "-c",
            "import logging, sys; logging.basicConfig(stream=sys.stderr); "
            f"logging.getLogger('probe').warning('stderr {dash} U+FFFD \\ufffd'); "
            f"print('stdout {dash}')",
        ]

        result = _test_hooks._default_run(probe, timeout_seconds=GENEROUS_SECONDS)

        assert result["returncode"] == 0
        assert result["stdout"].strip() == f"stdout {dash}"
        assert result["stderr"].strip() == f"WARNING:probe:stderr {dash} U+FFFD �"

    def test_run_sets_named_variables_after_withholding_and_keeps_the_rest(self) -> None:
        """A real child reads back what it was given (MCPs board task f4cd489f).

        A set name reaches the child with its value, a name both withheld and
        set arrives with the set value because setting is applied second, and
        PATH, neither withheld nor set, still arrives.
        """
        probe = [
            sys.executable,
            "-c",
            "import os; print(os.environ['FLEET_PROBE'], os.environ['FLEET_BOTH'], "
            "'PATH' in os.environ)",
        ]
        result = _test_hooks._default_run(
            probe,
            timeout_seconds=GENEROUS_SECONDS,
            unset_env=("FLEET_BOTH",),
            set_env=(("FLEET_PROBE", "published"), ("FLEET_BOTH", "second")),
        )

        assert result["returncode"] == 0, result["stderr"]
        assert result["stdout"].split() == ["published", "second", "True"]

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

    def test_the_real_environment_reader_normalises_blank_to_unset(self) -> None:
        """It delegates to the monorepo's one permitted environment reader.

        Rebinding THAT reader's own hook rather than setting a real variable
        is what keeps this package from growing a second ``os.environ``
        access -- the ``env`` guard rule names ``platform_core.config``
        explicitly rather than exempting anyone -- and it exercises the
        delegation rather than assuming it.

        The blank case is the one that matters: an exported-but-empty
        credential must read as unset, or a blank api key reaches the queue
        and the failure arrives as a 401 from a server instead of a named
        refusal here.
        """
        config_test_hooks.get_env = {"SET": "present", "BLANK": "   "}.get

        assert _test_hooks._default_env("SET") == "present"
        assert _test_hooks._default_env("BLANK") is None
        assert _test_hooks._default_env("ABSENT") is None

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

    def test_live_load_sums_only_running_rows_on_that_node(self, tmp_path: pathlib.Path) -> None:
        """Each live run's 6 granted workers at the project's 1.5 GB."""
        path = tmp_path / "ledger.jsonl"
        records.append_ledger(path, _row(run_id="a", node="lavender", outcome="running"))
        records.append_ledger(path, _row(run_id="b", node="lavender", outcome="passed"))
        records.append_ledger(path, _row(run_id="c", node="loki", outcome="running"))
        records.append_ledger(path, _row(run_id="d", node="loki", outcome="running"))

        assert records.live_load(path, node="lavender", projects=_PROJECTS) == LiveLoad(
            runs=1, workers=6, ram_gb=9.0
        )
        assert records.live_load(path, node="loki", projects=_PROJECTS) == LiveLoad(
            runs=2, workers=12, ram_gb=18.0
        )
        assert records.live_load(path, node="sedona", projects=_PROJECTS) == LiveLoad(
            runs=0, workers=0, ram_gb=0.0
        )

    def test_a_live_run_of_an_unregistered_project_is_refused(self, tmp_path: pathlib.Path) -> None:
        """What its workers hold is unknown, and guessing low overloads the node."""
        path = tmp_path / "ledger.jsonl"
        records.append_ledger(path, _row(run_id="a", node="lavender", outcome="running"))

        with pytest.raises(AppError) as excinfo:
            records.live_load(path, node="lavender", projects={})

        assert excinfo.value.code is FleetErrorCode.WORKSPACE_PROJECT_UNKNOWN
        assert excinfo.value.message.startswith(
            "live run a on lavender is of services/Model-Trainer, which fleet.json no longer"
        )


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

    def test_every_ssh_carries_the_keepalive_and_a_deadline(self) -> None:
        """The two bounds a dead or silent peer cannot escape (board tasks
        35940277 and 41ac6ed2): ssh's own keepalive for a peer that went
        away, and the seam's deadline for one that answers and never
        finishes. Both the run and the send carry them."""
        runner = FakeRun([ok(""), ok("")])
        _test_hooks.run = runner

        remote.run_ssh("pendragon", ("hostname",))
        remote.send_script(
            "pendragon", "C:/tmp/probe.ps1", "Get-Date", platform=NodePlatform.WINDOWS
        )

        for call in runner.calls:
            # Every `-o Key=Value` pair the invocation carries, by key.
            settings = [call[index + 1] for index, word in enumerate(call) if word == "-o"]
            options = dict(setting.split("=", 1) for setting in settings)
            assert options["ServerAliveInterval"] == "15"
            assert options["ServerAliveCountMax"] == "4"
            assert options["ConnectTimeout"] == "10"
        assert runner.timeouts == [remote.SSH_TIMEOUT_SECONDS] * 2 == [120, 120]

    def test_a_peer_that_stops_answering_is_unreachable_with_the_elapsed_seconds(self) -> None:
        _test_hooks.run = FakeRun([timed_out(120)])

        with pytest.raises(AppError) as excinfo:
            remote.run_ssh("pendragon", ("powershell", "-File", "observe.ps1"))

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert excinfo.value.message == (
            "ssh to pendragon timed out while running `powershell -File observe.ps1`: "
            "timed out after 120 s"
        )

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

        remote.send_script(
            "lavender", "C:/tmp/probe.ps1", "Write-Host 'hi'", platform=NodePlatform.WINDOWS
        )

        assert runner.stdin[0] == b"Write-Host 'hi'"
        assert "Set-Content" in runner.calls[0][-1]
        assert "C:/tmp/probe.ps1" in runner.calls[0][-1]

    def test_a_linux_node_is_written_through_mkdir_and_cat(self) -> None:
        """The same act in the other dialect: one argument for the remote
        login shell, the parent made in the same round trip."""
        runner = FakeRun([ok("")])
        _test_hooks.run = runner

        remote.send_script(
            "diphtheria", "/home/c/stage/probe.sh", "echo hi", platform=NodePlatform.LINUX
        )

        assert runner.stdin[0] == b"echo hi"
        assert runner.calls[0][-1] == (
            "mkdir -p \"$(dirname '/home/c/stage/probe.sh')\" && cat > '/home/c/stage/probe.sh'"
        )

    def test_an_unreachable_node_during_send_says_so(self) -> None:
        _test_hooks.run = FakeRun([failed(255, "no route to host")])

        with pytest.raises(AppError) as excinfo:
            remote.send_script("pendragon", "C:/tmp/probe.ps1", "x", platform=NodePlatform.WINDOWS)

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE

    def test_a_write_that_fails_is_a_dispatch_failure(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "access denied")])

        with pytest.raises(AppError) as excinfo:
            remote.send_script("lavender", "C:/tmp/probe.ps1", "x", platform=NodePlatform.WINDOWS)

        assert excinfo.value.code is FleetErrorCode.DISPATCH_FAILED
        assert "access denied" in excinfo.value.message

    def test_run_script_sends_then_runs_by_path(self) -> None:
        """The bytes that run are the bytes that were sent."""
        runner = FakeRun([ok(""), ok("output")])
        _test_hooks.run = runner

        assert (
            remote.run_script("lavender", "C:/tmp/p.ps1", "body", platform=NodePlatform.WINDOWS)
            == "output"
        )
        assert runner.stdin[0] == b"body"
        assert runner.calls[1][-6:-1] == dialect_windows.POWERSHELL_INVOCATION
        assert runner.calls[1][-1] == "C:/tmp/p.ps1"

    def test_run_script_on_a_linux_node_runs_through_bin_sh(self) -> None:
        runner = FakeRun([ok(""), ok("output")])
        _test_hooks.run = runner

        assert (
            remote.run_script("diphtheria", "/s/p.sh", "body", platform=NodePlatform.LINUX)
            == "output"
        )
        assert runner.calls[1][-2:] == ("/bin/sh", "/s/p.sh")
