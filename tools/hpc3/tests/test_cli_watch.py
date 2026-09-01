"""Tests for the watch CLI: what a job is doing, and what it charged.

Split out of ``test_cli.py`` when that file passed the 600-line ceiling. The
split is by role rather than by size: watch is the only command handed job
IDs instead of a run document, which is what makes its budget lookup
interesting and is why the two-project cases live here.
"""

from __future__ import annotations

import pathlib
import time
from collections.abc import Callable

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue, dump_json_str

from hpc3.cli import _test_hooks
from hpc3.cli import submit as submit_cli
from hpc3.cli import watch as watch_cli
from tests.against_hpc3 import decode_job_status
from tests.conftest import (
    FakeRun,
    budget_document,
    cluster,
    project_config,
    script_healthy_cluster,
    workspace_document,
    write_file,
    write_workspace,
)

_ROW = (
    "55519937|abl.verify|free-gpu32|COMPLETED|48|"
    "billing=11,cpu=11,gres/gpu=1,mem=64G,node=1|hpc3-gpu-n54-00"
)


def _write_json(path: pathlib.Path, payload: JSONValue) -> None:
    """Write a JSON document for the CLI to read.

    Args:
        path: File to write.
        payload: Document to serialise.
    """
    write_file(path, dump_json_str(payload).encode("utf-8"))


def _run_payload(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid run document.

    Args:
        **overrides: Fields to add or replace.

    Returns:
        The document.
    """
    document: dict[str, JSONValue] = {
        "project": "abl",
        "name": "arm-b-42",
        "command": "python train.py",
        "artifact": None,
        "experiment": {"arm": "B", "seed": "42"},
    }
    document.update(overrides)
    return document


def _config(tmp_path: pathlib.Path, **overrides: JSONValue) -> str:
    """Write a workspace and return its path.

    Args:
        tmp_path: Directory holding the documents.
        **overrides: Workspace fields to replace.

    Returns:
        The path, ready to pass as ``--config``.
    """
    return write_workspace(tmp_path / "hpc3.json", workspace_document(**overrides))


def _watch_args(tmp_path: pathlib.Path, jobs: str, *, gpu_hours: float = 100.0) -> list[str]:
    """Build a full watch argument list.

    Args:
        tmp_path: Directory holding the workspace.
        jobs: The ``--job`` value.
        gpu_hours: GPU-hour cap for the project's budget.

    Returns:
        Arguments excluding the program name.
    """
    budget = budget_document(gpu_hours=gpu_hours, units=100.0)
    return [
        "--config",
        _config(tmp_path, projects={"abl": project_config(budget=budget)}),
        "--job",
        jobs,
    ]


class TestWatchCli:
    def test_it_reports_state_and_real_cost(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        fake_run.add("sacct", stdout=_ROW + "\n")

        assert watch_cli.main(_watch_args(tmp_path, "55519937")) == 0
        assert emitted[0].startswith("final 55519937 abl.verify COMPLETED 48s 0.0000 SU")
        assert emitted[1] == "total 0.0000 SU across 1 row(s)"
        assert emitted[3] == "states COMPLETED=1"

    def test_a_row_from_a_billing_partition_reports_its_real_cost(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """Watch reads accounting, which reports whatever ran under the
        account -- not only what this package would have submitted. billing=11
        for 48s at UsageFactor 1.0 is 0.1467 SU."""
        billed = _ROW.replace("|free-gpu32|", "|gpu32|")
        fake_run.add("sacct", stdout=billed + "\n")

        assert watch_cli.main(_watch_args(tmp_path, "55519937")) == 0
        assert emitted[0].startswith("final 55519937 abl.verify COMPLETED 48s 0.1467 SU")
        assert emitted[1] == "total 0.1467 SU across 1 row(s)"

    def test_it_reports_gpu_hours(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        fake_run.add("sacct", stdout=_ROW + "\n")
        watch_cli.main(_watch_args(tmp_path, "55519937"))
        assert emitted[2] == "gpu-hours 0.01"

    def test_a_job_from_another_project_is_checked_against_that_project(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """Two projects, two caps, one sacct call.

        ``abl`` declares 100 GPU-hours and ``sirius`` declares 0.001, so a
        row belonging to each passes and fails respectively. Before the cap
        moved onto the project, whichever single workspace cap was passed
        decided both -- which is how watching an `mi` job with the cleargbm
        document enforced 0.5 GPU-hours on work submitted under 12.0.
        """
        projects: JSONValue = {
            "abl": project_config(budget=budget_document(gpu_hours=100.0)),
            "sirius": project_config(budget=budget_document(gpu_hours=0.001)),
        }
        fake_run.add(
            "sacct",
            stdout=_ROW + "\n55519938|sirius.batch7|free-gpu32|COMPLETED|48|gres/gpu=1|n1\n",
        )

        with pytest.raises(AppError) as excinfo:
            watch_cli.main(
                ["--config", _config(tmp_path, projects=projects), "--job", "55519937,55519938"]
            )
        assert excinfo.value.code is Hpc3ErrorCode.BUDGET_CONSUMPTION_EXCEEDED
        assert "budget OK abl" in emitted

    def test_a_job_this_workspace_never_submitted_is_named_not_judged(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """A bare job name carries no project, so no declared cap governs it.

        Reported rather than checked against whichever cap happened to be
        loaded: on a cluster 102 people share, a row accounting returns is
        not necessarily ours, and judging it against our cap invents a
        finding.
        """
        fake_run.add("sacct", stdout="1|arm|free-gpu|COMPLETED|10|billing=4,gres/gpu=1|n1\n")

        assert watch_cli.main(_watch_args(tmp_path, "1", gpu_hours=0.001)) == 0
        assert emitted[-1] == "NO DECLARED BUDGET 1"

    def test_an_overrun_is_reported_after_the_rows(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """The rows are worth seeing whether or not the budget was broken."""
        fake_run.add("sacct", stdout=_ROW + "\n")

        with pytest.raises(AppError) as excinfo:
            watch_cli.main(_watch_args(tmp_path, "55519937", gpu_hours=0.001))
        assert excinfo.value.code is Hpc3ErrorCode.BUDGET_CONSUMPTION_EXCEEDED
        assert emitted[0].startswith("final 55519937")

    def test_the_cap_it_enforces_is_the_one_the_submitter_projected_against(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """One document, so the two commands cannot disagree about the ceiling."""
        _write_json(tmp_path / "run.json", _run_payload())
        script_healthy_cluster(fake_run)
        config = _config(
            tmp_path, projects={"abl": project_config(budget=budget_document(gpu_hours=0.001))}
        )

        with pytest.raises(AppError) as submitted:
            submit_cli.main(["--config", config, "--run", str(tmp_path / "run.json")])
        assert submitted.value.code is Hpc3ErrorCode.BUDGET_PROJECTION_EXCEEDED

        fake_run.add("sacct", stdout=_ROW + "\n")
        with pytest.raises(AppError) as watched:
            watch_cli.main(["--config", config, "--job", "55519937"])
        assert watched.value.code is Hpc3ErrorCode.BUDGET_CONSUMPTION_EXCEEDED

    def test_a_pending_job_is_marked_live_with_no_node(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        fake_run.add("sacct", stdout="1|arm|free-gpu|PENDING|0||\n")
        watch_cli.main(_watch_args(tmp_path, "1"))
        assert emitted[0].startswith("live  1 arm PENDING 0s")
        assert emitted[0].endswith("@ -")

    def test_an_unknown_job_is_a_failure_not_an_empty_report(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        fake_run.add("sacct", stdout="")
        with pytest.raises(ValueError, match="knows no job"):
            watch_cli.main(_watch_args(tmp_path, "999"))
        assert emitted == []

    def test_the_config_flag_is_not_optional(self) -> None:
        with pytest.raises(ValueError, match="--config is required"):
            watch_cli.main(["--job", "1"])


class TestFormatStatus:
    def test_a_terminal_row_is_marked_final(self) -> None:
        status = decode_job_status(
            {
                "job_id": "1",
                "name": "arm",
                "partition": "free-gpu",
                "state": "COMPLETED",
                "elapsed_seconds": 1,
                "billing_tres": 0,
                "gpu_count": 1,
                "cpu_count": 8,
                "node_list": "n1",
            }
        )
        assert watch_cli.format_status(status, cluster()).startswith("final ")

    def test_a_requeued_row_is_marked_live_because_protection_worked(self) -> None:
        status = decode_job_status(
            {
                "job_id": "1",
                "name": "arm",
                "partition": "free-gpu",
                "state": "REQUEUED",
                "elapsed_seconds": 1,
                "billing_tres": 0,
                "gpu_count": 1,
                "cpu_count": 8,
                "node_list": "n1",
            }
        )
        assert watch_cli.format_status(status, cluster()).startswith("live  ")


class TestMultiJobWatch:
    def test_one_sacct_call_covers_the_whole_sweep(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """Six separate calls would observe six different moments."""
        rows = "\n".join(
            f"10{i}|rung-s{i}|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|node{i}" for i in range(3)
        )
        fake_run.add("sacct", stdout=rows + "\n")

        code = watch_cli.main(_watch_args(tmp_path, "101,102,103", gpu_hours=10.0))
        assert code == 0
        assert len(fake_run.calls) == 1
        assert "sacct -j 101,102,103" in fake_run.calls[0].remote_command

    def test_it_tallies_states_across_the_sweep(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        rows = (
            "101|a|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|n1\n"
            "102|b|free-gpu|RUNNING|30|billing=8,gres/gpu=1|n2\n"
            "103|c|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|n3\n"
        )
        fake_run.add("sacct", stdout=rows)
        watch_cli.main(_watch_args(tmp_path, "101,102,103", gpu_hours=10.0))
        assert emitted[5] == "states COMPLETED=2 RUNNING=1"

    def test_an_id_accounting_does_not_know_is_named_not_hidden(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        fake_run.add("sacct", stdout="101|a|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|n1\n")
        watch_cli.main(_watch_args(tmp_path, "101,999", gpu_hours=10.0))
        assert emitted[4] == "NOT FOUND 999"

    def test_trailing_commas_are_ignored(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        fake_run.add("sacct", stdout="101|a|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|n1\n")
        watch_cli.main(_watch_args(tmp_path, "101,,", gpu_hours=10.0))
        assert "sacct -j 101 " in fake_run.calls[0].remote_command

    def test_a_job_argument_of_only_commas_is_refused(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        with pytest.raises(ValueError, match="at least one job id"):
            watch_cli.main(_watch_args(tmp_path, ",,"))

    def test_a_non_positive_poll_cadence_is_refused(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        with pytest.raises(ValueError, match="--poll-seconds must be positive"):
            watch_cli.main([*_watch_args(tmp_path, "101"), "--poll-seconds", "0"])


class TestFollowMode:
    """The follow mode replaces the hand-rolled ssh polling loop per session.

    Six such loops were written in one day of panel-driving, and one of them
    declared a running panel drained on a shell quoting bug -- the failure
    class a tested flag exists to remove.
    """

    def _slept(self) -> tuple[list[float], Callable[[], None]]:
        """Swap the sleep hook for a recorder.

        Returns:
            The record of requested waits, and the restorer.
        """
        recorded: list[float] = []
        original = _test_hooks.sleep
        _test_hooks.sleep = recorded.append
        return recorded, lambda: setattr(_test_hooks, "sleep", original)

    def test_it_polls_to_terminal_and_emits_only_transitions(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        running = "55519937|abl.verify|free-gpu32|RUNNING|30|billing=11,gres/gpu=1|n1\n"
        fake_run.add("sacct", stdout=running, once=True)
        fake_run.add("sacct", stdout=running, once=True)
        fake_run.add("sacct", stdout=_ROW + "\n")
        slept, restore = self._slept()
        try:
            code = watch_cli.main(
                [*_watch_args(tmp_path, "55519937"), "--until-done", "1", "--poll-seconds", "5"]
            )
        finally:
            restore()
        assert code == 0
        status_lines = [line for line in emitted if line.startswith(("live ", "final "))]
        # Three reads, two states: the unchanged middle read emits nothing.
        assert len(status_lines) == 2
        assert "RUNNING" in status_lines[0]
        assert "COMPLETED" in status_lines[1]
        assert slept == [5, 5]
        assert emitted[-1] == "budget OK abl"

    def test_a_late_appearing_job_is_waited_for_and_said_so(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        fake_run.add("sacct", stdout="", once=True)
        fake_run.add("sacct", stdout=_ROW + "\n")
        slept, restore = self._slept()
        try:
            code = watch_cli.main([*_watch_args(tmp_path, "55519937"), "--until-done", "1"])
        finally:
            restore()
        assert code == 0
        assert emitted[0] == "accounting knows none of 1 job(s) yet; waiting"
        assert slept == [60]

    def test_ids_accounting_never_learns_are_ruled_wrong_not_waited_on_forever(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        fake_run.add("sacct", stdout="")
        slept, restore = self._slept()
        try:
            with pytest.raises(ValueError, match="sacct knows no job"):
                watch_cli.main([*_watch_args(tmp_path, "999"), "--until-done", "1"])
        finally:
            restore()
        # Four waits, then the fifth empty read raises rather than sleeping.
        assert slept == [60, 60, 60, 60]

    def test_the_default_sleep_hook_really_waits(self) -> None:
        """The production binding is a real wait, executed here against the
        clock -- everywhere else the suite swaps it, so without this the one
        line production runs would be the one line nothing ever ran. The
        floor sits under the request because Windows' timer granularity can
        return a few milliseconds early."""
        _test_hooks.reset_hooks()
        started = time.monotonic()
        _test_hooks.sleep(0.05)
        assert time.monotonic() - started >= 0.03

    def test_the_wait_holds_while_a_requested_id_is_still_absent(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """All rows terminal is not enough: a member accounting has not seen
        yet is still in flight, and returning without it would be the false
        drain this mode exists to prevent."""
        one_of_two = "101|abl.a|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|n1\n"
        both = one_of_two + "102|abl.b|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|n2\n"
        fake_run.add("sacct", stdout=one_of_two, once=True)
        fake_run.add("sacct", stdout=both)
        slept, restore = self._slept()
        try:
            code = watch_cli.main(
                [*_watch_args(tmp_path, "101,102", gpu_hours=10.0), "--until-done", "1"]
            )
        finally:
            restore()
        assert code == 0
        assert slept == [60]
        assert "states COMPLETED=2" in emitted
