"""Tests for cancellation.

``scancel`` is silent about a job that had already finished, so the whole
point of this layer is distinguishing "stopped your running job" from "did
nothing at all". Both look identical from the command's own output.
"""

from __future__ import annotations

import pathlib

import pytest

from hpc3.cli import cancel as cancel_cli
from hpc3.core.cancel import summarise
from tests.against_hpc3 import cancel
from tests.conftest import FakeRun, write_workspace


def _row(job_id: str, state: str, *, elapsed: int = 60) -> str:
    """Build one accounting row.

    Args:
        job_id: Job id.
        state: State to report.
        elapsed: Elapsed seconds.

    Returns:
        A pipe-delimited sacct row.
    """
    return f"{job_id}|arm|free-gpu|{state}|{elapsed}|billing=8,cpu=8,gres/gpu=1|n1"


class TestCancel:
    def test_a_running_job_is_reported_as_stopped(self, fake_run: FakeRun) -> None:
        """RUNNING before, CANCELLED after -- the real sequence, not one answer twice."""
        fake_run.add("sacct", stdout=_row("101", "RUNNING") + "\n", once=True)
        fake_run.add("sacct", stdout=_row("101", "CANCELLED") + "\n")
        outcomes = cancel("hpc3", ["101"])

        assert len(outcomes) == 1
        assert outcomes[0].was_running is True
        assert outcomes[0].state == "CANCELLED"

    def test_a_finished_job_is_not_claimed_as_stopped(self, fake_run: FakeRun) -> None:
        """The cancel succeeded and did nothing; saying otherwise gets believed."""
        fake_run.add("sacct", stdout=_row("101", "COMPLETED") + "\n")
        outcomes = cancel("hpc3", ["101"])

        assert outcomes[0].was_running is False
        assert outcomes[0].state == "COMPLETED"

    def test_a_pending_job_counts_as_live(self, fake_run: FakeRun) -> None:
        fake_run.add("sacct", stdout=_row("101", "PENDING", elapsed=0) + "\n", once=True)
        fake_run.add("sacct", stdout=_row("101", "CANCELLED", elapsed=0) + "\n")
        assert cancel("hpc3", ["101"])[0].was_running is True

    def test_it_reads_state_before_and_after_the_cancel(self, fake_run: FakeRun) -> None:
        fake_run.add("sacct", stdout=_row("101", "RUNNING") + "\n")
        cancel("hpc3", ["101"])

        commands = fake_run.commands()
        assert commands[0].startswith("sacct -j 101")
        assert commands[1] == "scancel 101"
        assert commands[2].startswith("sacct -j 101")

    def test_several_ids_are_cancelled_in_one_call(self, fake_run: FakeRun) -> None:
        rows = _row("101", "RUNNING") + "\n" + _row("102", "RUNNING") + "\n"
        fake_run.add("sacct", stdout=rows)
        cancel("hpc3", ["101", "102"])
        assert "scancel 101 102" in fake_run.commands()

    def test_no_ids_is_refused(self, fake_run: FakeRun) -> None:
        """A bare scancel takes every job the user has."""
        with pytest.raises(ValueError, match="at least one job id"):
            cancel("hpc3", [])
        assert fake_run.calls == []

    def test_a_job_accounting_never_knew_yields_no_outcome(self, fake_run: FakeRun) -> None:
        fake_run.add("sacct", stdout="")
        assert cancel("hpc3", ["999"]) == []


class TestSummarise:
    def test_it_separates_stopped_from_already_over(self, fake_run: FakeRun) -> None:
        before = _row("101", "RUNNING") + "\n" + _row("102", "COMPLETED") + "\n"
        after = _row("101", "CANCELLED") + "\n" + _row("102", "COMPLETED") + "\n"
        fake_run.add("sacct", stdout=before, once=True)
        fake_run.add("sacct", stdout=after)
        stopped, already_over = summarise(cancel("hpc3", ["101", "102"]))
        assert (stopped, already_over) == (1, 1)

    def test_no_outcomes_summarise_to_zero(self) -> None:
        assert summarise([]) == (0, 0)


class TestCancelCli:
    def _args(self, tmp_path: pathlib.Path, jobs: str) -> list[str]:
        """Build a full cancel argument list.

        Args:
            tmp_path: Directory holding the workspace.
            jobs: The ``--job`` value.

        Returns:
            Arguments excluding the program name.
        """
        return ["--config", write_workspace(tmp_path / "hpc3.json"), "--job", jobs]

    def test_it_names_what_each_job_became(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        before = _row("101", "RUNNING") + "\n" + _row("102", "COMPLETED") + "\n"
        after = _row("101", "CANCELLED") + "\n" + _row("102", "COMPLETED") + "\n"
        fake_run.add("sacct", stdout=before, once=True)
        fake_run.add("sacct", stdout=after)

        assert cancel_cli.main(self._args(tmp_path, "101,102")) == 0
        assert emitted == [
            "101 stopped CANCELLED",
            "102 already finished as COMPLETED",
            "stopped 1, already finished 1",
        ]

    def test_it_cancels_on_the_host_the_workspace_names(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """Cancelling on the wrong cluster silently leaves the job running."""
        fake_run.add("sacct", stdout=_row("101", "RUNNING") + "\n", once=True)
        fake_run.add("sacct", stdout=_row("101", "CANCELLED") + "\n")
        cancel_cli.main(self._args(tmp_path, "101"))
        assert [call.argv[-2] for call in fake_run.calls] == ["hpc3", "hpc3", "hpc3"]

    def test_an_unknown_job_is_a_failure_not_a_silent_success(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        fake_run.add("sacct", stdout="")
        with pytest.raises(ValueError, match="knows no job"):
            cancel_cli.main(self._args(tmp_path, "999"))
        assert emitted == []

    def test_an_empty_job_argument_is_refused(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        with pytest.raises(ValueError, match="at least one job id"):
            cancel_cli.main(self._args(tmp_path, ",,"))

    def test_a_missing_job_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--job is required"):
            cancel_cli.main(["--config", write_workspace(tmp_path / "hpc3.json")])

    def test_a_missing_config_flag_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--config is required"):
            cancel_cli.main(["--job", "101"])

    def test_the_entrypoint_reads_the_process_arguments(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], argv: list[str]
    ) -> None:
        fake_run.add("sacct", stdout=_row("101", "RUNNING") + "\n", once=True)
        fake_run.add("sacct", stdout=_row("101", "CANCELLED") + "\n")
        argv[:] = ["prog", *self._args(tmp_path, "101")]

        with pytest.raises(SystemExit) as excinfo:
            cancel_cli.entrypoint()
        assert excinfo.value.code == 0
        assert emitted[-1] == "stopped 1, already finished 0"
