"""Tests for the triage CLI: the whole reconciliation, end to end.

Split out of ``test_cli_sweep.py`` on 2026-09-06 when that file passed the
600-line ceiling. The two had nothing in common but a fixture: one exercises
submitting many members at once, the other exercises reconciling the ledger
against the cluster afterwards.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str

from hpc3.cli import submit as submit_cli
from hpc3.cli import triage as triage_cli
from hpc3.core import _test_hooks as core_hooks
from tests.conftest import (
    PREFLIGHT_LINE,
    FakeRun,
    ledger_row,
    project_config,
    workspace_document,
    write_file,
    write_workspace,
)


class TestTriageCli:
    def _config(self, tmp_path: pathlib.Path, **overrides: JSONValue) -> list[str]:
        """Build a triage argument list.

        Args:
            tmp_path: Directory holding the workspace.
            **overrides: Workspace fields to replace.

        Returns:
            Arguments excluding the program name.
        """
        config = write_workspace(tmp_path / "hpc3.json", workspace_document(**overrides))
        return ["--config", config]

    def _ledger(self, tmp_path: pathlib.Path, job_id: str = "101") -> None:
        """Write a one-entry ledger where the workspace says it lives.

        Args:
            tmp_path: Directory to write in.
            job_id: Job id to record.
        """
        write_file(
            tmp_path / "ledger.jsonl",
            dump_json_str(ledger_row(job_id=job_id, name="abl.arm")).encode("utf-8") + b"\n",
        )

    def test_an_empty_ledger_is_clean(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        assert triage_cli.main(self._config(tmp_path)) == 0
        assert "ledger is empty" in emitted[0]

    def test_it_reads_the_same_ledger_submit_wrote(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        """The condition a --ledger flag would let you get wrong."""
        write_file(
            tmp_path / "run.json",
            dump_json_str(
                {
                    "project": "abl",
                    "name": "arm-b-42",
                    "command": "python train.py",
                    "artifact": None,
                    "experiment": {"arm": "B"},
                }
            ).encode("utf-8"),
        )
        config = write_workspace(tmp_path / "hpc3.json", workspace_document())
        fake_run.add("test -d", stdout="PRESENT\n")
        fake_run.add("--test-only", stdout=PREFLIGHT_LINE + "\nrc=0\n")
        fake_run.add("sbatch", stdout="Submitted batch job 55519937\n")
        submit_cli.main(["--config", config, "--run", str(tmp_path / "run.json")])

        fake_run.add("sacct", stdout="55519937|abl.arm-b-42|free-gpu|RUNNING|60|billing=8|n1\n")
        fake_run.add("squeue --me", stdout="55519937|abl.arm-b-42|RUNNING\n")
        fake_run.add("squeue", stdout="")
        fake_run.add("date +%s", stdout="now 1000\n55519937 990\n")

        assert triage_cli.main(["--config", config]) == 0
        assert emitted[-1] == (
            "1 recorded, 1 on the cluster, 1 open, 1 not finished, 0 finding(s), 0 newly closed"
        )

    def _array_ledger(self, tmp_path: pathlib.Path, base: str = "55765275", count: int = 6) -> None:
        """Write a ledger holding every task of one array, as submit records them.

        Args:
            tmp_path: Directory to write in.
            base: The array's base job id.
            count: How many tasks to record.
        """
        rows = [
            dump_json_str(ledger_row(job_id=f"{base}_{index}", name=f"abl.arm-s{index}"))
            for index in range(count)
        ]
        write_file(tmp_path / "ledger.jsonl", ("\n".join(rows) + "\n").encode("utf-8"))

    def test_accounting_is_asked_by_array_base_id_not_by_task_id(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """`sacct -j 55765275_0` returns NOTHING while task 0 sits inside a
        pending aggregate (measured, HPC3 2026-09-06), so a query built from
        the ledger's per-task ids cannot see the row it needs. Six tasks
        collapse to one asked id.
        """
        self._array_ledger(tmp_path)
        fake_run.add("sacct", stdout="55765275_[0-5]|abl.arm|free-gpu|PENDING|0||None assigned\n")
        fake_run.add("squeue", stdout="")
        fake_run.add("date +%s", stdout="now 1000\n")

        assert triage_cli.main(self._config(tmp_path)) == 0
        sacct = next(c for c in fake_run.commands() if c.startswith("sacct"))
        assert "-j 55765275 " in sacct
        assert "55765275_0" not in sacct

    def test_a_pending_array_raises_nothing_and_its_tasks_stay_live(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """Twelve findings against two healthy queued arrays was the bug."""
        self._array_ledger(tmp_path)
        fake_run.add("sacct", stdout="55765275_[0-5]|abl.arm|free-gpu|PENDING|0||None assigned\n")
        fake_run.add("squeue --me", stdout="")
        fake_run.add("squeue", stdout="")
        fake_run.add("date +%s", stdout="now 1000\n")

        assert triage_cli.main(self._config(tmp_path)) == 0
        assert emitted[-1] == (
            "6 recorded, 0 on the cluster, 6 open, 6 not finished, 0 finding(s), 0 newly closed"
        )

    def test_the_queue_is_asked_about_expanded_tasks_never_the_bracket(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """`squeue -h -j '55765275_[0-5]'` exits 1 with "Invalid job id"
        (measured, HPC3 2026-09-06), and run_remote turns a non-zero exit into
        an AppError that ends the run. A pending row is precisely where the
        aggregate appears, so this is the path that would have broken.
        """
        self._array_ledger(tmp_path)
        fake_run.add("sacct", stdout="55765275_[0-5]|abl.arm|free-gpu|PENDING|0||None assigned\n")
        fake_run.add("squeue --me", stdout="")
        fake_run.add("squeue", stdout="")
        fake_run.add("date +%s", stdout="now 1000\n")

        assert triage_cli.main(self._config(tmp_path)) == 0
        asked = [c for c in fake_run.commands() if " -t PD " in c]
        assert len(asked) == 1
        assert "55765275_[0-5]" not in asked[0]
        assert "-j 55765275_0,55765275_1,55765275_2,55765275_3,55765275_4,55765275_5 " in asked[0]

    def test_an_array_cancelled_while_pending_closes_every_task(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """`55765275_[0-5]|CANCELLED by 2422328`, live on HPC3 2026-09-06.

        A closure keyed on the aggregate matches no ledger entry, so without
        expansion the six tasks are never filtered out and are re-reported on
        every subsequent run, forever.
        """
        self._array_ledger(tmp_path)
        cancelled = "55765275_[0-5]|abl.arm|free-gpu|CANCELLED by 2422328|0||None assigned\n"
        fake_run.add("sacct", stdout=cancelled)
        fake_run.add("squeue", stdout="")
        assert triage_cli.main(self._config(tmp_path)) == 0
        assert emitted[-1] == (
            "6 recorded, 0 on the cluster, 6 open, 0 not finished, 0 finding(s), 6 newly closed"
        )

        second = FakeRun()
        core_hooks.run = second
        second.add("sacct", stdout="")
        second.add("squeue", stdout="")
        assert triage_cli.main(self._config(tmp_path)) == 0
        assert emitted[-1] == "6 recorded, all closed; nothing left to reconcile"

    def test_a_healthy_finished_job_reports_nothing_wrong(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        self._ledger(tmp_path)
        fake_run.add("sacct", stdout="101|abl.arm|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|n1\n")
        fake_run.add("squeue --me", stdout="101|abl.arm|COMPLETING\n")
        fake_run.add("squeue", stdout="")

        assert triage_cli.main(self._config(tmp_path)) == 0
        assert emitted[-1] == (
            "1 recorded, 1 on the cluster, 1 open, 0 not finished, 0 finding(s), 1 newly closed"
        )

    def test_a_finished_job_is_closed_and_then_never_re_reported(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """The retention bug end to end: the second run does not ask about it.

        A completed job whose accounting row has aged out is, to a query,
        indistinguishable from a job that never existed. The first run records
        that it ended; the second never asks, so it cannot be reported.
        """
        self._ledger(tmp_path)
        fake_run.add("sacct", stdout="101|abl.arm|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|n1\n")
        fake_run.add("squeue", stdout="")
        assert triage_cli.main(self._config(tmp_path)) == 0

        # Accounting has now forgotten it -- the state after retention expires.
        second = FakeRun()
        core_hooks.run = second
        second.add("sacct", stdout="")
        second.add("squeue", stdout="")

        assert triage_cli.main(self._config(tmp_path)) == 0
        assert emitted[-1] == "1 recorded, all closed; nothing left to reconcile"
        # The LEDGER-side question is never asked again -- no sacct, no
        # id-restricted squeue. The account enumeration still is, and must be:
        # a fully closed ledger says nothing about what the cluster is holding
        # now, and this early return used to exit 0 before asking.
        # The bitstring prefix rides every parsed squeue; the command's
        # identity is the word after it.
        assert [c.split(" ")[1] for c in second.commands()] == ["squeue"]
        assert "--me" in second.commands()[0]

    def test_the_queue_is_not_asked_about_a_job_that_has_finished(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """Measured on the real cluster, not deduced.

        squeue holds a job for minutes after it ends and then forgets it.
        ``squeue -j`` on an id it no longer holds exits 1 with "Invalid job id
        specified" -- observed against a job that had COMPLETED perfectly --
        and that would be reported as a failed remote command. Since the
        blocked-job check only concerns pending jobs, the queue is asked only
        about ids accounting says are queued.
        """
        self._ledger(tmp_path)
        fake_run.add("sacct", stdout="101|abl.arm|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|n1\n")

        assert triage_cli.main(self._config(tmp_path)) == 0
        # The id-restricted query is the one that must not run. The account
        # enumeration is also a squeue and always runs, so asserting on the
        # bare word would now pass for the wrong reason.
        assert not any("squeue -h -j" in command for command in fake_run.commands())

    def test_the_queue_is_asked_only_about_the_ids_that_are_pending(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        write_file(
            tmp_path / "ledger.jsonl",
            b"".join(
                dump_json_str(ledger_row(job_id=job_id, name=f"abl.arm-{job_id}")).encode("utf-8")
                + b"\n"
                for job_id in ("101", "102")
            ),
        )
        fake_run.add(
            "sacct",
            stdout=(
                "101|abl.arm-101|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|n1\n"
                "102|abl.arm-102|free-gpu|PENDING|0||\n"
            ),
        )
        fake_run.add("squeue --me", stdout="102|abl.arm-102|PENDING\n")
        fake_run.add("squeue", stdout="102|abl.arm-102|Resources\n")
        fake_run.add("date +%s", stdout="now 1000\n")

        triage_cli.main(self._config(tmp_path))
        queue_calls = [c for c in fake_run.commands() if "squeue -h -j" in c]
        assert len(queue_calls) == 1
        assert "-j 102" in queue_calls[0]
        assert "101" not in queue_calls[0]

    def test_a_blocked_job_is_found_and_exits_non_zero(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """The 261-job failure mode: pending on something that never resolves."""
        self._ledger(tmp_path)
        fake_run.add("sacct", stdout="101|abl.arm|free-gpu|PENDING|0||\n")
        fake_run.add("squeue", stdout="101|abl.arm|DependencyNeverSatisfied\n")
        fake_run.add("date +%s", stdout="now 1000\n")

        assert triage_cli.main(self._config(tmp_path)) == 1
        assert emitted[0].startswith("BLOCKED 101 abl.arm:")
        assert "DependencyNeverSatisfied" in emitted[0]

    def test_an_unaccounted_job_is_found(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """We recorded submitting it and the cluster has never heard of it."""
        self._ledger(tmp_path)
        fake_run.add("sacct", stdout="")
        fake_run.add("squeue", stdout="")
        fake_run.add("date +%s", stdout="now 1000\n")

        assert triage_cli.main(self._config(tmp_path)) == 1
        assert emitted[0].startswith("UNACCOUNTED 101 abl.arm:")

    def test_an_unclaimed_job_is_found(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """The mirror: the cluster is holding it and the ledger never heard of it.

        The row is the one HPC3 really returned on 2026-08-28 -- an image
        build started by the raw ``ssh <host> sbatch`` this package's own
        README prescribes, which is how twenty-one of them ran unrecorded.
        """
        self._ledger(tmp_path)
        fake_run.add(
            "squeue --me", stdout="101|abl.arm|RUNNING\n55645549|img.abl-sif-v22|RUNNING\n"
        )
        fake_run.add("sacct", stdout="101|abl.arm|free-gpu|RUNNING|60|billing=8,gres/gpu=1|n1\n")
        fake_run.add("squeue", stdout="")
        fake_run.add("date +%s", stdout="now 1000\n101 990\n")

        assert triage_cli.main(self._config(tmp_path)) == 1
        assert emitted[0].startswith("UNCLAIMED 55645549 img.abl-sif-v22:")

    def test_an_empty_ledger_does_not_excuse_the_cluster_from_answering(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """The bug this check was added around.

        With no ledger the command used to emit "nothing has been submitted
        from this machine" and return 0 without asking the cluster anything --
        so a machine whose ledger was empty while seven jobs ran reported the
        strongest form of the condition as health.
        """
        fake_run.add("squeue --me", stdout="55645549|img.abl-sif-v22|RUNNING\n")

        assert triage_cli.main(self._config(tmp_path)) == 1
        assert "ledger is empty" in emitted[0]
        assert emitted[1].startswith("UNCLAIMED 55645549 img.abl-sif-v22:")

    def test_a_fully_closed_ledger_does_not_excuse_it_either(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """The second early return, which had the same hole as the first."""
        self._ledger(tmp_path)
        fake_run.add("sacct", stdout="101|abl.arm|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|n1\n")
        assert triage_cli.main(self._config(tmp_path)) == 0

        second = FakeRun()
        core_hooks.run = second
        second.add("squeue --me", stdout="55645549|img.abl-sif-v22|RUNNING\n")

        assert triage_cli.main(self._config(tmp_path)) == 1
        assert "nothing left to reconcile" in emitted[-2]
        assert emitted[-1].startswith("UNCLAIMED 55645549 img.abl-sif-v22:")

    def test_a_cluster_holding_only_recorded_jobs_is_clean(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        self._ledger(tmp_path)
        fake_run.add("squeue --me", stdout="101|abl.arm|RUNNING\n")
        fake_run.add("sacct", stdout="101|abl.arm|free-gpu|RUNNING|60|billing=8,gres/gpu=1|n1\n")
        fake_run.add("squeue", stdout="")
        fake_run.add("date +%s", stdout="now 1000\n101 990\n")

        assert triage_cli.main(self._config(tmp_path)) == 0

    def _closed_history(self, tmp_path: pathlib.Path, *, elapsed: int) -> None:
        """Record two finished jobs on the project's own partition.

        Args:
            tmp_path: Directory holding the ledger.
            elapsed: Seconds the LONGER of them ran; job 101 gets this and
                job 102 gets half, so the evidence a finding names is
                unambiguous rather than a tie broken by job id.
        """
        write_file(
            tmp_path / "ledger.jsonl",
            b"".join(
                dump_json_str(ledger_row(job_id=job_id, name=f"abl.arm-{job_id}")).encode("utf-8")
                + b"\n"
                for job_id in ("101", "102")
            ),
        )
        write_file(
            tmp_path / "ledger.jsonl.closed",
            b"".join(
                dump_json_str(
                    {
                        "job_id": job_id,
                        "state": "COMPLETED",
                        "closed_at": "2026-08-28T21:00:00+00:00",
                        "elapsed_seconds": elapsed if job_id == "101" else elapsed // 2,
                    }
                ).encode("utf-8")
                + b"\n"
                for job_id in ("101", "102")
            ),
        )

    def test_a_project_asking_for_far_more_than_it_uses_is_found(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """turkic-lstm asked for 720 minutes and finished in 27, and every
        check passed because the budget cap had been derived from the
        request."""
        self._closed_history(tmp_path, elapsed=60)

        assert triage_cli.main(self._config(tmp_path, projects={"abl": project_config()})) == 1
        reported = [line for line in emitted if line.startswith("OVERSIZED")]
        assert len(reported) == 1
        assert reported[0].startswith("OVERSIZED 101 abl:")
        assert "min or less would not be remarked on" in reported[0]

    def test_it_is_reported_even_when_every_job_is_closed(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """A fully closed ledger is a project whose work has all FINISHED,
        which is when its runtimes are most worth comparing against its
        request. This check sat after that early return for one revision and
        was silent in exactly the steady state it describes."""
        self._closed_history(tmp_path, elapsed=60)

        assert triage_cli.main(self._config(tmp_path, projects={"abl": project_config()})) == 1
        assert any("nothing left to reconcile" in line for line in emitted)
        assert any(line.startswith("OVERSIZED") for line in emitted)

    def test_a_right_sized_project_is_not_remarked_on(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        self._closed_history(tmp_path, elapsed=60 * 60)

        assert triage_cli.main(self._config(tmp_path, projects={"abl": project_config()})) == 0
        assert not any(line.startswith("OVERSIZED") for line in emitted)

    def test_a_silent_running_job_is_found(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        self._ledger(tmp_path)
        fake_run.add("sacct", stdout="101|abl.arm|free-gpu|RUNNING|9000|billing=8,gres/gpu=1|n1\n")
        fake_run.add("squeue", stdout="")
        fake_run.add("date +%s", stdout="now 100000\n101 1000\n")

        assert triage_cli.main(self._config(tmp_path)) == 1
        assert emitted[0].startswith("SILENT 101 abl.arm:")

    def test_the_quiet_threshold_comes_from_the_workspace(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """The same log age is healthy or silent depending only on the config."""
        self._ledger(tmp_path)
        fake_run.add("sacct", stdout="101|abl.arm|free-gpu|RUNNING|9000|billing=8,gres/gpu=1|n1\n")
        fake_run.add("squeue", stdout="")
        fake_run.add("date +%s", stdout="now 5000\n101 1000\n")

        assert triage_cli.main(self._config(tmp_path, quiet_seconds=7200)) == 0

    def test_a_workspace_with_a_zero_threshold_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Zero would report every running job as silent."""
        with pytest.raises(JSONTypeError, match="at least 1"):
            triage_cli.main(self._config(tmp_path, quiet_seconds=0))

    def test_the_config_flag_is_not_optional(self) -> None:
        with pytest.raises(ValueError, match="--config is required"):
            triage_cli.main([])

    def test_the_entrypoint_reads_the_process_arguments(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], argv: list[str]
    ) -> None:
        argv[:] = ["prog", *self._config(tmp_path)]
        with pytest.raises(SystemExit) as excinfo:
            triage_cli.entrypoint()
        assert excinfo.value.code == 0
