"""Tests for the sweep and triage CLIs."""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str

from hpc3.cli import submit as submit_cli
from hpc3.cli import sweep as sweep_cli
from hpc3.cli import triage as triage_cli
from hpc3.core import _test_hooks as core_hooks
from tests.against_hpc3 import read_ledger
from tests.conftest import (
    PREFLIGHT_LINE,
    FakeRun,
    budget_document,
    gpus,
    ledger_row,
    project_config,
    workspace_document,
    write_file,
    write_workspace,
)


def _payload(count: int = 3, **overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid sweep document.

    Args:
        count: How many members.
        **overrides: Fields to add or replace.

    Returns:
        The document.
    """
    members: list[JSONValue] = [
        {"suffix": f"s{i}", "command": f"python train.py --seed {i}", "artifact": None}
        for i in range(count)
    ]
    document: dict[str, JSONValue] = {
        "project": "abl",
        "name": "rung",
        "members": members,
        "experiment": {"rung": "774M"},
    }
    document.update(overrides)
    return document


def _write(path: pathlib.Path, payload: JSONValue) -> None:
    """Write a document for the CLI to read.

    Args:
        path: File to write.
        payload: Document to serialise.
    """
    write_file(path, dump_json_str(payload).encode("utf-8"))


def _args(tmp_path: pathlib.Path, **overrides: JSONValue) -> list[str]:
    """Build a full sweep argument list.

    Args:
        tmp_path: Directory holding the documents.
        **overrides: Workspace fields to replace.

    Returns:
        Arguments excluding the program name.
    """
    config = write_workspace(tmp_path / "hpc3.json", workspace_document(**overrides))
    return ["--config", config, "--run", str(tmp_path / "s.json")]


def _healthy(fake: FakeRun) -> None:
    """Script a cluster that admits every member.

    Args:
        fake: The runner to script.
    """
    fake.add("test -d", stdout="PRESENT\n")
    fake.add("--test-only", stdout=PREFLIGHT_LINE + "\nrc=0\n")


class TestSweepCli:
    def test_it_reports_every_member_and_a_watch_command(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        _write(tmp_path / "s.json", _payload())
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 101\n")

        assert sweep_cli.main(_args(tmp_path)) == 0
        assert emitted[0].startswith("budget OK: projected 1.5 GPU-hours")
        assert emitted[1:4] == [
            "submitted 101_0 abl.rung-s0",
            "submitted 101_1 abl.rung-s1",
            "submitted 101_2 abl.rung-s2",
        ]
        assert emitted[-1].endswith("--job 101_0,101_1,101_2")

    def test_it_reports_where_every_members_log_landed(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        _write(tmp_path / "s.json", _payload())
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 1\n")
        sweep_cli.main(_args(tmp_path))
        assert "logs /pub/w/abl/logs" in emitted

    def test_every_member_reaches_the_ledger(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        _write(tmp_path / "s.json", _payload())
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 101\n")

        sweep_cli.main(_args(tmp_path))
        entries = read_ledger(tmp_path / "ledger.jsonl")
        assert [e["job_id"] for e in entries] == ["101_0", "101_1", "101_2"]

    def test_a_budget_overrun_stops_it_before_the_cluster(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        _write(tmp_path / "s.json", _payload())
        with pytest.raises(AppError) as excinfo:
            sweep_cli.main(
                _args(
                    tmp_path,
                    projects={"abl": project_config(budget=budget_document(gpu_hours=1.0))},
                )
            )
        assert excinfo.value.code is Hpc3ErrorCode.BUDGET_PROJECTION_EXCEEDED
        assert fake_run.calls == []
        assert emitted == []

    def test_an_oversized_sweep_never_reaches_the_cluster(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        _write(tmp_path / "s.json", _payload(count=25))
        with pytest.raises(AppError) as excinfo:
            sweep_cli.main(_args(tmp_path))
        assert excinfo.value.code is Hpc3ErrorCode.SWEEP_EXCEEDS_GPU_CEILING
        assert fake_run.calls == []

    def test_the_report_names_the_partition_and_that_it_is_free(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        _write(tmp_path / "s.json", _payload(count=2))
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 1\n")
        sweep_cli.main(
            _args(
                tmp_path,
                projects={"abl": project_config(partition="free-gpu32", gpu=gpus("L40S"))},
            )
        )
        assert any("free-gpu32 (free)" in line for line in emitted)

    def test_the_config_flag_is_not_optional(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--config is required"):
            sweep_cli.main(["--run", str(tmp_path / "s.json")])

    def test_the_entrypoint_reads_the_process_arguments(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        argv: list[str],
        frozen_clock: str,
    ) -> None:
        _write(tmp_path / "s.json", _payload(count=1))
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 9\n")
        argv[:] = ["prog", *_args(tmp_path)]

        with pytest.raises(SystemExit) as excinfo:
            sweep_cli.entrypoint()
        assert excinfo.value.code == 0
        assert emitted[1] == "submitted 9_0 abl.rung-s0"


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
        assert [c.split(" ", 1)[0] for c in second.commands()] == ["squeue"]
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
        assert not any(command.startswith("squeue -h -j") for command in fake_run.commands())

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
        queue_calls = [c for c in fake_run.commands() if c.startswith("squeue -h -j")]
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
