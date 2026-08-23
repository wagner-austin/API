"""Tests for sweep submission, its audit events, and its ledger records."""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue

from hpc3.contracts.sweep import SweepSpec
from hpc3.core import audit
from hpc3.core.sweep import submit_sweep
from tests.against_hpc3 import decode_sweep_spec, read_ledger
from tests.conftest import PREFLIGHT_LINE, FakeRun, LoggedEvent, cluster

_AT = "2026-08-22T16:00:00+00:00"


def _sweep(count: int = 3, **overrides: JSONValue) -> SweepSpec:
    """Build a decoded sweep.

    Args:
        count: How many members.
        **overrides: Template fields to replace.

    Returns:
        A validated sweep.
    """
    base: dict[str, JSONValue] = {
        "project": "abl",
        "name": "rung",
        "partition": "free-gpu",
        "gpu": "A100",
        "gpu_count": 1,
        "cpus": 8,
        "mem_gb": 96,
        "minutes": 30,
        "requeue": False,
        "checkpoint_steps": 0,
        "accept_billing": False,
        "env_path": "/pub/envs/abl-pinned",
        "pinned_packages": {},
        "deterministic": False,
        "experiment": {"rung": "774M"},
        "command": "python train.py",
    }
    base.update(overrides)
    members: list[JSONValue] = [
        {"suffix": f"s{i}", "command": f"python train.py --seed {i}"} for i in range(count)
    ]
    return decode_sweep_spec({"base": base, "members": members})


def _healthy(fake: FakeRun) -> None:
    """Script a cluster that admits every member.

    Args:
        fake: The runner to script.
    """
    fake.add("test -d", stdout="PRESENT\n")
    fake.add("--test-only", stdout=PREFLIGHT_LINE + "\nrc=0\n")


def _run(spec: SweepSpec, tmp_path: pathlib.Path) -> list[str]:
    """Submit a sweep with the standard test wiring.

    Args:
        spec: Sweep to submit.
        tmp_path: Directory for the ledger.

    Returns:
        The submitted job ids.
    """
    return [
        member.job_id
        for member in submit_sweep(
            spec,
            host="hpc3",
            script_dir="/j",
            log_dir="/l",
            ledger_path=tmp_path / "ledger.jsonl",
            submitted_at=_AT,
            cluster=cluster(),
        )
    ]


class TestSubmitSweep:
    def test_it_submits_every_member_and_returns_their_ids(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        _healthy(fake_run)
        fake_run.add("sbatch abl.rung-s0", stdout="Submitted batch job 101\n")
        fake_run.add("sbatch abl.rung-s1", stdout="Submitted batch job 102\n")
        fake_run.add("sbatch abl.rung-s2", stdout="Submitted batch job 103\n")

        assert _run(_sweep(), tmp_path) == ["101", "102", "103"]

    def test_every_member_is_reported_by_the_name_squeue_will_show(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """The operator searches for what was printed; it must match the row."""
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 1\n")
        members = submit_sweep(
            _sweep(),
            host="hpc3",
            script_dir="/j",
            log_dir="/l",
            ledger_path=tmp_path / "ledger.jsonl",
            submitted_at=_AT,
            cluster=cluster(),
        )
        assert [m.name for m in members] == ["abl.rung-s0", "abl.rung-s1", "abl.rung-s2"]

    def test_each_member_gets_its_own_script(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 1\n")
        _run(_sweep(), tmp_path)

        writes = [c for c in fake_run.commands() if c.startswith("cat >")]
        assert writes == [
            "cat > '/j/abl.rung-s0.sbatch'",
            "cat > '/j/abl.rung-s1.sbatch'",
            "cat > '/j/abl.rung-s2.sbatch'",
        ]

    def test_each_script_carries_that_members_command(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 1\n")
        _run(_sweep(), tmp_path)

        payloads = [c.stdin_bytes for c in fake_run.calls if c.stdin_bytes is not None]
        assert len(payloads) == 3
        assert b"--seed 0" in payloads[0]
        assert b"--seed 1" in payloads[1]
        assert b"--seed 2" in payloads[2]

    def test_a_failure_partway_leaves_earlier_members_findable(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, logged: list[LoggedEvent]
    ) -> None:
        """No rollback -- and the ones already running are in the ledger."""
        _healthy(fake_run)
        fake_run.add("sbatch abl.rung-s0", stdout="Submitted batch job 101\n")
        fake_run.add("sbatch abl.rung-s1", returncode=1, stderr="Invalid account\n")

        with pytest.raises(AppError) as excinfo:
            _run(_sweep(), tmp_path)

        assert excinfo.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED
        recorded = read_ledger(tmp_path / "ledger.jsonl")
        assert [entry["job_id"] for entry in recorded] == ["101"]
        assert [e.event for e in logged if e.event == audit.SWEEP_SUBMITTED] == []


class TestSweepLedger:
    def test_every_member_is_recorded(self, tmp_path: pathlib.Path, fake_run: FakeRun) -> None:
        _healthy(fake_run)
        fake_run.add("sbatch abl.rung-s0", stdout="Submitted batch job 101\n")
        fake_run.add("sbatch abl.rung-s1", stdout="Submitted batch job 102\n")
        fake_run.add("sbatch abl.rung-s2", stdout="Submitted batch job 103\n")
        _run(_sweep(), tmp_path)

        recorded = read_ledger(tmp_path / "ledger.jsonl")
        assert [entry["job_id"] for entry in recorded] == ["101", "102", "103"]
        assert [entry["name"] for entry in recorded] == [
            "abl.rung-s0",
            "abl.rung-s1",
            "abl.rung-s2",
        ]
        assert {entry["project"] for entry in recorded} == {"abl"}
        assert {entry["log_dir"] for entry in recorded} == {"/l"}


class TestSweepAuditEvents:
    def test_each_member_emits_a_job_event(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, logged: list[LoggedEvent]
    ) -> None:
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 7\n")
        _run(_sweep(), tmp_path)

        jobs = [e for e in logged if e.event == audit.JOB_SUBMITTED]
        assert len(jobs) == 3
        assert jobs[0].fields["job_name"] == "abl.rung-s0"
        assert jobs[0].fields["project"] == "abl"
        assert jobs[0].fields["bills"] is False

    def test_the_sweep_event_lands_once_after_every_member(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, logged: list[LoggedEvent]
    ) -> None:
        _healthy(fake_run)
        fake_run.add("sbatch abl.rung-s0", stdout="Submitted batch job 101\n")
        fake_run.add("sbatch abl.rung-s1", stdout="Submitted batch job 102\n")
        fake_run.add("sbatch abl.rung-s2", stdout="Submitted batch job 103\n")
        _run(_sweep(), tmp_path)

        sweeps = [e for e in logged if e.event == audit.SWEEP_SUBMITTED]
        assert len(sweeps) == 1
        assert sweeps[0].fields == {
            "host": "hpc3",
            "project": "abl",
            "base_name": "abl.rung",
            "members": 3,
            "job_ids": "101,102,103",
        }
        assert logged[-1].event == audit.SWEEP_SUBMITTED

    def test_a_billing_sweep_records_that_it_bills(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, logged: list[LoggedEvent]
    ) -> None:
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 1\n")
        _run(
            _sweep(count=2, partition="free-gpu32", gpu="L40S", accept_billing=True),
            tmp_path,
        )
        jobs = [e for e in logged if e.event == audit.JOB_SUBMITTED]
        assert all(e.fields["bills"] is True for e in jobs)
