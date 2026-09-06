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
from tests.conftest import PREFLIGHT_LINE, FakeRun, LoggedEvent, cluster, gpus

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
        "gpu": gpus("A100"),
        "cpus": 8,
        "mem_gb": 96,
        "minutes": 30,
        "requeue": False,
        "checkpoint_steps": 0,
        "env_path": "/pub/envs/abl-pinned",
        "pinned_packages": {},
        "deterministic": False,
        "experiment": {"rung": "774M"},
        "command": "python train.py",
        "artifact": None,
    }
    base.update(overrides)
    members: list[JSONValue] = [
        {"suffix": f"s{i}", "command": f"python train.py --seed {i}", "artifact": None}
        for i in range(count)
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
            submitter="fable-brain-audit-0903",
            cluster=cluster(),
            charge_account="",
        )
    ]


class TestSubmitSweep:
    def test_the_whole_sweep_goes_up_as_one_array_call(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """One sbatch, every document position, task ids derived from it.

        The member-by-member loop this replaced cost three SSH round trips
        per member (~13s each, rusted ab48, 2026-09-01) against a cluster
        that scheduled everything instantly."""
        _healthy(fake_run)
        fake_run.add("sbatch --array=", stdout="Submitted batch job 101\n")

        assert _run(_sweep(), tmp_path) == ["101_0", "101_1", "101_2"]
        submits = [
            c for c in fake_run.commands() if "sbatch --array=" in c and "--test-only" not in c
        ]
        assert submits == ["cd /j && sbatch --array=0-2 abl.rung.sbatch"]

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
            submitter="fable-brain-audit-0903",
            cluster=cluster(),
            charge_account="",
        )
        assert [m.name for m in members] == ["abl.rung-s0", "abl.rung-s1", "abl.rung-s2"]

    def test_the_whole_sweep_lands_in_one_script(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """One upload, and the script IS the member table."""
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 1\n")
        _run(_sweep(), tmp_path)

        writes = [c for c in fake_run.commands() if c.startswith("cat >")]
        assert writes == ["cat > '/j/abl.rung.sbatch'"]

    def test_the_script_carries_every_members_command_dispatched_by_index(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 1\n")
        _run(_sweep(), tmp_path)

        payloads = [c.stdin_bytes for c in fake_run.calls if c.stdin_bytes is not None]
        assert len(payloads) == 1
        script = payloads[0].decode("utf-8")
        assert 'case "${SLURM_ARRAY_TASK_ID}" in' in script
        assert "--seed 0" in script
        assert "--seed 1" in script
        assert "--seed 2" in script
        # The dispatch order is the document order, so a sparse --array on a
        # later convergence pass selects the same members it always did.
        assert script.index("--seed 0") < script.index("--seed 1") < script.index("--seed 2")

    def test_a_refused_submission_leaves_nothing_behind(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, logged: list[LoggedEvent]
    ) -> None:
        """All-or-nothing, which is STRONGER than the old loop's no-rollback:
        the loop could die on member four leaving three live, while the array
        either submits whole or refuses whole -- so a refusal here means no
        job, no ledger row, and no audit event to reconcile."""
        _healthy(fake_run)
        fake_run.add("sbatch --array=", returncode=1, stderr="Invalid account\n")

        with pytest.raises(AppError) as excinfo:
            _run(_sweep(), tmp_path)

        assert excinfo.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED
        assert not (tmp_path / "ledger.jsonl").exists()
        assert [e.event for e in logged if e.event == audit.SWEEP_SUBMITTED] == []


class TestSweepLedger:
    def test_every_member_is_recorded_under_its_task_id(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """One submission act, one ledger row per member anyway: the task id
        is what sacct and squeue will call each member, and the ledger is
        where its per-member name durably lives."""
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 101\n")
        _run(_sweep(), tmp_path)

        recorded = read_ledger(tmp_path / "ledger.jsonl")
        assert [entry["job_id"] for entry in recorded] == ["101_0", "101_1", "101_2"]
        assert [entry["name"] for entry in recorded] == [
            "abl.rung-s0",
            "abl.rung-s1",
            "abl.rung-s2",
        ]
        assert {entry["project"] for entry in recorded} == {"abl"}
        assert {entry["log_dir"] for entry in recorded} == {"/l"}


class TestSweepAuditEvents:
    def test_one_submission_act_emits_one_sweep_event_and_no_job_events(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, logged: list[LoggedEvent]
    ) -> None:
        """Per-member job events described member-by-member submissions that
        no longer happen; telemetry of acts that did not occur would be a
        false trail. The one event carries every task id."""
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 101\n")
        _run(_sweep(), tmp_path)

        assert [e for e in logged if e.event == audit.JOB_SUBMITTED] == []
        sweeps = [e for e in logged if e.event == audit.SWEEP_SUBMITTED]
        assert len(sweeps) == 1
        assert sweeps[0].fields == {
            "host": "hpc3",
            "project": "abl",
            "base_name": "abl.rung",
            "members": 3,
            "job_ids": "101_0,101_1,101_2",
            "partition": "free-gpu",
            "usage_factor": 0.0,
        }
        assert logged[-1].event == audit.SWEEP_SUBMITTED

    def test_the_sweep_event_records_the_factor_the_array_went_out_under(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, logged: list[LoggedEvent]
    ) -> None:
        """The billing factor moved from the retired per-member events onto
        the sweep event; every member shares the template's partition, so one
        factor covers the array."""
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 1\n")
        _run(_sweep(count=2, partition="free-gpu32", gpu=gpus("L40S")), tmp_path)
        sweeps = [e for e in logged if e.event == audit.SWEEP_SUBMITTED]
        assert [e.fields["usage_factor"] for e in sweeps] == [0.0]
        assert [e.fields["partition"] for e in sweeps] == ["free-gpu32"]
