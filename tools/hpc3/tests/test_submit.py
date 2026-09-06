"""Tests for submission: validate, queue, record -- in that order, always.

Every test scripts the preflight steps as well as the ``sbatch``, because
submission preflights unconditionally. That is the point: there is no code
path to the cluster that skips validation, so there is no test that can
pretend otherwise.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue

from hpc3.contracts.job import JobSpec
from hpc3.contracts.ledger import LedgerEntry
from hpc3.core import audit
from hpc3.core.submit import parse_job_id, submit
from tests.against_hpc3 import decode_job_spec, read_ledger
from tests.conftest import (
    PREFLIGHT_LINE,
    FakeRun,
    LoggedEvent,
    cluster,
    gpus,
    script_healthy_cluster,
)

_AT = "2026-08-22T16:00:00+00:00"
_SUBMITTER = "fable-brain-audit-0903"


def _spec(**overrides: JSONValue) -> JobSpec:
    """Build a decoded job spec.

    Args:
        **overrides: Fields to replace.

    Returns:
        A validated spec.
    """
    base: dict[str, JSONValue] = {
        "project": "abl",
        "name": "arm-b-42",
        "partition": "free-gpu",
        "gpu": gpus("A100"),
        "cpus": 4,
        "mem_gb": 16,
        "minutes": 30,
        "requeue": False,
        "checkpoint_steps": 0,
        "env_path": "/pub/envs/abl-pinned",
        "pinned_packages": {},
        "deterministic": False,
        "experiment": {"arm": "B", "seed": "42"},
        "command": "python train.py",
        "artifact": None,
    }
    base.update(overrides)
    return decode_job_spec(base)


def _submit(tmp_path: pathlib.Path) -> str:
    """Submit with the standard test wiring.

    Args:
        tmp_path: Directory for the ledger.

    Returns:
        The submitted job's id.
    """
    return submit(
        _spec(),
        host="hpc3",
        script_dir="/pub/wagnera3/jobs",
        log_dir="/pub/wagnera3/logs",
        ledger_path=tmp_path / "ledger.jsonl",
        submitted_at=_AT,
        submitter=_SUBMITTER,
        cluster=cluster(),
        charge_account="",
    )


class TestParseJobId:
    def test_it_reads_the_announcement(self) -> None:
        assert parse_job_id("Submitted batch job 55519937\n") == "55519937"

    def test_it_finds_the_line_among_warnings(self) -> None:
        output = "sbatch: warning: quota low\nSubmitted batch job 55519937\n"
        assert parse_job_id(output) == "55519937"

    def test_no_announcement_is_a_failure(self) -> None:
        """sbatch can exit zero while printing only a warning."""
        with pytest.raises(AppError) as excinfo:
            parse_job_id("sbatch: warning: quota low\n")
        assert excinfo.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED

    def test_a_non_numeric_id_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            parse_job_id("Submitted batch job unknown\n")
        assert excinfo.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED

    def test_empty_output_is_refused(self) -> None:
        with pytest.raises(AppError):
            parse_job_id("")


class TestSubmit:
    def test_it_validates_then_queues_then_records(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        script_healthy_cluster(fake_run)
        assert _submit(tmp_path) == "55519937"

        commands = fake_run.commands()
        env_probe = next(i for i, c in enumerate(commands) if "test -d" in c)
        dry_run = next(i for i, c in enumerate(commands) if "--test-only" in c)
        real = next(i for i, c in enumerate(commands) if "sbatch " in c and "--test-only" not in c)
        assert env_probe < dry_run < real

    def test_submission_cannot_skip_preflight(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """A rejected job is never queued, and there is no flag to force it."""
        fake_run.add("test -d", stdout="PRESENT\n")
        fake_run.add("--test-only", stdout="allocation failure: bad account\nrc=1\n")

        with pytest.raises(AppError) as excinfo:
            _submit(tmp_path)
        assert excinfo.value.code is Hpc3ErrorCode.PREFLIGHT_REJECTED
        assert not any("sbatch " in c and "--test-only" not in c for c in fake_run.commands())

    def test_a_missing_environment_stops_it(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        fake_run.add("test -d", stdout="ABSENT\n")
        with pytest.raises(AppError) as excinfo:
            _submit(tmp_path)
        assert excinfo.value.code is Hpc3ErrorCode.ENV_PATH_MISSING

    def test_the_script_is_uploaded_once_not_twice(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """Preflight and submit share one upload, so they cannot drift."""
        script_healthy_cluster(fake_run)
        _submit(tmp_path)
        uploads = [c for c in fake_run.commands() if c.startswith("cat >")]
        assert uploads == ["cat > '/pub/wagnera3/jobs/abl.arm-b-42.sbatch'"]

    def test_the_uploaded_script_is_lf_encoded_utf8(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """A CRLF shebang makes the kernel report the interpreter as missing."""
        script_healthy_cluster(fake_run)
        _submit(tmp_path)

        written = next(c.stdin_bytes for c in fake_run.calls if c.stdin_bytes is not None)
        assert written.startswith(b"#!/bin/bash -l\n")
        assert b"\r" not in written
        assert b"--gres=gpu:A100:1" in written

    def test_a_rejected_submission_propagates(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        fake_run.add("test -d", stdout="PRESENT\n")
        fake_run.add("--test-only", stdout=PREFLIGHT_LINE + "\nrc=0\n")
        fake_run.add("sbatch", returncode=1, stderr="Invalid account\n")
        with pytest.raises(AppError) as excinfo:
            _submit(tmp_path)
        assert excinfo.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED


class TestSubmitRecordsTheJob:
    def test_the_ledger_entry_lands_with_everything_needed_to_find_it(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        script_healthy_cluster(fake_run)
        _submit(tmp_path)

        recorded = read_ledger(tmp_path / "ledger.jsonl")
        assert recorded == [
            LedgerEntry(
                job_id="55519937",
                project="abl",
                name="abl.arm-b-42",
                host="hpc3",
                partition="free-gpu",
                submitted_at=_AT,
                log_dir="/pub/wagnera3/logs",
                deterministic=False,
                experiment={"arm": "B", "seed": "42"},
                image_digest="",
                submitter=_SUBMITTER,
                artifact=None,
            )
        ]

    def test_the_entry_carries_what_the_run_was_not_only_which_row_it_held(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """A job id finds the job; this is what says which result it produced."""
        script_healthy_cluster(fake_run)
        _submit(tmp_path)
        assert read_ledger(tmp_path / "ledger.jsonl")[0]["experiment"] == {
            "arm": "B",
            "seed": "42",
        }

    def test_a_failed_submission_records_nothing(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """A job that does not exist must not appear in the record."""
        fake_run.add("test -d", stdout="ABSENT\n")
        with pytest.raises(AppError):
            _submit(tmp_path)
        assert read_ledger(tmp_path / "ledger.jsonl") == []


class TestSubmitAudit:
    def test_a_successful_submission_is_recorded(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, logged: list[LoggedEvent]
    ) -> None:
        script_healthy_cluster(fake_run)
        _submit(tmp_path)

        assert [event.event for event in logged] == [audit.JOB_SUBMITTED]
        assert logged[0].fields["job_id"] == "55519937"
        assert logged[0].fields["usage_factor"] == 0.0

    def test_a_failed_submission_logs_nothing(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, logged: list[LoggedEvent]
    ) -> None:
        fake_run.add("test -d", stdout="ABSENT\n")
        with pytest.raises(AppError):
            _submit(tmp_path)
        assert logged == []
