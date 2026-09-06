"""Tests for the collision this package could not see.

THE REAL EVENT, on 2026-08-28. A ``uz`` member was preempted, resubmitted as
``bases-uz-r2``, and then -- while r2 was still queued -- resubmitted again as
``bases-uz-r3`` against a rebuilt image. Both declared
``/pub/wagnera3/LSTM/checkpoints/uz_best.pt``. It was caught by eye and one
was cancelled; nothing in submit, sweep or the contracts would have caught it.

Worse than a crash, because it succeeds: both jobs report COMPLETED, the
checkpoint exists, and its provenance names one of them.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue, dump_json_str

from hpc3.contracts.account import AccountJob, decode_account_job
from hpc3.contracts.job import JobSpec
from hpc3.contracts.ledger import LedgerEntry
from hpc3.core.inflight import check_artifact_is_free, claimed_artifacts
from hpc3.core.submit import submit
from tests.against_hpc3 import decode_job_spec, decode_ledger_entry
from tests.conftest import (
    FakeRun,
    cluster,
    gpus,
    ledger_row,
    script_healthy_cluster,
    write_file,
)

_UZ = "/pub/wagnera3/LSTM/checkpoints/uz_best.pt"
_AT = "2026-08-28T21:00:00+00:00"


def _entry(job_id: str, name: str, artifact: str | None) -> LedgerEntry:
    """Build a ledger entry declaring an artifact.

    Args:
        job_id: Job id.
        name: Qualified job name.
        artifact: Path the job writes, or None.

    Returns:
        A validated entry.
    """
    return decode_ledger_entry(ledger_row(job_id=job_id, name=name, artifact=artifact))


def _account(job_id: str, name: str = "turkic-lstm.bases-uz-r2") -> AccountJob:
    """Build a row from the account enumeration.

    Args:
        job_id: Job id.
        name: Job name.

    Returns:
        A validated account job.
    """
    return decode_account_job({"job_id": job_id, "name": name, "state": "PENDING"})


class TestWhatALiveJobIsWriting:
    def test_a_live_jobs_artifact_is_claimed(self) -> None:
        claimed = claimed_artifacts(
            [_entry("101", "turkic-lstm.bases-uz-r2", _UZ)], [_account("101")]
        )
        assert claimed == {_UZ: "turkic-lstm.bases-uz-r2"}

    def test_a_job_the_cluster_no_longer_holds_claims_nothing(self) -> None:
        """Which is what makes a resume legal: the preempted run is gone."""
        assert claimed_artifacts([_entry("101", "turkic-lstm.bases-uz", _UZ)], []) == {}

    def test_a_live_job_with_no_artifact_claims_nothing(self) -> None:
        """Every cleargbm member runs --no-save-model and writes no file."""
        claimed = claimed_artifacts([_entry("101", "cleargbm.p6-1", None)], [_account("101")])
        assert claimed == {}

    def test_only_the_live_ones_count(self) -> None:
        entries = [
            _entry("101", "turkic-lstm.bases-uz-r2", _UZ),
            _entry("102", "turkic-lstm.bases-tr", "/pub/tr_best.pt"),
        ]
        assert claimed_artifacts(entries, [_account("102")]) == {
            "/pub/tr_best.pt": "turkic-lstm.bases-tr"
        }


class TestTheRefusal:
    def test_the_uz_race_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            check_artifact_is_free(
                _UZ, {_UZ: "turkic-lstm.bases-uz-r2"}, name="turkic-lstm.bases-uz-r3"
            )
        assert excinfo.value.code is Hpc3ErrorCode.ARTIFACT_ALREADY_IN_FLIGHT

    def test_the_refusal_names_the_job_to_cancel(self) -> None:
        """A refusal that does not say what to do gets worked around."""
        with pytest.raises(AppError) as excinfo:
            check_artifact_is_free(
                _UZ, {_UZ: "turkic-lstm.bases-uz-r2"}, name="turkic-lstm.bases-uz-r3"
            )
        assert "Cancel turkic-lstm.bases-uz-r2" in str(excinfo.value)

    def test_an_unclaimed_artifact_is_admitted(self) -> None:
        check_artifact_is_free(_UZ, {"/pub/tr_best.pt": "turkic-lstm.bases-tr"}, name="x.y")

    def test_a_job_writing_no_file_is_admitted(self) -> None:
        check_artifact_is_free(None, {_UZ: "turkic-lstm.bases-uz-r2"}, name="x.y")

    def test_nothing_claimed_admits_everything(self) -> None:
        check_artifact_is_free(_UZ, {}, name="x.y")


class TestSubmitEnforcesIt:
    """The check has no flag, and runs before anything reaches the cluster."""

    def _spec(self, artifact: str | None) -> JobSpec:
        """Build a decoded spec declaring an artifact.

        Args:
            artifact: Path the job writes, or None.

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
            "experiment": {"arm": "B"},
            # The artifact must appear in the command -- a separate contract
            # rule, and a good one: the ledger publishes this path as where
            # the result will be.
            "command": (
                "python train.py" if artifact is None else f"python train.py --out {artifact}"
            ),
            "artifact": artifact,
        }
        return decode_job_spec(base)

    def _ledger(self, tmp_path: pathlib.Path, artifact: str | None) -> pathlib.Path:
        """Write a one-row ledger claiming an artifact.

        Args:
            tmp_path: Directory to write in.
            artifact: The recorded job's artifact.

        Returns:
            Path to the ledger.
        """
        path = tmp_path / "ledger.jsonl"
        row = ledger_row(job_id="55646157", name="abl.arm-a-1", artifact=artifact)
        write_file(path, dump_json_str(row).encode("utf-8") + b"\n")
        return path

    def _submit(self, tmp_path: pathlib.Path, artifact: str | None) -> str:
        """Submit a spec declaring an artifact.

        Args:
            tmp_path: Directory for the ledger.
            artifact: Path the new job would write.

        Returns:
            The submitted job's id.
        """
        return submit(
            self._spec(artifact),
            host="hpc3",
            script_dir="/pub/wagnera3/jobs",
            log_dir="/pub/wagnera3/logs",
            ledger_path=self._ledger(tmp_path, artifact),
            submitted_at=_AT,
            submitter="",
            cluster=cluster(),
            charge_account="",
        )

    def test_a_racing_submission_never_reaches_the_cluster(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        script_healthy_cluster(fake_run)
        fake_run.add("squeue --me", stdout="55646157|abl.arm-a-1|PENDING\n")

        with pytest.raises(AppError) as excinfo:
            self._submit(tmp_path, _UZ)
        assert excinfo.value.code is Hpc3ErrorCode.ARTIFACT_ALREADY_IN_FLIGHT
        assert not any("sbatch" in command for command in fake_run.commands())

    def test_it_is_checked_before_preflight(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """The cheaper question, and the one whose answer means do not submit
        at all rather than this job is malformed."""
        script_healthy_cluster(fake_run)
        fake_run.add("squeue --me", stdout="55646157|abl.arm-a-1|PENDING\n")

        with pytest.raises(AppError):
            self._submit(tmp_path, _UZ)
        assert not any("--test-only" in command for command in fake_run.commands())

    def test_the_same_artifact_is_fine_once_the_holder_has_gone(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """Which is exactly what a resume is: the preempted job is not live."""
        fake_run.add("squeue --me", stdout="")
        script_healthy_cluster(fake_run)

        assert self._submit(tmp_path, _UZ) == "55519937"
