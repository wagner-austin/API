"""Tests for submitting an image build through the tool that records it.

The gap these close: every other job reached the cluster through ``submit()``,
which preflights and writes the ledger row before returning. The image build
reached it through a raw ``ssh <host> 'cd <dir> && sbatch build.sbatch'`` that
the README itself prescribed, so twenty-one builds ran unrecorded. The
directives parsed here are the real ones from ``/pub/wagnera3/images/v22`` on
2026-08-28, whose job -- 55645549, ``img.abl-sif-v22`` -- was the first thing
``hpc3-triage``'s new reverse check reported.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.clusters.hpc3 import HPC3
from hpc3.core import ledger
from hpc3.core.image_submit import (
    check_name_agrees,
    parse_build_directives,
    submit_build,
)
from tests.conftest import FakeRun

_REAL_SCRIPT = """#!/bin/bash -l
#SBATCH -J img.abl-sif-v22
#SBATCH -p free
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /pub/wagnera3/images/v22/build-%j.out
#SBATCH --requeue

cd /pub/wagnera3/images/v22
bash build.sh
"""

_PREFLIGHT_OK = (
    "sbatch: Job 55645549 to start at 2026-08-28T13:57:28 using 8 "
    "processors on nodes hpc3-15-23 in partition free\nrc=0\n"
)

_AT = "2026-08-28T20:57:00+00:00"


class TestReadingTheScriptThatWillRun:
    def test_the_real_v22_directives_are_read(self) -> None:
        directives = parse_build_directives(_REAL_SCRIPT)
        assert directives == {"job_name": "img.abl-sif-v22", "partition": "free"}

    def test_the_partition_is_read_rather_than_assumed(self) -> None:
        """BUILD_PARTITION says what this package renders today; the file on
        the cluster is what actually runs."""
        script = _REAL_SCRIPT.replace("#SBATCH -p free", "#SBATCH -p standard")
        assert parse_build_directives(script)["partition"] == "standard"

    def test_a_script_with_no_job_name_is_refused(self) -> None:
        script = _REAL_SCRIPT.replace("#SBATCH -J img.abl-sif-v22\n", "")
        with pytest.raises(AppError) as excinfo:
            parse_build_directives(script)
        assert excinfo.value.code is Hpc3ErrorCode.IMAGE_BUILD_SCRIPT_UNREADABLE
        assert "job name" in str(excinfo.value)

    def test_a_script_with_no_partition_is_refused(self) -> None:
        script = _REAL_SCRIPT.replace("#SBATCH -p free\n", "")
        with pytest.raises(AppError) as excinfo:
            parse_build_directives(script)
        assert excinfo.value.code is Hpc3ErrorCode.IMAGE_BUILD_SCRIPT_UNREADABLE
        assert "partition" in str(excinfo.value)

    def test_an_empty_directive_is_refused_rather_than_read_as_blank(self) -> None:
        script = _REAL_SCRIPT.replace("#SBATCH -J img.abl-sif-v22", "#SBATCH -J ")
        with pytest.raises(AppError) as excinfo:
            parse_build_directives(script)
        assert "empty job name" in str(excinfo.value)

    def test_the_first_directive_wins_because_slurm_takes_the_first(self) -> None:
        script = _REAL_SCRIPT.replace("#SBATCH -p free\n", "#SBATCH -p free\n#SBATCH -p standard\n")
        assert parse_build_directives(script)["partition"] == "free"


class TestTheNameMustAgree:
    def test_a_matching_name_is_admitted(self) -> None:
        check_name_agrees(declared="abl.image-v22", rendered="abl.image-v22")

    def test_the_measured_mismatch_is_refused(self) -> None:
        """v22 really was rendered `img.abl-sif-v22`, which reads as a project
        called `img` that no workspace declares."""
        with pytest.raises(AppError) as excinfo:
            check_name_agrees(declared="abl.image-v22", rendered="img.abl-sif-v22")
        assert excinfo.value.code is Hpc3ErrorCode.IMAGE_BUILD_NAME_MISMATCH

    def test_the_refusal_shows_the_render_that_would_fix_it(self) -> None:
        """A refusal that does not show the fix gets worked around."""
        with pytest.raises(AppError) as excinfo:
            check_name_agrees(declared="abl.image-v22", rendered="img.abl-sif-v22")
        assert "--job-name abl.image-v22" in str(excinfo.value)


class TestSubmittingAndRecording:
    def _run(self, fake_run: FakeRun, *, script: str = _REAL_SCRIPT) -> None:
        """Script a healthy cluster for a build submission.

        Args:
            fake_run: The runner to script.
            script: Build script the cluster returns for the ``cat``.
        """
        fake_run.add("cat ", stdout=script)
        fake_run.add("--test-only", stdout=_PREFLIGHT_OK)
        fake_run.add("sbatch", stdout="Submitted batch job 55645549\n")

    def _submit(self, tmp_path: pathlib.Path, label: str = "img.abl-sif-v22") -> str:
        """Submit a build against a temporary ledger.

        Args:
            tmp_path: Directory holding the ledger.
            label: Qualified name to record.

        Returns:
            The submitted job's id.
        """
        return submit_build(
            host="hpc3",
            image_dir="/pub/wagnera3/images/v22",
            project="abl",
            label=label,
            artifact="/pub/wagnera3/images/v22/abl.sif",
            ledger_path=tmp_path / "ledger.jsonl",
            submitted_at=_AT,
            cluster=HPC3,
        )

    def test_the_job_id_is_returned(self, tmp_path: pathlib.Path, fake_run: FakeRun) -> None:
        self._run(fake_run)
        assert self._submit(tmp_path) == "55645549"

    def test_the_ledger_row_is_written(self, tmp_path: pathlib.Path, fake_run: FakeRun) -> None:
        """The whole point: a build that leaves a record."""
        self._run(fake_run)
        self._submit(tmp_path)
        entries = ledger.read(tmp_path / "ledger.jsonl", HPC3)
        assert [(e["job_id"], e["project"], e["name"]) for e in entries] == [
            ("55645549", "abl", "img.abl-sif-v22")
        ]

    def test_the_row_carries_the_image_as_its_artifact(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """So hpc3-trace can answer which job built a given image, which
        nothing could answer before."""
        self._run(fake_run)
        self._submit(tmp_path)
        entry = ledger.read(tmp_path / "ledger.jsonl", HPC3)[0]
        assert entry["artifact"] == "/pub/wagnera3/images/v22/abl.sif"

    def test_the_row_records_the_partition_the_script_declares(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        self._run(fake_run)
        self._submit(tmp_path)
        assert ledger.read(tmp_path / "ledger.jsonl", HPC3)[0]["partition"] == "free"

    def test_the_row_claims_no_determinism_and_no_image(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """A build is not a numerical run, and it produces the image rather
        than running inside one."""
        self._run(fake_run)
        self._submit(tmp_path)
        entry = ledger.read(tmp_path / "ledger.jsonl", HPC3)[0]
        assert entry["deterministic"] is False
        assert entry["image_digest"] == ""

    def test_the_logs_are_findable_from_the_row_alone(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """A build writes build-<id>.out rather than <name>-<id>.out, so the
        directory is the part of the convention that still holds."""
        self._run(fake_run)
        self._submit(tmp_path)
        entry = ledger.read(tmp_path / "ledger.jsonl", HPC3)[0]
        assert entry["log_dir"] == "/pub/wagnera3/images/v22"

    def test_it_preflights_before_queueing(self, tmp_path: pathlib.Path, fake_run: FakeRun) -> None:
        """The same non-skippable prefix `submit` has, against the same bytes."""
        self._run(fake_run)
        self._submit(tmp_path)
        commands = fake_run.commands()
        assert [i for i, c in enumerate(commands) if "--test-only" in c] == [1]
        assert commands.index("cd /pub/wagnera3/images/v22 && sbatch build.sbatch") == 2

    def test_a_rejected_build_never_reaches_the_queue(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        fake_run.add("cat ", stdout=_REAL_SCRIPT)
        fake_run.add("--test-only", stdout="sbatch: error: Invalid partition\nrc=1\n")

        with pytest.raises(AppError) as excinfo:
            self._submit(tmp_path)
        assert excinfo.value.code is Hpc3ErrorCode.PREFLIGHT_REJECTED
        assert not any(c.endswith("sbatch build.sbatch") for c in fake_run.commands())

    def test_a_mismatched_name_never_reaches_the_queue(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """Refused after reading the script and before anything is sent."""
        self._run(fake_run)
        with pytest.raises(AppError) as excinfo:
            self._submit(tmp_path, label="abl.image-v22")
        assert excinfo.value.code is Hpc3ErrorCode.IMAGE_BUILD_NAME_MISMATCH
        assert not any("--test-only" in c for c in fake_run.commands())

    def test_nothing_is_recorded_when_the_build_is_refused(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """A ledger row for a job that was never queued is worse than none."""
        self._run(fake_run)
        with pytest.raises(AppError):
            self._submit(tmp_path, label="abl.image-v22")
        assert not (tmp_path / "ledger.jsonl").exists()
