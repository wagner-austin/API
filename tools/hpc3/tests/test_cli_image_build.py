"""Tests for ``hpc3-image-build``: the last raw ssh in the recipe, closed.

Steps 1-3 of "adopting an image" were commands of this package. Step 4 was
``ssh hpc3 'cd <dir> && sbatch build.sbatch'``, typed by hand, and that is why
twenty-one builds hold no ledger row. This command is step 4.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.cli import image_build as image_build_cli
from hpc3.clusters.hpc3 import HPC3
from hpc3.core import ledger
from tests.conftest import FakeRun, LoggedEvent, write_workspace

_SCRIPT = """#!/bin/bash -l
#SBATCH -J abl.image-v22
#SBATCH -p free
#SBATCH -c 8
#SBATCH -t 02:00:00
"""

_PREFLIGHT_OK = (
    "sbatch: Job 55645549 to start at 2026-08-28T13:57:28 using 8 "
    "processors on nodes hpc3-15-23 in partition free\nrc=0\n"
)


def _args(tmp_path: pathlib.Path, **overrides: str) -> list[str]:
    """Build an image-build argument list.

    Args:
        tmp_path: Directory holding the workspace.
        **overrides: Flag values to replace.

    Returns:
        Arguments excluding the program name.
    """
    config = write_workspace(tmp_path / "hpc3.json")
    flags: dict[str, str] = {
        "--config": config,
        "--project": "abl",
        "--name": "image-v22",
        "--image-dir": "/pub/wagnera3/images/v22",
        "--image-name": "abl.sif",
    }
    flags.update(overrides)
    return [token for flag, value in flags.items() for token in (flag, value)]


def _healthy(fake_run: FakeRun, *, script: str = _SCRIPT) -> None:
    """Script a cluster that accepts the build.

    Args:
        fake_run: The runner to script.
        script: Build script the cluster returns for the ``cat``.
    """
    fake_run.add("cat ", stdout=script)
    fake_run.add("--test-only", stdout=_PREFLIGHT_OK)
    fake_run.add("sbatch", stdout="Submitted batch job 55645549\n")


class TestItSubmitsAndRecords:
    def test_it_reports_the_submitted_build(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        _healthy(fake_run)
        assert image_build_cli.main(_args(tmp_path)) == 0
        assert emitted[0] == "submitted 55645549 abl.image-v22"

    def test_it_names_the_image_being_built(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        _healthy(fake_run)
        image_build_cli.main(_args(tmp_path))
        assert emitted[1] == "  building /pub/wagnera3/images/v22/abl.sif"

    def test_it_names_the_log_the_build_will_write(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        """build-<id>.out, not <name>-<id>.out -- a build names its own logs."""
        _healthy(fake_run)
        image_build_cli.main(_args(tmp_path))
        assert emitted[2] == "  logs /pub/wagnera3/images/v22/build-55645549.out"

    def test_the_ledger_row_lands_where_the_workspace_says(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        """The same ledger triage reads, which is what stops the build showing
        as unclaimed for the two hours it runs."""
        _healthy(fake_run)
        image_build_cli.main(_args(tmp_path))
        entries = ledger.read(tmp_path / "ledger.jsonl", HPC3)
        assert [(e["job_id"], e["project"], e["name"]) for e in entries] == [
            ("55645549", "abl", "abl.image-v22")
        ]
        assert entries[0]["submitted_at"] == frozen_clock

    def test_the_watch_line_carries_the_config_that_was_used(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        _healthy(fake_run)
        image_build_cli.main(_args(tmp_path))
        assert emitted[3].startswith("watch: hpc3-watch --config ")
        assert emitted[3].endswith("--job 55645549")


class TestWhatItRefuses:
    def test_a_mistyped_project_never_reaches_the_cluster(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """A ledger row naming the wrong project is still the defect this
        command exists to stop -- it is now caught against the SCRIPT.

        This used to look the project up in the workspace registry. That check
        was redundant AND wrong: redundant because the label a mistyped
        project produces disagrees with the rendered script's own job name,
        which `check_name_agrees` refuses before preflight; wrong because a
        project being ONBOARDED is not in the registry at all, so the lookup
        refused the one command that could produce the image registration
        requires.

        Agreement with the rendered bytes is the stronger check: it catches
        the typo AND a renderer and submitter that have drifted apart.
        """
        _healthy(fake_run)

        with pytest.raises(AppError) as excinfo:
            image_build_cli.main(_args(tmp_path, **{"--project": "sirius"}))

        assert excinfo.value.code is Hpc3ErrorCode.IMAGE_BUILD_NAME_MISMATCH
        # Exactly one remote call: reading the script. The refusal lands
        # before the --test-only preflight and before the real sbatch, so
        # nothing was queued and no ledger row was written. Asserted on the
        # call COUNT rather than on the absence of "sbatch" in a command,
        # because the script being read is itself named build.sbatch.
        assert len(fake_run.calls) == 1
        assert fake_run.calls[0].remote_command.startswith("cat ")

    def test_a_script_naming_a_different_job_is_refused(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """The v22 case: rendered `img.abl-sif-v22`, recorded `abl.image-v22`.
        The row would name a job no squeue search finds."""
        _healthy(fake_run, script=_SCRIPT.replace("abl.image-v22", "img.abl-sif-v22"))
        with pytest.raises(AppError) as excinfo:
            image_build_cli.main(_args(tmp_path))
        assert excinfo.value.code is Hpc3ErrorCode.IMAGE_BUILD_NAME_MISMATCH
        assert not (tmp_path / "ledger.jsonl").exists()

    def test_a_directory_with_no_build_script_is_refused(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        fake_run.add("cat ", stdout="", returncode=1, stderr="No such file or directory\n")
        with pytest.raises(AppError) as excinfo:
            image_build_cli.main(_args(tmp_path))
        assert excinfo.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED

    def test_a_build_slurm_would_refuse_never_reaches_the_queue(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        fake_run.add("cat ", stdout=_SCRIPT)
        fake_run.add("--test-only", stdout="sbatch: error: Invalid partition\nrc=1\n")
        with pytest.raises(AppError) as excinfo:
            image_build_cli.main(_args(tmp_path))
        assert excinfo.value.code is Hpc3ErrorCode.PREFLIGHT_REJECTED
        assert not (tmp_path / "ledger.jsonl").exists()

    def test_every_flag_is_required(self, tmp_path: pathlib.Path) -> None:
        """Each one is a fact the ledger row needs and cannot infer."""
        for flag in ("--config", "--project", "--name", "--image-dir", "--image-name"):
            args = _args(tmp_path)
            index = args.index(flag)
            del args[index : index + 2]
            with pytest.raises(ValueError, match=f"{flag} is required"):
                image_build_cli.main(args)

    def test_an_unknown_flag_is_refused_rather_than_ignored(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="unknown argument"):
            image_build_cli.main([*_args(tmp_path), "--host", "hpc3"])


class TestTheEntrypoint:
    def test_it_reads_the_process_arguments(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        argv: list[str],
        frozen_clock: str,
    ) -> None:
        _healthy(fake_run)
        argv[:] = ["prog", *_args(tmp_path)]
        with pytest.raises(SystemExit) as excinfo:
            image_build_cli.entrypoint()
        assert excinfo.value.code == 0


class TestTheAuditEvent:
    def test_it_records_the_build_with_its_artifact(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        logged: list[LoggedEvent],
        frozen_clock: str,
    ) -> None:
        """ "which job built this image" had no answer at all before."""
        _healthy(fake_run)
        image_build_cli.main(_args(tmp_path))
        assert [event.event for event in logged] == ["hpc3_job_submitted"]
        assert logged[0].fields["kind"] == "image-build"
        assert logged[0].fields["artifact"] == "/pub/wagnera3/images/v22/abl.sif"
        assert logged[0].fields["job_name"] == "abl.image-v22"
        assert logged[0].fields["usage_factor"] == 0.0
