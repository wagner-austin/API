"""Building an archive, sending it, and the scripts that run on the node.

Only the ssh boundary is faked. The archive is a real tar built by the real
tar binary over a real temporary tree and the digest is a real SHA-256, so
what is exercised is the transport as it will actually behave -- including
the exclusion that keeps one machine's ``.venv`` off another.

THE SCRIPT TESTS BELOW ARE MOSTLY REGRESSIONS, and all of them come from one
dispatch. On 2026-09-04 the first ``fleet-run`` that reached a node registered
a scheduled task whose ``-Argument`` was the eleven characters ``-Command
"cd`` and whose WORKING DIRECTORY was the remaining two hundred, because the
build was interpolated into a single-quoted PowerShell string that contained
single quotes. It could not be started. Nothing failed: PowerShell exited 0
and the ledger recorded a run that did not exist.
"""

from __future__ import annotations

import base64
import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.cli import cancel
from fleet.core import _test_hooks, dispatch, manifest, staging
from tests.conftest import DEMO_DEPENDENCY, DEMO_PROJECT, DEMO_RUN_ID, FakeRun, ok


class TestArchive:
    def test_it_builds_a_real_archive_and_excludes_the_venv(
        self, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """.venv leads the exclusion list for the reason the package exists.

        One machine's has absolute paths baked into it, and it is the exact
        thing two dispatches must not share.
        """
        destination = tmp_path / "tree.tgz"

        payload = staging.archive(repo, manifest.build_tree(repo, DEMO_PROJECT), destination)

        assert destination.is_file()
        # A real gzip member, not merely some bytes: 1f 8b is the magic, and
        # the payload must be exactly what landed on disk or the digest the
        # node is asked to match would be of something else.
        assert payload[:2] == b"\x1f\x8b"
        assert payload == destination.read_bytes()
        listing = _test_hooks.run(["tar", "-tzf", str(destination)])["stdout"]
        assert "Makefile" in listing
        assert ".venv" not in listing

    def test_the_dependency_a_lockfile_resolves_against_is_inside(
        self, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """The defect the first real dispatch found, as a regression.

        ``tools/fleet`` was staged as one directory. Its pyproject declares
        ``platform-core`` at ``../../libs/platform_core``, so poetry on the
        node could not have resolved the lockfile at all -- and would have
        reported that as the project's fault.
        """
        destination = tmp_path / "tree.tgz"

        staging.archive(repo, manifest.build_tree(repo, DEMO_PROJECT), destination)

        listing = _test_hooks.run(["tar", "-tzf", str(destination)])["stdout"]
        assert f"{DEMO_DEPENDENCY}/pyproject.toml" in listing

    def test_the_shared_launcher_directory_is_inside(
        self, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """Every Makefile's test target calls ..\\..\\scripts\\run-tests.ps1."""
        destination = tmp_path / "tree.tgz"

        staging.archive(repo, manifest.build_tree(repo, DEMO_PROJECT), destination)

        listing = _test_hooks.run(["tar", "-tzf", str(destination)])["stdout"]
        for path in manifest.SHARED_PATHS:
            assert path in listing

    def test_the_extracted_layout_keeps_the_dependency_relative_to_the_project(
        self, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """``../base`` has to resolve on the node exactly as it does here.

        Unpacked into a stage directory, the members keep their repo-relative
        names, so a manifest's relative path needs no rewriting.
        """
        destination = tmp_path / "tree.tgz"
        staging.archive(repo, manifest.build_tree(repo, DEMO_PROJECT), destination)
        unpacked = tmp_path / "unpacked"
        unpacked.mkdir()

        _test_hooks.run(["tar", "-xzmf", str(destination), "-C", str(unpacked)])

        declared = (unpacked / DEMO_PROJECT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'path = "../base"' in declared
        assert (unpacked / DEMO_PROJECT / ".." / "base" / "pyproject.toml").resolve().is_file()

    def test_a_project_that_does_not_exist_is_refused(
        self, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        with pytest.raises(AppError) as excinfo:
            staging.archive(repo, ("libs/absent",), tmp_path / "tree.tgz")

        assert excinfo.value.code is FleetErrorCode.STAGE_ARCHIVE_UNREADABLE

    def test_an_archive_with_no_members_is_refused(self, repo: pathlib.Path) -> None:
        """It would stage, extract to nothing, and fail at make instead."""
        with pytest.raises(ValueError) as excinfo:
            staging.archive(repo, (), repo / "tree.tgz")

        assert "at least one member" in str(excinfo.value)

    def test_the_digest_is_a_full_length_sha256(self) -> None:
        assert len(staging.digest(b"payload")) == 64

    def test_encoding_round_trips_through_base64(self) -> None:
        """Base64 because raw bytes do not survive ssh into PowerShell."""
        payload = bytes(range(256))

        assert base64.b64decode(staging.encode(payload)) == payload


class TestStage:
    def test_a_verified_archive_is_unpacked(self) -> None:
        payload = b"archive-bytes"
        runner = FakeRun(
            [
                ok(""),  # send mkdir script
                ok(""),  # run mkdir
                ok(""),  # send the base64
                ok(""),  # send reassemble script
                ok(staging.digest(payload)),  # run reassemble -> digest
                ok(""),  # send extract script
                ok(""),  # run extract
            ]
        )
        _test_hooks.run = runner

        target = staging.stage(
            "lavender", run_id=DEMO_RUN_ID, stage_root="C:/fleet/stage", payload=payload
        )

        assert target == f"C:/fleet/stage/{DEMO_RUN_ID}"
        assert any(b"tar -xzmf" in (sent or b"") for sent in runner.stdin)

    def test_a_mismatched_digest_refuses_before_unpacking(self) -> None:
        """Nothing is extracted, so no unverified tree lands where make looks."""
        runner = FakeRun([ok(""), ok(""), ok(""), ok(""), ok("0" * 64)])
        _test_hooks.run = runner

        with pytest.raises(AppError) as excinfo:
            staging.stage(
                "lavender", run_id=DEMO_RUN_ID, stage_root="C:/fleet/stage", payload=b"bytes"
            )

        assert excinfo.value.code is FleetErrorCode.STAGE_DIGEST_MISMATCH
        assert "nothing has been unpacked" in excinfo.value.message
        assert not any(b"tar -xzmf" in (sent or b"") for sent in runner.stdin)


class TestBuildScript:
    def test_it_runs_the_recipe_in_the_project(self) -> None:
        body = dispatch.build_script(target="C:/s/run-1", project=DEMO_PROJECT, workers=6)

        assert f"Set-Location -LiteralPath 'C:/s/run-1/{DEMO_PROJECT}'" in body
        assert "make check" in body

    def test_it_pins_the_worker_count(self) -> None:
        body = dispatch.build_script(target="C:/s/run-1", project=DEMO_PROJECT, workers=6)

        assert "PYTEST_XDIST_AUTO_NUM_WORKERS = '6'" in body

    def test_it_records_the_status_last(self) -> None:
        """The result file's absence is how a run is known to be unfinished.

        Written after the recipe, so it can never exist while make is still
        going -- which is what lets `fleet-collect` treat absence as running.
        """
        body = dispatch.build_script(target="C:/s/run-1", project=DEMO_PROJECT, workers=6)
        lines = [line for line in body.splitlines() if line.strip()]

        assert lines[-1].startswith("$LASTEXITCODE")
        assert dispatch.RESULT_NAME in lines[-1]

    def test_it_reads_the_exit_code_and_not_the_success_flag(self) -> None:
        """`make` writes to stderr on a passing run; under redirection that
        sets $? false in PS 5.1 while $LASTEXITCODE stays correct."""
        body = dispatch.build_script(target="C:/s/run-1", project=DEMO_PROJECT, workers=6)

        assert "$LASTEXITCODE" in body
        assert "$?" not in body


class TestRegisterScript:
    def test_it_registers_and_starts_a_scheduled_task(self) -> None:
        """Not an ssh child. Windows OpenSSH puts that in a job object that
        dies with the connection, and this command returns immediately."""
        body = dispatch.register_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert "Register-ScheduledTask" in body
        assert "Start-ScheduledTask" in body

    def test_it_sets_priority_four(self) -> None:
        """Priority 7 is the Register-ScheduledTask default and sets LOW I/O.

        A run that inherits it crawls, and the symptom reads as a slow node
        rather than a misconfigured launch.
        """
        body = dispatch.register_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert "-Priority 4" in body
        assert "[TimeSpan]::Zero" in body
        assert "-LogonType S4U" in body

    def test_it_runs_the_build_by_path_and_never_inlines_it(self) -> None:
        """THE REGRESSION. Interpolating the build into -Argument split the
        task in two: PowerShell ended the single-quoted string at the first
        inner quote and bound the rest to -WorkingDirectory. Measured on
        sedona 2026-09-04; the task could not be started at all.
        """
        body = dispatch.register_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert f'-File "C:/s/run-1/{dispatch.BUILD_SCRIPT_NAME}"' in body
        assert "-Command" not in body
        assert "make check" not in body

    def test_the_argument_string_contains_no_single_quotes(self) -> None:
        """The mechanical form of the same defect, asserted directly.

        -Argument is passed as a single-quoted PowerShell string, so ANY
        single quote inside it terminates the argument early.
        """
        body = dispatch.register_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)
        argument = body.split("-Argument '", 1)[1].split("'\n", 1)[0]

        assert "'" not in argument

    def test_it_survives_the_lid_being_shut(self) -> None:
        """Two of the three nodes are laptops and both battery settings
        default to refusing: without these a dispatch to an unplugged sedona
        registers a task that never runs, and reports nothing.
        """
        body = dispatch.register_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert "-AllowStartIfOnBatteries" in body
        assert "-DontStopIfGoingOnBatteries" in body

    def test_it_waits_for_the_task_to_actually_start(self) -> None:
        """Start-ScheduledTask reports a refusal as a NON-terminating error.

        On 2026-09-04 it failed with 'Element not found', PowerShell exited 0,
        and the dispatch was recorded as running. The script now watches for
        the task to leave SCHED_S_TASK_HAS_NOT_RUN and throws if it does not.
        """
        body = dispatch.register_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert "$ErrorActionPreference = 'Stop'" in body
        assert str(dispatch.TASK_HAS_NOT_RUN) in body
        assert "throw" in body

    def test_the_task_name_is_the_one_cancel_stops(self) -> None:
        """Both come from dispatch.task_name, so a rename cannot make
        fleet-cancel report success having stopped nothing."""
        registered = dispatch.register_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert dispatch.task_name(DEMO_RUN_ID) in registered
        assert dispatch.task_name(DEMO_RUN_ID) in cancel.stop_script(DEMO_RUN_ID)


class TestOtherScripts:
    def test_the_extract_script_keeps_the_node_s_clock(self) -> None:
        """Without -m, a tree from a fast clock makes targets look fresh.

        The build then does nothing, which reads as a suite that passed
        instantly.
        """
        assert "-xzmf" in staging.extract_script("C:/s/run-1")

    def test_the_result_script_prints_nothing_while_running(self) -> None:
        """Absence is the signal, so an unfinished run is not read as exit 0."""
        body = dispatch.result_script("C:/s/run-1")

        assert "Test-Path" in body

    def test_the_stop_script_never_prompts(self) -> None:
        """There is nobody at the node to answer, and a prompt would hang."""
        assert "-Confirm:$false" in cancel.stop_script(DEMO_RUN_ID)
