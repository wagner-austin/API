"""Building an archive and getting it onto a node, verified before it is used.

Only the ssh boundary is faked. The archive is a real tar built by the real
tar binary over a real temporary tree and the digest is a real SHA-256, so
what is exercised is the transport as it will actually behave -- including the
exclusions that keep one machine's ``.venv``, and the previous dispatch's own
archive, off another.

The scripts a node is HANDED are tested in ``test_launch.py``, beside the
module that renders them.
"""

from __future__ import annotations

import base64
import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.core import _test_hooks, manifest, staging
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

    def test_a_previous_dispatch_s_archive_is_not_carried(
        self, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """Measured 2026-09-04: five consecutive dispatches of tools/fleet
        produced archives of 185 KB, 1.4 MB, 4.5 MB, 13.5 MB and 20.7 MB,
        each one staging its predecessors out of `runs/`. Nothing on the node
        needs that directory -- the shared launcher creates it."""
        stale = repo / DEMO_PROJECT / "runs"
        stale.mkdir()
        (stale / "previous.tgz").write_text("x" * 8192, encoding="utf-8")
        destination = tmp_path / "tree.tgz"

        staging.archive(repo, manifest.build_tree(repo, DEMO_PROJECT), destination)

        listing = _test_hooks.run(["tar", "-tzf", str(destination)])["stdout"]
        assert "previous.tgz" not in listing
        assert "runs" not in listing

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

    def test_a_records_directory_that_does_not_exist_yet_is_created(
        self, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """The ordinary first run in a fresh workspace. The archive is built
        before any record is appended, so nothing has made the directory --
        and tar will not make it, so the dispatch failed at tar with a message
        about a path rather than about staging."""
        destination = tmp_path / "never-made" / "deeper" / "tree.tgz"

        staging.archive(repo, manifest.build_tree(repo, DEMO_PROJECT), destination)

        assert destination.is_file()

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

    def test_the_extract_script_keeps_the_node_s_clock(self) -> None:
        """Without -m, a tree from a fast clock makes targets look fresh.

        The build then does nothing, which reads as a suite that passed
        instantly.
        """
        assert "-xzmf" in staging.extract_script("C:/s/run-1")

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
