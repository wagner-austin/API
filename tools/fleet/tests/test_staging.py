"""Building an archive and getting it onto a node, verified before it is used.

Only the ssh boundary is faked. The archive is a real tar built by the real
tar binary over a real temporary tree and the digest is a real SHA-256, so
what is exercised is the transport as it will actually behave -- including the
exclusion that keeps one machine's ``.venv`` off another, and the absence of
one that would have hidden a project's committed records.

The scripts a node is HANDED are tested in ``test_launch.py``, beside the
module that renders them.
"""

from __future__ import annotations

import base64
import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.core import (
    _test_hooks,
    dialect,
    dialect_linux,
    dialect_windows,
    manifest,
    names,
    staging,
)
from tests.conftest import DEMO_DEPENDENCY, DEMO_PROJECT, DEMO_RUN_ID, FakeRun, ok

#: The commit a staged companion is the export of, in these tests.
COMPANION_SHA = "9f1c0b7a2d3e4f5061728394a5b6c7d8e9f01234"

#: The deadline for the real tar calls these tests make over a tiny tree:
#: no listing here comes near it, so a result is about the archive and
#: never about the clock.
TAR_LISTING_SECONDS = 60


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
        listing = _test_hooks.run(
            ["tar", "-tzf", str(destination)], timeout_seconds=TAR_LISTING_SECONDS
        )["stdout"]
        assert "Makefile" in listing
        assert ".venv" not in listing

    def test_a_committed_runs_directory_is_carried(
        self, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """`runs` was excluded for an hour to stop archives compounding, and
        that hid 294 committed run documents under tools/hpc3/runs which its
        suite reads -- four tests failed on lavender for a reason that read as
        hpc3's fault. The archives were the error; they live outside the
        repository now, so nothing about a project's tree is hidden."""
        committed = repo / DEMO_PROJECT / "runs"
        committed.mkdir()
        (committed / "registry.json").write_text("{}\n", encoding="utf-8")
        destination = tmp_path / "tree.tgz"

        staging.archive(repo, manifest.build_tree(repo, DEMO_PROJECT), destination)

        listing = _test_hooks.run(
            ["tar", "-tzf", str(destination)], timeout_seconds=TAR_LISTING_SECONDS
        )["stdout"]
        assert "registry.json" in listing

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

        listing = _test_hooks.run(
            ["tar", "-tzf", str(destination)], timeout_seconds=TAR_LISTING_SECONDS
        )["stdout"]
        assert f"{DEMO_DEPENDENCY}/pyproject.toml" in listing

    def test_the_shared_launcher_directory_is_inside(
        self, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """Every Makefile includes scripts/make/shell.mk and calls tools/maketools."""
        destination = tmp_path / "tree.tgz"

        staging.archive(repo, manifest.build_tree(repo, DEMO_PROJECT), destination)

        listing = _test_hooks.run(
            ["tar", "-tzf", str(destination)], timeout_seconds=TAR_LISTING_SECONDS
        )["stdout"]
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

        _test_hooks.run(
            ["tar", "-xzmf", str(destination), "-C", str(unpacked)],
            timeout_seconds=TAR_LISTING_SECONDS,
        )

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
                ok(""),  # send the git-init script
                ok(""),  # run git init
            ]
        )
        _test_hooks.run = runner

        target = staging.stage(
            "lavender",
            platform="windows",
            run_id=DEMO_RUN_ID,
            stage_root="C:/fleet/stage",
            payload=payload,
        )

        assert target == f"C:/fleet/stage/{DEMO_RUN_ID}"
        assert any(b"tar -xzmf" in (sent or b"") for sent in runner.stdin)
        # Every script went out under the Windows dialect's name and runner.
        assert runner.calls[0][-1].endswith(f"mkdir-{DEMO_RUN_ID}.ps1' -Encoding utf8\"")
        assert runner.calls[1][-6:-1] == dialect_windows.POWERSHELL_INVOCATION

    def test_a_linux_node_is_staged_in_sh(self) -> None:
        """The same five acts, in the other dialect: written through a
        mkdir-and-cat command, run by /bin/sh, reassembled with base64 and
        sha256sum, and the shared tar and git steps unchanged."""
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
                ok(""),  # send the git-init script
                ok(""),  # run git init
            ]
        )
        _test_hooks.run = runner

        target = staging.stage(
            "diphtheria",
            platform="linux",
            run_id=DEMO_RUN_ID,
            stage_root="/home/corvis/fleet/stage",
            payload=payload,
        )

        assert target == f"/home/corvis/fleet/stage/{DEMO_RUN_ID}"
        assert runner.calls[0][-1] == (
            f"mkdir -p \"$(dirname '/home/corvis/fleet/stage/mkdir-{DEMO_RUN_ID}.sh')\" && "
            f"cat > '/home/corvis/fleet/stage/mkdir-{DEMO_RUN_ID}.sh'"
        )
        made = f"/home/corvis/fleet/stage/mkdir-{DEMO_RUN_ID}.sh"
        assert runner.calls[1][-2:] == ("/bin/sh", made)
        assert runner.calls[3][-1].endswith(f"{target}/reassemble.sh'")
        assert b"sha256sum" in (runner.stdin[3] or b"")
        # Both carry sh's prologue, whose `set -e` is what ends the script at
        # a command that failed -- the other dialect has to be asked for that
        # and was not, which is the silent stage this pair now pins.
        assert runner.stdin[5] == (
            f"{dialect_linux.PROLOGUE}tar -xzmf '{target}/tree.tgz' -C '{target}'\n".encode()
        )
        assert (
            runner.stdin[7]
            == (
                f"{dialect_linux.PROLOGUE}git -C '{target}' init --quiet\n"
                f"git -C '{target}' add --all\n"
                f"git -C '{target}' -c user.name='fleet' -c user.email='fleet@corvis.invalid' "
                f"commit --quiet --message 'fleet export {DEMO_RUN_ID}'\n"
            ).encode()
        )
        assert not any("powershell" in argument for call in runner.calls for argument in call)

    def test_the_staged_tree_becomes_a_git_repository(self) -> None:
        """Ruff applies .gitignore ONLY inside a git repository, so without
        this a staged build lints every path the repository deliberately
        excludes. Measured on lavender 2026-09-04: tools/hpc3 reported 902
        errors in build artifacts, and the same tree with `git init` run in
        it reported All checks passed."""
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
                ok(""),  # send the git-init script
                ok(""),  # run git init
            ]
        )
        _test_hooks.run = runner

        staging.stage(
            "lavender",
            platform="windows",
            run_id=DEMO_RUN_ID,
            stage_root="C:/fleet/stage",
            payload=payload,
        )

        sent = [payload or b"" for payload in runner.stdin]
        assert any(b"git -C" in body and b"init" in body for body in sent)
        # After the tree lands: before extraction there is nothing for the
        # ignore rules to cover, and the .gitignore that gives them content
        # arrives with the tree.
        target = f"C:/fleet/stage/{DEMO_RUN_ID}"
        spoken = dialect.for_platform("windows")
        assert sent.index(
            spoken.checked_script(
                dialect.extract_commands(f"{target}/{names.ARCHIVE_NAME}", target)
            ).encode()
        ) < (
            sent.index(
                spoken.checked_script(
                    dialect.init_repository_commands(target, DEMO_RUN_ID)
                ).encode()
            )
        )

    def test_the_extract_script_keeps_the_node_s_clock(self) -> None:
        """Without -m, a tree from a fast clock makes targets look fresh.

        The build then does nothing, which reads as a suite that passed
        instantly.
        """
        assert "-xzmf" in dialect.extract_commands("C:/s/run-1/tree.tgz", "C:/s/run-1")[0]

    def test_a_mismatched_digest_refuses_before_unpacking(self) -> None:
        """Nothing is extracted, so no unverified tree lands where make looks."""
        runner = FakeRun([ok(""), ok(""), ok(""), ok(""), ok("0" * 64)])
        _test_hooks.run = runner

        with pytest.raises(AppError) as excinfo:
            staging.stage(
                "lavender",
                platform="windows",
                run_id=DEMO_RUN_ID,
                stage_root="C:/fleet/stage",
                payload=b"bytes",
            )

        assert excinfo.value.code is FleetErrorCode.STAGE_DIGEST_MISMATCH
        assert "nothing has been unpacked" in excinfo.value.message
        assert not any(b"tar -xzmf" in (sent or b"") for sent in runner.stdin)


class TestStagingACompanion:
    """The repository a project's check reads BESIDE its export (MCPs board
    task 0515040d): what the node is asked to do, in what order, and what the
    directory it lands in is guaranteed to hold."""

    def test_the_tree_lands_beside_the_exports_and_is_committed_at_its_sha(self) -> None:
        """``<stage_root>/<directory>`` is ``../<directory>`` from an export
        root, which is where a workstation keeps the same checkout, so the
        recipe names one path on either machine. The commit is what a check
        reading ``git show HEAD:`` needs, and it is made here because a node
        has no HEAD until one is."""
        payload = b"companion-bytes"
        runner = FakeRun([ok("")] * 6 + [ok(staging.digest(payload))] + [ok("")] * 4)
        _test_hooks.run = runner

        where = staging.stage_companion(
            "lavender",
            platform="windows",
            stage_root="C:/fleet/stage",
            directory="MCPs",
            sha=COMPANION_SHA,
            payload=payload,
        )

        sent = [body or b"" for body in runner.stdin]
        assert where == "C:/fleet/stage/MCPs"
        assert sent[0] == dialect_windows.WindowsDialect().reset_directory_script(where).encode()
        assert sent[4] == staging.encode(payload).encode()
        spoken = dialect.for_platform("windows")
        assert (
            sent[7]
            == spoken.checked_script(
                dialect.extract_commands(f"C:/fleet/stage/MCPs.stage/{names.ARCHIVE_NAME}", where)
            ).encode()
        )
        assert (
            sent[9]
            == spoken.checked_script(
                dialect.companion_repository_commands(where, COMPANION_SHA)
            ).encode()
        )

    def test_the_archive_never_enters_the_tree_that_is_committed(self) -> None:
        """A ``tree.tgz`` at the root of a staged workspace would be a file the
        workspace does not have, sitting in the tree a check compares against
        it, so the transport files stay in a staging directory beside it."""
        payload = b"companion-bytes"
        runner = FakeRun([ok("")] * 6 + [ok(staging.digest(payload))] + [ok("")] * 4)
        _test_hooks.run = runner

        staging.stage_companion(
            "lavender",
            platform="windows",
            stage_root="C:/fleet/stage",
            directory="MCPs",
            sha=COMPANION_SHA,
            payload=payload,
        )

        written = [argument for call in runner.calls for argument in call]
        assert not any(f"C:/fleet/stage/MCPs/{names.ENCODED_NAME}" in text for text in written)
        assert any(f"C:/fleet/stage/MCPs.stage/{names.ENCODED_NAME}" in text for text in written)

    def test_the_directory_is_replaced_rather_than_unpacked_over(self) -> None:
        """Every run carrying a companion writes the same directory, so a file
        the workspace has since deleted would otherwise survive into a tree
        that is then committed AS the workspace."""
        payload = b"companion-bytes"
        runner = FakeRun([ok("")] * 6 + [ok(staging.digest(payload))] + [ok("")] * 4)
        _test_hooks.run = runner

        staging.stage_companion(
            "diphtheria",
            platform="linux",
            stage_root="/home/corvis/fleet/stage",
            directory="MCPs",
            sha=COMPANION_SHA,
            payload=payload,
        )

        sent = [body or b"" for body in runner.stdin]
        assert b"rm -rf '/home/corvis/fleet/stage/MCPs'" in sent[0]
        assert sent.index(sent[0]) < sent.index(
            dialect.for_platform("linux")
            .checked_script(
                dialect.extract_commands(
                    f"/home/corvis/fleet/stage/MCPs.stage/{names.ARCHIVE_NAME}",
                    "/home/corvis/fleet/stage/MCPs",
                )
            )
            .encode()
        )

    def test_a_mismatched_digest_names_the_companion_and_unpacks_nothing(self) -> None:
        """The message says WHICH archive disagreed, because a dispatch now
        carries more than one and a mismatch that named none would leave a
        reader looking at the wrong transfer."""
        runner = FakeRun([ok("")] * 6 + [ok("0" * 64)])
        _test_hooks.run = runner

        with pytest.raises(AppError) as excinfo:
            staging.stage_companion(
                "lavender",
                platform="windows",
                stage_root="C:/fleet/stage",
                directory="MCPs",
                sha=COMPANION_SHA,
                payload=b"bytes",
            )

        assert excinfo.value.code is FleetErrorCode.STAGE_DIGEST_MISMATCH
        assert f"the MCPs companion at {COMPANION_SHA}" in excinfo.value.message
        assert not any(b"tar -xzmf" in (sent or b"") for sent in runner.stdin)
