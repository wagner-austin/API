"""Refusing to stage bytes the repository cannot produce.

The property under test is narrow and easy to state wrongly: the check fires
on TRACKED-AND-DIFFERING and on nothing else. An untracked file has no
repository copy to disagree with, so reporting it would refuse every corpus
and every image this package has ever staged.

The fake runner matches by the LAST argv element, which for these commands is
``HEAD:./<name>`` for the committed side and ``<name>`` for the working side.
The bare name is a substring of the other, so every test here scripts the
``HEAD:./`` rule FIRST -- first match wins. A test that scripted them the
other way round would answer the rev-parse with the working hash and pass
while checking nothing.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.stage import StagedFile, StageManifest
from hpc3.core.reproducible import (
    committed_blob,
    require_sources_reproducible,
    unreproducible,
    working_blob,
)
from tests.conftest import FakeRun

_COMMITTED = "1111111111111111111111111111111111111111"
_WORKING = "2222222222222222222222222222222222222222"
_PROVENANCE = {"what": "test", "why": "test"}


def _manifest(*names: str) -> StageManifest:
    """Build a manifest naming the given files.

    Args:
        names: File names to include.

    Returns:
        The manifest. Digests and sizes are irrelevant here -- this check
        never reads them, which is itself worth pinning: it compares the file
        against the repository, not against the manifest's own claims.
    """
    return StageManifest(
        destination="/pub/wagnera3/x",
        files=[StagedFile(name=name, sha256="0" * 64, size_bytes=1) for name in names],
        provenance=_PROVENANCE,
    )


class TestAskingTheRepository:
    """What git is asked, and what an absent answer means."""

    def test_an_untracked_path_reports_no_committed_blob(self, fake_run: FakeRun) -> None:
        """Empty is an ANSWER, not a failure.

        `rev-parse HEAD:./x` exits non-zero when the path is not in HEAD.
        That is the ordinary case for a corpus staged from outside the tree,
        and treating it as an error would refuse every one of them.
        """
        fake_run.add("HEAD:./corpus.jsonl", returncode=128, stderr="fatal: path does not exist")

        assert committed_blob(pathlib.Path("/src"), "corpus.jsonl") == ""

    def test_a_tracked_path_reports_its_committed_blob(self, fake_run: FakeRun) -> None:
        """The hash HEAD records, stripped of its newline."""
        fake_run.add("HEAD:./spec.json", stdout=f"{_COMMITTED}\n")

        assert committed_blob(pathlib.Path("/src"), "spec.json") == _COMMITTED

    def test_the_working_blob_is_hashed_by_git(self, fake_run: FakeRun) -> None:
        """Blob hashes, not contents: equal hashes means equal bytes."""
        fake_run.add("spec.json", stdout=f"{_WORKING}\n")

        assert working_blob(pathlib.Path("/src"), "spec.json") == _WORKING

    def test_a_file_git_cannot_hash_is_refused_rather_than_guessed(self, fake_run: FakeRun) -> None:
        """Unreadable, or no git at all.

        Staging on a guess about which of those it was would record a digest
        this check exists to make trustworthy.
        """
        fake_run.add("spec.json", returncode=128, stderr="fatal: not a git repository")

        with pytest.raises(AppError) as excinfo:
            _ = working_blob(pathlib.Path("/src"), "spec.json")

        assert excinfo.value.code is Hpc3ErrorCode.STAGE_SOURCE_NOT_REPRODUCIBLE
        assert "not a git repository" in excinfo.value.message


class TestWhichFilesAreReported:
    """Tracked-and-differing, and nothing else."""

    def test_a_tracked_file_that_differs_is_reported(self, fake_run: FakeRun) -> None:
        """The defect: a digest only this checkout could produce."""
        fake_run.add("HEAD:./spec.json", stdout=f"{_COMMITTED}\n")
        fake_run.add("spec.json", stdout=f"{_WORKING}\n")

        assert unreproducible(pathlib.Path("/src"), _manifest("spec.json")) == ("spec.json",)

    def test_a_tracked_file_that_matches_is_not_reported(self, fake_run: FakeRun) -> None:
        """The ordinary committed case."""
        fake_run.add("HEAD:./spec.json", stdout=f"{_COMMITTED}\n")
        fake_run.add("spec.json", stdout=f"{_COMMITTED}\n")

        assert unreproducible(pathlib.Path("/src"), _manifest("spec.json")) == ()

    def test_an_untracked_file_is_never_reported(self, fake_run: FakeRun) -> None:
        """A corpus staged from outside the tree must always pass.

        This is the case that would make the check useless if it fired: every
        corpus and every image this package stages lives outside the
        repository, and refusing them would refuse the package's main job.
        """
        fake_run.add("HEAD:./corpus.jsonl", returncode=128)
        fake_run.add("corpus.jsonl", stdout=f"{_WORKING}\n")

        assert unreproducible(pathlib.Path("/src"), _manifest("corpus.jsonl")) == ()

    def test_only_the_differing_file_of_several_is_named(self, fake_run: FakeRun) -> None:
        """Order follows the manifest, and clean neighbours stay unnamed."""
        fake_run.add("HEAD:./good.json", stdout=f"{_COMMITTED}\n")
        fake_run.add("HEAD:./bad.json", stdout=f"{_COMMITTED}\n")
        fake_run.add("good.json", stdout=f"{_COMMITTED}\n")
        fake_run.add("bad.json", stdout=f"{_WORKING}\n")

        found = unreproducible(pathlib.Path("/src"), _manifest("good.json", "bad.json"))

        assert found == ("bad.json",)


class TestTheRefusal:
    """What a caller is told, and when nothing is said at all."""

    def test_a_clean_manifest_passes_silently(self, fake_run: FakeRun) -> None:
        """No exception, no output: the check is invisible when it holds."""
        fake_run.add("HEAD:./spec.json", stdout=f"{_COMMITTED}\n")
        fake_run.add("spec.json", stdout=f"{_COMMITTED}\n")

        require_sources_reproducible(pathlib.Path("/src"), _manifest("spec.json"))

    def test_the_refusal_names_every_file_and_how_to_fix_it(self, fake_run: FakeRun) -> None:
        """A traceable code and a message that says what to do next.

        The CRLF hint is in the message deliberately: every real instance of
        this defect measured on 2026-09-09 was a text file checked out with
        CRLF against an LF blob, and a reader told only 'differs' would go
        looking for an edit they never made.
        """
        fake_run.add("HEAD:./spec.json", stdout=f"{_COMMITTED}\n")
        fake_run.add("spec.json", stdout=f"{_WORKING}\n")

        with pytest.raises(AppError) as excinfo:
            require_sources_reproducible(pathlib.Path("/src"), _manifest("spec.json"))

        assert excinfo.value.code is Hpc3ErrorCode.STAGE_SOURCE_NOT_REPRODUCIBLE
        assert "spec.json" in excinfo.value.message
        assert "gitattributes" in excinfo.value.message
