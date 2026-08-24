"""Tests for the three services a cluster run supplies for itself.

The refusals carry the weight here. On a compute node there is nothing to fall
back to: no data bank to fetch a missing corpus from, no upload service to
re-request an artifact from. Every one of these failures has to be a refusal
with a message naming the fix, because the alternative is a job that burns
hours and then cannot say what was wrong.
"""

from __future__ import annotations

import hashlib
import pathlib
import tarfile

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.cluster.stores import LocalArtifacts, StagedCorpus

_CORPUS = b"the quick brown fox\n" * 64
_DIGEST = hashlib.sha256(_CORPUS).hexdigest()


def _staged(tmp_path: pathlib.Path) -> pathlib.Path:
    """Place a corpus the way hpc3-stage would.

    Args:
        tmp_path: Test directory.

    Returns:
        The staging directory.
    """
    root = tmp_path / "corpora"
    root.mkdir()
    (root / _DIGEST).write_bytes(_CORPUS)
    return root


class TestStagedCorpus:
    def test_it_resolves_a_staged_file_by_its_digest(self, tmp_path: pathlib.Path) -> None:
        found = StagedCorpus(_staged(tmp_path)).fetch(_DIGEST)
        assert found.read_bytes() == _CORPUS

    def test_a_missing_corpus_is_refused_with_the_code_the_service_uses(
        self, tmp_path: pathlib.Path
    ) -> None:
        with pytest.raises(AppError) as excinfo:
            StagedCorpus(_staged(tmp_path)).fetch("0" * 64)
        assert excinfo.value.code is ModelTrainerErrorCode.CORPUS_NOT_FOUND

    def test_the_refusal_names_what_is_actually_staged(self, tmp_path: pathlib.Path) -> None:
        """The fix is a staging step the operator forgot, so the message has
        to show what IS there -- a digest that differs in one character
        cannot be spotted any other way."""
        with pytest.raises(AppError) as excinfo:
            StagedCorpus(_staged(tmp_path)).fetch("0" * 64)
        assert _DIGEST in excinfo.value.message
        assert "hpc3-stage" in excinfo.value.message

    def test_an_absent_staging_directory_is_refused_not_crashed(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Nothing staged at all is the likeliest first-run mistake, and it
        must read the same as a missing file rather than as a stack trace."""
        with pytest.raises(AppError) as excinfo:
            StagedCorpus(tmp_path / "never-made").fetch(_DIGEST)
        assert excinfo.value.code is ModelTrainerErrorCode.CORPUS_NOT_FOUND
        assert "Staged there: []" in excinfo.value.message

    def test_a_directory_named_like_a_digest_is_not_a_corpus(self, tmp_path: pathlib.Path) -> None:
        root = tmp_path / "corpora"
        (root / _DIGEST).mkdir(parents=True)
        with pytest.raises(AppError):
            StagedCorpus(root).fetch(_DIGEST)


class TestLocalArtifacts:
    def _run_dir(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """Build a directory standing in for a finished run's output.

        Args:
            tmp_path: Test directory.

        Returns:
            The directory to pack.
        """
        run = tmp_path / "run-abc"
        (run / "nested").mkdir(parents=True)
        (run / "config.json").write_text('{"seed": 42}', encoding="utf-8")
        (run / "nested" / "weights.bin").write_bytes(b"\x00\x01\x02" * 100)
        return run

    def test_the_filename_carries_the_artifact_name_and_its_digest(
        self, tmp_path: pathlib.Path
    ) -> None:
        store = LocalArtifacts(tmp_path / "artifacts")
        result = store.upload_artifact(
            self._run_dir(tmp_path), artifact_name="run-abc", request_id="r1"
        )
        expected = tmp_path / "artifacts" / f"run-abc-{result['sha256'][:12]}.tar.gz"
        assert expected.is_file()

    def test_no_partial_file_is_left_behind(self, tmp_path: pathlib.Path) -> None:
        """The tarball is written under a .partial name and renamed once its
        digest is known, so a reader never sees a name it can trust holding
        bytes still being written."""
        store = LocalArtifacts(tmp_path / "artifacts")
        store.upload_artifact(self._run_dir(tmp_path), artifact_name="run-abc", request_id="r1")
        assert list((tmp_path / "artifacts").glob("*.partial")) == []

    def test_the_reported_digest_is_of_the_bytes_on_disk(self, tmp_path: pathlib.Path) -> None:
        """Computed from the written file rather than reported from memory,
        so the value a run records is a fact about the artifact and is
        checkable by the same command that checks a staged input."""
        store = LocalArtifacts(tmp_path / "artifacts")
        result = store.upload_artifact(
            self._run_dir(tmp_path), artifact_name="run-abc", request_id="r1"
        )
        written = (tmp_path / "artifacts" / f"run-abc-{result['sha256'][:12]}.tar.gz").read_bytes()
        assert result["sha256"] == hashlib.sha256(written).hexdigest()
        assert result["file_id"] == result["sha256"]
        assert result["size"] == len(written)

    def test_it_creates_its_output_directory(self, tmp_path: pathlib.Path) -> None:
        store = LocalArtifacts(tmp_path / "deep" / "artifacts")
        store.upload_artifact(self._run_dir(tmp_path), artifact_name="a", request_id="r1")
        assert len(list((tmp_path / "deep" / "artifacts").glob("a-*.tar.gz"))) == 1

    def test_an_artifact_round_trips_through_the_store(self, tmp_path: pathlib.Path) -> None:
        store = LocalArtifacts(tmp_path / "artifacts")
        result = store.upload_artifact(
            self._run_dir(tmp_path), artifact_name="run-abc", request_id="r1"
        )
        extracted = store.download_artifact(
            result["file_id"],
            dest_dir=tmp_path / "out",
            request_id="r2",
            expected_root="run-abc",
        )
        assert (extracted / "config.json").read_text(encoding="utf-8") == '{"seed": 42}'
        assert (extracted / "nested" / "weights.bin").read_bytes() == b"\x00\x01\x02" * 100

    def test_a_second_run_of_one_project_does_not_erase_the_first(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Every run of a project picks the same artifact NAME. Keying the
        file on the name alone made the second overwrite the first, and the
        first's recorded file_id then named bytes that were gone -- a
        silently unreachable artifact, which this test exists to catch."""
        store = LocalArtifacts(tmp_path / "artifacts")
        run = self._run_dir(tmp_path)
        first = store.upload_artifact(run, artifact_name="same-name", request_id="r1")

        (run / "config.json").write_text('{"seed": 7}', encoding="utf-8")
        second = store.upload_artifact(run, artifact_name="same-name", request_id="r2")
        assert first["file_id"] != second["file_id"]

        older = store.download_artifact(
            first["file_id"],
            dest_dir=tmp_path / "first",
            request_id="r3",
            expected_root=run.name,
        )
        assert (older / "config.json").read_text(encoding="utf-8") == '{"seed": 42}'

    def test_it_finds_an_artifact_by_content_rather_than_by_its_name(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A caller holds a file_id, which is a digest. Searching by name
        would need it to also remember what the run was called."""
        store = LocalArtifacts(tmp_path / "artifacts")
        run = self._run_dir(tmp_path)
        result = store.upload_artifact(run, artifact_name="whatever", request_id="r1")
        found = store.download_artifact(
            result["file_id"],
            dest_dir=tmp_path / "out",
            request_id="r2",
            expected_root=run.name,
        )
        assert found.is_dir()

    def test_an_unknown_digest_is_refused(self, tmp_path: pathlib.Path) -> None:
        store = LocalArtifacts(tmp_path / "artifacts")
        store.upload_artifact(self._run_dir(tmp_path), artifact_name="a", request_id="r1")
        with pytest.raises(AppError) as excinfo:
            store.download_artifact(
                "f" * 64, dest_dir=tmp_path / "out", request_id="r2", expected_root="a"
            )
        assert excinfo.value.code is ModelTrainerErrorCode.ARTIFACT_DOWNLOAD_FAILED

    def test_downloading_from_an_empty_store_is_refused(self, tmp_path: pathlib.Path) -> None:
        store = LocalArtifacts(tmp_path / "never-written")
        with pytest.raises(AppError) as excinfo:
            store.download_artifact(
                "f" * 64, dest_dir=tmp_path / "out", request_id="r2", expected_root="a"
            )
        assert excinfo.value.code is ModelTrainerErrorCode.ARTIFACT_DOWNLOAD_FAILED

    def test_a_tarball_without_the_expected_root_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Extraction succeeding is not the same as getting what was asked
        for, and a caller that trusted the path would read an empty run."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        other = tmp_path / "something-else"
        other.mkdir()
        (other / "f.txt").write_text("x", encoding="utf-8")
        target = artifacts / "mislabelled.tar.gz"
        with tarfile.open(target, "w:gz") as tar:
            tar.add(other, arcname=other.name)

        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        store = LocalArtifacts(artifacts)
        with pytest.raises(AppError) as excinfo:
            store.download_artifact(
                digest, dest_dir=tmp_path / "out", request_id="r", expected_root="run-abc"
            )
        assert excinfo.value.code is ModelTrainerErrorCode.ARTIFACT_DOWNLOAD_FAILED
        assert "run-abc" in excinfo.value.message
