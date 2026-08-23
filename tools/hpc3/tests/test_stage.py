"""Tests for staging, including the remote re-verification.

The central assertion is that a local-only check is insufficient: a file that
verifies locally and arrives corrupted must still fail. That is the whole
reason the digest is computed twice.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.stage import StagedFile, StageManifest
from hpc3.core import audit
from hpc3.core.stage import stage_manifest, stage_one
from tests.conftest import FakeRun, LoggedEvent, write_file

_B = b"the marker predicts extraction accuracy.\n"
_C = b"the diluted corpus degrades the item set.\n"
_B_DIGEST = hashlib.sha256(_B).hexdigest()
_C_DIGEST = hashlib.sha256(_C).hexdigest()

_DEST = "/pub/wagnera3/corpora"


def _record(name: str, payload: bytes) -> StagedFile:
    """Build a staged-file record describing a payload.

    Args:
        name: Filename on both sides.
        payload: The bytes the record describes.

    Returns:
        A matching record.
    """
    return StagedFile(
        name=name, sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload)
    )


_PROVENANCE = {"wiki_commit": "176bb8c", "emitter": "emit_corpus.py"}


class TestStageOne:
    def test_a_good_file_is_written_and_verified(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        write_file(tmp_path / "armB.txt", _B)
        fake_run.add("sha256sum", stdout=f"{_B_DIGEST}  {_DEST}/armB.txt\n")

        placed = stage_one("hpc3", tmp_path, _DEST, _record("armB.txt", _B))

        assert placed == f"{_DEST}/armB.txt"
        assert fake_run.calls[0].stdin_bytes == _B
        assert fake_run.commands() == [
            f"cat > '{_DEST}/armB.txt'",
            f"sha256sum '{_DEST}/armB.txt'",
        ]

    def test_corruption_in_transit_is_caught_by_the_remote_digest(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """The local file is perfect; what arrived is not. Only step three sees it."""
        write_file(tmp_path / "armB.txt", _B)
        fake_run.add("sha256sum", stdout=f"{_C_DIGEST}  {_DEST}/armB.txt\n")

        with pytest.raises(AppError) as excinfo:
            stage_one("hpc3", tmp_path, _DEST, _record("armB.txt", _B))

        assert excinfo.value.code is Hpc3ErrorCode.DIGEST_MISMATCH
        assert "on the cluster" in excinfo.value.message

    def test_a_wrong_local_file_never_reaches_the_network(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        write_file(tmp_path / "armB.txt", _C)
        with pytest.raises(AppError) as excinfo:
            stage_one("hpc3", tmp_path, _DEST, _record("armB.txt", _B))
        assert excinfo.value.code is Hpc3ErrorCode.DIGEST_MISMATCH
        assert fake_run.calls == []

    def test_unparsable_remote_output_is_a_command_failure(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        write_file(tmp_path / "armB.txt", _B)
        fake_run.add("sha256sum", stdout="sha256sum: command not found\n")
        with pytest.raises(AppError) as excinfo:
            stage_one("hpc3", tmp_path, _DEST, _record("armB.txt", _B))
        assert excinfo.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED


class TestStageManifest:
    def test_it_creates_the_destination_then_places_each_file(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        write_file(tmp_path / "armB.txt", _B)
        write_file(tmp_path / "armC.txt", _C)
        fake_run.add("sha256sum '/pub/wagnera3/corpora/armB.txt'", stdout=f"{_B_DIGEST}  x\n")
        fake_run.add("sha256sum '/pub/wagnera3/corpora/armC.txt'", stdout=f"{_C_DIGEST}  x\n")

        manifest = StageManifest(
            destination=_DEST,
            files=[_record("armB.txt", _B), _record("armC.txt", _C)],
            provenance=_PROVENANCE,
        )
        placed = stage_manifest("hpc3", tmp_path, manifest)

        assert placed == [f"{_DEST}/armB.txt", f"{_DEST}/armC.txt"]
        assert fake_run.commands()[0] == f"mkdir -p '{_DEST}'"

    def test_the_first_bad_file_stops_the_run(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """No partial success: a corpus staged in part is not a corpus."""
        write_file(tmp_path / "armB.txt", _B)
        fake_run.add("sha256sum", stdout=f"{_B_DIGEST}  x\n")

        manifest = StageManifest(
            destination=_DEST,
            files=[_record("armB.txt", _B), _record("armC.txt", _C)],
            provenance=_PROVENANCE,
        )
        with pytest.raises(AppError) as excinfo:
            stage_manifest("hpc3", tmp_path, manifest)

        assert excinfo.value.code is Hpc3ErrorCode.MANIFEST_FILE_MISSING
        assert "armC.txt" in excinfo.value.message


class TestStageAudit:
    def test_a_fully_verified_stage_is_recorded_once(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, logged: list[LoggedEvent]
    ) -> None:
        write_file(tmp_path / "armB.txt", _B)
        write_file(tmp_path / "armC.txt", _C)
        fake_run.add("sha256sum '/pub/wagnera3/corpora/armB.txt'", stdout=f"{_B_DIGEST}  x\n")
        fake_run.add("sha256sum '/pub/wagnera3/corpora/armC.txt'", stdout=f"{_C_DIGEST}  x\n")

        manifest = StageManifest(
            destination=_DEST,
            files=[_record("armB.txt", _B), _record("armC.txt", _C)],
            provenance=_PROVENANCE,
        )
        stage_manifest("hpc3", tmp_path, manifest)

        assert [event.event for event in logged] == [audit.FILES_STAGED]
        assert logged[0].fields == {
            "host": "hpc3",
            "destination": _DEST,
            "files": 2,
            "provenance": "emitter=emit_corpus.py wiki_commit=176bb8c",
        }

    def test_a_partial_stage_records_nothing(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, logged: list[LoggedEvent]
    ) -> None:
        """One event per file would record a partial stage as successes."""
        write_file(tmp_path / "armB.txt", _B)
        fake_run.add("sha256sum", stdout=f"{_B_DIGEST}  x\n")

        manifest = StageManifest(
            destination=_DEST,
            files=[_record("armB.txt", _B), _record("armC.txt", _C)],
            provenance=_PROVENANCE,
        )
        with pytest.raises(AppError):
            stage_manifest("hpc3", tmp_path, manifest)
        assert logged == []
