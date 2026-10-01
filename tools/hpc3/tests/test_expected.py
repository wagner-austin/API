"""Tests for the check that a manifest names the RIGHT bytes.

Every other check in this package verifies transport: the bytes that arrived
are the bytes that left. This one verifies identity, and it is the only one
that can catch the ablation's actual failure mode -- a corpus regenerated from
the wrong source state, whose manifest agrees with itself perfectly and is
comparable to nothing already published.

The digests below are the real ones: ``07ab4976...`` is arm B over the 733
prose-bearing pages at wiki commit ``176bb8c``, recorded in
``runs/manifests/corpus-B.json`` before any of this existed.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue
from platform_core.stage_manifest import StageManifest, decode_stage_manifest

from hpc3.core.expected import check_expected, read_expected_digests
from tests.conftest import write_file

_ARM_B = "07ab4976" + "a" * 56
_ARM_C = "4c91fbc1" + "b" * 56
_REGENERATED = "deadbeef" + "c" * 56

_PROVENANCE: dict[str, JSONValue] = {
    "wiki_commit": "176bb8c",
    "emitter": "extraction-eval/emit_corpus.py",
    "emitter_flags": "--seed 0 --dilution oscar_en.txt --dilution-ratio 7.0",
}


def _manifest(digest: str = _ARM_B) -> StageManifest:
    """Build a decoded manifest naming one file.

    Args:
        digest: The digest to claim for it.

    Returns:
        The validated manifest.
    """
    return decode_stage_manifest(
        {
            "destination": "/pub/wagnera3/abl/corpora",
            "files": [{"name": "armB.txt", "sha256": digest, "size_bytes": 4096}],
            "provenance": _PROVENANCE,
        }
    )


class TestReadExpectedDigests:
    def test_it_reads_a_plain_list(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "file_ids.txt"
        write_file(path, f"{_ARM_B}\n{_ARM_C}\n".encode())
        assert read_expected_digests(path) == {_ARM_B, _ARM_C}

    def test_it_reads_digests_out_of_a_json_record(self, tmp_path: pathlib.Path) -> None:
        """runs/manifests/corpus-B.json works without conversion."""
        path = tmp_path / "corpus-B.json"
        write_file(path, f'{{"target_sha256": "{_ARM_B}", "pages": 733}}'.encode())
        assert read_expected_digests(path) == {_ARM_B}

    def test_it_reads_sha256sum_output(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "sums"
        write_file(path, f"{_ARM_B}  armB.txt\n{_ARM_C}  armC.txt\n".encode())
        assert read_expected_digests(path) == {_ARM_B, _ARM_C}

    def test_an_uppercase_digest_is_not_read(self, tmp_path: pathlib.Path) -> None:
        """The manifest contract stores lowercase; a re-cased token is not it."""
        path = tmp_path / "sums"
        write_file(path, (_ARM_B.upper() + "\n" + _ARM_C + "\n").encode())
        assert read_expected_digests(path) == {_ARM_C}

    def test_a_record_naming_no_digest_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Otherwise every file is refused with a message about the manifest."""
        path = tmp_path / "notes.txt"
        write_file(path, b"the corpora live in runs/\n")
        with pytest.raises(AppError) as excinfo:
            read_expected_digests(path)
        assert excinfo.value.code is Hpc3ErrorCode.STAGED_DIGEST_UNEXPECTED
        assert "names no sha256 digest" in excinfo.value.message

    def test_a_too_short_token_is_not_a_digest(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "sums"
        write_file(path, b"07ab4976\n")
        with pytest.raises(AppError):
            read_expected_digests(path)


class TestCheckExpected:
    def test_the_published_corpus_passes(self, tmp_path: pathlib.Path) -> None:
        check_expected(_manifest(), {_ARM_B, _ARM_C}, source=tmp_path / "rec")

    def test_a_corpus_regenerated_from_the_wrong_state_is_refused(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The void-experiment case: self-consistent manifest, wrong bytes."""
        with pytest.raises(AppError) as excinfo:
            check_expected(_manifest(_REGENERATED), {_ARM_B, _ARM_C}, source=tmp_path / "rec")
        assert excinfo.value.code is Hpc3ErrorCode.STAGED_DIGEST_UNEXPECTED

    def test_the_message_names_the_file_the_digest_and_the_record(
        self, tmp_path: pathlib.Path
    ) -> None:
        record = tmp_path / "file_ids.txt"
        with pytest.raises(AppError) as excinfo:
            check_expected(_manifest(_REGENERATED), {_ARM_B}, source=record)
        assert "armB.txt" in excinfo.value.message
        assert _REGENERATED in excinfo.value.message
        assert str(record) in excinfo.value.message

    def test_extra_digests_in_the_record_admit_nothing(self, tmp_path: pathlib.Path) -> None:
        """The check is one-way: a fuller record does not loosen it."""
        with pytest.raises(AppError):
            check_expected(_manifest(_REGENERATED), {_ARM_B, _ARM_C}, source=tmp_path / "r")
