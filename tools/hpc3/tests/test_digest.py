"""Tests for digest computation and verification.

Real files on disk rather than fakes: the production ``read_bytes`` and
``file_exists`` hooks work on real paths, so exercising them costs nothing
and proves the code against the filesystem it will actually meet.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.stage import StagedFile
from hpc3.core.digest import (
    check_remote_digest,
    parse_remote_digest,
    read_and_verify,
    sha256_hex,
)
from tests.conftest import write_file

_PAYLOAD = b"the marker predicts extraction accuracy.\n"
_DIGEST = hashlib.sha256(_PAYLOAD).hexdigest()


def _record(**overrides: str | int) -> StagedFile:
    """Build a staged-file record matching the payload.

    Args:
        **overrides: Fields to replace.

    Returns:
        A record describing ``_PAYLOAD`` unless overridden.
    """
    name = overrides.get("name", "armB.txt")
    sha = overrides.get("sha256", _DIGEST)
    size = overrides.get("size_bytes", len(_PAYLOAD))
    assert isinstance(name, str)
    assert isinstance(sha, str)
    assert isinstance(size, int)
    return StagedFile(name=name, sha256=sha, size_bytes=size)


class TestSha256Hex:
    def test_it_matches_hashlib(self) -> None:
        assert sha256_hex(_PAYLOAD) == _DIGEST

    def test_it_is_lowercase_hex_of_the_right_length(self) -> None:
        digest = sha256_hex(b"")
        assert len(digest) == 64
        assert digest == digest.lower()


class TestReadAndVerify:
    def test_a_matching_file_returns_its_bytes(self, tmp_path: pathlib.Path) -> None:
        write_file(tmp_path / "armB.txt", _PAYLOAD)
        assert read_and_verify(tmp_path, _record()) == _PAYLOAD

    def test_a_missing_file_names_the_manifest_entry(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(AppError) as excinfo:
            read_and_verify(tmp_path, _record())
        assert excinfo.value.code is Hpc3ErrorCode.MANIFEST_FILE_MISSING
        assert "armB.txt" in excinfo.value.message

    def test_a_truncated_file_is_reported_by_length(self, tmp_path: pathlib.Path) -> None:
        write_file(tmp_path / "armB.txt", _PAYLOAD[:10])
        with pytest.raises(AppError) as excinfo:
            read_and_verify(tmp_path, _record())
        assert excinfo.value.code is Hpc3ErrorCode.DIGEST_MISMATCH
        assert "truncated" in excinfo.value.message

    def test_a_same_length_different_file_is_reported_as_wrong_not_truncated(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The 733-vs-773 case: right size, wrong corpus."""
        other = b"x" * len(_PAYLOAD)
        write_file(tmp_path / "armB.txt", other)
        with pytest.raises(AppError) as excinfo:
            read_and_verify(tmp_path, _record())
        assert excinfo.value.code is Hpc3ErrorCode.DIGEST_MISMATCH
        assert "wrong file" in excinfo.value.message


class TestParseRemoteDigest:
    def test_it_reads_sha256sum_output(self) -> None:
        assert parse_remote_digest(f"{_DIGEST}  /pub/x/armB.txt\n", "armB.txt") == _DIGEST

    def test_it_reads_a_digest_with_no_filename(self) -> None:
        assert parse_remote_digest(f"{_DIGEST}\n", "armB.txt") == _DIGEST

    def test_empty_output_is_a_command_failure_not_a_mismatch(self) -> None:
        with pytest.raises(AppError) as excinfo:
            parse_remote_digest("   \n", "armB.txt")
        assert excinfo.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED

    def test_a_short_first_token_is_a_command_failure(self) -> None:
        with pytest.raises(AppError) as excinfo:
            parse_remote_digest("sha256sum: not found\n", "armB.txt")
        assert excinfo.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED


class TestCheckRemoteDigest:
    def test_matching_digests_return_the_verified_value(self) -> None:
        assert check_remote_digest("armB.txt", _DIGEST, _DIGEST) == _DIGEST

    def test_a_mismatch_names_both_digests(self) -> None:
        other = sha256_hex(b"different")
        with pytest.raises(AppError) as excinfo:
            check_remote_digest("armB.txt", _DIGEST, other)
        assert excinfo.value.code is Hpc3ErrorCode.DIGEST_MISMATCH
        assert _DIGEST in excinfo.value.message
        assert other in excinfo.value.message
