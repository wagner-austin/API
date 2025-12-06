from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from platform_ml.tarball import TarballError, create_tarball, extract_tarball


def test_create_tarball_raises_on_missing_dir(tmp_path: Path) -> None:
    src = tmp_path / "nope"
    with pytest.raises(TarballError):
        create_tarball(src, tmp_path / "x.tgz", root_name="root")


def test_extract_tarball_input_validation(tmp_path: Path) -> None:
    # Create tarball to use for extraction tests
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("x", encoding="utf-8")
    tar = tmp_path / "x.tgz"
    create_tarball(src, tar, root_name="root")
    # Non-file source raises
    with pytest.raises(TarballError):
        extract_tarball(tmp_path / "missing.tgz", tmp_path / "d", expected_root="root")
    # Empty root not allowed
    with pytest.raises(TarballError):
        extract_tarball(tar, tmp_path / "d2", expected_root="")


# Coverage for line 24: create_tarball with empty root_name
def test_create_tarball_empty_root_name(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("x", encoding="utf-8")
    with pytest.raises(TarballError, match="root_name must be non-empty"):
        create_tarball(src, tmp_path / "x.tgz", root_name="")


# Coverage for line 49: tarball with absolute path entry
def test_extract_tarball_rejects_absolute_path(tmp_path: Path) -> None:
    # Manually create a malformed tarball with an absolute path
    bad_tar = tmp_path / "bad.tar.gz"
    with tarfile.open(bad_tar.as_posix(), mode="w:gz") as tf:
        # Add a file with absolute path
        data = b"malicious"
        import io

        info = tarfile.TarInfo(name="/etc/passwd")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    with pytest.raises(TarballError, match="archive contains absolute or empty path"):
        extract_tarball(bad_tar, tmp_path / "out", expected_root="root")


# Coverage for line 49: tarball with empty path entry
def test_extract_tarball_rejects_empty_path(tmp_path: Path) -> None:
    bad_tar = tmp_path / "bad_empty.tar.gz"
    with tarfile.open(bad_tar.as_posix(), mode="w:gz") as tf:
        # Add an entry with empty name
        import io

        info = tarfile.TarInfo(name="")
        info.size = 0
        tf.addfile(info, io.BytesIO(b""))
    with pytest.raises(TarballError, match="archive contains absolute or empty path"):
        extract_tarball(bad_tar, tmp_path / "out", expected_root="root")


# Coverage for line 55: extraction succeeds but expected root dir missing
def test_extract_tarball_missing_root_dir(tmp_path: Path) -> None:
    # Create a tarball that has files under the expected prefix but no directory entry
    bad_tar = tmp_path / "no_root_dir.tar.gz"
    with tarfile.open(bad_tar.as_posix(), mode="w:gz") as tf:
        # Add a file that passes prefix check (starts with "expected/")
        # but the actual "expected" directory won't exist after extraction
        # because we only add files, not the directory itself, and the file
        # path will pass validation but won't create "expected" as a dir
        import io

        info = tarfile.TarInfo(name="expected")  # This is the root - matches expected
        info.type = tarfile.REGTYPE  # Make it a regular file, not a directory
        info.size = 5
        tf.addfile(info, io.BytesIO(b"hello"))
    # After extraction, "expected" exists but as a file, not a directory
    with pytest.raises(TarballError, match="extraction did not produce the expected root"):
        extract_tarball(bad_tar, tmp_path / "out", expected_root="expected")
