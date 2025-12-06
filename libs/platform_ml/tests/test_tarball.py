from __future__ import annotations

from pathlib import Path

import pytest

from platform_ml.tarball import TarballError, create_tarball, extract_tarball


def test_create_and_extract_tarball(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "nested").mkdir(parents=True)
    (src / "nested" / "a.txt").write_text("hello", encoding="utf-8")
    (src / "b.bin").write_bytes(b"\x00\x01")

    tar = tmp_path / "artifacts" / "x.tar.gz"
    out = create_tarball(src, tar, root_name="artifact-x")
    assert out.exists() and out.is_file()
    dest = tmp_path / "dest"
    root = extract_tarball(out, dest, expected_root="artifact-x")
    assert root == dest / "artifact-x"
    assert (root / "nested" / "a.txt").read_text(encoding="utf-8") == "hello"
    assert (root / "b.bin").read_bytes() == b"\x00\x01"


def test_extract_tarball_rejects_bad_root(tmp_path: Path) -> None:
    src = tmp_path / "s"
    src.mkdir()
    (src / "file.txt").write_text("x", encoding="utf-8")
    tar = tmp_path / "x.tgz"
    create_tarball(src, tar, root_name="good")
    with pytest.raises(TarballError):
        extract_tarball(tar, tmp_path / "d", expected_root="other")
