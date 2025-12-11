from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from platform_core.errors import AppError, ErrorCode

from data_bank_api import storage as storage_mod
from data_bank_api.storage import Storage


def _storage(tmp_path: Path) -> Storage:
    root = tmp_path / "files"
    return Storage(root=root, min_free_gb=0)


def test_path_for_invalid_file_id_raises(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    with pytest.raises(AppError) as exc_info:
        # too short and non-hex
        s.head("zz")
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.INVALID_INPUT


def test_head_and_open_range_not_found(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    with pytest.raises(AppError) as exc_info:
        s.head("abcd1234")
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.NOT_FOUND
    with pytest.raises(AppError) as exc_info2:
        s.open_range("abcd1234", 0, None)
    err2: AppError[ErrorCode] = exc_info2.value
    assert err2.code == ErrorCode.NOT_FOUND


def test_open_range_invalid_and_unsatisfiable(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    # write a small file
    meta = s.save_stream(io.BytesIO(b"0123456789"), "text/plain")
    # invalid: end < start
    with pytest.raises(AppError) as exc_info:
        s.open_range(meta["file_id"], 5, 2)
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.RANGE_NOT_SATISFIABLE
    # unsatisfiable: start > last
    with pytest.raises(AppError) as exc_info2:
        s.open_range(meta["file_id"], 100, None)
    err2: AppError[ErrorCode] = exc_info2.value
    assert err2.code == ErrorCode.RANGE_NOT_SATISFIABLE


def test_get_size_not_found(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    with pytest.raises(AppError) as exc_info:
        s.get_size("ffffeeee")
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.NOT_FOUND


def _raise_oserror_fail(*_args: str, **_kwargs: str) -> None:
    raise OSError("fail")


def _raise_oserror_unlink(*_args: str, **_kwargs: str) -> None:
    raise OSError("unlink fail")


def test_save_stream_cleanup_unlink(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    # force os.replace to raise so tmp file remains for cleanup
    storage_mod._os_replace = _raise_oserror_fail
    with pytest.raises(OSError):
        s.save_stream(io.BytesIO(b"data"), "text/plain")
    # ensure no upload_* tmp files remain
    parts = (tmp_path / "files" / "aa" / "bb").glob("upload_*")
    assert list(parts) == []


def test_save_stream_cleanup_unlink_error(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    storage_mod._os_replace = _raise_oserror_fail
    # make unlink also fail to exercise except branch in cleanup
    storage_mod._os_unlink = _raise_oserror_unlink
    with pytest.raises(OSError):
        s.save_stream(io.BytesIO(b"data"), "text/plain")


def test_insufficient_space_guard(tmp_path: Path) -> None:
    # configure high min_free to trigger guard
    s = Storage(root=tmp_path / "files", min_free_gb=1_000_000)
    with pytest.raises(AppError) as exc_info:
        s.save_stream(io.BytesIO(b"x"), "text/plain")
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.INSUFFICIENT_STORAGE


def test_meta_path_invalid_file_id_raises(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    with pytest.raises(AppError) as exc_info:
        _ = s._meta_path_for("zz")
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.INVALID_INPUT


def test_head_fallback_without_sidecar(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    meta = s.save_stream(io.BytesIO(b"abcdef"), "text/plain")
    mpath = s._meta_path_for(meta["file_id"])
    assert mpath.exists()
    mpath.unlink()
    with pytest.raises(AppError) as exc_info:
        s.head(meta["file_id"])
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.INVALID_INPUT


def test_read_sidecar_oserror_branch(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    meta = s.save_stream(io.BytesIO(b"xyz"), "text/plain")
    mpath = s._meta_path_for(meta["file_id"])
    mpath.unlink()
    mpath.mkdir()
    with pytest.raises(AppError) as exc_info:
        s.head(meta["file_id"])
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.INVALID_INPUT


def test_meta_path_valid_return(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    meta = s.save_stream(io.BytesIO(b"q"), "text/plain")
    meta_path = s._meta_path_for(meta["file_id"])
    # Ensure helper returns the actual path
    assert meta_path.exists()


def test_sidecar_replace_and_cleanup_unlink_error(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    # Patch os.replace and os.unlink to fail for meta temp files
    real_replace = storage_mod._default_os_replace
    real_unlink = storage_mod._default_os_unlink

    def _replace(src: str, dst: str) -> None:
        from pathlib import Path as PathMod

        if PathMod(src).name.startswith("meta_"):
            raise OSError("meta replace fail")
        real_replace(src, dst)

    def _unlink(path: str) -> None:
        from pathlib import Path as PathMod

        if PathMod(path).name.startswith("meta_"):
            raise OSError("meta unlink fail")
        real_unlink(path)

    storage_mod._os_replace = _replace
    storage_mod._os_unlink = _unlink
    # Metadata writes are mandatory; save_stream should fail if metadata fails
    # Cleanup will also fail but should be silently handled
    with pytest.raises(OSError, match="meta replace fail"):
        s.save_stream(io.BytesIO(b"sidecar"), "text/plain")


def test_sidecar_cleanup_unlink_error_after_successful_save(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    real_unlink = storage_mod._default_os_unlink

    def _unlink(path: str) -> None:
        from pathlib import Path as PathMod

        if PathMod(path).name.startswith("meta_"):
            raise OSError("meta unlink fail")
        real_unlink(path)

    storage_mod._os_unlink = _unlink
    # Upload should succeed even if cleanup fails
    meta = s.save_stream(io.BytesIO(b"test"), "text/plain")
    assert len(meta["sha256"]) == 64


def test_sidecar_present_but_invalid_values(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    meta = s.save_stream(io.BytesIO(b"abcdef"), "text/plain")
    mpath = s._meta_path_for(meta["file_id"])
    mpath.write_text("sha256=z\ncontent_type=\ncreated_at=\n", encoding="utf-8")
    with pytest.raises(AppError) as exc_info:
        s.head(meta["file_id"])
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.INVALID_INPUT


def test_sidecar_present_empty_file(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    meta = s.save_stream(io.BytesIO(b"abcdef"), "text/plain")
    mpath = s._meta_path_for(meta["file_id"])
    mpath.write_text("", encoding="utf-8")
    with pytest.raises(AppError) as exc_info:
        s.head(meta["file_id"])
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.INVALID_INPUT


def test_sidecar_present_unrelated_line(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    meta = s.save_stream(io.BytesIO(b"abcdef"), "text/plain")
    mpath = s._meta_path_for(meta["file_id"])
    mpath.write_text("ignored=1\n", encoding="utf-8")
    with pytest.raises(AppError) as exc_info:
        s.head(meta["file_id"])
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.INVALID_INPUT


def test_delete_removes_blob_and_sidecar(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    meta = s.save_stream(io.BytesIO(b"abcdef"), "text/plain")
    path = s._path_for(meta["file_id"])
    meta_path = s._meta_path_for(meta["file_id"])
    assert path.exists()
    assert meta_path.exists()

    deleted = s.delete(meta["file_id"])

    assert deleted is True
    assert not path.exists()
    assert not meta_path.exists()


def test_delete_cleans_stale_sidecar(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    meta = s.save_stream(io.BytesIO(b"abcdef"), "text/plain")
    path = s._path_for(meta["file_id"])
    meta_path = s._meta_path_for(meta["file_id"])
    # Simulate blob missing but sidecar present
    path.unlink()
    assert not path.exists()
    assert meta_path.exists()

    deleted = s.delete(meta["file_id"])

    assert deleted is True
    assert not meta_path.exists()


def test_open_range_early_break_on_empty_chunk(tmp_path: Path) -> None:
    # Cover storage.py line 211: break when chunk is empty before all data read
    # This simulates a file that reports larger size but has less data
    s = _storage(tmp_path)
    # Create a small file
    original_data = b"x" * 100
    meta = s.save_stream(io.BytesIO(original_data), "application/octet-stream")
    file_id = meta["file_id"]
    target_path = s._path_for(file_id)

    # Create a fake stat function that reports inflated size for our file
    def fake_stat(path: Path) -> os.stat_result:
        real_stat = path.stat()
        # Only inflate size for our specific file
        if path == target_path:
            return os.stat_result(
                (
                    real_stat.st_mode,
                    real_stat.st_ino,
                    real_stat.st_dev,
                    real_stat.st_nlink,
                    real_stat.st_uid,
                    real_stat.st_gid,
                    10000,  # Fake size much larger than actual
                    real_stat.st_atime,
                    real_stat.st_mtime,
                    real_stat.st_ctime,
                )
            )
        return real_stat

    storage_mod._path_stat = fake_stat

    # Request a range based on the fake size
    it, _start, _last = s.open_range(file_id, 0, 9999)

    # Consume the iterator - should hit break when read returns empty
    chunks_received = list(it)

    # Should have received partial data before break
    assert chunks_received
    total = sum(len(c) for c in chunks_received)
    # Should be only the actual file size (100), not the reported size (10000)
    assert total == 100
    assert total < 10000
