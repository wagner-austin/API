from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable, Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, TypedDict

from platform_core.errors import AppError, ErrorCode
from platform_core.logging import get_logger


# =========================================================================
# Test hooks for OS operations - tests can override to inject failures
# =========================================================================
def _default_os_replace(src: str, dst: str) -> None:
    """Default implementation using os.replace."""
    os.replace(src, dst)


def _default_os_unlink(path: str) -> None:
    """Default implementation using os.unlink."""
    os.unlink(path)


def _default_path_stat(path: Path) -> os.stat_result:
    """Default implementation using path.stat()."""
    return path.stat()


# Hook for os.replace. Tests can inject failures.
_os_replace: Callable[[str, str], None] = _default_os_replace

# Hook for os.unlink. Tests can inject failures.
_os_unlink: Callable[[str], None] = _default_os_unlink

# Hook for Path.stat. Tests can inject fake stat results.
_path_stat: Callable[[Path], os.stat_result] = _default_path_stat


class FileMetadata(TypedDict):
    file_id: str
    size_bytes: int
    sha256: str
    content_type: str
    created_at: str | None


def _is_hex(s: str) -> bool:
    return all(c in "0123456789abcdef" for c in s)


class Storage:
    def __init__(self: Storage, root: Path, min_free_gb: int, *, max_file_bytes: int = 0) -> None:
        self._root = root
        self._min_free_bytes = int(min_free_gb) * 1024 * 1024 * 1024
        self._max_file_bytes = int(max_file_bytes) if max_file_bytes is not None else 0
        self._logger = get_logger(__name__)

    def _path_for(self: Storage, file_id: str) -> Path:
        fid = file_id.strip().lower()
        if len(fid) < 4 or not _is_hex(fid):
            raise AppError(ErrorCode.INVALID_INPUT, "invalid file_id", 400)
        sub1, sub2 = fid[:2], fid[2:4]
        return self._root / sub1 / sub2 / f"{fid}.bin"

    def _meta_path_for(self: Storage, file_id: str) -> Path:
        fid = file_id.strip().lower()
        if len(fid) < 4 or not _is_hex(fid):
            raise AppError(ErrorCode.INVALID_INPUT, "invalid file_id", 400)
        sub1, sub2 = fid[:2], fid[2:4]
        return self._root / sub1 / sub2 / f"{fid}.meta"

    def _read_sidecar(self: Storage, file_id: str) -> tuple[str, str, str | None]:
        """Read sidecar metadata; missing or invalid metadata is an error."""
        mpath = self._meta_path_for(file_id)
        if not mpath.exists() or not mpath.is_file():
            raise AppError(ErrorCode.INVALID_INPUT, "metadata missing", 400)
        text = mpath.read_text(encoding="utf-8")
        sha = ""
        ctype = ""
        created_at: str | None = None
        for line in text.splitlines():
            if line.startswith("sha256="):
                v = line[len("sha256=") :].strip()
                if _is_hex(v) and len(v) == 64:
                    sha = v
            elif line.startswith("content_type="):
                v2 = line[len("content_type=") :].strip()
                if v2 != "":
                    ctype = v2
            elif line.startswith("created_at="):
                v3 = line[len("created_at=") :].strip()
                if v3 != "":
                    created_at = v3
        if sha == "" or ctype == "":
            raise AppError(ErrorCode.INVALID_INPUT, "metadata incomplete", 400)
        return sha, ctype, created_at

    def _ensure_free_space(self: Storage) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(self._root)
        free_bytes = int(usage.free)
        if free_bytes < self._min_free_bytes:
            raise AppError(ErrorCode.INSUFFICIENT_STORAGE, "insufficient free space", 507)

    def save_stream(self: Storage, stream: BinaryIO, content_type: str) -> FileMetadata:
        """Save stream to storage using server-generated sha256 file_id.

        Writes to a temp file in the storage root, computes sha256 and total size,
        enforces max size if configured, then atomically renames to the final
        hierarchical path. Also writes a small sidecar metadata file containing
        content_type and created_at for faster HEAD/INFO.
        """
        self._ensure_free_space()
        self._root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="upload_", dir=str(self._root))
        size = 0
        h = hashlib.sha256()
        try:
            with os.fdopen(fd, "wb") as f:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    size += len(chunk)
                    if self._max_file_bytes > 0 and size > self._max_file_bytes:
                        raise AppError(ErrorCode.PAYLOAD_TOO_LARGE, "file too large", 413)
                    h.update(chunk)
                f.flush()
                os.fsync(f.fileno())
            file_id = h.hexdigest()
            target = self._path_for(file_id)
            target_parent = target.parent
            target_parent.mkdir(parents=True, exist_ok=True)
            _os_replace(tmp, str(target))
            created_at = datetime.now(tz=UTC).isoformat()
            # Write sidecar metadata atomically under the final directory
            meta_tmp_fd, meta_tmp = tempfile.mkstemp(prefix="meta_", dir=str(target_parent))
            try:
                with os.fdopen(meta_tmp_fd, "w", encoding="utf-8") as mf:
                    mf.write(f"sha256={file_id}\n")
                    mf.write(f"content_type={content_type}\n")
                    mf.write(f"created_at={created_at}\n")
                    mf.flush()
                    os.fsync(mf.fileno())
                _os_replace(meta_tmp, str(self._meta_path_for(file_id)))
            finally:
                try:
                    if os.path.exists(meta_tmp):
                        _os_unlink(meta_tmp)
                except OSError as exc:
                    get_logger("data_bank_api").debug(
                        "meta_tmp_cleanup_failed path=%s error=%s", meta_tmp, exc
                    )
            return {
                "file_id": file_id,
                "size_bytes": size,
                "sha256": file_id,
                "content_type": content_type,
                "created_at": created_at,
            }
        finally:
            try:
                if os.path.exists(tmp):
                    _os_unlink(tmp)
            except OSError as exc:
                get_logger("data_bank_api").debug("tmp_cleanup_failed path=%s error=%s", tmp, exc)

    def head(self: Storage, file_id: str) -> FileMetadata:
        path = self._path_for(file_id)
        if not path.exists() or not path.is_file():
            raise AppError(ErrorCode.NOT_FOUND, "file not found", 404)
        size = _path_stat(path).st_size
        sha, ctype, created_at = self._read_sidecar(file_id)
        return {
            "file_id": file_id.strip().lower(),
            "size_bytes": size,
            "sha256": sha,
            "content_type": ctype,
            "created_at": created_at,
        }

    def open_range(
        self: Storage, file_id: str, start: int, end_inclusive: int | None
    ) -> tuple[Generator[bytes, None, None], int, int]:
        path = self._path_for(file_id)
        if not path.exists() or not path.is_file():
            raise AppError(ErrorCode.NOT_FOUND, "file not found", 404)
        size = _path_stat(path).st_size
        if start < 0 or (end_inclusive is not None and end_inclusive < start):
            raise AppError(ErrorCode.RANGE_NOT_SATISFIABLE, "invalid range", 416)
        last = size - 1 if end_inclusive is None or end_inclusive > size - 1 else end_inclusive
        if start > last:
            raise AppError(ErrorCode.RANGE_NOT_SATISFIABLE, "unsatisfiable range", 416)

        def _iter() -> Generator[bytes, None, None]:
            with path.open("rb") as f:
                f.seek(start)
                to_read = last - start + 1
                while to_read > 0:
                    chunk = f.read(min(1024 * 1024, to_read))
                    if not chunk:
                        break
                    yield chunk
                    to_read -= len(chunk)

        return _iter(), start, last

    def delete(self: Storage, file_id: str) -> bool:
        path = self._path_for(file_id)
        meta_path = self._meta_path_for(file_id)
        existed = False
        try:
            path.unlink()
            existed = True
        except FileNotFoundError:
            # Blob already missing; proceed to sidecar cleanup below.
            pass

        if meta_path.exists():
            meta_path.unlink()
            existed = True

        return existed

    def get_size(self: Storage, file_id: str) -> int:
        path = self._path_for(file_id)
        if not path.exists() or not path.is_file():
            raise AppError(ErrorCode.NOT_FOUND, "file not found", 404)
        return int(_path_stat(path).st_size)
