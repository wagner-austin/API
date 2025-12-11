from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from platform_core.logging import get_logger

from ...config.settings import Settings


class CorpusCacheCleanupError(Exception):
    """Raised when corpus cache cleanup fails."""


class CorpusCacheCleanupResult:
    deleted_files: int
    bytes_freed: int

    def __init__(self: CorpusCacheCleanupResult, deleted_files: int, bytes_freed: int) -> None:
        self.deleted_files = deleted_files
        self.bytes_freed = bytes_freed


class _CacheEntry:
    path: Path
    size_bytes: int
    access_ts: float
    modified_ts: float

    def __init__(
        self: _CacheEntry, path: Path, size_bytes: int, access_ts: float, modified_ts: float
    ) -> None:
        self.path = path
        self.size_bytes = size_bytes
        self.access_ts = access_ts
        self.modified_ts = modified_ts


def _cache_key_lru(entry: _CacheEntry) -> float:
    return entry.access_ts


def _cache_key_oldest(entry: _CacheEntry) -> float:
    return entry.modified_ts


class CorpusCacheCleanupService:
    """Service for cleaning up the worker-local corpus cache directory.

    Semantics:
    - Cache entries are pure cache (canonical source is data-bank-api).
    - Every file can be deleted safely; the worker will re-download when needed.
    - No best-effort branches: filesystem errors during inspection or deletion
      raise CorpusCacheCleanupError and abort the cleanup run.
    """

    settings: Settings

    def __init__(self: CorpusCacheCleanupService, settings: Settings) -> None:
        self.settings = settings

    def clean(self: CorpusCacheCleanupService) -> CorpusCacheCleanupResult:
        logger = get_logger(__name__)
        cfg = self.settings["app"]["corpus_cache_cleanup"]
        if not cfg["enabled"]:
            logger.info(
                "Corpus cache cleanup skipped: disabled",
                extra={"event": "corpus_cache_cleanup_skipped", "reason": "disabled"},
            )
            return CorpusCacheCleanupResult(deleted_files=0, bytes_freed=0)

        cache_dir = Path(self.settings["app"]["data_root"]) / "corpus_cache"
        if not cache_dir.exists():
            logger.info(
                "Corpus cache cleanup: directory missing",
                extra={
                    "event": "corpus_cache_cleanup_completed",
                    "deleted_files": 0,
                    "bytes_freed": 0,
                    "reason": "directory_missing",
                },
            )
            return CorpusCacheCleanupResult(deleted_files=0, bytes_freed=0)

        if not cache_dir.is_dir():
            raise CorpusCacheCleanupError(f"corpus cache path is not a directory: {cache_dir}")

        logger.info(
            "Corpus cache cleanup started",
            extra={
                "event": "corpus_cache_cleanup_started",
                "path": str(cache_dir),
                "max_bytes": cfg["max_bytes"],
                "min_free_bytes": cfg["min_free_bytes"],
                "eviction_policy": cfg["eviction_policy"],
            },
        )

        from model_trainer.core import _test_hooks

        try:
            usage = _test_hooks.shutil_disk_usage(str(cache_dir))
        except OSError as exc:
            logger.error(
                "Failed to read disk usage for corpus cache",
                extra={
                    "event": "corpus_cache_cleanup_failed",
                    "path": str(cache_dir),
                    "error": str(exc),
                },
            )
            raise CorpusCacheCleanupError(f"disk_usage failed for {cache_dir}: {exc}") from exc

        free_bytes = int(usage.free)
        entries, total_bytes = self._scan_cache_dir(cache_dir)

        if total_bytes <= cfg["max_bytes"] and free_bytes >= cfg["min_free_bytes"]:
            logger.info(
                "Corpus cache cleanup no-op: thresholds satisfied",
                extra={
                    "event": "corpus_cache_cleanup_completed",
                    "deleted_files": 0,
                    "bytes_freed": 0,
                    "total_bytes": total_bytes,
                    "free_bytes": free_bytes,
                },
            )
            return CorpusCacheCleanupResult(deleted_files=0, bytes_freed=0)

        deleted_files = 0
        bytes_freed = 0

        policy: Literal["lru", "oldest"] = cfg["eviction_policy"]
        key_fn: Callable[[_CacheEntry], float]
        key_fn = _cache_key_lru if policy == "lru" else _cache_key_oldest

        for entry in sorted(entries, key=key_fn):
            if total_bytes <= cfg["max_bytes"] and free_bytes >= cfg["min_free_bytes"]:
                break
            try:
                _test_hooks.path_unlink(entry.path)
            except OSError as exc:
                logger.error(
                    "Failed to delete corpus cache file",
                    extra={
                        "event": "corpus_cache_cleanup_failed",
                        "path": str(entry.path),
                        "error": str(exc),
                    },
                )
                raise CorpusCacheCleanupError(f"failed to delete {entry.path}: {exc}") from exc
            deleted_files += 1
            bytes_freed += entry.size_bytes
            total_bytes -= entry.size_bytes
            free_bytes += entry.size_bytes

        logger.info(
            "Corpus cache cleanup completed",
            extra={
                "event": "corpus_cache_cleanup_completed",
                "deleted_files": deleted_files,
                "bytes_freed": bytes_freed,
                "total_bytes_after": total_bytes,
                "free_bytes_after": free_bytes,
            },
        )
        return CorpusCacheCleanupResult(deleted_files=deleted_files, bytes_freed=bytes_freed)

    def _scan_cache_dir(
        self: CorpusCacheCleanupService,
        cache_dir: Path,
    ) -> tuple[list[_CacheEntry], int]:
        from model_trainer.core import _test_hooks

        entries: list[_CacheEntry] = []
        total = 0
        try:
            with _test_hooks.os_scandir(str(cache_dir)) as it:
                for dirent in it:
                    if not dirent.is_file():
                        continue
                    stat = dirent.stat()
                    size = int(stat.st_size)
                    access_ts = float(stat.st_atime or stat.st_mtime)
                    modified_ts = float(stat.st_mtime)
                    entry = _CacheEntry(
                        path=Path(dirent.path),
                        size_bytes=size,
                        access_ts=access_ts,
                        modified_ts=modified_ts,
                    )
                    entries.append(entry)
                    total += size
        except OSError as exc:
            raise CorpusCacheCleanupError(f"failed to scan corpus cache: {exc}") from exc
        return entries, total
