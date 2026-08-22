"""Cache locking and categorical-encoding serialization for the parquet cache."""

from __future__ import annotations

import time
from pathlib import Path
from types import TracebackType

from covenant_ml.datasets.types import (
    CategoricalEncoding,
)


def _get_lock_dir(cache_dir: Path) -> Path:
    """Get the lock directory path for a cache directory.

    Args:
        cache_dir: Path to the cache directory (.../.cache/<hash>).

    Returns:
        Path to the corresponding lock directory (.../.cache/<hash>.lock).
    """
    return cache_dir.parent / f"{cache_dir.name}.lock"


class _CacheLock:
    """Filesystem lock using an exclusive directory.

    Creates a lock directory to acquire the lock and removes it to release.
    Directory creation is atomic on local filesystems across platforms, which
    makes this a simple, dependency-free cross-platform lock.

    Args:
        cache_dir: Target cache directory for which we coordinate access.
    """

    _lock_dir: Path
    _acquired: bool

    def __init__(self, cache_dir: Path) -> None:
        self._lock_dir = _get_lock_dir(cache_dir)
        self._acquired = False

    def acquire(self, timeout_seconds: float = 30.0, poll_seconds: float = 0.05) -> None:
        """Acquire the lock, waiting up to timeout_seconds.

        Raises:
            TimeoutError: If the lock cannot be acquired within the timeout.
        """
        deadline = time.monotonic() + timeout_seconds
        # Ensure parent exists (e.g., .cache directory)
        self._lock_dir.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self._lock_dir.mkdir()
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out acquiring cache lock: {self._lock_dir}"
                    ) from None
                time.sleep(poll_seconds)
                continue
            self._acquired = True
            return

    def release(self) -> None:
        """Release the lock if held."""
        if self._acquired:
            # Remove lock directory if it still exists
            if self._lock_dir.exists():
                self._lock_dir.rmdir()
            self._acquired = False

    # Context manager helpers
    def __enter__(self) -> _CacheLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def _serialize_encodings(encodings: tuple[CategoricalEncoding, ...]) -> str:
    """Serialize categorical encodings to string.

    Args:
        encodings: Tuple of categorical encodings.

    Returns:
        String representation for storage.
    """
    if not encodings:
        return "[]"

    parts: list[str] = []
    for enc in encodings:
        mapping_parts: list[str] = []
        for val, code in enc["mapping"]:
            # Escape special characters
            escaped_val = val.replace("\\", "\\\\").replace("|", "\\|").replace(",", "\\,")
            mapping_parts.append(f"{escaped_val}:{code}")
        mapping_str = ",".join(mapping_parts)
        parts.append(f"{enc['column_name']}|{enc['n_categories']}|{mapping_str}")

    return ";".join(parts)


def _parse_encodings(encodings_str: str) -> tuple[CategoricalEncoding, ...]:
    """Parse categorical encodings from string.

    Args:
        encodings_str: String representation from storage.

    Returns:
        Tuple of categorical encodings.
    """
    if encodings_str == "[]" or not encodings_str:
        return ()

    result: list[CategoricalEncoding] = []
    for part in encodings_str.split(";"):
        if not part:
            continue

        sections = part.split("|")
        if len(sections) < 3:
            continue

        column_name = sections[0]
        n_categories = int(sections[1])
        mapping_str = "|".join(sections[2:])  # Rejoin in case of escaped pipes

        mapping: list[tuple[str, int]] = []
        if mapping_str:
            # Parse mapping entries, handling escaped characters
            entries = _split_escaped(mapping_str, ",")
            for entry in entries:
                if ":" in entry:
                    val, code_str = entry.rsplit(":", 1)
                    # Unescape special characters
                    val = val.replace("\\,", ",").replace("\\|", "|").replace("\\\\", "\\")
                    mapping.append((val, int(code_str)))

        result.append(
            CategoricalEncoding(
                column_name=column_name,
                mapping=tuple(mapping),
                n_categories=n_categories,
            )
        )

    return tuple(result)


def _split_escaped(text: str, delimiter: str) -> list[str]:
    """Split string by delimiter, respecting escaped delimiters.

    Args:
        text: String to split.
        delimiter: Delimiter character.

    Returns:
        List of split parts.
    """
    result: list[str] = []
    current: list[str] = []
    i = 0

    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            # Escaped character - include both
            current.append(text[i])
            current.append(text[i + 1])
            i += 2
        elif text[i] == delimiter:
            result.append("".join(current))
            current = []
            i += 1
        else:
            current.append(text[i])
            i += 1

    result.append("".join(current))
    return result


def _serialize_string_tuple(values: tuple[str, ...]) -> str:
    """Serialize tuple of strings to string.

    Args:
        values: Tuple of strings.

    Returns:
        String representation for storage.
    """
    if not values:
        return ""

    escaped: list[str] = []
    for val in values:
        esc = val.replace("\\", "\\\\").replace("|", "\\|")
        escaped.append(esc)

    return "|".join(escaped)


def _parse_string_tuple(text: str) -> tuple[str, ...]:
    """Parse tuple of strings from string.

    Args:
        text: String representation from storage.

    Returns:
        Tuple of strings.
    """
    if not text:
        return ()

    parts = _split_escaped(text, "|")
    result: list[str] = []
    for part in parts:
        unescaped = part.replace("\\|", "|").replace("\\\\", "\\")
        result.append(unescaped)

    return tuple(result)
