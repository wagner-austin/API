"""Public test utilities for data-bank-api consumers."""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import BinaryIO

from platform_core.errors import AppError, ErrorCode

from data_bank_api._test_hooks import StorageFactoryProtocol, StorageProtocol
from data_bank_api.storage import FileMetadata, Storage


class FakeStorage:
    """Fake Storage that delegates to real storage but allows method overrides.

    Configure error injection by setting method attributes to callables that raise.
    """

    def __init__(self, root: Path, min_free_gb: int, *, max_file_bytes: int = 0) -> None:
        self._delegate = Storage(root, min_free_gb, max_file_bytes=max_file_bytes)
        # Method overrides - set to callable to replace behavior
        self.head_override: Callable[[str], FileMetadata] | None = None
        self.get_size_override: Callable[[str], int] | None = None
        self.delete_override: Callable[[str], bool] | None = None
        self.save_stream_override: Callable[[BinaryIO, str], FileMetadata] | None = None
        self.open_range_override: (
            Callable[[str, int, int | None], tuple[Generator[bytes, None, None], int, int]] | None
        ) = None
        self.ensure_free_space_override: Callable[[], None] | None = None

    def head(self, file_id: str) -> FileMetadata:
        if self.head_override is not None:
            return self.head_override(file_id)
        return self._delegate.head(file_id)

    def get_size(self, file_id: str) -> int:
        if self.get_size_override is not None:
            return self.get_size_override(file_id)
        return self._delegate.get_size(file_id)

    def delete(self, file_id: str) -> bool:
        if self.delete_override is not None:
            return self.delete_override(file_id)
        return self._delegate.delete(file_id)

    def save_stream(self, stream: BinaryIO, content_type: str) -> FileMetadata:
        # Check ensure_free_space_override first (simulates _ensure_free_space failing)
        if self.ensure_free_space_override is not None:
            self.ensure_free_space_override()
        if self.save_stream_override is not None:
            return self.save_stream_override(stream, content_type)
        return self._delegate.save_stream(stream, content_type)

    def open_range(
        self, file_id: str, start: int, end_inclusive: int | None
    ) -> tuple[Generator[bytes, None, None], int, int]:
        if self.open_range_override is not None:
            return self.open_range_override(file_id, start, end_inclusive)
        return self._delegate.open_range(file_id, start, end_inclusive)


def make_fake_storage_factory(
    storage: FakeStorage,
) -> StorageFactoryProtocol:
    """Create a factory that returns the given FakeStorage instance."""

    def _factory(root: Path, min_free_gb: int, *, max_file_bytes: int = 0) -> StorageProtocol:
        return storage

    return _factory


# Common error factories for convenience
def raise_insufficient_storage() -> None:
    """Raise 507 insufficient storage error."""
    raise AppError(ErrorCode.INSUFFICIENT_STORAGE, "insufficient space", 507)


def raise_not_found(file_id: str) -> FileMetadata:
    """Raise 404 not found error."""
    raise AppError(ErrorCode.NOT_FOUND, "file not found", 404)


def raise_not_found_int(file_id: str) -> int:
    """Raise 404 not found error (for get_size)."""
    raise AppError(ErrorCode.NOT_FOUND, "file not found", 404)


def raise_invalid_input(file_id: str) -> FileMetadata:
    """Raise 400 invalid input error."""
    raise AppError(ErrorCode.INVALID_INPUT, "invalid input", 400)


def raise_range_not_satisfiable(
    file_id: str, start: int, end: int | None
) -> tuple[Generator[bytes, None, None], int, int]:
    """Raise 416 range not satisfiable error."""
    raise AppError(ErrorCode.RANGE_NOT_SATISFIABLE, "range not satisfiable", 416)


def raise_not_found_range(
    file_id: str, start: int, end: int | None
) -> tuple[Generator[bytes, None, None], int, int]:
    """Raise 404 not found for range requests."""
    raise AppError(ErrorCode.NOT_FOUND, "file vanished", 404)
