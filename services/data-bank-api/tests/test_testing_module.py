"""Tests for data_bank_api.testing module - FakeStorage and error helpers."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from platform_core.errors import AppError, ErrorCode

from data_bank_api.testing import (
    FakeStorage,
    make_fake_storage_factory,
    raise_insufficient_storage,
    raise_invalid_input,
    raise_not_found,
    raise_not_found_int,
    raise_not_found_range,
    raise_range_not_satisfiable,
)


def test_fake_storage_get_size_delegates_when_no_override(tmp_path: Path) -> None:
    """FakeStorage.get_size delegates to real storage when no override set."""
    storage = FakeStorage(tmp_path, 0, max_file_bytes=0)
    # Create a file via delegate
    stream = io.BytesIO(b"test content")
    meta = storage.save_stream(stream, "text/plain")
    # get_size should delegate to real storage
    size = storage.get_size(meta["file_id"])
    assert size == 12


def test_fake_storage_delete_delegates_when_no_override(tmp_path: Path) -> None:
    """FakeStorage.delete delegates to real storage when no override set."""
    storage = FakeStorage(tmp_path, 0, max_file_bytes=0)
    # Create a file
    stream = io.BytesIO(b"to delete")
    meta = storage.save_stream(stream, "text/plain")
    # delete should delegate to real storage
    result = storage.delete(meta["file_id"])
    assert result is True


def test_fake_storage_delete_uses_override_when_set(tmp_path: Path) -> None:
    """FakeStorage.delete uses override when set."""
    storage = FakeStorage(tmp_path, 0, max_file_bytes=0)

    def _always_false(file_id: str) -> bool:
        return False

    storage.delete_override = _always_false
    # Should return False from override, not actually delete
    result = storage.delete("any-id")
    assert result is False


def test_fake_storage_open_range_delegates_when_no_override(tmp_path: Path) -> None:
    """FakeStorage.open_range delegates to real storage when no override set."""
    storage = FakeStorage(tmp_path, 0, max_file_bytes=0)
    # Create a file
    stream = io.BytesIO(b"range content")
    meta = storage.save_stream(stream, "text/plain")
    # open_range should delegate to real storage
    gen, _start, _end = storage.open_range(meta["file_id"], 0, 4)
    chunks = list(gen)
    assert b"".join(chunks) == b"range"


def test_make_fake_storage_factory_returns_provided_instance(tmp_path: Path) -> None:
    """make_fake_storage_factory returns the exact FakeStorage instance."""
    storage = FakeStorage(tmp_path, 0, max_file_bytes=0)
    factory = make_fake_storage_factory(storage)
    # Factory should return the same instance regardless of args
    result = factory(Path("/other"), 999, max_file_bytes=1000)
    assert result is storage


def test_raise_insufficient_storage_raises_507() -> None:
    """raise_insufficient_storage raises AppError with 507 http_status."""
    with pytest.raises(AppError) as exc_info:
        raise_insufficient_storage()
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.INSUFFICIENT_STORAGE
    assert err.http_status == 507


def test_raise_not_found_raises_404() -> None:
    """raise_not_found raises AppError with 404 http_status."""
    with pytest.raises(AppError) as exc_info:
        raise_not_found("some-file-id")
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.NOT_FOUND
    assert err.http_status == 404


def test_raise_not_found_int_raises_404() -> None:
    """raise_not_found_int raises AppError with 404 http_status."""
    with pytest.raises(AppError) as exc_info:
        raise_not_found_int("some-file-id")
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.NOT_FOUND
    assert err.http_status == 404


def test_raise_invalid_input_raises_400() -> None:
    """raise_invalid_input raises AppError with 400 http_status."""
    with pytest.raises(AppError) as exc_info:
        raise_invalid_input("some-file-id")
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.INVALID_INPUT
    assert err.http_status == 400


def test_raise_range_not_satisfiable_raises_416() -> None:
    """raise_range_not_satisfiable raises AppError with 416 http_status."""
    with pytest.raises(AppError) as exc_info:
        raise_range_not_satisfiable("file-id", 0, 100)
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.RANGE_NOT_SATISFIABLE
    assert err.http_status == 416


def test_raise_not_found_range_raises_404() -> None:
    """raise_not_found_range raises AppError with 404 http_status."""
    with pytest.raises(AppError) as exc_info:
        raise_not_found_range("file-id", 0, 100)
    err: AppError[ErrorCode] = exc_info.value
    assert err.code == ErrorCode.NOT_FOUND
    assert err.http_status == 404
