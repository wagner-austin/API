"""Test hooks for data-bank-api - allows injecting test dependencies."""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import BinaryIO, Protocol

from platform_core.config import _optional_env_str
from platform_core.data_bank_client import DataBankClient
from platform_core.data_bank_protocol import FileUploadResponse
from platform_workers.redis import RedisStrProto, redis_for_kv
from platform_workers.rq_harness import WorkerConfig

from data_bank_api.api.config import JobParams
from data_bank_api.core.corpus_download import ensure_corpus_file as _default_ensure_corpus
from data_bank_api.storage import FileMetadata, Storage


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None:
        """Run the worker with the given config."""
        ...


class StorageProtocol(Protocol):
    """Protocol for Storage - allows injecting fakes for testing."""

    def head(self, file_id: str) -> FileMetadata: ...
    def get_size(self, file_id: str) -> int: ...
    def delete(self, file_id: str) -> bool: ...
    def save_stream(self, stream: BinaryIO, content_type: str) -> FileMetadata: ...
    def open_range(
        self, file_id: str, start: int, end_inclusive: int | None
    ) -> tuple[Generator[bytes, None, None], int, int]: ...


class StorageFactoryProtocol(Protocol):
    """Protocol for Storage factory."""

    def __call__(
        self, root: Path, min_free_gb: int, *, max_file_bytes: int = 0
    ) -> StorageProtocol: ...


def _default_get_env(key: str) -> str | None:
    """Production implementation - reads from os.environ."""
    return _optional_env_str(key)


def _default_storage_factory(
    root: Path, min_free_gb: int, *, max_file_bytes: int = 0
) -> StorageProtocol:
    """Production implementation - creates real Storage."""
    return Storage(root, min_free_gb, max_file_bytes=max_file_bytes)


# Module-level injectable runner for testing.
# Tests set this BEFORE running worker_entry as __main__.
# Because this is a separate module, it persists across runpy.run_module.
test_runner: WorkerRunnerProtocol | None = None

# Hook for environment variable access. Tests can override to provide fake values.
get_env: Callable[[str], str | None] = _default_get_env

# Hook for Redis client creation. Tests can override with FakeRedis.
redis_factory: Callable[[str], RedisStrProto] = redis_for_kv

# Hook for Storage creation. Tests can override to inject FakeStorage.
storage_factory: StorageFactoryProtocol = _default_storage_factory


# =========================================================================
# Hooks for jobs module
# =========================================================================


class LocalCorpusServiceProtocol(Protocol):
    """Protocol for LocalCorpusService - allows injecting fakes for testing."""

    def __init__(self, data_dir: str) -> None: ...

    def stream(self, spec: JobParams) -> Generator[str, None, None]: ...


class DataBankUploaderProtocol(Protocol):
    """Protocol for DataBankClient upload method - allows injecting fakes for testing."""

    def upload(
        self,
        file_id: str,
        stream: BinaryIO,
        *,
        content_type: str,
        request_id: str | None,
    ) -> FileUploadResponse: ...


class EnsureCorpusProtocol(Protocol):
    """Protocol for ensure_corpus_file function."""

    def __call__(
        self,
        *,
        source: str,
        language: str,
        data_dir: str,
        max_sentences: int,
        transliterate: bool,
        confidence_threshold: float,
    ) -> None: ...


class LocalCorpusServiceFactoryProtocol(Protocol):
    """Protocol for LocalCorpusService class factory."""

    def __call__(self, data_dir: str) -> LocalCorpusServiceProtocol: ...


class DataBankUploaderFactoryProtocol(Protocol):
    """Protocol for DataBankClient factory - allows injecting fakes for testing."""

    def __call__(
        self, api_url: str, api_key: str, *, timeout_seconds: float
    ) -> DataBankUploaderProtocol: ...


def _default_local_corpus_factory(data_dir: str) -> LocalCorpusServiceProtocol:
    """Production implementation - creates real LocalCorpusService."""
    # Import lazily to avoid circular import (jobs imports from _test_hooks)
    from data_bank_api.api.jobs import LocalCorpusService

    return LocalCorpusService(data_dir)


def _default_data_bank_uploader_factory(
    api_url: str, api_key: str, *, timeout_seconds: float
) -> DataBankUploaderProtocol:
    """Production implementation - creates real DataBankClient."""
    return DataBankClient(api_url, api_key, timeout_seconds=timeout_seconds)


# Hook for ensure_corpus_file. Tests can override to no-op.
ensure_corpus_file: EnsureCorpusProtocol = _default_ensure_corpus

# Hook for LocalCorpusService factory. Tests can inject fake services.
local_corpus_service_factory: LocalCorpusServiceFactoryProtocol = _default_local_corpus_factory

# Hook for DataBankClient factory. Tests can inject fake clients.
data_bank_client_factory: DataBankUploaderFactoryProtocol = _default_data_bank_uploader_factory
