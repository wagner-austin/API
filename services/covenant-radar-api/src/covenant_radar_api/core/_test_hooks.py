"""Hooks for container factories - production defaults, tests override.

Production code initializes these to real implementations at module level.
Tests replace them with fakes before exercising the code under test.
No conditionals needed - just call the hook directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from covenant_persistence import ConnectionProtocol
from platform_core.data_bank_client import DataBankClient, HeadInfo
from platform_workers.redis import RedisStrProto, redis_for_kv
from platform_workers.rq_harness import (
    RQClientQueue,
    _RedisBytesClient,
    redis_raw_for_rq,
    rq_queue,
)


class KvClientFactoryProtocol(Protocol):
    """Protocol for key-value client factory."""

    def __call__(self, url: str) -> RedisStrProto:
        """Create KV client from URL."""
        ...


class ConnectionFactoryProtocol(Protocol):
    """Protocol for psycopg connect factory."""

    def __call__(self, dsn: str) -> ConnectionProtocol:
        """Create database connection from DSN."""
        ...


class RqClientFactoryProtocol(Protocol):
    """Protocol for RQ client factory."""

    def __call__(self, url: str) -> _RedisBytesClient:
        """Create RQ client from URL."""
        ...


class QueueFactoryProtocol(Protocol):
    """Protocol for rq_queue factory."""

    def __call__(self, name: str, connection: _RedisBytesClient) -> RQClientQueue:
        """Create RQ queue from name and connection."""
        ...


class PsycopgModuleProtocol(Protocol):
    """Protocol for psycopg module with connect method."""

    def connect(self, dsn: str, autocommit: bool = False) -> ConnectionProtocol:
        """Connect to database."""
        ...


class LoadPsycopgModuleHook(Protocol):
    """Protocol for psycopg module loader hook."""

    def __call__(self) -> PsycopgModuleProtocol:
        """Load psycopg module."""
        ...


def _real_load_psycopg_module() -> PsycopgModuleProtocol:
    """Import psycopg and return it behind its protocol.

    Returns:
        The psycopg module.
    """
    module: PsycopgModuleProtocol = __import__("psycopg")
    return module


# Hook for loading psycopg module - tests rebind it to return a fake module.
load_psycopg_module: LoadPsycopgModuleHook = _real_load_psycopg_module


def _psycopg_connect_autocommit(dsn: str) -> ConnectionProtocol:
    """Connect to postgres with autocommit enabled.

    Uses autocommit=True to prevent failed transactions from blocking
    subsequent queries. Each statement commits immediately.
    """
    module = load_psycopg_module()
    conn: ConnectionProtocol = module.connect(dsn, autocommit=True)
    return conn


# =============================================================================
# Data Bank Downloader Hook
# =============================================================================


class DataBankDownloaderProtocol(Protocol):
    """Protocol for data-bank file downloader interface.

    Defines the methods needed for downloading models from data-bank-api.
    Tests inject fakes that implement this protocol.
    """

    def download_to_path(
        self,
        file_id: str,
        dest: Path,
        *,
        resume: bool = True,
        request_id: str | None = None,
        verify_etag: bool = True,
        chunk_size: int = 1024 * 1024,
    ) -> HeadInfo:
        """Download file to local path.

        Args:
            file_id: ID of file in data-bank.
            dest: Destination path for downloaded file.
            resume: Resume partial download if dest exists.
            request_id: Optional correlation ID.
            verify_etag: Verify SHA256 hash matches ETag.
            chunk_size: Chunk size for streaming download.

        Returns:
            HeadInfo with size, etag, and content_type.

        Raises:
            NotFoundError: If file_id does not exist.
            DataBankClientError: On transport or validation errors.
        """
        ...


class DataBankDownloaderFactoryProtocol(Protocol):
    """Protocol for data-bank downloader factory function.

    Creates downloader instances from URL and API key.
    """

    def __call__(self, base_url: str, api_key: str) -> DataBankDownloaderProtocol:
        """Create data-bank downloader instance.

        Args:
            base_url: Base URL for data-bank-api.
            api_key: API key for authentication.

        Returns:
            Downloader instance implementing DataBankDownloaderProtocol.
        """
        ...


def _default_data_bank_client_factory(base_url: str, api_key: str) -> DataBankDownloaderProtocol:
    """Production data-bank client factory.

    Args:
        base_url: Base URL for data-bank-api.
        api_key: API key for authentication.

    Returns:
        Real DataBankClient instance (from platform_core).
    """
    client: DataBankDownloaderProtocol = DataBankClient(base_url, api_key)
    return client


# Factory hooks - initialized to production implementations.
# Tests replace these with fakes before calling container code.
# Production defaults call real external services (redis, postgres).
kv_factory: KvClientFactoryProtocol = redis_for_kv
connection_factory: ConnectionFactoryProtocol = _psycopg_connect_autocommit
rq_client_factory: RqClientFactoryProtocol = redis_raw_for_rq
queue_factory: QueueFactoryProtocol = rq_queue
data_bank_client_factory: DataBankDownloaderFactoryProtocol = _default_data_bank_client_factory
