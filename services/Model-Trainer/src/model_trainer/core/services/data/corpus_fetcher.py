from __future__ import annotations

import time
from pathlib import Path
from typing import Final, Protocol

from platform_core.data_bank_client import DataBankClient
from platform_core.logging import get_logger

from model_trainer.core import _test_hooks


class _Logger(Protocol):
    def info(
        self: _Logger, msg: str, *, extra: dict[str, str | int | float] | None = None
    ) -> None: ...
    def error(
        self: _Logger, msg: str, *, extra: dict[str, str | int | float] | None = None
    ) -> None: ...


_logger: Final[_Logger] = get_logger(__name__)


class CorpusFetcher:
    """Downloads corpus files from data-bank-api with caching and structured logging."""

    def __init__(self: CorpusFetcher, api_url: str, api_key: str, cache_dir: Path) -> None:
        self._api_url: Final[str] = api_url.rstrip("/")
        self._api_key: Final[str] = api_key
        self._cache_dir: Final[Path] = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self: CorpusFetcher, file_id: str) -> Path:
        """Fetch a corpus file from the data bank API, with caching.

        Args:
            file_id: The unique identifier for the corpus file.

        Returns:
            Path to the cached corpus file.

        Raises:
            httpx.HTTPStatusError: For HTTP errors (404, 401, 503, etc).
            httpx.TimeoutException: If request times out.
            httpx.ConnectError: If connection fails.
            RuntimeError: If downloaded file size doesn't match expected size.
        """
        cache_path = self._cache_dir / f"{file_id}.txt"
        if cache_path.exists():
            _logger.info("Corpus cache hit", extra={"file_id": file_id, "path": str(cache_path)})
            return cache_path

        _logger.info(
            "Starting corpus fetch from data bank",
            extra={"file_id": file_id, "api_url": self._api_url},
        )
        start_time = time.time()

        url = f"{self._api_url}/files/{file_id}"
        _logger.info("Sending HEAD request to data bank", extra={"file_id": file_id, "url": url})
        # Use hook for httpx client to enable testing with fake transport
        httpx_client = _test_hooks.httpx_client_factory(timeout_seconds=30.0)
        client = DataBankClient(
            base_url=self._api_url,
            api_key=self._api_key,
            timeout_seconds=30.0,
            client=httpx_client,
        )
        head = client.head(file_id, request_id=file_id)
        expected_size = int(head["size"])
        _logger.info(
            "HEAD request successful",
            extra={"file_id": file_id, "expected_size": expected_size},
        )

        temp_path = cache_path.with_suffix(".tmp")
        start = temp_path.stat().st_size if temp_path.exists() else 0
        if start > 0:
            _logger.info(
                "Resuming partial download",
                extra={"file_id": file_id, "resume_from": start},
            )

        _logger.info(
            "Starting corpus download",
            extra={"file_id": file_id, "url": url, "expected_size": expected_size},
        )
        # Download with resume and verify ETag integrity
        client.download_to_path(
            file_id,
            temp_path,
            resume=True,
            request_id=file_id,
            verify_etag=True,
        )

        if temp_path.stat().st_size != expected_size:
            _logger.error(
                "Downloaded file size mismatch",
                extra={
                    "file_id": file_id,
                    "expected_size": expected_size,
                    "actual_size": temp_path.stat().st_size,
                },
            )
            raise RuntimeError(
                f"Size mismatch: expected {expected_size}, got {temp_path.stat().st_size}"
            )

        temp_path.replace(cache_path)
        elapsed = time.time() - start_time
        _logger.info(
            "Corpus fetch completed successfully",
            extra={
                "file_id": file_id,
                "size": expected_size,
                "elapsed_seconds": round(elapsed, 2),
            },
        )
        return cache_path
