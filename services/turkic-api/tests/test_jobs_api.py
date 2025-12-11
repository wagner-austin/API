"""Tests for turkic-api job routes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from platform_core.config import config_test_hooks
from platform_core.data_bank_client import DataBankClientError, HeadInfo, NotFoundError
from platform_core.testing import make_fake_env
from platform_core.turkic_jobs import turkic_job_key
from platform_workers.testing import FakeQueue, FakeRedis

from turkic_api import _test_hooks
from turkic_api.api.config import Settings
from turkic_api.api.main import RedisCombinedProtocol, create_app
from turkic_api.api.models import parse_job_status_json
from turkic_api.api.types import QueueProtocol


def make_fake_head_info(
    *, size: int = 1, etag: str = "etag", content_type: str = "text/plain; charset=utf-8"
) -> HeadInfo:
    """Create a fake HeadInfo TypedDict for testing."""
    return HeadInfo(size=size, etag=etag, content_type=content_type)


class FakeDataBankDownloader:
    """Fake data bank client for downloading."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        timeout_seconds: float,
        head_status: int = 200,
        stream_status: int = 200,
        chunks: list[bytes] | None = None,
    ) -> None:
        self._head_status = head_status
        self._stream_status = stream_status
        self._chunks = chunks if chunks is not None else [b"hello\nworld\n"]

    def head(self, file_id: str, *, request_id: str | None = None) -> HeadInfo:
        if self._head_status == 404:
            raise NotFoundError("not found")
        if self._head_status >= 400:
            raise DataBankClientError("error")
        return make_fake_head_info()

    def stream_download(
        self,
        file_id: str,
        *,
        request_id: str | None = None,
        chunk_size: int = 8192,
    ) -> httpx.Response:
        # Create a mock httpx.Response-compatible object
        # httpx.Response requires minimal attributes for our usage
        return httpx.Response(
            status_code=self._stream_status,
            content=b"".join(self._chunks),
        )


def _make_client(tmp_path: Path) -> tuple[TestClient, FakeRedis]:
    """Create a test client with fake dependencies."""
    env = make_fake_env(
        {
            "TURKIC_DATA_DIR": str(tmp_path),
            "TURKIC_DATA_BANK_API_URL": "http://db",
            "TURKIC_DATA_BANK_API_KEY": "k",
            "TURKIC_REDIS_URL": "redis://test:6379/0",
        }
    )
    config_test_hooks.get_env = env

    r = FakeRedis()
    q = FakeQueue()

    def _redis_provider(settings: Settings) -> RedisCombinedProtocol:
        return r

    def _queue_provider() -> QueueProtocol:
        return q

    app = create_app(
        redis_provider=_redis_provider,
        queue_provider=_queue_provider,
    )
    return TestClient(app), r


def _seed_job(redis: FakeRedis, job_id: str, status: str, tmp_path: Path) -> None:
    """Seed a job in Redis with the given status."""
    redis.hset(
        turkic_job_key(job_id),
        {
            "user_id": "42",
            "status": status,
            "progress": "100" if status == "completed" else "0",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        },
    )
    if status == "completed":
        redis.hset(
            turkic_job_key(job_id),
            {
                "file_id": f"{job_id}-fid",
                "upload_status": "uploaded",
                "message": "done",
                "error": "",
            },
        )


def _setup_data_bank_downloader(
    *,
    head_status: int = 200,
    stream_status: int = 200,
    chunks: list[bytes] | None = None,
) -> None:
    """Set up the data bank downloader hook with a fake."""

    def _factory(api_url: str, api_key: str, *, timeout_seconds: float) -> FakeDataBankDownloader:
        return FakeDataBankDownloader(
            api_url,
            api_key,
            timeout_seconds=timeout_seconds,
            head_status=head_status,
            stream_status=stream_status,
            chunks=chunks,
        )

    _test_hooks.data_bank_downloader_factory = _factory


def test_job_status_not_found(tmp_path: Path) -> None:
    """Test getting status of non-existent job returns 404."""
    client, redis = _make_client(tmp_path)
    resp = client.get("/api/v1/jobs/doesnotexist")
    assert resp.status_code == 404
    redis.assert_only_called({"hgetall"})


def test_job_status_found(tmp_path: Path) -> None:
    """Test getting status of existing job returns 200."""
    client, rstub = _make_client(tmp_path)
    _seed_job(rstub, "abc", "processing", tmp_path)
    resp = client.get("/api/v1/jobs/abc")
    assert resp.status_code == 200
    status = parse_job_status_json(resp.text)
    assert status["job_id"] == "abc"
    assert status["status"] == "processing"
    assert status["progress"] in range(101)
    rstub.assert_only_called({"hset", "expire", "hgetall"})


def test_job_result_not_ready(tmp_path: Path) -> None:
    """Test getting result of queued job returns 425."""
    client, rstub = _make_client(tmp_path)
    _seed_job(rstub, "j1", "queued", tmp_path)
    resp = client.get("/api/v1/jobs/j1/result")
    assert resp.status_code == 425
    rstub.assert_only_called({"hset", "expire", "hgetall"})


def test_job_result_completed(tmp_path: Path) -> None:
    """Test getting result of completed job streams content."""
    client, rstub = _make_client(tmp_path)
    _seed_job(rstub, "j2", "completed", tmp_path)
    _setup_data_bank_downloader()

    resp = client.get("/api/v1/jobs/j2/result")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    headers: Mapping[str, str] = resp.headers
    cd = headers.get("content-disposition") or ""
    assert "attachment;" in cd
    assert "hello" in resp.text
    rstub.assert_only_called({"hset", "expire", "hgetall"})


def test_job_result_missing_file_is_expired(tmp_path: Path) -> None:
    """Test that completed job with empty file_id returns 410."""
    client, redis_stub = _make_client(tmp_path)
    redis_stub.hset(
        turkic_job_key("j3"),
        {
            "user_id": "42",
            "status": "completed",
            "progress": "100",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "file_id": "",
            "upload_status": "",
            "message": "",
            "error": "",
        },
    )
    resp = client.get("/api/v1/jobs/j3/result")
    assert resp.status_code == 410
    redis_stub.assert_only_called({"hset", "expire", "hgetall"})


def test_job_result_data_bank_config_missing(tmp_path: Path) -> None:
    """Test that missing data bank config returns 500."""
    # Override env to remove data bank URL
    env = make_fake_env(
        {
            "TURKIC_DATA_DIR": str(tmp_path),
            "TURKIC_DATA_BANK_API_URL": "",
            "TURKIC_DATA_BANK_API_KEY": "k",
            "TURKIC_REDIS_URL": "redis://test:6379/0",
        }
    )
    config_test_hooks.get_env = env

    # Must create the client after setting config so that app loads empty URL
    r = FakeRedis()
    q = FakeQueue()

    def _redis_provider(settings: Settings) -> RedisCombinedProtocol:
        return r

    def _queue_provider() -> QueueProtocol:
        return q

    app = create_app(
        redis_provider=_redis_provider,
        queue_provider=_queue_provider,
    )
    client = TestClient(app)

    _seed_job(r, "j4", "completed", tmp_path)
    resp = client.get("/api/v1/jobs/j4/result")
    assert resp.status_code == 500
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_job_result_head_404(tmp_path: Path) -> None:
    """Test that 404 on head returns 410."""
    client, rstub = _make_client(tmp_path)
    _seed_job(rstub, "j5", "completed", tmp_path)
    _setup_data_bank_downloader(head_status=404)
    resp = client.get("/api/v1/jobs/j5/result")
    assert resp.status_code == 410
    rstub.assert_only_called({"hset", "expire", "hgetall"})


def test_job_result_head_error(tmp_path: Path) -> None:
    """Test that error on head returns 502."""
    client, rstub = _make_client(tmp_path)
    _seed_job(rstub, "j6", "completed", tmp_path)
    _setup_data_bank_downloader(head_status=500)
    resp = client.get("/api/v1/jobs/j6/result")
    assert resp.status_code == 502
    rstub.assert_only_called({"hset", "expire", "hgetall"})


def test_job_result_stream_404(tmp_path: Path) -> None:
    """Test that 404 on stream returns 410."""
    client, rstub = _make_client(tmp_path)
    _seed_job(rstub, "j7", "completed", tmp_path)
    _setup_data_bank_downloader(stream_status=404)
    resp = client.get("/api/v1/jobs/j7/result")
    assert resp.status_code == 410
    rstub.assert_only_called({"hset", "expire", "hgetall"})


def test_job_result_stream_error(tmp_path: Path) -> None:
    """Test that error on stream returns 502."""
    client, rstub = _make_client(tmp_path)
    _seed_job(rstub, "j8", "completed", tmp_path)
    _setup_data_bank_downloader(stream_status=500)
    resp = client.get("/api/v1/jobs/j8/result")
    assert resp.status_code == 502
    rstub.assert_only_called({"hset", "expire", "hgetall"})
