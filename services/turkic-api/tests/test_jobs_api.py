from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from platform_core.turkic_jobs import turkic_job_key
from platform_workers.testing import FakeQueue, FakeRedis

from turkic_api.api.config import Settings
from turkic_api.api.main import RedisCombinedProtocol, create_app
from turkic_api.api.models import parse_job_status_json
from turkic_api.api.types import QueueProtocol


def _make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, FakeRedis]:
    monkeypatch.setenv("TURKIC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TURKIC_DATA_BANK_API_URL", "http://db")
    monkeypatch.setenv("TURKIC_DATA_BANK_API_KEY", "k")
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
    # Minimal fields to emulate JobService
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


def _stub_data_bank(
    monkeypatch: pytest.MonkeyPatch,
    *,
    head_status: int = 200,
    stream_status: int = 200,
    chunks: list[bytes] | None = None,
) -> None:
    from platform_core.data_bank_client import (
        DataBankClient,
        DataBankClientError,
        HeadInfo,
        NotFoundError,
    )

    class _Resp:
        def __init__(self, status_code: int, data: list[bytes]) -> None:
            self.status_code = status_code
            self._data = data

        def iter_bytes(self) -> list[bytes]:
            return list(self._data)

        def close(self) -> None:
            return None

    def _head(self: DataBankClient, file_id: str, *, request_id: str | None = None) -> HeadInfo:
        if head_status == 404:
            raise NotFoundError("not found")
        if head_status >= 400:
            raise DataBankClientError("error")
        return {"size": 1, "etag": "etag", "content_type": "text/plain; charset=utf-8"}

    def _stream_download(
        self: DataBankClient,
        file_id: str,
        *,
        request_id: str | None = None,
        chunk_size: int = 8192,
    ) -> _Resp:
        data = chunks if chunks is not None else [b"hello\nworld\n"]
        return _Resp(stream_status, data)

    monkeypatch.setattr(DataBankClient, "head", _head, raising=True)
    monkeypatch.setattr(DataBankClient, "stream_download", _stream_download, raising=True)


def test_job_status_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, redis = _make_client(tmp_path, monkeypatch)
    resp = client.get("/api/v1/jobs/doesnotexist")
    assert resp.status_code == 404
    redis.assert_only_called({"hgetall"})


def test_job_status_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, rstub = _make_client(tmp_path, monkeypatch)
    _seed_job(rstub, "abc", "processing", tmp_path)
    resp = client.get("/api/v1/jobs/abc")
    assert resp.status_code == 200
    status = parse_job_status_json(resp.text)
    assert status["job_id"] == "abc"
    assert status["status"] == "processing"
    assert status["progress"] in range(101)
    rstub.assert_only_called({"hset", "expire", "hgetall"})


def test_job_result_not_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, rstub = _make_client(tmp_path, monkeypatch)
    _seed_job(rstub, "j1", "queued", tmp_path)
    resp = client.get("/api/v1/jobs/j1/result")
    assert resp.status_code == 425
    rstub.assert_only_called({"hset", "expire", "hgetall"})


def test_job_result_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, rstub = _make_client(tmp_path, monkeypatch)
    _seed_job(rstub, "j2", "completed", tmp_path)
    _stub_data_bank(monkeypatch)

    resp = client.get("/api/v1/jobs/j2/result")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    headers: Mapping[str, str] = resp.headers
    cd = headers.get("content-disposition") or ""
    assert "attachment;" in cd
    assert "hello" in resp.text
    rstub.assert_only_called({"hset", "expire", "hgetall"})


def test_job_result_missing_file_is_expired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, redis_stub = _make_client(tmp_path, monkeypatch)
    # Seed completed status but do not create file
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


def test_job_result_data_bank_config_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, rstub = _make_client(tmp_path, monkeypatch)
    monkeypatch.delenv("TURKIC_DATA_BANK_API_URL", raising=False)
    _seed_job(rstub, "j4", "completed", tmp_path)
    resp = client.get("/api/v1/jobs/j4/result")
    assert resp.status_code == 500
    rstub.assert_only_called({"hset", "expire", "hgetall"})


def test_job_result_head_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, rstub = _make_client(tmp_path, monkeypatch)
    _seed_job(rstub, "j5", "completed", tmp_path)
    _stub_data_bank(monkeypatch, head_status=404)
    resp = client.get("/api/v1/jobs/j5/result")
    assert resp.status_code == 410
    rstub.assert_only_called({"hset", "expire", "hgetall"})


def test_job_result_head_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, rstub = _make_client(tmp_path, monkeypatch)
    _seed_job(rstub, "j6", "completed", tmp_path)
    _stub_data_bank(monkeypatch, head_status=500)
    resp = client.get("/api/v1/jobs/j6/result")
    assert resp.status_code == 502
    rstub.assert_only_called({"hset", "expire", "hgetall"})


def test_job_result_stream_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, rstub = _make_client(tmp_path, monkeypatch)
    _seed_job(rstub, "j7", "completed", tmp_path)
    _stub_data_bank(monkeypatch, stream_status=404)
    resp = client.get("/api/v1/jobs/j7/result")
    assert resp.status_code == 410
    rstub.assert_only_called({"hset", "expire", "hgetall"})


def test_job_result_stream_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, rstub = _make_client(tmp_path, monkeypatch)
    _seed_job(rstub, "j8", "completed", tmp_path)
    _stub_data_bank(monkeypatch, stream_status=500)
    resp = client.get("/api/v1/jobs/j8/result")
    assert resp.status_code == 502
    rstub.assert_only_called({"hset", "expire", "hgetall"})
