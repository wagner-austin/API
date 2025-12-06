from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import BinaryIO, Final, TypedDict

import pytest
from platform_core.data_bank_client import DataBankClientError
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.job_events import (
    decode_job_event,
    is_completed,
    is_failed,
    is_progress,
    is_started,
)
from platform_core.job_types import job_key
from platform_workers.testing import FakeRedis

from data_bank_api.api.config import Settings
from data_bank_api.api.jobs import JobParams, LoggerLike, process_corpus_impl


class _LogExtraDict(TypedDict, total=False):
    job_id: str


class _FakeLogger(LoggerLike):
    """Minimal logger stub compatible with LoggerLike protocol."""

    def __init__(self) -> None:
        self.records: list[dict[str, str | _LogExtraDict]] = []

    def info(self, msg: str, *, extra: _LogExtraDict | None = None) -> None:
        self.records.append({"level": "info", "msg": msg, "extra": extra or {}})

    def error(self, msg: str, *, extra: _LogExtraDict | None = None) -> None:
        self.records.append({"level": "error", "msg": msg, "extra": extra or {}})


class _MockDataBankClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 60.0,
    ) -> None:
        pass

    def upload(
        self,
        file_id: str,
        stream: BinaryIO,
        *,
        content_type: str = "application/octet-stream",
        request_id: str | None = None,
    ) -> FileUploadResponse:
        return {
            "file_id": "deadbeef",
            "size": 3,
            "sha256": "a" * 64,
            "content_type": "text/plain",
            "created_at": "2025-01-01T00:00:00Z",
        }


def _settings(tmp_path: Path, *, url: str = "", key: str = "") -> Settings:
    return Settings(
        data_dir=str(tmp_path),
        environment="test",
        data_bank_api_url=url,
        data_bank_api_key=key,
    )


def test_jobs_uploads_to_data_bank_and_sets_file_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange a tiny corpus stream and environment
    job_id: Final[str] = "job-123"

    # Patch corpus ensure + streaming
    def _noop_ensure1(
        *,
        source: str,
        language: str,
        data_dir: str,
        max_sentences: int,
        transliterate: bool,
        confidence_threshold: float,
    ) -> None:
        return None

    monkeypatch.setattr("data_bank_api.core.corpus_download.ensure_corpus_file", _noop_ensure1)

    class _Svc:
        def __init__(self, _data_dir: str) -> None:
            pass

        def stream(self, _spec: JobParams) -> Generator[str, None, None]:
            yield from ["a", "b", "c"]

    import data_bank_api.api.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "LocalCorpusService", _Svc)
    monkeypatch.setattr(jobs_mod, "DataBankClient", _MockDataBankClient)

    # Redis in-memory
    r = FakeRedis()
    s = _settings(tmp_path, url="http://db", key="K")
    log = _FakeLogger()

    # Act
    out = process_corpus_impl(
        job_id,
        params={
            "source": "oscar",
            "language": "kk",
            "max_sentences": 3,
            "transliterate": False,
            "confidence_threshold": 0.9,
        },
        redis=r,
        settings=s,
        logger=log,
    )

    # file_id persisted in redis
    data = r.hgetall(job_key("databank", job_id))
    assert data.get("file_id") == "deadbeef"
    events = [decode_job_event(record.payload) for record in r.published]
    assert any(is_started(ev) for ev in events)
    assert any(is_progress(ev) for ev in events)
    assert any(is_completed(ev) for ev in events)
    assert not any(is_failed(ev) for ev in events)
    # function result reflects completion
    assert out["status"] == "completed"


def test_jobs_upload_handles_missing_file_id_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange minimal environment and noop corpus
    def _noop_ensure(
        *,
        source: str,
        language: str,
        data_dir: str,
        max_sentences: int,
        transliterate: bool,
        confidence_threshold: float,
    ) -> None:
        return None

    monkeypatch.setattr("data_bank_api.core.corpus_download.ensure_corpus_file", _noop_ensure)

    class _Svc2:
        def __init__(self, _data_dir: str) -> None:
            pass

        def stream(self, _spec: JobParams) -> Generator[str, None, None]:
            yield from ["x"]

    class _BadClient:
        def __init__(
            self,
            base_url: str,
            api_key: str,
            *,
            timeout_seconds: float = 60.0,
        ) -> None:
            pass

        def upload(
            self,
            file_id: str,
            stream: BinaryIO,
            *,
            content_type: str = "application/octet-stream",
            request_id: str | None = None,
        ) -> FileUploadResponse:
            raise DataBankClientError("upload response missing file_id")

    import data_bank_api.api.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "LocalCorpusService", _Svc2)
    monkeypatch.setattr(jobs_mod, "DataBankClient", _BadClient)

    r = FakeRedis()
    s = _settings(tmp_path, url="http://db", key="K")
    with pytest.raises(DataBankClientError, match="missing file_id"):
        process_corpus_impl(
            "job-2",
            params={
                "source": "oscar",
                "language": "kk",
                "max_sentences": 1,
                "transliterate": False,
                "confidence_threshold": 0.9,
            },
            redis=r,
            settings=s,
            logger=_FakeLogger(),
        )
    events = [decode_job_event(record.payload) for record in r.published]
    assert any(is_started(ev) for ev in events)
    assert any(is_progress(ev) for ev in events)
    assert any(is_failed(ev) for ev in events)
    assert not any(is_completed(ev) for ev in events)


def test_jobs_upload_config_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Arrange minimal environment and noop corpus
    def _noop_ensure(
        *,
        source: str,
        language: str,
        data_dir: str,
        max_sentences: int,
        transliterate: bool,
        confidence_threshold: float,
    ) -> None:
        return None

    monkeypatch.setattr("data_bank_api.core.corpus_download.ensure_corpus_file", _noop_ensure)

    class _Svc3:
        def __init__(self, _data_dir: str) -> None:
            pass

        def stream(self, _spec: JobParams) -> Generator[str, None, None]:
            yield from ["x"]

    import data_bank_api.api.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "LocalCorpusService", _Svc3)

    r = FakeRedis()
    # Empty config triggers error before client is created
    s = _settings(tmp_path, url="", key="")
    with pytest.raises(DataBankClientError, match="configuration missing"):
        process_corpus_impl(
            "job-cfg",
            params={
                "source": "oscar",
                "language": "kk",
                "max_sentences": 1,
                "transliterate": False,
                "confidence_threshold": 0.9,
            },
            redis=r,
            settings=s,
            logger=_FakeLogger(),
        )
    events = [decode_job_event(record.payload) for record in r.published]
    assert any(is_started(ev) for ev in events)
    assert any(is_failed(ev) for ev in events)
    assert not any(is_completed(ev) for ev in events)
    assert not any(is_progress(ev) for ev in events)
