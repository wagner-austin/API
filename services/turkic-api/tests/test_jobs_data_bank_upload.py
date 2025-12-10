from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.data_bank_client import DataBankClientError
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.logging import get_logger
from platform_core.turkic_jobs import turkic_job_key
from platform_workers.testing import FakeRedis
from tests.conftest import make_probs

import turkic_api.api.jobs as jobs_mod
from turkic_api.api.config import Settings


def _seed_processing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Svc:
        def __init__(self, _root: str) -> None: ...
        def stream(self, _spec: str | int | float | bool | None) -> Generator[str, None, None]:
            yield "hello"

    monkeypatch.setenv("TURKIC_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(jobs_mod, "LocalCorpusService", _Svc)

    def _to_ipa(s: str, _l: str) -> str:
        return s

    monkeypatch.setattr(jobs_mod, "to_ipa", _to_ipa)

    def _ensure(
        *_a: str | int | float | bool | None, **_k: str | int | float | bool | None
    ) -> Path:
        return tmp_path / "corpus" / "oscar_kk.txt"

    monkeypatch.setattr(jobs_mod, "ensure_corpus_file", _ensure)

    # Mock langid model loader
    class _LangModel:
        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            return (("__label__kk",), make_probs(1.0))

    def _load_langid(data_dir: str, prefer_218e: bool = True) -> _LangModel:
        return _LangModel()

    monkeypatch.setattr(jobs_mod, "load_langid_model", _load_langid)


def test_upload_success_records_file_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _seed_processing(monkeypatch, tmp_path)

    # Mock DataBankClient to return a successful response
    class _MockClient:
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
                "size": 10,
                "sha256": "abc",
                "content_type": "text/plain",
                "created_at": "2024-01-01T00:00:00Z",
            }

    monkeypatch.setattr(jobs_mod, "DataBankClient", _MockClient)

    redis = FakeRedis()
    settings = Settings(
        redis_url="redis://localhost:6379/0",
        data_dir=str(tmp_path),
        environment="test",
        data_bank_api_url="http://db",
        data_bank_api_key="k",
    )
    logger = get_logger(__name__)

    out = jobs_mod.process_corpus_impl(
        "jid1",
        {
            "user_id": 42,
            "source": "oscar",
            "language": "kk",
            "script": None,
            "max_sentences": 1,
            "transliterate": True,
            "confidence_threshold": 0.9,
        },
        redis=redis,
        settings=settings,
        logger=logger,
    )
    assert out["status"] == "completed"
    h = redis._hashes.get(turkic_job_key("jid1"), {})
    assert h.get("file_id") == "deadbeef"
    assert h.get("upload_status") == "uploaded"
    meta = redis._hashes.get(f"{turkic_job_key('jid1')}:file", {})
    assert meta.get("size") == "10"
    assert meta.get("sha256") == "abc"
    assert meta.get("created_at") == "2024-01-01T00:00:00Z"
    redis.assert_only_called({"hset", "expire", "publish"})


@pytest.mark.parametrize("error_msg", ["HTTP 400", "HTTP 401", "HTTP 403", "HTTP 500"])
def test_upload_failure_breaks_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_msg: str
) -> None:
    _seed_processing(monkeypatch, tmp_path)

    class _MockClient:
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
            raise DataBankClientError(error_msg)

    monkeypatch.setattr(jobs_mod, "DataBankClient", _MockClient)

    redis = FakeRedis()
    settings = Settings(
        redis_url="redis://localhost:6379/0",
        data_dir=str(tmp_path),
        environment="test",
        data_bank_api_url="http://db",
        data_bank_api_key="k",
    )
    logger = get_logger(__name__)

    with pytest.raises(DataBankClientError, match=error_msg):
        jobs_mod.process_corpus_impl(
            "jid2",
            {
                "user_id": 42,
                "source": "oscar",
                "language": "kk",
                "script": None,
                "max_sentences": 1,
                "transliterate": True,
                "confidence_threshold": 0.9,
            },
            redis=redis,
            settings=settings,
            logger=logger,
        )
    redis.assert_only_called({"hset", "expire", "publish"})


def test_upload_missing_file_id_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _seed_processing(monkeypatch, tmp_path)

    class _MockClient:
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

    monkeypatch.setattr(jobs_mod, "DataBankClient", _MockClient)

    redis = FakeRedis()
    settings = Settings(
        redis_url="redis://localhost:6379/0",
        data_dir=str(tmp_path),
        environment="test",
        data_bank_api_url="http://db",
        data_bank_api_key="k",
    )
    logger = get_logger(__name__)

    with pytest.raises(DataBankClientError, match="missing file_id"):
        jobs_mod.process_corpus_impl(
            "jid3",
            {
                "user_id": 42,
                "source": "oscar",
                "language": "kk",
                "script": None,
                "max_sentences": 1,
                "transliterate": True,
                "confidence_threshold": 0.9,
            },
            redis=redis,
            settings=settings,
            logger=logger,
        )
    redis.assert_only_called({"hset", "expire", "publish"})


def test_upload_config_missing_marks_job_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_processing(monkeypatch, tmp_path)

    # Leave data_bank_api_url and key empty to trigger config error
    redis = FakeRedis()
    settings = Settings(
        redis_url="redis://localhost:6379/0",
        data_dir=str(tmp_path),
        environment="test",
        data_bank_api_url="",
        data_bank_api_key="",
    )
    logger = get_logger(__name__)

    with pytest.raises(DataBankClientError, match="data-bank configuration missing"):
        jobs_mod.process_corpus_impl(
            "jid_cfg",
            {
                "user_id": 42,
                "source": "oscar",
                "language": "kk",
                "script": None,
                "max_sentences": 1,
                "transliterate": True,
                "confidence_threshold": 0.9,
            },
            redis=redis,
            settings=settings,
            logger=logger,
        )

    h = redis._hashes.get(turkic_job_key("jid_cfg"))
    if h is None:
        pytest.fail("expected job hash")
    assert h.get("status") == "failed"
    assert h.get("message") == "upload_failed"
    assert h.get("error") == "config_missing"
    redis.assert_only_called({"hset", "expire", "publish"})
