"""Tests for data bank upload functionality in jobs module."""

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
from turkic_api import _test_hooks
from turkic_api.api.config import Settings
from turkic_api.api.jobs import process_corpus_impl
from turkic_api.core.models import ProcessSpec


class FakeCorpusService:
    """Fake corpus service that yields a single line."""

    def __init__(self, _root: str) -> None:
        pass

    def stream(self, _spec: ProcessSpec) -> Generator[str, None, None]:
        yield "hello"


class FakeLangModel:
    """Fake language ID model that always returns kk with 1.0 probability."""

    def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
        return (("__label__kk",), make_probs(1.0))


def _setup_processing_hooks(tmp_path: Path) -> None:
    """Set up test hooks for job processing tests."""
    _test_hooks.local_corpus_service_factory = lambda _data_dir: FakeCorpusService(_data_dir)
    _test_hooks.to_ipa = lambda s, _l: s

    def _fake_ensure_corpus(
        spec: ProcessSpec,
        data_dir: str,
        script: str | None = None,
        *,
        langid_model: _test_hooks.LangIdModelProtocol | None = None,
    ) -> Path:
        return tmp_path / "corpus" / "oscar_kk.txt"

    _test_hooks.ensure_corpus_file = _fake_ensure_corpus
    _test_hooks.load_langid_model = lambda _data_dir, prefer_218e=True: FakeLangModel()


class FakeDataBankClientSuccess:
    """Fake data bank client that returns a successful upload response."""

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


def test_upload_success_records_file_id(tmp_path: Path) -> None:
    """Test that successful upload records file metadata."""
    _setup_processing_hooks(tmp_path)
    _test_hooks.data_bank_client_factory = (
        lambda api_url, api_key, timeout_seconds: FakeDataBankClientSuccess(
            api_url, api_key, timeout_seconds=timeout_seconds
        )
    )

    redis = FakeRedis()
    settings = Settings(
        redis_url="redis://localhost:6379/0",
        data_dir=str(tmp_path),
        environment="test",
        data_bank_api_url="http://db",
        data_bank_api_key="k",
    )
    logger = get_logger(__name__)

    out = process_corpus_impl(
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


class FakeDataBankClientError:
    """Fake data bank client that raises an error on upload."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 60.0,
        error_msg: str = "HTTP 400",
    ) -> None:
        self._error_msg = error_msg

    def upload(
        self,
        file_id: str,
        stream: BinaryIO,
        *,
        content_type: str = "application/octet-stream",
        request_id: str | None = None,
    ) -> FileUploadResponse:
        raise DataBankClientError(self._error_msg)


@pytest.mark.parametrize("error_msg", ["HTTP 400", "HTTP 401", "HTTP 403", "HTTP 500"])
def test_upload_failure_breaks_job(tmp_path: Path, error_msg: str) -> None:
    """Test that upload failure raises DataBankClientError."""
    _setup_processing_hooks(tmp_path)

    def _make_error_client(
        api_url: str, api_key: str, *, timeout_seconds: float
    ) -> FakeDataBankClientError:
        return FakeDataBankClientError(
            api_url, api_key, timeout_seconds=timeout_seconds, error_msg=error_msg
        )

    _test_hooks.data_bank_client_factory = _make_error_client

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
        process_corpus_impl(
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


def test_upload_missing_file_id_raises(tmp_path: Path) -> None:
    """Test that missing file_id in response raises error."""
    _setup_processing_hooks(tmp_path)

    def _make_error_client(
        api_url: str, api_key: str, *, timeout_seconds: float
    ) -> FakeDataBankClientError:
        return FakeDataBankClientError(
            api_url,
            api_key,
            timeout_seconds=timeout_seconds,
            error_msg="upload response missing file_id",
        )

    _test_hooks.data_bank_client_factory = _make_error_client

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
        process_corpus_impl(
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


def test_upload_config_missing_marks_job_failed(tmp_path: Path) -> None:
    """Test that missing config marks job as failed."""
    _setup_processing_hooks(tmp_path)

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
        process_corpus_impl(
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
