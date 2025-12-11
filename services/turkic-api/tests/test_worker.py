from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.logging import get_logger
from platform_core.turkic_jobs import turkic_job_key
from platform_workers.testing import FakeRedis
from tests.conftest import make_probs

from turkic_api import _test_hooks
from turkic_api.api.config import Settings
from turkic_api.api.jobs import process_corpus_impl


class _FakeDataBankClient:
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
            "size": 1,
            "sha256": "abc",
            "content_type": "text/plain",
            "created_at": "2024-01-01T00:00:00Z",
        }


class _FakeLangModel:
    def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
        return (("__label__kk",), make_probs(1.0))


def test_process_corpus_impl_creates_file_and_updates_status(tmp_path: Path) -> None:
    redis = FakeRedis()
    settings = Settings(
        redis_url="redis://localhost:6379/0",
        data_dir=str(tmp_path),
        environment="test",
        data_bank_api_url="http://db",
        data_bank_api_key="k",
    )
    logger = get_logger(__name__)

    # Seed a local corpus file matching spec: oscar_kk.txt
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    (corpus_dir / "oscar_kk.txt").write_text("Қазақстан\n", encoding="utf-8")

    # Set up test hooks for DataBankClient and langid model
    def _fake_data_bank_factory(
        api_url: str, api_key: str, *, timeout_seconds: float
    ) -> _FakeDataBankClient:
        return _FakeDataBankClient(api_url, api_key, timeout_seconds=timeout_seconds)

    _test_hooks.data_bank_client_factory = _fake_data_bank_factory

    def _fake_load_model(data_dir: str, prefer_218e: bool = True) -> _FakeLangModel:
        return _FakeLangModel()

    _test_hooks.load_langid_model = _fake_load_model

    from turkic_api.api.jobs import JobParams

    params: JobParams = {
        "user_id": 42,
        "source": "oscar",
        "language": "kk",
        "script": None,
        "max_sentences": 10,
        "transliterate": True,
        "confidence_threshold": 0.95,
    }
    result = process_corpus_impl("w1", params, redis=redis, settings=settings, logger=logger)

    # No local result file when streaming upload
    out = tmp_path / "results" / "w1.txt"
    assert not out.exists()

    # Redis status updated
    h = redis._hashes.get(turkic_job_key("w1"))
    if h is None:
        pytest.fail("expected job hash")
    assert h.get("status") == "completed"
    assert h.get("progress") == "100"
    assert result["status"] == "completed"
    redis.assert_only_called({"hset", "publish"})
