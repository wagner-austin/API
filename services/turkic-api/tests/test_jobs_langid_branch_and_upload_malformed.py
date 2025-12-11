"""Tests for langid branch and malformed upload error handling."""

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


class FakeLangIdModel:
    """Fake language ID model."""

    def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
        return (("__label__kk",), make_probs(1.0))


class FailingDataBankClient:
    """Fake data bank client that raises error on upload."""

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


def _setup_hooks(tmp_path: Path) -> None:
    """Set up test hooks for this test module."""
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
    _test_hooks.load_langid_model = lambda _data_dir, prefer_218e=True: FakeLangIdModel()
    _test_hooks.data_bank_client_factory = (
        lambda api_url, api_key, timeout_seconds: FailingDataBankClient(
            api_url, api_key, timeout_seconds=timeout_seconds
        )
    )


def test_langid_branch_and_malformed_file_id(tmp_path: Path) -> None:
    """Test langid branch with malformed file_id error from data bank."""
    _setup_hooks(tmp_path)

    redis = FakeRedis()
    settings = Settings(
        redis_url="redis://localhost:6379/0",
        data_dir=str(tmp_path),
        environment="prod",
        data_bank_api_url="http://db",
        data_bank_api_key="k",
    )
    logger = get_logger(__name__)

    with pytest.raises(DataBankClientError, match="upload response missing file_id"):
        process_corpus_impl(
            "jj",
            {
                "user_id": 42,
                "source": "oscar",
                "language": "kk",
                "script": None,
                "max_sentences": 1,
                "transliterate": True,
                "confidence_threshold": 1.0,
            },
            redis=redis,
            settings=settings,
            logger=logger,
        )
    redis.assert_only_called({"hset", "expire", "publish"})
