"""Tests for process_corpus entry points in jobs module."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray
from platform_core.config import config_test_hooks
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.json_utils import JSONValue
from platform_core.testing import make_fake_env
from platform_workers.testing import FakeRedis

from tests.conftest import make_probs
from turkic_api import _test_hooks
from turkic_api.api.jobs import _decode_process_corpus, process_corpus
from turkic_api.core.models import ProcessSpec


class FakeDataBankClient:
    """Fake data bank client for testing."""

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


class FakeCorpusService:
    """Fake corpus service that yields a single line."""

    def __init__(self, _root: str) -> None:
        pass

    def stream(self, _spec: ProcessSpec) -> Generator[str, None, None]:
        yield "hello"


class FakeLangModel:
    """Fake language ID model."""

    def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
        return (("__label__kk",), make_probs(1.0))


def _setup_job_hooks(tmp_path: Path, stub: FakeRedis) -> None:
    """Set up test hooks for job processing tests."""
    env = make_fake_env(
        {
            "TURKIC_DATA_DIR": str(tmp_path),
            "TURKIC_DATA_BANK_API_URL": "http://db",
            "TURKIC_DATA_BANK_API_KEY": "k",
            "TURKIC_REDIS_URL": "redis://test:6379/0",
        }
    )
    config_test_hooks.get_env = env

    _test_hooks.redis_factory = lambda _url: stub
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
    _test_hooks.data_bank_client_factory = (
        lambda api_url, api_key, timeout_seconds: FakeDataBankClient(
            api_url, api_key, timeout_seconds=timeout_seconds
        )
    )


def test_process_corpus_entry(tmp_path: Path) -> None:
    """Test _decode_process_corpus entry point."""
    stub = FakeRedis()
    _setup_job_hooks(tmp_path, stub)

    params: dict[str, JSONValue] = {
        "user_id": 42,
        "source": "oscar",
        "language": "kk",
        "max_sentences": 1,
        "transliterate": True,
        "confidence_threshold": 0.9,
    }

    result = _decode_process_corpus("e1", params)
    assert result["status"] == "completed"
    assert stub.closed is True
    # With streaming upload, no local result file is written.
    assert not (tmp_path / "results" / "e1.txt").exists()
    stub.assert_only_called({"hset", "expire", "publish", "close"})


def test_process_corpus_public_entry(tmp_path: Path) -> None:
    """Test the public process_corpus entry point delegates correctly."""
    stub = FakeRedis()
    _setup_job_hooks(tmp_path, stub)

    # Use Mapping[str, JSONValue] compatible input (simulating RQ payload)
    params: dict[str, JSONValue] = {
        "user_id": 42,
        "source": "oscar",
        "language": "kk",
        "max_sentences": 1,
        "transliterate": True,
        "confidence_threshold": 0.9,
    }

    # Call the PUBLIC entry point
    result = process_corpus("e2", params)
    assert result["status"] == "completed"
    assert stub.closed is True
    stub.assert_only_called({"hset", "expire", "publish", "close"})
