from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.data_bank_client import DataBankClientError
from platform_core.logging import get_logger
from platform_workers.testing import FakeRedis
from tests.conftest import make_probs

import turkic_api.api.jobs as jobs_mod
from turkic_api.api.config import Settings
from turkic_api.core.langid import LangIdModel


def _seed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Svc:
        def __init__(self, _root: str) -> None: ...
        def stream(self, _spec: str | int | float | bool | None) -> Generator[str, None, None]:
            yield "hello"

    monkeypatch.setattr(jobs_mod, "LocalCorpusService", _Svc)

    def _to_ipa(s: str, _l: str) -> str:
        return s

    monkeypatch.setattr(jobs_mod, "to_ipa", _to_ipa)

    def _ensure(
        *_a: str | int | float | bool | None, **_k: str | int | float | bool | None
    ) -> Path:
        return tmp_path / "corpus" / "oscar_kk.txt"

    monkeypatch.setattr(jobs_mod, "ensure_corpus_file", _ensure)


def test_langid_branch_and_malformed_file_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed(monkeypatch, tmp_path)

    # Force non-test environment to load model, then stub loader
    def _load(_data_dir: str) -> LangIdModel:
        class _Model:
            def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
                return (("__label__kk",), make_probs(1.0))

        return _Model()

    monkeypatch.setattr(jobs_mod, "load_langid_model", _load)

    # Mock DataBankClient to raise an error on upload
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
        ) -> None:
            raise DataBankClientError("upload response missing file_id")

    monkeypatch.setattr(jobs_mod, "DataBankClient", _MockClient)

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
        jobs_mod.process_corpus_impl(
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
