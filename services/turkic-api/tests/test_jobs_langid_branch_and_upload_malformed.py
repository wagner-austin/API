from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import BinaryIO

import pytest
from platform_core.data_bank_client import DataBankClientError
from platform_core.logging import get_logger

import turkic_api.api.jobs as jobs_mod
from turkic_api.api.config import Settings
from turkic_api.core.langid import LangIdModel


class _RedisStub:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.published: list[tuple[str, str]] = []

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        cur = self.hashes.get(key, {})
        cur.update(mapping)
        self.hashes[key] = cur
        return 1

    def hgetall(self, key: str) -> dict[str, str]:
        return self.hashes.get(key, {}).copy()

    def publish(self, channel: str, message: str) -> int:
        self.published.append((channel, message))
        return 1

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def set(self, key: str, value: str) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return None

    def hget(self, key: str, field: str) -> str | None:
        return None

    def sismember(self, key: str, member: str) -> bool:
        return False

    def close(self) -> None:
        pass

    def sadd(self, key: str, *values: str) -> int:
        return len(values)

    def scard(self, key: str) -> int:
        return len(self.hashes.get(key, {}))


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
            def predict(self, text: str, k: int = 1) -> tuple[list[str], list[float]]:
                return (["__label__kk"], [1.0])

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

    redis = _RedisStub()
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
