from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import BinaryIO

import pytest
from platform_core.data_bank_protocol import FileUploadResponse

import turkic_api.api.jobs as jobs_mod


class _RedisStub:
    def __init__(self) -> None:
        self.closed = False
        self.hashes: dict[str, dict[str, str]] = {}
        self.published: list[tuple[str, str]] = []

    def close(self) -> None:
        self.closed = True

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

    def sadd(self, key: str, *values: str) -> int:
        return len(values)

    def scard(self, key: str) -> int:
        return len(self.hashes.get(key, {}))


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
            "size": 1,
            "sha256": "abc",
            "content_type": "text/plain",
            "created_at": "2024-01-01T00:00:00Z",
        }


def test_process_corpus_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Ensure data dir and data-bank config
    monkeypatch.setenv("TURKIC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TURKIC_DATA_BANK_API_URL", "http://db")
    monkeypatch.setenv("TURKIC_DATA_BANK_API_KEY", "k")

    # Stub _get_redis_client
    stub = _RedisStub()

    def _get_client(_url: str) -> _RedisStub:
        return stub

    monkeypatch.setattr(jobs_mod, "_get_redis_client", _get_client)

    # Stub corpus and transliteration
    class _Svc:
        def __init__(self, _root: str) -> None: ...
        def stream(self, _spec: str | int | float | bool | None) -> Generator[str, None, None]:
            yield "hello"

    monkeypatch.setattr(jobs_mod, "LocalCorpusService", _Svc)

    def _to_ipa(s: str, _l: str) -> str:
        return s

    monkeypatch.setattr(jobs_mod, "to_ipa", _to_ipa)

    # Avoid network in test: pretend corpus file exists (single patch)
    def _ensure(
        *_a: str | int | float | bool | None, **_k: str | int | float | bool | None
    ) -> Path:
        return tmp_path / "corpus" / "oscar_kk.txt"

    monkeypatch.setattr(jobs_mod, "ensure_corpus_file", _ensure)

    # Mock langid model loader
    class _LangModel:
        def predict(self, text: str, k: int = 1) -> tuple[list[str], list[float]]:
            return (["__label__kk"], [1.0])

    def _load_langid(data_dir: str, prefer_218e: bool = True) -> _LangModel:
        return _LangModel()

    monkeypatch.setattr(jobs_mod, "load_langid_model", _load_langid)

    # Mock DataBankClient
    monkeypatch.setattr(jobs_mod, "DataBankClient", _MockDataBankClient)

    from turkic_api.core.models import UnknownJson

    params: dict[str, UnknownJson] = {
        "user_id": 42,
        "source": "oscar",
        "language": "kk",
        "max_sentences": 1,
        "transliterate": True,
        "confidence_threshold": 0.9,
    }

    result = jobs_mod._decode_process_corpus("e1", params)
    assert result["status"] == "completed"
    assert stub.closed is True
    # With streaming upload, no local result file is written.
    assert not (tmp_path / "results" / "e1.txt").exists()


def test_process_corpus_public_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test the public process_corpus entry point delegates correctly."""
    # Ensure data dir and data-bank config
    monkeypatch.setenv("TURKIC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TURKIC_DATA_BANK_API_URL", "http://db")
    monkeypatch.setenv("TURKIC_DATA_BANK_API_KEY", "k")

    # Stub _get_redis_client
    stub = _RedisStub()

    def _get_client(_url: str) -> _RedisStub:
        return stub

    monkeypatch.setattr(jobs_mod, "_get_redis_client", _get_client)

    # Stub corpus and transliteration
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

    # Mock langid model loader
    class _LangModel2:
        def predict(self, text: str, k: int = 1) -> tuple[list[str], list[float]]:
            return (["__label__kk"], [1.0])

    def _load_langid2(data_dir: str, prefer_218e: bool = True) -> _LangModel2:
        return _LangModel2()

    monkeypatch.setattr(jobs_mod, "load_langid_model", _load_langid2)

    # Mock DataBankClient
    monkeypatch.setattr(jobs_mod, "DataBankClient", _MockDataBankClient)

    from platform_core.json_utils import JSONValue

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
    result = jobs_mod.process_corpus("e2", params)
    assert result["status"] == "completed"
    assert stub.closed is True
