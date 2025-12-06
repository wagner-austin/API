from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pytest
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.logging import get_logger
from platform_core.turkic_jobs import turkic_job_key

import turkic_api.api.jobs as jobs_mod
from turkic_api.api.config import Settings
from turkic_api.api.jobs import process_corpus_impl


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


class _RedisStub:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.published: list[tuple[str, str]] = []

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        # Merge mapping into existing hash to mimic Redis semantics
        cur = self.hashes.get(key, {})
        cur.update(mapping)
        self.hashes[key] = cur
        return 1

    def hgetall(self, key: str) -> dict[str, str]:
        return self.hashes.get(key, {}).copy()

    def publish(self, channel: str, message: str) -> int:
        self.published.append((channel, message))
        return 1

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


def test_process_corpus_impl_creates_file_and_updates_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis = _RedisStub()
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

    # Mock DataBankClient
    monkeypatch.setattr(jobs_mod, "DataBankClient", _MockDataBankClient)

    # Provide a trivial langid model to satisfy filtering when threshold>0
    class _LangModel:
        def predict(self, text: str, k: int = 1) -> tuple[list[str], list[float]]:
            return (["__label__kk"], [1.0])

    def _load_model(data_dir: str, prefer_218e: bool = True) -> _LangModel:
        return _LangModel()

    monkeypatch.setattr(jobs_mod, "load_langid_model", _load_model)

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
    h = redis.hashes.get(turkic_job_key("w1"))
    if h is None:
        pytest.fail("expected job hash")
    assert h.get("status") == "completed"
    assert h.get("progress") == "100"
    assert result["status"] == "completed"
