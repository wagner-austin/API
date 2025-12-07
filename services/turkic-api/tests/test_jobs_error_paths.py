from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import BinaryIO

import pytest
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.logging import get_logger
from platform_core.turkic_jobs import turkic_job_key
from platform_workers.testing import FakeRedis

import turkic_api.api.jobs as jobs_mod
from turkic_api.api.config import Settings
from turkic_api.core.langid import LangIdModel
from turkic_api.core.models import ProcessSpec


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


def test_process_spec_type_errors() -> None:
    # Test validation errors in _decode_job_params
    with pytest.raises(TypeError, match="source and language"):
        jobs_mod._decode_job_params({"user_id": 42, "source": 1, "language": 2})

    with pytest.raises(TypeError, match="max_sentences"):
        jobs_mod._decode_job_params(
            {"user_id": 42, "source": "oscar", "language": "kk", "max_sentences": "x"}
        )

    with pytest.raises(TypeError, match="transliterate"):
        jobs_mod._decode_job_params(
            {
                "user_id": 42,
                "source": "oscar",
                "language": "kk",
                "max_sentences": 1,
                "transliterate": "y",
            }
        )

    with pytest.raises(TypeError, match="confidence_threshold"):
        jobs_mod._decode_job_params(
            {
                "user_id": 42,
                "source": "oscar",
                "language": "kk",
                "max_sentences": 1,
                "transliterate": True,
                "confidence_threshold": "no",
            }
        )


def test_invalid_source_or_language() -> None:
    # Test validation errors in _decode_job_params
    with pytest.raises(ValueError, match="Invalid source or language"):
        jobs_mod._decode_job_params(
            {
                "user_id": 42,
                "source": "bogus",
                "language": "kk",
                "max_sentences": 1,
                "transliterate": True,
                "confidence_threshold": 0.9,
            }
        )


def test_progress_updates_every_50(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    redis = FakeRedis()
    settings = Settings(
        redis_url="redis://localhost:6379/0",
        data_dir=str(tmp_path),
        environment="test",
        data_bank_api_url="http://db",
        data_bank_api_key="k",
    )
    logger = get_logger(__name__)

    class _Svc:
        def __init__(self, _data_dir: str) -> None: ...
        def stream(self, _spec: ProcessSpec) -> Generator[str, None, None]:
            for i in range(100):
                yield f"line {i}"

    monkeypatch.setattr(jobs_mod, "LocalCorpusService", _Svc)

    def _to_ipa(s: str, _l: str) -> str:
        return s

    monkeypatch.setattr(jobs_mod, "to_ipa", _to_ipa)

    def _ensure(
        spec: ProcessSpec,
        data_dir: str,
        script: str | None = None,
        *,
        langid_model: LangIdModel | None = None,
    ) -> Path:
        return tmp_path / "corpus" / "oscar_kk.txt"

    monkeypatch.setattr(jobs_mod, "ensure_corpus_file", _ensure)

    monkeypatch.setattr(jobs_mod, "DataBankClient", _MockDataBankClient)

    params: jobs_mod.JobParams = {
        "user_id": 42,
        "source": "oscar",
        "language": "kk",
        "script": None,
        "max_sentences": 1000,
        "transliterate": True,
        "confidence_threshold": 0.0,
    }

    result = jobs_mod.process_corpus_impl(
        "p1", params, redis=redis, settings=settings, logger=logger
    )
    h = redis._hashes.get(turkic_job_key("p1"))
    if h is None:
        pytest.fail("expected job hash")
    assert h.get("status") == "completed"
    assert result["status"] == "completed"
    redis.assert_only_called({"hset", "expire", "publish"})


def test_invalid_script_type_raises() -> None:
    with pytest.raises(TypeError, match="script must be a string or null"):
        jobs_mod._decode_job_params(
            {
                "user_id": 42,
                "source": "oscar",
                "language": "kk",
                "max_sentences": 1,
                "transliterate": True,
                "confidence_threshold": 0.9,
                "script": 123,
            }
        )


def test_invalid_script_value_raises() -> None:
    with pytest.raises(ValueError, match="Invalid script"):
        jobs_mod._decode_job_params(
            {
                "user_id": 42,
                "source": "oscar",
                "language": "kk",
                "max_sentences": 1,
                "transliterate": True,
                "confidence_threshold": 0.9,
                "script": "Greek",
            }
        )


def test_valid_script_normalizes_and_passes() -> None:
    """Test that valid script values are normalized correctly in _decode_job_params."""
    result = jobs_mod._decode_job_params(
        {
            "user_id": 42,
            "source": "oscar",
            "language": "kk",
            "max_sentences": 1,
            "transliterate": True,
            "confidence_threshold": 0.9,
            "script": "latn",
        }
    )
    assert result["script"] == "Latn"


def test_blank_script_string_is_treated_as_none() -> None:
    """Test that blank script strings are treated as None in _decode_job_params."""
    result = jobs_mod._decode_job_params(
        {
            "user_id": 42,
            "source": "oscar",
            "language": "kk",
            "max_sentences": 1,
            "transliterate": True,
            "confidence_threshold": 0.9,
            "script": "   ",
        }
    )
    assert result["script"] is None


def test_normalize_script_with_value() -> None:
    """Test _normalize_script with a valid script value."""
    assert jobs_mod._normalize_script("latn") == "Latn"
    assert jobs_mod._normalize_script("CYRL") == "Cyrl"
    assert jobs_mod._normalize_script("ArAb") == "Arab"


def test_normalize_script_with_whitespace_only() -> None:
    """Test _normalize_script treats whitespace-only as None."""
    assert jobs_mod._normalize_script("   ") is None
    assert jobs_mod._normalize_script("\t\n") is None


def test_normalize_script_with_none() -> None:
    """Test _normalize_script returns None for None input."""
    assert jobs_mod._normalize_script(None) is None


def test_get_redis_client_returns_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_client = FakeRedis()

    def fake_for_kv(url: str) -> FakeRedis:
        assert url == "redis://example"
        return stub_client

    monkeypatch.setattr(jobs_mod, "redis_for_kv", fake_for_kv)
    client = jobs_mod._get_redis_client("redis://example")
    assert type(client).__name__ == "FakeRedis"
    stub_client.assert_only_called(set())
