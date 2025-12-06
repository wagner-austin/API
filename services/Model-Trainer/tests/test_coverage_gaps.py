"""Tests to cover remaining coverage gaps for 100% statement and branch coverage."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import Settings, load_settings
from model_trainer.core.infra.paths import tokenizer_logs_path
from model_trainer.core.infra.redis_utils import get_with_retry, set_with_retry
from model_trainer.core.logging.utils import narrow_log_level
from model_trainer.core.services.data.corpus import list_text_files

# --- paths.py coverage: line 22 (tokenizer_logs_path) ---


def test_tokenizer_logs_path_returns_expected_path() -> None:
    """Cover tokenizer_logs_path function (paths.py line 22)."""
    settings = load_settings()
    tok_id = "test-tok-123"
    result = tokenizer_logs_path(settings, tok_id)
    assert result.name == "logs.jsonl"
    assert tok_id in str(result)


# --- logging/utils.py coverage: lines 16, 19-25 (narrow_log_level branches) ---


def test_narrow_log_level_debug() -> None:
    """Cover DEBUG branch."""
    assert narrow_log_level("DEBUG") == "DEBUG"


def test_narrow_log_level_info() -> None:
    """Cover INFO branch."""
    assert narrow_log_level("INFO") == "INFO"


def test_narrow_log_level_warning() -> None:
    """Cover WARNING branch (line 19)."""
    assert narrow_log_level("WARNING") == "WARNING"


def test_narrow_log_level_error() -> None:
    """Cover ERROR branch (line 21)."""
    assert narrow_log_level("ERROR") == "ERROR"


def test_narrow_log_level_critical() -> None:
    """Cover CRITICAL branch (line 23)."""
    assert narrow_log_level("CRITICAL") == "CRITICAL"


def test_narrow_log_level_unknown_defaults_to_info() -> None:
    """Cover default branch (line 25)."""
    assert narrow_log_level("UNKNOWN") == "INFO"
    assert narrow_log_level("") == "INFO"
    assert narrow_log_level("debug") == "INFO"  # Case sensitive


# --- redis_utils.py coverage: lines 18, 35 (non-Redis error re-raises) ---


class _NonRedisErrorClient(FakeRedis):
    """FakeRedis that raises a non-Redis exception on get/set."""

    def set(self, key: str, value: str) -> bool:
        raise ValueError("not a redis error")

    def get(self, key: str) -> str | None:
        raise ValueError("not a redis error")


def test_redis_utils_get_non_redis_error_reraises() -> None:
    """Cover redis_utils.py line 18 (non-Redis error re-raises in get_with_retry)."""
    client = _NonRedisErrorClient()
    with pytest.raises(ValueError, match="not a redis error"):
        get_with_retry(client, "key", attempts=3)


def test_redis_utils_set_non_redis_error_reraises() -> None:
    """Cover redis_utils.py line 35 (non-Redis error re-raises in set_with_retry)."""
    client = _NonRedisErrorClient()
    with pytest.raises(ValueError, match="not a redis error"):
        set_with_retry(client, "key", "value", attempts=3)


# --- corpus.py coverage: branch 15->14 (non-.txt files skipped) ---


def test_list_text_files_skips_non_txt_files(tmp_path: Path) -> None:
    """Cover corpus.py branch where non-.txt files are skipped (line 15->14)."""
    # Create a directory with mixed file types
    (tmp_path / "file1.txt").write_text("text file", encoding="utf-8")
    (tmp_path / "file2.json").write_text("{}", encoding="utf-8")
    (tmp_path / "file3.py").write_text("# python", encoding="utf-8")
    (tmp_path / "file4.text").write_text("another text", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("nested", encoding="utf-8")
    (tmp_path / "subdir" / "nested.md").write_text("# markdown", encoding="utf-8")

    result = list_text_files(str(tmp_path))

    # Should only include .txt and .text files
    assert len(result) == 3
    filenames = [Path(p).name for p in result]
    assert "file1.txt" in filenames
    assert "file4.text" in filenames
    assert "nested.txt" in filenames
    # Should NOT include non-txt files
    assert "file2.json" not in filenames
    assert "file3.py" not in filenames
    assert "nested.md" not in filenames


# --- container.py coverage: line 88 (wrong queue name raises) ---


def test_container_wrong_queue_name_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover container.py line 88 (ValueError when queue_name != TRAINER_QUEUE)."""

    from model_trainer.core.services.container import _create_enqueuer

    settings = load_settings()
    # Modify settings to have wrong queue name
    wrong_settings: Settings = {
        **settings,
        "rq": {
            **settings["rq"],
            "queue_name": "wrong-queue-name",
        },
    }
    with pytest.raises(ValueError, match="RQ queue must be trainer per platform alignment"):
        _create_enqueuer(wrong_settings)


# --- health.py coverage: line 57 (non-Redis error re-raises in readyz) ---


class _NonRedisExceptionClient(FakeRedis):
    """FakeRedis that raises a non-Redis exception on ping."""

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        raise TypeError("not a redis error")


def test_health_readyz_non_redis_error_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover health.py line 57 (non-Redis error re-raises in readyz).

    Uses TestClient to properly invoke the readyz endpoint through FastAPI.
    """
    from fastapi.testclient import TestClient

    from model_trainer.api.main import create_app
    from model_trainer.core.services import container as container_mod

    # Monkeypatch redis_for_kv to return our failing client
    def _fake_redis_for_kv(url: str) -> _NonRedisExceptionClient:
        return _NonRedisExceptionClient()

    monkeypatch.setattr(container_mod, "redis_for_kv", _fake_redis_for_kv)

    app = create_app()
    client = TestClient(app, raise_server_exceptions=True)

    with pytest.raises(TypeError, match="not a redis error"):
        client.get("/readyz")


# --- tokenizer_worker.py coverage: line 24 (empty Redis URL raises) ---


def test_tokenizer_worker_empty_redis_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover tokenizer_worker.py line 24 (RuntimeError when Redis URL is empty)."""
    from model_trainer.worker.tokenizer_worker import _redis_client

    settings = load_settings()
    # Modify settings to have empty Redis URL
    empty_redis_settings: Settings = {
        **settings,
        "redis": {
            **settings["redis"],
            "url": "   ",  # Empty/whitespace URL
        },
    }
    with pytest.raises(RuntimeError, match="Redis URL must not be empty"):
        _redis_client(empty_redis_settings)


# --- train_job.py coverage: (non-Redis error when recording error) ---
# This is covered via integration tests in test_training_worker_error_handling.py
# The pattern requires triggering a non-Redis error during _handle_train_error after
# a training failure, which is tested through the full worker flow.
