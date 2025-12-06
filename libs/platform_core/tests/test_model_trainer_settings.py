from __future__ import annotations

from typing import Literal

import pytest

from platform_core.config import ModelTrainerSettings, load_model_trainer_settings


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = [
        "LOGGING__LEVEL",
        "REDIS__ENABLED",
        "REDIS__URL",
        "RQ__QUEUE_NAME",
        "RQ__JOB_TIMEOUT_SEC",
        "RQ__RESULT_TTL_SEC",
        "RQ__FAILURE_TTL_SEC",
        "RQ__RETRY_MAX",
        "RQ__RETRY_INTERVALS_SEC",
        "APP__DATA_ROOT",
        "APP__ARTIFACTS_ROOT",
        "APP__RUNS_ROOT",
        "APP__LOGS_ROOT",
        "APP__THREADS",
        "APP__TOKENIZER_SAMPLE_MAX_LINES",
        "APP__DATA_BANK_API_URL",
        "APP__DATA_BANK_API_KEY",
        "APP__CLEANUP__ENABLED",
        "APP__CLEANUP__VERIFY_UPLOAD",
        "APP__CLEANUP__GRACE_PERIOD_SECONDS",
        "APP__CLEANUP__DRY_RUN",
        "APP__CORPUS_CACHE_CLEANUP__ENABLED",
        "APP__CORPUS_CACHE_CLEANUP__MAX_BYTES",
        "APP__CORPUS_CACHE_CLEANUP__MIN_FREE_BYTES",
        "APP__CORPUS_CACHE_CLEANUP__EVICTION_POLICY",
        "APP__TOKENIZER_CLEANUP__ENABLED",
        "APP__TOKENIZER_CLEANUP__MIN_UNUSED_DAYS",
        "SECURITY__API_KEY",
        "APP_ENV",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_model_trainer_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    cfg: ModelTrainerSettings = load_model_trainer_settings()
    assert cfg["logging"]["level"] == "INFO"
    assert cfg["redis"] == {"enabled": True, "url": "redis://redis:6379/0"}
    assert cfg["rq"]["queue_name"] == "trainer"
    assert cfg["rq"]["job_timeout_sec"] == 86_400
    assert cfg["rq"]["result_ttl_sec"] == 86_400
    assert cfg["rq"]["failure_ttl_sec"] == 7 * 86_400
    assert cfg["rq"]["retry_max"] == 1
    assert cfg["rq"]["retry_intervals_sec"] == "300"
    assert cfg["app"]["data_root"] == "/data"
    assert cfg["app"]["artifacts_root"] == "/data/artifacts"
    assert cfg["app"]["runs_root"] == "/data/runs"
    assert cfg["app"]["logs_root"] == "/data/logs"
    assert cfg["app"]["threads"] == 0
    assert cfg["app"]["tokenizer_sample_max_lines"] == 10000
    assert cfg["app"]["data_bank_api_url"] == ""
    assert cfg["app"]["data_bank_api_key"] == ""
    assert cfg["app"]["cleanup"]["enabled"] is True
    assert cfg["app"]["cleanup"]["verify_upload"] is True
    assert cfg["app"]["cleanup"]["grace_period_seconds"] == 0
    assert cfg["app"]["cleanup"]["dry_run"] is False
    assert cfg["app"]["corpus_cache_cleanup"]["enabled"] is False
    assert cfg["app"]["corpus_cache_cleanup"]["eviction_policy"] == "lru"
    assert cfg["app"]["tokenizer_cleanup"]["enabled"] is False
    assert cfg["app"]["tokenizer_cleanup"]["min_unused_days"] == 30
    assert cfg["security"]["api_key"] == ""
    assert cfg["app_env"] == "dev"


def test_model_trainer_settings_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("LOGGING__LEVEL", "DEBUG")
    monkeypatch.setenv("REDIS__ENABLED", "false")
    monkeypatch.setenv("REDIS__URL", "redis://override:6379/1")
    monkeypatch.setenv("RQ__QUEUE_NAME", "trainer")
    monkeypatch.setenv("RQ__JOB_TIMEOUT_SEC", "100")
    monkeypatch.setenv("RQ__RESULT_TTL_SEC", "200")
    monkeypatch.setenv("RQ__FAILURE_TTL_SEC", "300")
    monkeypatch.setenv("RQ__RETRY_MAX", "2")
    monkeypatch.setenv("RQ__RETRY_INTERVALS_SEC", "10,20")
    monkeypatch.setenv("APP__DATA_ROOT", "/tmp/data")
    monkeypatch.setenv("APP__ARTIFACTS_ROOT", "/tmp/artifacts")
    monkeypatch.setenv("APP__RUNS_ROOT", "/tmp/runs")
    monkeypatch.setenv("APP__LOGS_ROOT", "/tmp/logs")
    monkeypatch.setenv("APP__THREADS", "4")
    monkeypatch.setenv("APP__TOKENIZER_SAMPLE_MAX_LINES", "500")
    monkeypatch.setenv("APP__DATA_BANK_API_URL", "http://db")
    monkeypatch.setenv("APP__DATA_BANK_API_KEY", "k")
    monkeypatch.setenv("APP__CLEANUP__ENABLED", "false")
    monkeypatch.setenv("APP__CLEANUP__VERIFY_UPLOAD", "false")
    monkeypatch.setenv("APP__CLEANUP__GRACE_PERIOD_SECONDS", "5")
    monkeypatch.setenv("APP__CLEANUP__DRY_RUN", "true")
    monkeypatch.setenv("APP__CORPUS_CACHE_CLEANUP__ENABLED", "true")
    monkeypatch.setenv("APP__CORPUS_CACHE_CLEANUP__MAX_BYTES", "123")
    monkeypatch.setenv("APP__CORPUS_CACHE_CLEANUP__MIN_FREE_BYTES", "456")
    monkeypatch.setenv("APP__CORPUS_CACHE_CLEANUP__EVICTION_POLICY", "oldest")
    monkeypatch.setenv("APP__TOKENIZER_CLEANUP__ENABLED", "true")
    monkeypatch.setenv("APP__TOKENIZER_CLEANUP__MIN_UNUSED_DAYS", "7")
    monkeypatch.setenv("SECURITY__API_KEY", "sekret")
    monkeypatch.setenv("APP_ENV", "prod")

    cfg = load_model_trainer_settings()
    assert cfg["logging"]["level"] == "DEBUG"
    assert cfg["redis"] == {"enabled": False, "url": "redis://override:6379/1"}
    assert cfg["rq"]["queue_name"] == "trainer"
    assert cfg["rq"]["job_timeout_sec"] == 100
    assert cfg["rq"]["result_ttl_sec"] == 200
    assert cfg["rq"]["failure_ttl_sec"] == 300
    assert cfg["rq"]["retry_max"] == 2
    assert cfg["rq"]["retry_intervals_sec"] == "10,20"
    assert cfg["app"]["data_root"] == "/tmp/data"
    assert cfg["app"]["artifacts_root"] == "/tmp/artifacts"
    assert cfg["app"]["runs_root"] == "/tmp/runs"
    assert cfg["app"]["logs_root"] == "/tmp/logs"
    assert cfg["app"]["threads"] == 4
    assert cfg["app"]["tokenizer_sample_max_lines"] == 500
    assert cfg["app"]["data_bank_api_url"] == "http://db"
    assert cfg["app"]["data_bank_api_key"] == "k"
    assert cfg["app"]["cleanup"]["enabled"] is False
    assert cfg["app"]["cleanup"]["verify_upload"] is False
    assert cfg["app"]["cleanup"]["grace_period_seconds"] == 5
    assert cfg["app"]["cleanup"]["dry_run"] is True
    assert cfg["app"]["corpus_cache_cleanup"]["enabled"] is True
    assert cfg["app"]["corpus_cache_cleanup"]["max_bytes"] == 123
    assert cfg["app"]["corpus_cache_cleanup"]["min_free_bytes"] == 456
    assert cfg["app"]["corpus_cache_cleanup"]["eviction_policy"] == "oldest"
    assert cfg["app"]["tokenizer_cleanup"]["enabled"] is True
    assert cfg["app"]["tokenizer_cleanup"]["min_unused_days"] == 7
    assert cfg["security"]["api_key"] == "sekret"
    assert cfg["app_env"] == "prod"

    app_env_literal: Literal["dev", "prod"] = cfg["app_env"]
    assert app_env_literal in ("dev", "prod")


def test_model_trainer_settings_log_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test all log level branches are covered."""
    _clear_env(monkeypatch)

    # Test WARNING
    monkeypatch.setenv("LOGGING__LEVEL", "WARNING")
    cfg = load_model_trainer_settings()
    assert cfg["logging"]["level"] == "WARNING"

    # Test ERROR
    monkeypatch.setenv("LOGGING__LEVEL", "ERROR")
    cfg = load_model_trainer_settings()
    assert cfg["logging"]["level"] == "ERROR"

    # Test CRITICAL
    monkeypatch.setenv("LOGGING__LEVEL", "CRITICAL")
    cfg = load_model_trainer_settings()
    assert cfg["logging"]["level"] == "CRITICAL"

    # Test invalid level falls back to INFO
    monkeypatch.setenv("LOGGING__LEVEL", "INVALID")
    cfg = load_model_trainer_settings()
    assert cfg["logging"]["level"] == "INFO"


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
