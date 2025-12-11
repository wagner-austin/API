from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Final

import pytest
from platform_core.job_types import JobStatusLiteral
from platform_core.trainer_keys import artifact_file_id_key
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import (
    AppConfig,
    CleanupConfig,
    CorpusCacheCleanupConfig,
    LoggingConfig,
    RedisConfig,
    RQConfig,
    SecurityConfig,
    Settings,
    TokenizerCleanupConfig,
)
from model_trainer.core.services.storage.artifact_cleanup import (
    ArtifactCleanupService,
    CleanupError,
)
from model_trainer.worker.trainer_job_store import TrainerJobStore


def _settings_with_cleanup(
    enabled: bool,
    verify_upload: bool = True,
    dry_run: bool = False,
    grace_period_seconds: int = 0,
) -> Settings:
    cleanup_cfg: CleanupConfig = {
        "enabled": enabled,
        "verify_upload": verify_upload,
        "grace_period_seconds": grace_period_seconds,
        "dry_run": dry_run,
    }
    corpus_cache: CorpusCacheCleanupConfig = {
        "enabled": False,
        "max_bytes": 0,
        "min_free_bytes": 0,
        "eviction_policy": "lru",
    }
    tokenizer_cleanup: TokenizerCleanupConfig = {
        "enabled": False,
        "min_unused_days": 0,
    }
    app: AppConfig = {
        "data_root": "/tmp",
        "artifacts_root": "/tmp",
        "runs_root": "/tmp/runs",
        "logs_root": "/tmp/logs",
        "threads": 1,
        "tokenizer_sample_max_lines": 100,
        "data_bank_api_url": "http://localhost",
        "data_bank_api_key": "test",
        "cleanup": cleanup_cfg,
        "corpus_cache_cleanup": corpus_cache,
        "tokenizer_cleanup": tokenizer_cleanup,
    }
    logging_cfg: LoggingConfig = {"level": "INFO"}
    redis: RedisConfig = {"enabled": False, "url": ""}
    rq: RQConfig = {
        "queue_name": "test",
        "job_timeout_sec": 300,
        "result_ttl_sec": 300,
        "failure_ttl_sec": 300,
        "retry_max": 0,
        "retry_intervals_sec": "0",
    }
    security: SecurityConfig = {"api_key": "test"}
    settings: Settings = {
        "app_env": "dev",
        "logging": logging_cfg,
        "redis": redis,
        "rq": rq,
        "app": app,
        "security": security,
        "wandb": {"enabled": False, "project": "test"},
    }
    return settings


def _service(settings: Settings, redis_client: RedisStrProto) -> ArtifactCleanupService:
    return ArtifactCleanupService(settings=settings, redis_client=redis_client)


def _save_status(redis_client: FakeRedis, run_id: str, status: JobStatusLiteral) -> None:
    now = datetime.utcnow()
    TrainerJobStore(redis_client).save(
        {
            "job_id": run_id,
            "user_id": 1,
            "status": status,
            "progress": 0,
            "message": status,
            "created_at": now,
            "updated_at": now,
            "error": None,
            "artifact_file_id": None,
        }
    )


def test_cleanup_disabled_returns_not_deleted(tmp_path: Path) -> None:
    settings = _settings_with_cleanup(enabled=False)
    r = FakeRedis()
    service = _service(settings, r)

    artifact_dir = tmp_path / "run"
    artifact_dir.mkdir()
    result = service.cleanup_run_artifacts("run-1", artifact_dir)

    assert result.deleted is False
    assert result.reason == "cleanup_disabled"
    assert artifact_dir.exists()
    r.assert_only_called(set())


def test_cleanup_directory_not_found_returns_not_deleted(tmp_path: Path) -> None:
    settings = _settings_with_cleanup(enabled=True)
    r = FakeRedis()
    service = _service(settings, r)

    missing_dir = tmp_path / "missing"
    result = service.cleanup_run_artifacts("run-2", missing_dir)

    assert result.deleted is False
    assert result.reason == "directory_not_found"
    r.assert_only_called(set())


def test_cleanup_no_file_id_in_redis_skips_deletion(tmp_path: Path) -> None:
    settings = _settings_with_cleanup(enabled=True)
    r = FakeRedis()
    # status is terminal but no file_id
    _save_status(r, "run-3", "completed")
    artifact_dir = tmp_path / "run-3"
    artifact_dir.mkdir()

    service = _service(settings, r)
    result = service.cleanup_run_artifacts("run-3", artifact_dir)

    assert result.deleted is False
    assert result.reason == "upload_not_verified"
    assert artifact_dir.exists()
    r.assert_only_called({"hset", "get"})


def test_cleanup_verify_upload_disabled_skips_redis_check(tmp_path: Path) -> None:
    settings = _settings_with_cleanup(enabled=True, verify_upload=False)
    r = FakeRedis()
    _save_status(r, "run-4", "completed")
    artifact_dir = tmp_path / "run-4"
    artifact_dir.mkdir()

    service = _service(settings, r)
    result = service.cleanup_run_artifacts("run-4", artifact_dir)

    assert result.deleted is True
    assert result.reason is None
    assert not artifact_dir.exists()
    r.assert_only_called({"hset", "hgetall"})


def test_cleanup_skips_when_run_not_terminal(tmp_path: Path) -> None:
    settings = _settings_with_cleanup(enabled=True)
    r = FakeRedis()
    r.set(artifact_file_id_key("run-5"), "fid-123")
    _save_status(r, "run-5", "processing")
    artifact_dir = tmp_path / "run-5"
    artifact_dir.mkdir()

    service = _service(settings, r)
    result = service.cleanup_run_artifacts("run-5", artifact_dir)

    assert result.deleted is False
    assert result.reason == "run_not_terminal"
    assert artifact_dir.exists()
    r.assert_only_called({"set", "hset", "get", "hgetall"})


def test_cleanup_dry_run_does_not_delete(tmp_path: Path) -> None:
    settings = _settings_with_cleanup(enabled=True, dry_run=True)

    r = FakeRedis()
    r.set(artifact_file_id_key("run-6"), "fid-456")
    _save_status(r, "run-6", "completed")
    artifact_dir = tmp_path / "run-6"
    artifact_dir.mkdir()
    (artifact_dir / "a.txt").write_text("x", encoding="utf-8")

    service = _service(settings, r)
    result = service.cleanup_run_artifacts("run-6", artifact_dir)

    assert result.deleted is False
    assert result.reason == "dry_run"
    assert artifact_dir.exists()
    r.assert_only_called({"set", "hset", "get", "hgetall"})


def test_cleanup_success_deletes_directory(tmp_path: Path) -> None:
    settings = _settings_with_cleanup(enabled=True)
    r = FakeRedis()
    r.set(artifact_file_id_key("run-7"), "fid-789")
    _save_status(r, "run-7", "completed")
    artifact_dir = tmp_path / "run-7"
    artifact_dir.mkdir()
    f1 = artifact_dir / "a.txt"
    f2 = artifact_dir / "sub" / "b.txt"
    f2.parent.mkdir()
    f1.write_text("hello", encoding="utf-8")
    f2.write_text("world", encoding="utf-8")

    service = _service(settings, r)
    result = service.cleanup_run_artifacts("run-7", artifact_dir)

    assert result.deleted is True
    assert result.reason is None
    assert not artifact_dir.exists()
    assert result.files_deleted == 2
    assert result.bytes_freed >= 10
    r.assert_only_called({"set", "hset", "get", "hgetall"})


def test_cleanup_deletion_failure_raises(tmp_path: Path) -> None:
    from model_trainer.core import _test_hooks

    settings = _settings_with_cleanup(enabled=True)
    r = FakeRedis()
    r.set(artifact_file_id_key("run-8"), "fid-000")
    _save_status(r, "run-8", "completed")
    artifact_dir = tmp_path / "run-8"
    artifact_dir.mkdir()

    delete_called: Final[list[bool]] = [False]

    def _fail_rmtree(path: str | Path) -> None:
        delete_called[0] = True
        raise OSError("boom")

    _test_hooks.shutil_rmtree = _fail_rmtree
    service = _service(settings, r)

    with pytest.raises(CleanupError):
        service.cleanup_run_artifacts("run-8", artifact_dir)

    assert delete_called[0] is True
    assert artifact_dir.exists()
    r.assert_only_called({"set", "hset", "get", "hgetall"})


def test_cleanup_grace_period_delays_before_delete(
    tmp_path: Path,
) -> None:
    from model_trainer.core import _test_hooks

    sleep_called: Final[list[float]] = []

    def _fake_sleep(seconds: float) -> None:
        sleep_called.append(seconds)

    _test_hooks.time_sleep = _fake_sleep

    settings = _settings_with_cleanup(
        enabled=True,
        verify_upload=False,
        grace_period_seconds=1,
    )
    r = FakeRedis()
    _save_status(r, "run-9", "completed")
    artifact_dir = tmp_path / "run-9"
    artifact_dir.mkdir()
    (artifact_dir / "a.txt").write_text("x", encoding="utf-8")

    service = _service(settings, r)
    result = service.cleanup_run_artifacts("run-9", artifact_dir)

    assert result.deleted is True
    assert not artifact_dir.exists()
    assert sleep_called == [1.0]
    r.assert_only_called({"hset", "hgetall"})


def test_calculate_size_and_count_handle_errors(tmp_path: Path) -> None:
    settings = _settings_with_cleanup(enabled=True, verify_upload=False)
    r = FakeRedis()
    r.set(artifact_file_id_key("run-9"), "fid-999")
    _save_status(r, "run-9", "completed")
    artifact_dir = tmp_path / "run-9"
    artifact_dir.mkdir()
    (artifact_dir / "a.txt").write_text("x", encoding="utf-8")

    service = _service(settings, r)
    size = service._calculate_directory_size(artifact_dir)
    count = service._count_files(artifact_dir)

    assert size > 0
    assert count == 1
    r.assert_only_called({"set", "hset"})
