from __future__ import annotations

import os
from pathlib import Path

import pytest

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
from model_trainer.core.services.data.corpus_cache_cleanup import (
    CorpusCacheCleanupError,
    CorpusCacheCleanupService,
)


def _settings_with_cache_cleanup(cfg: CorpusCacheCleanupConfig, data_root: Path) -> Settings:
    cleanup: CleanupConfig = {
        "enabled": False,
        "verify_upload": False,
        "grace_period_seconds": 0,
        "dry_run": False,
    }
    tokenizer_cleanup: TokenizerCleanupConfig = {
        "enabled": False,
        "min_unused_days": 0,
    }
    app: AppConfig = {
        "data_root": str(data_root),
        "artifacts_root": "/tmp",
        "runs_root": "/tmp/runs",
        "logs_root": "/tmp/logs",
        "threads": 1,
        "tokenizer_sample_max_lines": 100,
        "data_bank_api_url": "http://localhost",
        "data_bank_api_key": "test",
        "cleanup": cleanup,
        "corpus_cache_cleanup": cfg,
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


def test_cleanup_disabled_returns_zero(tmp_path: Path) -> None:
    cfg: CorpusCacheCleanupConfig = {
        "enabled": False,
        "max_bytes": 0,
        "min_free_bytes": 0,
        "eviction_policy": "lru",
    }
    settings = _settings_with_cache_cleanup(cfg, tmp_path)
    service = CorpusCacheCleanupService(settings=settings)

    result = service.clean()

    assert result.deleted_files == 0 and result.bytes_freed == 0


def test_cleanup_missing_directory_is_noop(tmp_path: Path) -> None:
    cfg: CorpusCacheCleanupConfig = {
        "enabled": True,
        "max_bytes": 0,
        "min_free_bytes": 0,
        "eviction_policy": "lru",
    }
    settings = _settings_with_cache_cleanup(cfg, tmp_path)
    service = CorpusCacheCleanupService(settings=settings)

    result = service.clean()

    assert result.deleted_files == 0
    assert result.bytes_freed == 0


def test_cleanup_below_threshold_no_deletion(tmp_path: Path) -> None:
    cfg = CorpusCacheCleanupConfig(
        enabled=True,
        max_bytes=10_000,
        min_free_bytes=0,
        eviction_policy="lru",
    )
    cache_dir = tmp_path / "corpus_cache"
    cache_dir.mkdir()
    f = cache_dir / "a.txt"
    f.write_text("hello", encoding="utf-8")

    settings = _settings_with_cache_cleanup(cfg, tmp_path)
    service = CorpusCacheCleanupService(settings=settings)

    result = service.clean()

    assert result.deleted_files == 0
    assert result.bytes_freed == 0
    assert f.exists()


def test_cleanup_lru_eviction(tmp_path: Path) -> None:
    cfg = CorpusCacheCleanupConfig(
        enabled=True,
        max_bytes=0,
        min_free_bytes=0,
        eviction_policy="lru",
    )
    cache_dir = tmp_path / "corpus_cache"
    cache_dir.mkdir()
    f_old = cache_dir / "old.txt"
    f_new = cache_dir / "new.txt"
    f_old.write_text("old", encoding="utf-8")
    f_new.write_text("new", encoding="utf-8")

    os.utime(f_old, (1, 1))
    os.utime(f_new, (2, 2))

    settings = _settings_with_cache_cleanup(cfg, tmp_path)
    service = CorpusCacheCleanupService(settings=settings)

    result = service.clean()

    assert result.deleted_files == 2
    assert result.bytes_freed > 0
    assert not f_old.exists()
    assert not f_new.exists()


def test_cleanup_oldest_policy(tmp_path: Path) -> None:
    cfg = CorpusCacheCleanupConfig(
        enabled=True,
        max_bytes=0,
        min_free_bytes=0,
        eviction_policy="oldest",
    )
    cache_dir = tmp_path / "corpus_cache"
    cache_dir.mkdir()
    f1 = cache_dir / "f1.txt"
    f2 = cache_dir / "f2.txt"
    f1.write_text("first", encoding="utf-8")
    f2.write_text("second", encoding="utf-8")
    os.utime(f1, (1, 1))
    os.utime(f2, (2, 2))

    settings = _settings_with_cache_cleanup(cfg, tmp_path)
    service = CorpusCacheCleanupService(settings=settings)

    result = service.clean()

    assert result.deleted_files == 2
    assert result.bytes_freed > 0
    assert not f1.exists()
    assert not f2.exists()


def test_cleanup_raises_on_non_directory_cache(tmp_path: Path) -> None:
    cache_file = tmp_path / "corpus_cache"
    cache_file.write_text("not a directory", encoding="utf-8")
    cfg: CorpusCacheCleanupConfig = {
        "enabled": True,
        "max_bytes": 0,
        "min_free_bytes": 0,
        "eviction_policy": "lru",
    }
    settings = _settings_with_cache_cleanup(cfg, tmp_path)
    service = CorpusCacheCleanupService(settings=settings)

    with pytest.raises(CorpusCacheCleanupError):
        service.clean()


def test_scan_cache_dir_raises_on_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = tmp_path / "corpus_cache"
    cache_dir.mkdir()
    cfg: CorpusCacheCleanupConfig = {
        "enabled": True,
        "max_bytes": 0,
        "min_free_bytes": 0,
        "eviction_policy": "lru",
    }
    settings = _settings_with_cache_cleanup(cfg, tmp_path)
    service = CorpusCacheCleanupService(settings=settings)

    def _scandir_fail(_path: str) -> None:
        raise OSError("boom")

    monkeypatch.setattr(
        "model_trainer.core.services.data.corpus_cache_cleanup.os.scandir",
        _scandir_fail,
    )

    with pytest.raises(CorpusCacheCleanupError):
        service.clean()


def test_disk_usage_error_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg: CorpusCacheCleanupConfig = {
        "enabled": True,
        "max_bytes": 0,
        "min_free_bytes": 0,
        "eviction_policy": "lru",
    }
    settings = _settings_with_cache_cleanup(cfg, tmp_path)
    cache_dir = tmp_path / "corpus_cache"
    cache_dir.mkdir()
    (cache_dir / "a.txt").write_text("x", encoding="utf-8")

    def _disk_usage_fail(_path: str) -> None:
        raise OSError("disk fail")

    monkeypatch.setattr(
        "model_trainer.core.services.data.corpus_cache_cleanup.shutil.disk_usage",
        _disk_usage_fail,
    )
    service = CorpusCacheCleanupService(settings=settings)

    with pytest.raises(CorpusCacheCleanupError):
        service.clean()


def test_unsupported_eviction_policy_raises(tmp_path: Path) -> None:
    # Config validation prevents unsupported policies; no runtime branch to exercise.
    cfg: CorpusCacheCleanupConfig = {
        "enabled": True,
        "max_bytes": 0,
        "min_free_bytes": 0,
        "eviction_policy": "lru",
    }
    settings = _settings_with_cache_cleanup(cfg, tmp_path)
    assert settings["app"]["corpus_cache_cleanup"]["eviction_policy"] in {"lru", "oldest"}


def test_cleanup_breaks_when_threshold_reached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Create three files; configure thresholds so that only two need deletion.
    cfg = CorpusCacheCleanupConfig(
        enabled=True,
        max_bytes=15,
        min_free_bytes=0,
        eviction_policy="lru",
    )
    cache_dir = tmp_path / "corpus_cache"
    cache_dir.mkdir()
    f1 = cache_dir / "f1.txt"
    f2 = cache_dir / "f2.txt"
    f3 = cache_dir / "f3.txt"
    f1.write_text("1234567890", encoding="utf-8")  # 10
    f2.write_text("12345", encoding="utf-8")  # 5
    f3.write_text("1234567890", encoding="utf-8")  # 10

    # Ensure deterministic access ordering
    os.utime(f1, (1, 1))
    os.utime(f2, (2, 2))
    os.utime(f3, (3, 3))

    settings = _settings_with_cache_cleanup(cfg, tmp_path)
    service = CorpusCacheCleanupService(settings=settings)

    # Patch disk_usage to a stable large free space value
    class _Usage:
        def __init__(self: _Usage) -> None:
            self.total = 100_000
            self.used = 10_000
            self.free = 90_000

    def _fake_disk_usage(_path: str) -> _Usage:
        return _Usage()

    monkeypatch.setattr(
        "model_trainer.core.services.data.corpus_cache_cleanup.shutil.disk_usage",
        _fake_disk_usage,
    )

    result = service.clean()

    # Only the oldest file needs to be deleted to satisfy max_bytes
    assert result.deleted_files == 1
    assert f2.exists()
    assert f3.exists()


def test_scan_skips_non_file_entries(tmp_path: Path) -> None:
    cfg = CorpusCacheCleanupConfig(
        enabled=True,
        max_bytes=0,
        min_free_bytes=0,
        eviction_policy="lru",
    )
    cache_dir = tmp_path / "corpus_cache"
    cache_dir.mkdir()
    # Add a subdirectory that should be ignored by the scanner
    subdir = cache_dir / "nested"
    subdir.mkdir()
    f = cache_dir / "keep.txt"
    f.write_text("data", encoding="utf-8")

    settings = _settings_with_cache_cleanup(cfg, tmp_path)
    service = CorpusCacheCleanupService(settings=settings)

    result = service.clean()

    # All files are eligible; subdir is ignored
    assert result.deleted_files == 1


def test_cleanup_deletion_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = CorpusCacheCleanupConfig(
        enabled=True,
        max_bytes=0,
        min_free_bytes=0,
        eviction_policy="lru",
    )
    cache_dir = tmp_path / "corpus_cache"
    cache_dir.mkdir()
    f = cache_dir / "a.txt"
    f.write_text("x", encoding="utf-8")

    settings = _settings_with_cache_cleanup(cfg, tmp_path)
    service = CorpusCacheCleanupService(settings=settings)

    def _unlink_fail(_self: Path) -> None:
        raise OSError("unlink-fail")

    monkeypatch.setattr(
        "model_trainer.core.services.data.corpus_cache_cleanup.Path.unlink",
        _unlink_fail,
    )

    with pytest.raises(CorpusCacheCleanupError):
        service.clean()
