from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from platform_core.job_types import JobStatusLiteral
from platform_workers.redis import RedisStrProto

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings
from model_trainer.core.infra.paths import models_dir
from model_trainer.worker.job_utils import (
    MATERIALIZED_RUN_KEEP,
    evict_materialized_runs,
)
from model_trainer.worker.trainer_job_store import TrainerJobStore


def _make_run_dir(models_root: Path, run_id: str, *, mtime: float) -> Path:
    """Create a materialized run directory with a chosen recency.

    Args:
        models_root: The models root.
        run_id: Run the directory belongs to.
        mtime: Modification time to stamp, deciding LRU order.

    Returns:
        The directory created.
    """
    run_dir = models_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model.bin").write_bytes(b"weights")
    os.utime(run_dir, (mtime, mtime))
    return run_dir


def _record_status(redis: RedisStrProto, run_id: str, status: JobStatusLiteral) -> None:
    """Persist a job status for a run.

    Args:
        redis: Store to write into.
        run_id: Run whose status to set. Run id and job id are the same value.
        status: Status to record.
    """
    now = datetime.utcnow()
    TrainerJobStore(redis).save(
        {
            "job_id": run_id,
            "user_id": 1,
            "status": status,
            "progress": 100,
            "message": None,
            "created_at": now,
            "updated_at": now,
            "error": None,
            "artifact_file_id": None,
        }
    )


def test_evicts_only_beyond_the_keep_window(
    settings_with_paths: Settings,
    fake_redis: RedisStrProto,
) -> None:
    """The newest KEEP directories survive; older completed ones do not."""
    models_root = models_dir(settings_with_paths)
    redis = fake_redis
    total = MATERIALIZED_RUN_KEEP + 2
    for index in range(total):
        run_id = f"run-{index}"
        _make_run_dir(models_root, run_id, mtime=1_000_000 + index)
        _record_status(redis, run_id, "completed")

    evicted = evict_materialized_runs(settings_with_paths, redis)

    # mtime ascends with index, so the two oldest are run-0 and run-1.
    assert sorted(evicted) == ["run-0", "run-1"]
    assert not (models_root / "run-0").exists()
    assert (models_root / f"run-{total - 1}").exists()


def test_never_evicts_a_non_terminal_run(
    settings_with_paths: Settings,
    fake_redis: RedisStrProto,
) -> None:
    """The actively-training run writes into this same root.

    Evicting it would delete a run out from under itself, so a directory whose
    status is not completed or failed is kept however old it is.
    """
    models_root = models_dir(settings_with_paths)
    redis = fake_redis
    _make_run_dir(models_root, "training-now", mtime=1)
    _record_status(redis, "training-now", "processing")
    for index in range(MATERIALIZED_RUN_KEEP + 1):
        run_id = f"done-{index}"
        _make_run_dir(models_root, run_id, mtime=2_000_000 + index)
        _record_status(redis, run_id, "completed")

    evicted = evict_materialized_runs(settings_with_paths, redis)

    assert "training-now" not in evicted
    assert (models_root / "training-now").exists()


def test_never_evicts_a_run_with_no_recorded_status(
    settings_with_paths: Settings,
    fake_redis: RedisStrProto,
) -> None:
    """An unknown status is not evidence the run is finished."""
    models_root = models_dir(settings_with_paths)
    redis = fake_redis
    _make_run_dir(models_root, "unknown", mtime=1)
    for index in range(MATERIALIZED_RUN_KEEP + 1):
        run_id = f"done-{index}"
        _make_run_dir(models_root, run_id, mtime=2_000_000 + index)
        _record_status(redis, run_id, "completed")

    evicted = evict_materialized_runs(settings_with_paths, redis)

    assert "unknown" not in evicted
    assert (models_root / "unknown").exists()


def test_evicts_a_failed_run(
    settings_with_paths: Settings,
    fake_redis: RedisStrProto,
) -> None:
    """Failed is terminal too: its artifacts are not being written to."""
    models_root = models_dir(settings_with_paths)
    redis = fake_redis
    _make_run_dir(models_root, "broke", mtime=1)
    _record_status(redis, "broke", "failed")
    for index in range(MATERIALIZED_RUN_KEEP):
        run_id = f"done-{index}"
        _make_run_dir(models_root, run_id, mtime=2_000_000 + index)
        _record_status(redis, run_id, "completed")

    evicted = evict_materialized_runs(settings_with_paths, redis)

    assert evicted == ("broke",)
    assert not (models_root / "broke").exists()


def test_evicts_nothing_when_within_the_window(
    settings_with_paths: Settings,
    fake_redis: RedisStrProto,
) -> None:
    """A cache under its bound is left alone."""
    models_root = models_dir(settings_with_paths)
    redis = fake_redis
    for index in range(MATERIALIZED_RUN_KEEP):
        run_id = f"done-{index}"
        _make_run_dir(models_root, run_id, mtime=1_000 + index)
        _record_status(redis, run_id, "completed")

    assert evict_materialized_runs(settings_with_paths, redis) == ()


def test_evicts_nothing_when_the_models_root_is_absent(
    settings_with_paths: Settings,
    fake_redis: RedisStrProto,
) -> None:
    """A worker that has never materialized anything must not fail here."""
    redis = fake_redis

    assert evict_materialized_runs(settings_with_paths, redis) == ()


def test_ignores_stray_files_beside_the_run_directories(
    settings_with_paths: Settings,
    fake_redis: RedisStrProto,
) -> None:
    """Only directories are cache entries; a stray file is not a run."""
    models_root = models_dir(settings_with_paths)
    models_root.mkdir(parents=True)
    (models_root / "README.txt").write_text("not a run", encoding="utf-8")
    redis = fake_redis
    for index in range(MATERIALIZED_RUN_KEEP + 1):
        run_id = f"done-{index}"
        _make_run_dir(models_root, run_id, mtime=1_000 + index)
        _record_status(redis, run_id, "completed")

    evicted = evict_materialized_runs(settings_with_paths, redis)

    assert evicted == ("done-0",)
    assert (models_root / "README.txt").exists()


def test_use_refreshes_recency_so_the_busy_run_is_not_evicted(
    settings_with_paths: Settings,
    fake_redis: RedisStrProto,
) -> None:
    """Recency must track USE, not the age of the original download.

    Without the touch, the run being chatted with every minute is the oldest
    directory on disk and is exactly the one thrown away.
    """
    models_root = models_dir(settings_with_paths)
    redis = fake_redis
    busy = _make_run_dir(models_root, "busy", mtime=1)
    _record_status(redis, "busy", "completed")
    for index in range(MATERIALIZED_RUN_KEEP):
        run_id = f"done-{index}"
        _make_run_dir(models_root, run_id, mtime=2_000_000 + index)
        _record_status(redis, run_id, "completed")

    _test_hooks.os_utime(busy)
    evicted = evict_materialized_runs(settings_with_paths, redis)

    assert "busy" not in evicted
    assert busy.exists()
