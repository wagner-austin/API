"""Tests for training orchestrator get_progress method."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_core.errors import AppError
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.progress import TrainingProgress
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator
from model_trainer.worker.progress_store import ProgressStore
from model_trainer.worker.trainer_job_store import TrainerJobStore


class _SettingsFactory(Protocol):
    def __call__(
        self,
        *,
        artifacts_root: str | None = ...,
        runs_root: str | None = ...,
        logs_root: str | None = ...,
        data_root: str | None = ...,
        data_bank_api_url: str | None = ...,
        data_bank_api_key: str | None = ...,
        threads: int | None = ...,
        redis_url: str | None = ...,
        app_env: Literal["dev", "prod"] | None = ...,
        security_api_key: str | None = ...,
    ) -> Settings: ...


def test_get_progress_with_progress_data(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    """Test get_progress returns progress when available."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    settings = settings_factory(
        artifacts_root=str(artifacts),
        data_root=str(tmp_path / "data"),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
    )

    fake = FakeRedis()
    enqueuer = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
    orchestrator = TrainingOrchestrator(
        settings=settings,
        redis_client=fake,
        enqueuer=enqueuer,
        model_registry=None,
    )

    # Save progress to Redis
    progress_store = ProgressStore(fake)
    progress: TrainingProgress = {
        "run_id": "run-with-progress",
        "phase": "training",
        "epoch": 5,
        "total_epochs": 10,
        "step": 250,
        "total_steps": 500,
        "train_loss": 0.75,
        "train_ppl": 2.1,
        "grad_norm": 0.12,
        "samples_per_sec": 56.0,
        "val_loss": 0.65,
        "val_ppl": 1.9,
        "updated_at": "2024-01-15T15:00:00",
    }
    progress_store.save(progress)

    # Get progress
    result = orchestrator.get_progress("run-with-progress")
    assert result["run_id"] == "run-with-progress"
    assert result["phase"] == "training"
    assert result["epoch"] == 5
    assert result["total_epochs"] == 10
    assert result["step"] == 250
    assert result["total_steps"] == 500
    assert result["train_loss"] == 0.75
    assert result["train_ppl"] == 2.1
    assert result["grad_norm"] == 0.12
    assert result["samples_per_sec"] == 56.0
    assert result["val_loss"] == 0.65
    assert result["val_ppl"] == 1.9
    assert result["updated_at"] == "2024-01-15T15:00:00"
    fake.assert_only_called({"set", "expire", "get"})


def test_get_progress_no_progress_but_job_exists(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    """Test get_progress returns queued state when job exists but no progress."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    settings = settings_factory(
        artifacts_root=str(artifacts),
        data_root=str(tmp_path / "data"),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
    )

    fake = FakeRedis()
    enqueuer = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
    orchestrator = TrainingOrchestrator(
        settings=settings,
        redis_client=fake,
        enqueuer=enqueuer,
        model_registry=None,
    )

    # Save job status (but no progress)
    from datetime import datetime

    job_store = TrainerJobStore(fake)
    job_store.save(
        {
            "job_id": "run-no-progress",
            "user_id": 1,
            "status": "queued",
            "progress": 0,
            "message": "queued",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "error": None,
            "artifact_file_id": None,
        }
    )

    # Get progress - should return initial state
    result = orchestrator.get_progress("run-no-progress")
    assert result["run_id"] == "run-no-progress"
    assert result["phase"] == "queued"
    assert result["epoch"] == 0
    assert result["total_epochs"] == 0
    assert result["step"] == 0
    assert result["total_steps"] == 0
    assert result["train_loss"] == 0.0
    assert result["train_ppl"] == 0.0
    assert result["grad_norm"] == 0.0
    assert result["samples_per_sec"] == 0.0
    assert result["val_loss"] is None
    assert result["val_ppl"] is None
    fake.assert_only_called({"get", "hset", "hgetall"})


def test_get_progress_run_not_found(tmp_path: Path, settings_factory: _SettingsFactory) -> None:
    """Test get_progress raises error when run not found."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    settings = settings_factory(
        artifacts_root=str(artifacts),
        data_root=str(tmp_path / "data"),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
    )

    fake = FakeRedis()
    enqueuer = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
    orchestrator = TrainingOrchestrator(
        settings=settings,
        redis_client=fake,
        enqueuer=enqueuer,
        model_registry=None,
    )

    # Get progress for non-existent run
    with pytest.raises(AppError):
        orchestrator.get_progress("run-nonexistent")
    fake.assert_only_called({"get", "hgetall"})
