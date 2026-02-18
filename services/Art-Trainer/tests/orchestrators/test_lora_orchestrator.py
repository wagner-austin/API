"""Tests for LoRA orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from platform_workers.testing import FakeRedis

from art_trainer.api.schemas.lora import LoraTrainRequest
from art_trainer.core.config.settings import Settings
from art_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from art_trainer.core.services.registries import BackendRegistry
from art_trainer.core.services.training.backend_factory import create_kohya_backend
from art_trainer.orchestrators.lora_orchestrator import LoraOrchestrator


def _make_test_settings(tmp_path: Path) -> Settings:
    """Create test settings.

    Args:
        tmp_path: Temporary directory path.

    Returns:
        Test Settings.
    """
    kohya_path = tmp_path / "kohya_ss"
    kohya_path.mkdir(parents=True)
    (kohya_path / "train_network.py").touch()

    app_env: Literal["dev", "prod"] = "dev"

    return {
        "app_env": app_env,
        "logging": {"level": "INFO"},
        "redis": {"enabled": True, "url": "redis://localhost:6379/0"},
        "rq": {
            "queue_name": "art-trainer",
            "job_timeout_sec": 86400,
            "result_ttl_sec": 86400,
            "failure_ttl_sec": 604800,
            "retry_max": 1,
            "retry_intervals_sec": "300",
        },
        "app": {
            "data_root": str(tmp_path / "data"),
            "output_root": str(tmp_path / "output"),
            "logs_root": str(tmp_path / "logs"),
            "data_bank_api_url": "http://localhost:8000",
            "data_bank_api_key": "test-key",
            "kohya_ss_path": str(kohya_path),
            "comfyui_lora_path": str(tmp_path / "comfyui" / "models" / "loras"),
            "blip_model_name": "Salesforce/blip-image-captioning-large",
            "caption_trigger_word": "sks person",
            "gemini_api_key": "",
            "openai_api_key": "",
        },
        "security": {"api_key": "test-api-key"},
    }


def _make_orchestrator(settings: Settings, redis: FakeRedis) -> LoraOrchestrator:
    """Create test orchestrator.

    Args:
        settings: Test settings.
        redis: Fake Redis client.

    Returns:
        Configured LoraOrchestrator.
    """
    rq_settings = RQSettings(
        job_timeout_sec=86400,
        result_ttl_sec=86400,
        failure_ttl_sec=604800,
        retry_max=1,
        retry_intervals=[300],
    )
    enqueuer = RQEnqueuer(redis_url=settings["redis"]["url"], settings=rq_settings)
    backends = {"kohya": create_kohya_backend}
    backend_registry = BackendRegistry(backends, settings)

    return LoraOrchestrator(
        settings=settings,
        redis_client=redis,
        enqueuer=enqueuer,
        backend_registry=backend_registry,
    )


def test_enqueue_training(tmp_path: Path, fake_redis: FakeRedis) -> None:
    """Test enqueue_training creates job and returns job_id."""
    settings = _make_test_settings(tmp_path)
    orchestrator = _make_orchestrator(settings, fake_redis)

    request: LoraTrainRequest = {
        "user_id": 123,
        "base_model": "sd15",
        "training_type": "style",
        "dataset_file_id": "file-abc-123",
        "steps": 1000,
        "learning_rate": 0.0001,
        "network_rank": 16,
        "network_alpha": 16,
        "resolution": 512,
        "batch_size": 1,
        "seed": 42,
        "caption_extension": ".txt",
        "shuffle_caption": True,
        "keep_tokens": 1,
    }

    response = orchestrator.enqueue_training(request)

    # job_id should be a UUID-like string
    assert response["job_id"].count("-") == 4  # UUID format has 4 dashes
    fake_redis.assert_only_called({"get", "set"})


def test_get_status_queued(tmp_path: Path, fake_redis: FakeRedis) -> None:
    """Test get_status returns queued for new job."""
    settings = _make_test_settings(tmp_path)
    orchestrator = _make_orchestrator(settings, fake_redis)

    request: LoraTrainRequest = {
        "user_id": 123,
        "base_model": "sdxl",
        "training_type": "character",
        "dataset_file_id": "file-xyz-789",
        "steps": 2000,
        "learning_rate": 0.0001,
        "network_rank": 32,
        "network_alpha": 16,
        "resolution": 1024,
        "batch_size": 1,
        "seed": 12345,
        "caption_extension": ".txt",
        "shuffle_caption": False,
        "keep_tokens": 2,
    }

    enqueue_response = orchestrator.enqueue_training(request)
    status_response = orchestrator.get_status(enqueue_response["job_id"])

    assert status_response["job_id"] == enqueue_response["job_id"]
    assert status_response["status"] == "queued"
    fake_redis.assert_only_called({"get", "set"})


def test_get_status_not_found(tmp_path: Path, fake_redis: FakeRedis) -> None:
    """Test get_status returns failed for unknown job."""
    settings = _make_test_settings(tmp_path)
    orchestrator = _make_orchestrator(settings, fake_redis)

    status_response = orchestrator.get_status("non-existent-job-id")

    assert status_response["job_id"] == "non-existent-job-id"
    assert status_response["status"] == "failed"
    assert status_response["message"] == "Job not found"
    fake_redis.assert_only_called({"get", "set"})


def test_get_progress(tmp_path: Path, fake_redis: FakeRedis) -> None:
    """Test get_progress returns progress for job."""
    settings = _make_test_settings(tmp_path)
    orchestrator = _make_orchestrator(settings, fake_redis)

    request: LoraTrainRequest = {
        "user_id": 456,
        "base_model": "flux",
        "training_type": "concept",
        "dataset_file_id": "file-progress-test",
        "steps": 3000,
        "learning_rate": 0.00005,
        "network_rank": 64,
        "network_alpha": 32,
        "resolution": 1024,
        "batch_size": 1,
        "seed": 99999,
        "caption_extension": ".caption",
        "shuffle_caption": True,
        "keep_tokens": 0,
    }

    enqueue_response = orchestrator.enqueue_training(request)
    progress = orchestrator.get_progress(enqueue_response["job_id"])

    assert progress["job_id"] == enqueue_response["job_id"]
    assert progress["phase"] == "queued"
    assert progress["total_steps"] == 3000
    fake_redis.assert_only_called({"get", "set"})


def test_cancel_job(tmp_path: Path, fake_redis: FakeRedis) -> None:
    """Test cancel_job sets cancellation flag."""
    settings = _make_test_settings(tmp_path)
    orchestrator = _make_orchestrator(settings, fake_redis)

    request: LoraTrainRequest = {
        "user_id": 789,
        "base_model": "sd15",
        "training_type": "style",
        "dataset_file_id": "file-cancel-test",
        "steps": 500,
        "learning_rate": 0.0001,
        "network_rank": 8,
        "network_alpha": 8,
        "resolution": 512,
        "batch_size": 2,
        "seed": 11111,
        "caption_extension": ".txt",
        "shuffle_caption": True,
        "keep_tokens": 1,
    }

    enqueue_response = orchestrator.enqueue_training(request)
    result = orchestrator.cancel_job(enqueue_response["job_id"])

    assert result is True
    fake_redis.assert_only_called({"get", "set"})


def test_get_status_running(tmp_path: Path, fake_redis: FakeRedis) -> None:
    """Test get_status returns running status when set."""
    settings = _make_test_settings(tmp_path)
    orchestrator = _make_orchestrator(settings, fake_redis)

    # Set running status directly
    fake_redis.set("art:job:status:running-job-id", "running")

    status_response = orchestrator.get_status("running-job-id")

    assert status_response["job_id"] == "running-job-id"
    assert status_response["status"] == "running"
    fake_redis.assert_only_called({"get", "set"})


def test_get_status_completed(tmp_path: Path, fake_redis: FakeRedis) -> None:
    """Test get_status returns completed status when set."""
    settings = _make_test_settings(tmp_path)
    orchestrator = _make_orchestrator(settings, fake_redis)

    # Set completed status directly
    fake_redis.set("art:job:status:completed-job-id", "completed")

    status_response = orchestrator.get_status("completed-job-id")

    assert status_response["job_id"] == "completed-job-id"
    assert status_response["status"] == "completed"
    fake_redis.assert_only_called({"get", "set"})


def test_get_status_cancelled(tmp_path: Path, fake_redis: FakeRedis) -> None:
    """Test get_status returns cancelled status when set."""
    settings = _make_test_settings(tmp_path)
    orchestrator = _make_orchestrator(settings, fake_redis)

    # Set cancelled status directly
    fake_redis.set("art:job:status:cancelled-job-id", "cancelled")

    status_response = orchestrator.get_status("cancelled-job-id")

    assert status_response["job_id"] == "cancelled-job-id"
    assert status_response["status"] == "cancelled"
    fake_redis.assert_only_called({"get", "set"})


def test_get_status_unknown_defaults_to_failed(tmp_path: Path, fake_redis: FakeRedis) -> None:
    """Test get_status returns failed for unknown status values."""
    settings = _make_test_settings(tmp_path)
    orchestrator = _make_orchestrator(settings, fake_redis)

    # Set unknown status directly
    fake_redis.set("art:job:status:unknown-status-job", "some_random_status")

    status_response = orchestrator.get_status("unknown-status-job")

    assert status_response["job_id"] == "unknown-status-job"
    assert status_response["status"] == "failed"
    fake_redis.assert_only_called({"get", "set"})


def test_get_progress_not_found(tmp_path: Path, fake_redis: FakeRedis) -> None:
    """Test get_progress returns default progress for unknown job."""
    settings = _make_test_settings(tmp_path)
    orchestrator = _make_orchestrator(settings, fake_redis)

    progress = orchestrator.get_progress("non-existent-job-id")

    assert progress["job_id"] == "non-existent-job-id"
    assert progress["phase"] == "failed"
    assert progress["step"] == 0
    assert progress["total_steps"] == 0
    fake_redis.assert_only_called({"get", "set"})


def test_get_progress_invalid_json(tmp_path: Path, fake_redis: FakeRedis) -> None:
    """Test get_progress returns default progress for invalid JSON."""
    settings = _make_test_settings(tmp_path)
    orchestrator = _make_orchestrator(settings, fake_redis)

    # Set invalid JSON (array instead of object)
    fake_redis.set("art:job:progress:invalid-json-job", "[1, 2, 3]")

    progress = orchestrator.get_progress("invalid-json-job")

    assert progress["job_id"] == "invalid-json-job"
    assert progress["phase"] == "failed"
    assert progress["step"] == 0
    assert progress["total_steps"] == 0
    fake_redis.assert_only_called({"get", "set"})


def test_get_status_completed_with_result(tmp_path: Path, fake_redis: FakeRedis) -> None:
    """Test get_status returns lora_file_id when completed with result."""
    settings = _make_test_settings(tmp_path)
    orchestrator = _make_orchestrator(settings, fake_redis)

    job_id = "completed-with-result-job"

    # Set completed status
    fake_redis.set(f"art:job:status:{job_id}", "completed")

    # Set result with lora_file_id and lora_name
    result_json = (
        '{"job_id": "completed-with-result-job", '
        '"lora_file_id": "file-lora-123", '
        '"lora_name": "my_trained_lora"}'
    )
    fake_redis.set(f"art:job:result:{job_id}", result_json)

    status_response = orchestrator.get_status(job_id)

    assert status_response["job_id"] == job_id
    assert status_response["status"] == "completed"
    assert status_response["lora_file_id"] == "file-lora-123"
    assert status_response["lora_name"] == "my_trained_lora"
    fake_redis.assert_only_called({"get", "set"})


def test_get_status_completed_without_result(tmp_path: Path, fake_redis: FakeRedis) -> None:
    """Test get_status returns None for lora_file_id when no result stored."""
    settings = _make_test_settings(tmp_path)
    orchestrator = _make_orchestrator(settings, fake_redis)

    job_id = "completed-no-result-job"

    # Set completed status but no result
    fake_redis.set(f"art:job:status:{job_id}", "completed")

    status_response = orchestrator.get_status(job_id)

    assert status_response["job_id"] == job_id
    assert status_response["status"] == "completed"
    assert status_response["lora_file_id"] is None
    assert status_response["lora_name"] is None
    fake_redis.assert_only_called({"get", "set"})


def test_get_status_completed_invalid_result_json(tmp_path: Path, fake_redis: FakeRedis) -> None:
    """Test get_status handles invalid result JSON gracefully."""
    settings = _make_test_settings(tmp_path)
    orchestrator = _make_orchestrator(settings, fake_redis)

    job_id = "completed-invalid-result-job"

    # Set completed status
    fake_redis.set(f"art:job:status:{job_id}", "completed")

    # Set invalid result JSON (array instead of object)
    fake_redis.set(f"art:job:result:{job_id}", '["not", "an", "object"]')

    status_response = orchestrator.get_status(job_id)

    assert status_response["job_id"] == job_id
    assert status_response["status"] == "completed"
    assert status_response["lora_file_id"] is None
    assert status_response["lora_name"] is None
    fake_redis.assert_only_called({"get", "set"})


def test_get_progress_completed_without_result(tmp_path: Path, fake_redis: FakeRedis) -> None:
    """Test get_progress returns None for lora_file_id when no result stored."""
    settings = _make_test_settings(tmp_path)
    orchestrator = _make_orchestrator(settings, fake_redis)

    job_id = "progress-completed-no-result"

    # Set progress to completed phase but no result stored
    progress_json = (
        '{"job_id": "progress-completed-no-result", '
        '"phase": "completed", '
        '"step": 500, '
        '"total_steps": 500, '
        '"loss": 0.03, '
        '"learning_rate": 0.0001, '
        '"updated_at": "2026-02-16T10:00:00Z"}'
    )
    fake_redis.set(f"art:job:progress:{job_id}", progress_json)

    # No result stored

    progress = orchestrator.get_progress(job_id)

    assert progress["job_id"] == job_id
    assert progress["phase"] == "completed"
    assert progress["step"] == 500
    assert progress["lora_file_id"] is None
    assert progress["lora_name"] is None
    fake_redis.assert_only_called({"get", "set"})


def test_get_progress_completed_with_result(tmp_path: Path, fake_redis: FakeRedis) -> None:
    """Test get_progress returns lora_file_id when completed with result."""
    settings = _make_test_settings(tmp_path)
    orchestrator = _make_orchestrator(settings, fake_redis)

    job_id = "progress-completed-with-result"

    # Set progress to completed phase (include all required fields)
    progress_json = (
        '{"job_id": "progress-completed-with-result", '
        '"phase": "completed", '
        '"step": 1000, '
        '"total_steps": 1000, '
        '"loss": 0.05, '
        '"learning_rate": 0.0001, '
        '"updated_at": "2026-02-16T12:00:00Z"}'
    )
    fake_redis.set(f"art:job:progress:{job_id}", progress_json)

    # Set result with lora_file_id and lora_name
    result_json = (
        '{"job_id": "progress-completed-with-result", '
        '"lora_file_id": "file-progress-lora-456", '
        '"lora_name": "progress_trained_lora"}'
    )
    fake_redis.set(f"art:job:result:{job_id}", result_json)

    progress = orchestrator.get_progress(job_id)

    assert progress["job_id"] == job_id
    assert progress["phase"] == "completed"
    assert progress["step"] == 1000
    assert progress["total_steps"] == 1000
    assert progress["lora_file_id"] == "file-progress-lora-456"
    assert progress["lora_name"] == "progress_trained_lora"
    fake_redis.assert_only_called({"get", "set"})
