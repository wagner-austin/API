"""Tests for LoRA training routes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str, require_int, require_str

from art_trainer.api.main import create_app
from art_trainer.api.schemas.lora import LoraTrainRequest
from art_trainer.core import _test_hooks
from art_trainer.core.config.settings import Settings


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


def _make_train_request(
    *,
    user_id: int = 123,
    base_model: Literal["sd15", "sdxl", "flux"] = "sd15",
    training_type: Literal["style", "character", "concept"] = "style",
    dataset_file_id: str = "file-123",
    steps: int = 1000,
    learning_rate: float = 0.0001,
    network_rank: int = 16,
    network_alpha: int = 16,
    resolution: int = 512,
    batch_size: int = 1,
    seed: int = 42,
    caption_extension: str = ".txt",
    shuffle_caption: bool = True,
    keep_tokens: int = 1,
) -> LoraTrainRequest:
    """Create test train request.

    Args:
        user_id: User ID.
        base_model: Base model type.
        training_type: Training type.
        dataset_file_id: Dataset file ID.
        steps: Number of steps.
        learning_rate: Learning rate.
        network_rank: Network rank.
        network_alpha: Network alpha.
        resolution: Resolution.
        batch_size: Batch size.
        seed: Random seed.
        caption_extension: Caption extension.
        shuffle_caption: Shuffle caption.
        keep_tokens: Keep tokens.

    Returns:
        LoraTrainRequest.
    """
    return {
        "user_id": user_id,
        "base_model": base_model,
        "training_type": training_type,
        "dataset_file_id": dataset_file_id,
        "steps": steps,
        "learning_rate": learning_rate,
        "network_rank": network_rank,
        "network_alpha": network_alpha,
        "resolution": resolution,
        "batch_size": batch_size,
        "seed": seed,
        "caption_extension": caption_extension,
        "shuffle_caption": shuffle_caption,
        "keep_tokens": keep_tokens,
    }


def test_lora_train_requires_api_key(tmp_path: Path) -> None:
    """Test POST /lora/train requires API key."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    app = create_app(settings)
    client = TestClient(app)

    request_body = _make_train_request()
    response = client.post("/lora/train", json=request_body)

    # Should fail without API key
    assert response.status_code == 401


def test_lora_train_with_api_key(tmp_path: Path) -> None:
    """Test POST /lora/train succeeds with API key."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    app = create_app(settings)
    client = TestClient(app)

    request_body = _make_train_request()
    response = client.post(
        "/lora/train",
        json=request_body,
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    data = load_json_str(response.text)
    if not isinstance(data, dict):
        raise AssertionError("Response body must be a JSON object")
    job_id = require_str(data, "job_id")
    # UUID format has 4 dashes
    assert job_id.count("-") == 4


def test_lora_get_status(tmp_path: Path) -> None:
    """Test GET /lora/{job_id} returns status."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    app = create_app(settings)
    client = TestClient(app)

    # First create a job
    request_body = _make_train_request(
        user_id=456,
        base_model="sdxl",
        training_type="character",
        dataset_file_id="file-456",
        steps=2000,
        network_rank=32,
        resolution=1024,
        seed=12345,
        shuffle_caption=False,
        keep_tokens=2,
    )
    train_response = client.post(
        "/lora/train",
        json=request_body,
        headers={"X-API-Key": "test-api-key"},
    )
    train_data = load_json_str(train_response.text)
    if not isinstance(train_data, dict):
        raise AssertionError("Response body must be a JSON object")
    job_id = require_str(train_data, "job_id")

    # Get status
    status_response = client.get(
        f"/lora/{job_id}",
        headers={"X-API-Key": "test-api-key"},
    )

    assert status_response.status_code == 200
    data = load_json_str(status_response.text)
    if not isinstance(data, dict):
        raise AssertionError("Response body must be a JSON object")
    response_job_id = require_str(data, "job_id")
    status = require_str(data, "status")
    assert response_job_id == job_id
    assert status == "queued"


def test_lora_get_progress(tmp_path: Path) -> None:
    """Test GET /lora/{job_id}/progress returns progress."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    app = create_app(settings)
    client = TestClient(app)

    # First create a job
    request_body = _make_train_request(
        user_id=789,
        base_model="flux",
        training_type="concept",
        dataset_file_id="file-789",
        steps=3000,
        learning_rate=0.00005,
        network_rank=64,
        network_alpha=32,
        resolution=1024,
        seed=99999,
        caption_extension=".caption",
        keep_tokens=0,
    )
    train_response = client.post(
        "/lora/train",
        json=request_body,
        headers={"X-API-Key": "test-api-key"},
    )
    train_data = load_json_str(train_response.text)
    if not isinstance(train_data, dict):
        raise AssertionError("Response body must be a JSON object")
    job_id = require_str(train_data, "job_id")

    # Get progress
    progress_response = client.get(
        f"/lora/{job_id}/progress",
        headers={"X-API-Key": "test-api-key"},
    )

    assert progress_response.status_code == 200
    data = load_json_str(progress_response.text)
    if not isinstance(data, dict):
        raise AssertionError("Response body must be a JSON object")
    response_job_id = require_str(data, "job_id")
    phase = require_str(data, "phase")
    total_steps = require_int(data, "total_steps")
    assert response_job_id == job_id
    assert phase == "queued"
    assert total_steps == 3000


def test_lora_cancel(tmp_path: Path) -> None:
    """Test POST /lora/{job_id}/cancel cancels job."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    app = create_app(settings)
    client = TestClient(app)

    # First create a job
    request_body = _make_train_request(
        user_id=111,
        dataset_file_id="file-cancel",
        steps=500,
        network_rank=8,
        network_alpha=8,
        batch_size=2,
        seed=11111,
    )
    train_response = client.post(
        "/lora/train",
        json=request_body,
        headers={"X-API-Key": "test-api-key"},
    )
    train_data = load_json_str(train_response.text)
    if not isinstance(train_data, dict):
        raise AssertionError("Response body must be a JSON object")
    job_id = require_str(train_data, "job_id")

    # Cancel the job
    cancel_response = client.post(
        f"/lora/{job_id}/cancel",
        headers={"X-API-Key": "test-api-key"},
    )

    assert cancel_response.status_code == 200
    data = load_json_str(cancel_response.text)
    if not isinstance(data, dict):
        raise AssertionError("Response body must be a JSON object")
    status = require_str(data, "status")
    assert status == "cancellation-requested"


def test_lora_train_non_object_body(tmp_path: Path) -> None:
    """Test POST /lora/train with non-object JSON body returns error."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    app = create_app(settings)
    client = TestClient(app)

    # Send array instead of object
    response = client.post(
        "/lora/train",
        content="[1, 2, 3]",
        headers={
            "X-API-Key": "test-api-key",
            "Content-Type": "application/json",
        },
    )

    # Should return 400 or 422 for invalid request
    assert response.status_code in [400, 422]
