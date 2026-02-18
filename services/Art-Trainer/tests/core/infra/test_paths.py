"""Tests for path utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from art_trainer.core.config.settings import Settings
from art_trainer.core.infra.paths import dataset_dir, lora_logs_path, lora_output_dir


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


def test_lora_output_dir(tmp_path: Path) -> None:
    """Test lora_output_dir returns correct path."""
    settings = _make_test_settings(tmp_path)
    result = lora_output_dir(settings, "job-123")

    expected = tmp_path / "output" / "job-123"
    assert result == expected


def test_lora_logs_path(tmp_path: Path) -> None:
    """Test lora_logs_path returns correct path."""
    settings = _make_test_settings(tmp_path)
    result = lora_logs_path(settings, "job-456")

    expected = tmp_path / "logs" / "job-456.log"
    assert result == expected


def test_dataset_dir(tmp_path: Path) -> None:
    """Test dataset_dir returns correct path."""
    settings = _make_test_settings(tmp_path)
    result = dataset_dir(settings, "job-789")

    expected = tmp_path / "data" / "datasets" / "job-789"
    assert result == expected
