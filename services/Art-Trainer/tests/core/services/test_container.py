"""Tests for ServiceContainer."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from art_trainer.core.config.settings import Settings
from art_trainer.core.services.container import ServiceContainer


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


def test_service_container_from_settings(tmp_path: Path) -> None:
    """Test ServiceContainer.from_settings creates container."""
    settings = _make_test_settings(tmp_path)
    container = ServiceContainer.from_settings(settings)

    assert container.settings == settings
    # Verify redis connection string is configured
    redis_url = container.settings["redis"]["url"]
    assert redis_url == "redis://localhost:6379/0"
    # Verify backend registry has kohya
    assert "kohya" in container.backend_registry.available_backends()
    # Verify orchestrator can get status for non-existent job
    status = container.lora_orchestrator.get_status("non-existent")
    assert status["status"] == "failed"


def test_service_container_backend_registry(tmp_path: Path) -> None:
    """Test ServiceContainer has kohya backend in registry."""
    settings = _make_test_settings(tmp_path)
    container = ServiceContainer.from_settings(settings)

    backends = container.backend_registry.available_backends()
    assert backends == ["kohya"]


def test_service_container_kohya_backend(tmp_path: Path) -> None:
    """Test ServiceContainer can create kohya backend."""
    settings = _make_test_settings(tmp_path)
    container = ServiceContainer.from_settings(settings)

    backend = container.backend_registry.get("kohya")
    if backend is None:
        raise AssertionError("Backend 'kohya' not found in registry")
    backend_name = backend.name()
    assert backend_name == "kohya_ss"
    is_available = backend.is_available()
    assert is_available is True


def test_backend_registry_get_unknown_returns_none(tmp_path: Path) -> None:
    """Test BackendRegistry.get returns None for unknown backend."""
    settings = _make_test_settings(tmp_path)
    container = ServiceContainer.from_settings(settings)

    backend = container.backend_registry.get("unknown_backend")
    assert backend is None
