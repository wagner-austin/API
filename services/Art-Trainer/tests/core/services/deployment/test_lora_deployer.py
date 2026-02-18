"""Tests for LoRA deployment service."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from art_trainer.core.config.settings import Settings
from art_trainer.core.services.deployment import _test_hooks
from art_trainer.core.services.deployment.lora_deployer import deploy_lora


class FakeFileCopier:
    """Fake file copier for tests."""

    calls: list[tuple[Path, Path]]

    def __init__(self) -> None:
        """Initialize fake file copier."""
        self.calls = []

    def __call__(self, src: Path, dst: Path) -> Path:
        """Copy a file.

        Args:
            src: Source file path.
            dst: Destination file path.

        Returns:
            Path to the copied file (dst).
        """
        self.calls.append((src, dst))
        return dst


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


def test_deploy_lora_success(tmp_path: Path) -> None:
    """Test deploy_lora copies file successfully."""
    settings = _make_test_settings(tmp_path)
    fake_copier = FakeFileCopier()
    _test_hooks.Hooks.file_copier = fake_copier

    # Create source LoRA file
    lora_path = tmp_path / "output" / "lora_test.safetensors"
    lora_path.parent.mkdir(parents=True)
    lora_path.touch()

    result = deploy_lora(settings, lora_path, "my_lora")

    assert result["success"] is True
    assert result["source_path"] == str(lora_path)
    expected_dest = tmp_path / "comfyui" / "models" / "loras" / "my_lora.safetensors"
    assert result["deployed_path"] == str(expected_dest)
    assert result["error_message"] is None

    # Verify copier was called
    assert len(fake_copier.calls) == 1
    assert fake_copier.calls[0] == (lora_path, expected_dest)


def test_deploy_lora_already_has_extension(tmp_path: Path) -> None:
    """Test deploy_lora with name already having .safetensors extension."""
    settings = _make_test_settings(tmp_path)
    fake_copier = FakeFileCopier()
    _test_hooks.Hooks.file_copier = fake_copier

    # Create source LoRA file
    lora_path = tmp_path / "output" / "lora_test.safetensors"
    lora_path.parent.mkdir(parents=True)
    lora_path.touch()

    result = deploy_lora(settings, lora_path, "my_lora.safetensors")

    assert result["success"] is True
    expected_dest = tmp_path / "comfyui" / "models" / "loras" / "my_lora.safetensors"
    assert result["deployed_path"] == str(expected_dest)


def test_deploy_lora_source_not_found(tmp_path: Path) -> None:
    """Test deploy_lora with non-existent source file."""
    settings = _make_test_settings(tmp_path)

    # Don't create the source file
    lora_path = tmp_path / "nonexistent.safetensors"

    result = deploy_lora(settings, lora_path, "my_lora")

    assert result["success"] is False
    assert result["source_path"] == str(lora_path)
    assert result["deployed_path"] is None
    assert "Source LoRA not found" in (result["error_message"] or "")


def test_deploy_lora_uses_default_copier(tmp_path: Path) -> None:
    """Test deploy_lora uses default file copier when hook not set."""
    settings = _make_test_settings(tmp_path)
    _test_hooks.Hooks.file_copier = None

    # Create source LoRA file with content
    lora_path = tmp_path / "output" / "lora_test.safetensors"
    lora_path.parent.mkdir(parents=True)
    lora_path.write_bytes(b"fake lora content")

    result = deploy_lora(settings, lora_path, "my_lora")

    assert result["success"] is True

    # Verify file was actually copied
    dest_path = tmp_path / "comfyui" / "models" / "loras" / "my_lora.safetensors"
    assert dest_path.exists()
    assert dest_path.read_bytes() == b"fake lora content"
