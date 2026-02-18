"""Tests for core/_test_hooks.py default implementations.

These tests exercise the default hook implementations that production code uses.
They verify the hook pattern works correctly without fakes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from art_trainer.core._test_hooks import (
    _default_load_settings,
    _default_lora_output_dir,
    _default_rq_queue,
    _default_rq_retry,
    _default_shutil_which,
)
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


def test_default_rq_queue() -> None:
    """Test _default_rq_queue creates a real RQ queue."""
    from platform_workers.rq_harness import RQClientQueue
    from platform_workers.testing import FakeRedisBytesClient

    connection = FakeRedisBytesClient()
    queue: RQClientQueue = _default_rq_queue("test-queue", connection)

    # Verify it returns a queue that satisfies the protocol
    # The queue should have an enqueue method callable
    enqueue_method = queue.enqueue
    assert callable(enqueue_method)


def test_default_rq_retry() -> None:
    """Test _default_rq_retry creates a real RQ retry."""
    from platform_workers.rq_harness import RQRetryLike

    retry: RQRetryLike = _default_rq_retry(max_retries=3, intervals=[10, 20, 30])

    # Verify it returns a retry that satisfies the protocol
    # The return value is used by RQ to configure retries
    if retry is None:
        raise AssertionError("Expected _default_rq_retry to return a retry object")
    # Retry objects should be truthy (usable in boolean context)
    assert retry, "Retry object should be truthy"


def test_default_load_settings() -> None:
    """Test _default_load_settings loads real settings."""
    settings = _default_load_settings()

    # Verify settings structure
    assert "app_env" in settings
    assert "redis" in settings
    assert "rq" in settings
    assert "app" in settings


def test_default_lora_output_dir(tmp_path: Path) -> None:
    """Test _default_lora_output_dir returns correct path."""
    settings = _make_test_settings(tmp_path)

    result = _default_lora_output_dir(settings, "test-job-123")

    expected = tmp_path / "output" / "test-job-123"
    assert result == expected


def test_default_shutil_which_finds_python() -> None:
    """Test _default_shutil_which finds python executable."""
    result = _default_shutil_which("python")

    # Python should be findable on any system where tests run
    # Verify it returns a path string containing python
    if result is None:
        raise AssertionError("Expected python to be found on PATH")
    assert result.lower().endswith("python.exe") or result.lower().endswith("python")


def test_default_shutil_which_returns_none_for_nonexistent() -> None:
    """Test _default_shutil_which returns None for nonexistent command."""
    result = _default_shutil_which("this_command_definitely_does_not_exist_xyz123")

    assert result is None, f"Expected None for nonexistent command, got {result}"
