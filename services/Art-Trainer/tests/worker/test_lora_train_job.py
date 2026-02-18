"""Tests for LoRA training worker job."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Literal

from platform_core.json_utils import JSONObject
from platform_workers.testing import FakeRedis

from art_trainer.core import _test_hooks
from art_trainer.core.config.settings import Settings
from art_trainer.core.contracts.queue_encoding import encode_lora_train_payload
from art_trainer.core.infra.redis_keys import status_key
from art_trainer.core.services.dataset import _test_hooks as dataset_test_hooks
from art_trainer.core.services.dataset._test_hooks import UploadResult
from art_trainer.core.services.training.backends.kohya import (
    _test_hooks as kohya_test_hooks,
)
from art_trainer.worker.lora_train_job import run_lora_train
from tests.core.services.training.backends.kohya.testing import (
    FakeConfigWriter,
    FakeKohyaRunner,
)


def _create_fake_dataset_zip() -> bytes:
    """Create a fake dataset ZIP with a dummy image and caption.

    Returns:
        ZIP file bytes.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add a dummy image (1x1 pixel PNG)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        zf.writestr("image001.png", png_bytes)
        zf.writestr("image001.txt", "test caption")
    return buffer.getvalue()


def _make_fake_http_get(dataset_zip_bytes: bytes) -> dataset_test_hooks.HttpGetProto:
    """Create a fake HTTP GET function.

    Args:
        dataset_zip_bytes: ZIP bytes to return.

    Returns:
        Fake HTTP GET function.
    """

    def fake_http_get(url: str, headers: dict[str, str]) -> bytes:
        return dataset_zip_bytes

    return fake_http_get


def _make_fake_http_upload() -> dataset_test_hooks.HttpUploadProto:
    """Create a fake HTTP upload function.

    Returns:
        Fake HTTP upload function.
    """

    def fake_http_upload(
        url: str,
        headers: dict[str, str],
        filename: str,
        content: bytes,
    ) -> UploadResult:
        result: UploadResult = {
            "file_id": "fake-lora-file-id",
            "filename": filename,
        }
        return result

    return fake_http_upload


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


def _make_test_payload() -> JSONObject:
    """Create test payload.

    Returns:
        Encoded test payload.
    """
    return encode_lora_train_payload(
        {
            "job_id": "test-worker-job",
            "user_id": 123,
            "base_model": "sd15",
            "training_type": "style",
            "dataset_file_id": "file-worker-test",
            "steps": 100,
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
    )


def test_run_lora_train_success(tmp_path: Path) -> None:
    """Test run_lora_train completes successfully."""
    settings = _make_test_settings(tmp_path)
    fake_redis = FakeRedis()

    # Set up hooks
    def load_settings() -> Settings:
        return settings

    def kv_factory(url: str) -> FakeRedis:
        return fake_redis

    _test_hooks.load_settings = load_settings
    _test_hooks.kv_store_factory = kv_factory

    # Set up dataset hooks
    dataset_zip_bytes = _create_fake_dataset_zip()
    dataset_test_hooks.http_get = _make_fake_http_get(dataset_zip_bytes)
    dataset_test_hooks.http_upload = _make_fake_http_upload()

    # Set up Kohya fakes
    fake_runner = FakeKohyaRunner(should_succeed=True, final_loss=0.05)
    fake_writer = FakeConfigWriter()
    kohya_test_hooks.Hooks.subprocess_runner = fake_runner
    kohya_test_hooks.Hooks.config_writer = fake_writer

    # Create output directory and fake output file
    output_dir = tmp_path / "output" / "test-worker-job"
    output_dir.mkdir(parents=True)
    (output_dir / "lora_test-worker-job.safetensors").touch()

    payload = _make_test_payload()
    run_lora_train(payload)

    # Verify status was set to completed
    status = fake_redis.get(status_key("test-worker-job"))
    assert status == "completed"
    fake_redis.assert_only_called({"get", "set"})


def test_run_lora_train_failure(tmp_path: Path) -> None:
    """Test run_lora_train handles failure correctly."""
    settings = _make_test_settings(tmp_path)
    fake_redis = FakeRedis()

    # Set up hooks
    def load_settings() -> Settings:
        return settings

    def kv_factory(url: str) -> FakeRedis:
        return fake_redis

    _test_hooks.load_settings = load_settings
    _test_hooks.kv_store_factory = kv_factory

    # Set up dataset hooks
    dataset_zip_bytes = _create_fake_dataset_zip()
    dataset_test_hooks.http_get = _make_fake_http_get(dataset_zip_bytes)
    dataset_test_hooks.http_upload = _make_fake_http_upload()

    # Set up Kohya fakes to fail
    fake_runner = FakeKohyaRunner(
        should_succeed=False,
        returncode=1,
        stderr="CUDA error",
    )
    fake_writer = FakeConfigWriter()
    kohya_test_hooks.Hooks.subprocess_runner = fake_runner
    kohya_test_hooks.Hooks.config_writer = fake_writer

    payload = _make_test_payload()
    run_lora_train(payload)

    # Verify status was set to failed
    status = fake_redis.get(status_key("test-worker-job"))
    assert status == "failed"
    fake_redis.assert_only_called({"get", "set"})


def test_run_lora_train_cancellation(tmp_path: Path) -> None:
    """Test run_lora_train respects cancellation."""
    settings = _make_test_settings(tmp_path)
    fake_redis = FakeRedis()

    # Pre-set cancellation flag
    from art_trainer.core.infra.redis_keys import cancel_key

    fake_redis.set(cancel_key("test-worker-job"), "1")

    # Set up hooks
    def load_settings() -> Settings:
        return settings

    def kv_factory(url: str) -> FakeRedis:
        return fake_redis

    _test_hooks.load_settings = load_settings
    _test_hooks.kv_store_factory = kv_factory

    # Set up dataset hooks
    dataset_zip_bytes = _create_fake_dataset_zip()
    dataset_test_hooks.http_get = _make_fake_http_get(dataset_zip_bytes)
    dataset_test_hooks.http_upload = _make_fake_http_upload()

    # Set up Kohya fakes (should not be called due to cancellation)
    fake_runner = FakeKohyaRunner(should_succeed=True)
    fake_writer = FakeConfigWriter()
    kohya_test_hooks.Hooks.subprocess_runner = fake_runner
    kohya_test_hooks.Hooks.config_writer = fake_writer

    payload = _make_test_payload()
    run_lora_train(payload)

    # Verify status was set to cancelled
    status = fake_redis.get(status_key("test-worker-job"))
    assert status == "cancelled"
    fake_redis.assert_only_called({"get", "set"})


def test_run_lora_train_success_no_output_file(tmp_path: Path) -> None:
    """Test run_lora_train completes when no output file exists."""
    settings = _make_test_settings(tmp_path)
    fake_redis = FakeRedis()

    # Set up hooks
    def load_settings() -> Settings:
        return settings

    def kv_factory(url: str) -> FakeRedis:
        return fake_redis

    _test_hooks.load_settings = load_settings
    _test_hooks.kv_store_factory = kv_factory

    # Set up dataset hooks
    dataset_zip_bytes = _create_fake_dataset_zip()
    dataset_test_hooks.http_get = _make_fake_http_get(dataset_zip_bytes)
    dataset_test_hooks.http_upload = _make_fake_http_upload()

    # Set up Kohya fakes - training succeeds but no output file
    fake_runner = FakeKohyaRunner(should_succeed=True, final_loss=0.05)
    fake_writer = FakeConfigWriter()
    kohya_test_hooks.Hooks.subprocess_runner = fake_runner
    kohya_test_hooks.Hooks.config_writer = fake_writer

    # Create output directory but NO output file
    output_dir = tmp_path / "output" / "test-worker-job"
    output_dir.mkdir(parents=True)
    # Don't create the .safetensors file

    payload = _make_test_payload()
    run_lora_train(payload)

    # Verify status was set to completed (training succeeded even without file)
    status = fake_redis.get(status_key("test-worker-job"))
    assert status == "completed"
    fake_redis.assert_only_called({"get", "set"})
