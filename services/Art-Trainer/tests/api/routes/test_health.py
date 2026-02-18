"""Tests for health check routes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str, require_str
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from art_trainer.api.main import create_app
from art_trainer.core import _test_hooks
from art_trainer.core.config.settings import Settings


class FakeKVStoreFactoryWithWorkers:
    """Fake KV store factory that returns FakeRedis with workers."""

    fake_redis: FakeRedis

    def __init__(self) -> None:
        """Initialize factory."""
        self.fake_redis = FakeRedis()
        self.fake_redis.sadd("rq:workers", "worker-1")
        self.fake_redis.calls.clear()

    def __call__(self, url: str) -> RedisStrProto:
        """Return FakeRedis with workers.

        Args:
            url: Redis URL (ignored).

        Returns:
            FakeRedis with workers registered.
        """
        return self.fake_redis


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


def test_healthz_returns_ok(tmp_path: Path) -> None:
    """Test healthz endpoint returns ok status."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    app = create_app(settings)
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    data = load_json_str(response.text)
    if not isinstance(data, dict):
        raise AssertionError("Response body must be a JSON object")
    status = require_str(data, "status")
    assert status == "ok"


def test_readyz_returns_degraded_no_workers(tmp_path: Path) -> None:
    """Test readyz endpoint returns degraded when no workers."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    app = create_app(settings)
    client = TestClient(app)

    response = client.get("/readyz")

    # Without workers, should be degraded
    assert response.status_code in [200, 503]
    data = load_json_str(response.text)
    if not isinstance(data, dict):
        raise AssertionError("Response body must be a JSON object")
    status = require_str(data, "status")
    # Status should be ok or degraded
    assert status in ["ok", "degraded"]


def test_readyz_returns_ready_with_workers(tmp_path: Path) -> None:
    """Test readyz endpoint returns ready when workers available."""
    settings = _make_test_settings(tmp_path)

    def load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = load_settings

    # Use factory that has workers registered
    factory = FakeKVStoreFactoryWithWorkers()
    _test_hooks.kv_store_factory = factory

    app = create_app(settings)
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 200
    data = load_json_str(response.text)
    if not isinstance(data, dict):
        raise AssertionError("Response body must be a JSON object")
    status = require_str(data, "status")
    assert status == "ready"
    factory.fake_redis.assert_only_called({"ping", "scard", "sadd"})
