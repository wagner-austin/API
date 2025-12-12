from __future__ import annotations

import secrets
from datetime import UTC, datetime
from pathlib import Path

import pytest
from platform_core.config import _test_hooks as config_test_hooks
from platform_core.digits_metrics_events import (
    DigitsConfigV1,
    encode_digits_metrics_event,
)
from platform_core.json_utils import JSONValue, dump_json_str
from platform_core.testing import make_fake_env
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

import handwriting_ai.jobs.digits as dj
from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import JobContextProtocol
from handwriting_ai.config import Settings
from handwriting_ai.inference.engine import build_fresh_state_dict
from handwriting_ai.training.train_config import TrainConfig, TrainingResult

pytestmark = pytest.mark.usefixtures("digits_redis")


def test_encode_event_compact_json() -> None:
    """Test that encode_digits_metrics_event produces compact JSON."""
    evt: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "r",
        "user_id": 1,
        "model_id": "m",
        "total_epochs": 5,
        "queue": "digits",
    }
    s = encode_digits_metrics_event(evt)
    assert s == dump_json_str(evt, compact=True)


def test_process_train_job_invalid_type_raises_after_logging() -> None:
    """Test that invalid job type raises an error."""
    payload: dict[str, JSONValue] = {"type": "wrong"}
    # Should raise when job type is invalid
    with pytest.raises((ValueError, TypeError, KeyError)):
        dj._decode_and_process_train_job(payload)


def test_load_settings_calls_config_load_settings() -> None:
    """Cover line 60: ensure _load_settings() returns a Settings dict."""
    result = dj._load_settings()
    # Should return a dict with expected keys
    assert "app" in result
    assert "digits" in result
    assert "security" in result


def test_process_train_job_happy_path_raises_on_artifact_upload(
    tmp_path: Path,
) -> None:
    """Test that missing data bank credentials raises RuntimeError."""
    # Set environment with REDIS_URL but without data bank credentials
    config_test_hooks.get_env = make_fake_env(
        {
            "REDIS_URL": "redis://test",
        }
    )

    # Stub _load_settings to use test paths
    def _fake_load_settings(*, create_dirs: bool = True) -> Settings:
        return {
            "app": {
                "data_root": tmp_path / "data",
                "artifacts_root": tmp_path / "artifacts",
                "logs_root": tmp_path / "logs",
                "threads": 0,
                "port": 8081,
            },
            "digits": {
                "model_dir": tmp_path / "models",
                "active_model": "mnist_resnet18_v1",
                "tta": False,
                "uncertain_threshold": 0.70,
                "max_image_mb": 2,
                "max_image_side_px": 1024,
                "predict_timeout_seconds": 5,
                "visualize_max_kb": 16,
                "retention_keep_runs": 3,
                "allowed_hosts": frozenset(["*"]),
            },
            "security": {"api_key": "", "api_key_enabled": False},
        }

    _test_hooks.load_settings = _fake_load_settings

    # Stub training to return a valid result
    def _fake_run_training(cfg: TrainConfig) -> TrainingResult:
        run_ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_rand = secrets.token_hex(3)
        run_id = f"{run_ts}-{run_rand}"
        sd = build_fresh_state_dict(arch="resnet18", n_classes=10)
        return {
            "model_id": cfg["model_id"],
            "state_dict": sd,
            "val_acc": 0.9,
            "metadata": {
                "run_id": run_id,
                "epochs": int(cfg["epochs"]),
                "batch_size": int(cfg["batch_size"]),
                "lr": float(cfg["lr"]),
                "seed": int(cfg["seed"]),
                "device": str(cfg["device"]),
                "optim": str(cfg["optim"]),
                "scheduler": str(cfg["scheduler"]),
                "augment": bool(cfg["augment"]),
            },
        }

    _test_hooks.run_training = _fake_run_training

    # Stub redis_factory to return a fake Redis
    stub = FakeRedis()

    def _fake_redis(url: str) -> FakeRedis:
        return stub

    _test_hooks.redis_factory = _fake_redis

    # Stub make_job_context to return a stub
    class _StubJobCtx:
        def publish_started(self) -> None:
            pass

        def publish_progress(
            self, progress: int, message: str | None = None, *, payload: JSONValue | None = None
        ) -> None:
            pass

        def publish_completed(self, result_id: str, result_bytes: int) -> None:
            pass

        def publish_failed(self, error_kind: str, message: str) -> None:
            pass

    def _fake_make_job_context(
        *,
        redis: RedisStrProto,
        domain: str,
        events_channel: str,
        job_id: str,
        user_id: int,
        queue_name: str,
    ) -> JobContextProtocol:
        return _StubJobCtx()

    _test_hooks.make_job_context = _fake_make_job_context

    payload: dict[str, JSONValue] = {
        "type": "digits.train.v1",
        "request_id": "r1",
        "user_id": 2,
        "model_id": "m1",
        "epochs": 1,
        "batch_size": 4,
        "lr": 0.001,
        "seed": 3,
        "augment": False,
        "notes": None,
    }

    # Should raise when artifact upload fails due to missing credentials
    with pytest.raises(RuntimeError, match="missing data bank API credentials"):
        dj._decode_and_process_train_job(payload)

    # Verify FakeRedis was used correctly
    stub.assert_only_called({"publish", "close"})
