from __future__ import annotations

import secrets
from datetime import UTC, datetime
from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue
from platform_workers.redis import RedisStrProto

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import JobContextProtocol
from handwriting_ai.config import Settings
from handwriting_ai.inference.engine import build_fresh_state_dict
from handwriting_ai.jobs import digits as dj
from handwriting_ai.training.train_config import TrainConfig, TrainingResult

pytestmark = pytest.mark.usefixtures("digits_redis")


class _StubJobCtx:
    def __init__(
        self,
        *,
        progress_side_effects: list[BaseException | None] | None = None,
        completed_exc: BaseException | None = None,
    ) -> None:
        self.publish_started_count = 0
        self.publish_progress_count = 0
        self.publish_completed_count = 0
        self.publish_failed_count = 0
        self._progress_side_effects = list(progress_side_effects or [])
        self._completed_exc = completed_exc

    def publish_started(self) -> None:
        self.publish_started_count += 1

    def publish_progress(
        self, progress: int, message: str | None = None, *, payload: JSONValue | None = None
    ) -> None:
        self.publish_progress_count += 1
        if self._progress_side_effects:
            eff = self._progress_side_effects.pop(0)
            if eff is not None:
                raise eff

    def publish_completed(self, result_id: str, result_bytes: int) -> None:
        self.publish_completed_count += 1
        if self._completed_exc is not None:
            raise self._completed_exc

    def publish_failed(self, error_kind: str, message: str) -> None:
        self.publish_failed_count += 1


def _make_test_settings(tmp_path: Path) -> Settings:
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


def _quick_training(cfg: TrainConfig) -> TrainingResult:
    run_ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_rand = secrets.token_hex(3)
    run_id = f"{run_ts}-{run_rand}"
    sd = build_fresh_state_dict(arch="resnet18", n_classes=10)
    return {
        "model_id": cfg["model_id"],
        "state_dict": sd,
        "val_acc": 0.5,
        "metadata": {
            "run_id": run_id,
            "epochs": int(cfg["epochs"]),
            "batch_size": int(cfg["batch_size"]),
            "lr": float(cfg["lr"]),
            "seed": int(cfg["seed"]),
            "device": "cpu",
            "precision": "fp32",
            "optim": str(cfg["optim"]),
            "scheduler": str(cfg["scheduler"]),
            "augment": bool(cfg["augment"]),
        },
    }


def test_process_train_job_keyboard_interrupt_publishes_interrupted_and_reraises(
    tmp_path: Path,
) -> None:
    job_ctx = _StubJobCtx(progress_side_effects=[None, None, None, KeyboardInterrupt()])

    def _fake_make_job_context(
        *,
        redis: RedisStrProto,
        domain: str,
        events_channel: str,
        job_id: str,
        user_id: int,
        queue_name: str,
    ) -> JobContextProtocol:
        return job_ctx

    _test_hooks.make_job_context = _fake_make_job_context
    _test_hooks.run_training = _quick_training

    settings = _make_test_settings(tmp_path)

    def _fake_load_settings(*, create_dirs: bool = True) -> Settings:
        return settings

    _test_hooks.load_settings = _fake_load_settings

    payload: dict[str, JSONValue] = {
        "type": "digits.train.v1",
        "request_id": "r-kb",
        "user_id": 7,
        "model_id": "m-kb",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.001,
        "seed": 1,
        "augment": False,
        "notes": None,
    }

    with pytest.raises(KeyboardInterrupt):
        dj._decode_and_process_train_job(payload)

    assert job_ctx.publish_started_count == 1
    assert job_ctx.publish_progress_count == 4


def test_process_train_job_completed_publish_failure(
    tmp_path: Path,
) -> None:
    job_ctx = _StubJobCtx(completed_exc=OSError("boom-completed"))

    def _fake_make_job_context(
        *,
        redis: RedisStrProto,
        domain: str,
        events_channel: str,
        job_id: str,
        user_id: int,
        queue_name: str,
    ) -> JobContextProtocol:
        return job_ctx

    _test_hooks.make_job_context = _fake_make_job_context
    _test_hooks.run_training = _quick_training

    settings = _make_test_settings(tmp_path)

    def _fake_load_settings(*, create_dirs: bool = True) -> Settings:
        return settings

    _test_hooks.load_settings = _fake_load_settings

    payload: dict[str, JSONValue] = {
        "type": "digits.train.v1",
        "request_id": "r2",
        "user_id": 7,
        "model_id": "m2",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.001,
        "seed": 1,
        "augment": False,
        "notes": None,
    }

    with pytest.raises(OSError, match="boom-completed"):
        dj._decode_and_process_train_job(payload)

    assert job_ctx.publish_started_count == 1
    assert job_ctx.publish_completed_count == 1


def test_process_train_job_completes_without_publisher(
    tmp_path: Path,
) -> None:
    """Test job completion when publisher is None (covers job_ctx is None branch)."""

    from platform_core.job_events import JobDomain

    def _fake_make_job_context(
        *,
        redis: RedisStrProto,
        domain: JobDomain,
        events_channel: str,
        job_id: str,
        user_id: int,
        queue_name: str,
    ) -> None:
        _ = (redis, domain, events_channel, job_id, user_id, queue_name)
        return

    _test_hooks.make_job_context = _fake_make_job_context
    _test_hooks.run_training = _quick_training

    settings = _make_test_settings(tmp_path)

    def _fake_load_settings(*, create_dirs: bool = True) -> Settings:
        return settings

    _test_hooks.load_settings = _fake_load_settings

    payload: dict[str, JSONValue] = {
        "type": "digits.train.v1",
        "request_id": "r-no-pub",
        "user_id": 7,
        "model_id": "m-no-pub",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.001,
        "seed": 1,
        "augment": False,
        "notes": None,
    }

    with pytest.raises(RuntimeError, match="JobContext could not be created"):
        dj._decode_and_process_train_job(payload)
