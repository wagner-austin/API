from __future__ import annotations

import secrets
from datetime import UTC, datetime
from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue

import handwriting_ai.jobs.digits as dj
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import build_fresh_state_dict
from handwriting_ai.training.train_config import (
    TrainConfig,
    TrainingResult,
    TrainingResultMetadata,
)

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
        self, progress: int, message: str | None = None, payload: JSONValue | None = None
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
    app: AppConfig = {
        "data_root": tmp_path / "data",
        "artifacts_root": tmp_path / "artifacts",
        "logs_root": tmp_path / "logs",
        "threads": 0,
        "port": 8081,
    }
    dig: DigitsConfig = {
        "model_dir": tmp_path / "models",
        "active_model": "mnist_resnet18_v1",
        "tta": False,
        "uncertain_threshold": 0.70,
        "max_image_mb": 2,
        "max_image_side_px": 1024,
        "predict_timeout_seconds": 5,
        "visualize_max_kb": 16,
        "retention_keep_runs": 3,
    }
    sec: SecurityConfig = {"api_key": ""}
    return {"app": app, "digits": dig, "security": sec}


def _quick_training(cfg: TrainConfig) -> TrainingResult:
    run_ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_rand = secrets.token_hex(3)
    run_id = f"{run_ts}-{run_rand}"
    sd = build_fresh_state_dict(arch="resnet18", n_classes=10)
    meta: TrainingResultMetadata = {
        "run_id": run_id,
        "epochs": int(cfg["epochs"]),
        "batch_size": int(cfg["batch_size"]),
        "lr": float(cfg["lr"]),
        "seed": int(cfg["seed"]),
        "device": str(cfg["device"]),
        "optim": str(cfg["optim"]),
        "scheduler": str(cfg["scheduler"]),
        "augment": bool(cfg["augment"]),
    }
    return {
        "model_id": cfg["model_id"],
        "state_dict": sd,
        "val_acc": 0.5,
        "metadata": meta,
    }


def test_process_train_job_keyboard_interrupt_publishes_interrupted_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_ctx = _StubJobCtx(progress_side_effects=[None, None, None, KeyboardInterrupt()])

    def make_job_context_mock(*args: JSONValue, **kwargs: JSONValue) -> _StubJobCtx:
        return job_ctx

    monkeypatch.setattr(dj, "make_job_context", make_job_context_mock, raising=True)
    monkeypatch.setattr(dj, "_run_training", _quick_training)

    settings = _make_test_settings(tmp_path)
    monkeypatch.setattr(dj, "_load_settings", staticmethod(lambda: settings))

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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_ctx = _StubJobCtx(completed_exc=OSError("boom-completed"))

    def make_job_context_mock(*args: JSONValue, **kwargs: JSONValue) -> _StubJobCtx:
        return job_ctx

    monkeypatch.setattr(dj, "make_job_context", make_job_context_mock, raising=True)
    monkeypatch.setattr(dj, "_run_training", _quick_training)

    settings = _make_test_settings(tmp_path)
    monkeypatch.setattr(dj, "_load_settings", staticmethod(lambda: settings))

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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test job completion when publisher is None (covers job_ctx is None branch)."""

    def _none_ctx(*args: JSONValue, **kwargs: JSONValue) -> None:
        return None

    monkeypatch.setattr(dj, "make_job_context", _none_ctx, raising=True)
    monkeypatch.setattr(dj, "_run_training", _quick_training)

    settings = _make_test_settings(tmp_path)
    monkeypatch.setattr(dj, "_load_settings", staticmethod(lambda: settings))

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
