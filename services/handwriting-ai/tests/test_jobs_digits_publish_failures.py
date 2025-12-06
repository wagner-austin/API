from __future__ import annotations

import secrets
from datetime import UTC, datetime

import pytest

import handwriting_ai.jobs.digits as dj
from handwriting_ai.inference.engine import build_fresh_state_dict
from handwriting_ai.training.train_config import (
    TrainConfig,
    TrainingResult,
    TrainingResultMetadata,
)

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None

pytestmark = pytest.mark.usefixtures("digits_redis")


class _RaisingRedis:
    def publish(self, channel: str, message: str) -> int:
        raise OSError("fail")

    def close(self) -> None:
        return None


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
        "val_acc": 0.1,
        "metadata": meta,
    }


def test_process_train_job_publish_failures_raise_after_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _redis_for_kv(_: str) -> _RaisingRedis:
        return _RaisingRedis()

    monkeypatch.setattr(dj, "redis_for_kv", _redis_for_kv, raising=True)
    monkeypatch.setattr(dj, "_run_training", _quick_training, raising=True)

    payload: dict[str, UnknownJson] = {
        "type": "digits.train.v1",
        "request_id": "r1",
        "user_id": 9,
        "model_id": "m1",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.001,
        "seed": 3,
        "augment": False,
        "notes": None,
    }

    # Should raise when publisher fails during started event
    with pytest.raises(OSError, match="fail"):
        dj._decode_and_process_train_job(payload)
