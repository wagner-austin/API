from __future__ import annotations

import secrets
from datetime import UTC, datetime

import pytest
from platform_workers.testing import FakeRedisPublishError

import handwriting_ai.jobs.digits as dj
from handwriting_ai import _test_hooks
from handwriting_ai.inference.engine import build_fresh_state_dict
from handwriting_ai.training.train_config import TrainConfig, TrainingResult

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None

pytestmark = pytest.mark.usefixtures("digits_redis")


def _quick_training(cfg: TrainConfig) -> TrainingResult:
    run_ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_rand = secrets.token_hex(3)
    run_id = f"{run_ts}-{run_rand}"
    sd = build_fresh_state_dict(arch="resnet18", n_classes=10)
    return {
        "model_id": cfg["model_id"],
        "state_dict": sd,
        "val_acc": 0.1,
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


def test_process_train_job_publish_failures_raise_after_logging() -> None:
    fr = FakeRedisPublishError()

    def _redis_for_error(url: str) -> FakeRedisPublishError:
        _ = url
        return fr

    _test_hooks.redis_factory = _redis_for_error
    _test_hooks.run_training = _quick_training

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
    with pytest.raises(OSError, match="publish failure"):
        dj._decode_and_process_train_job(payload)
    fr.assert_only_called({"publish", "close"})
