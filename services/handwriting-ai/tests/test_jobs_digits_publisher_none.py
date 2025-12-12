from __future__ import annotations

import secrets
from datetime import UTC, datetime

import pytest

import handwriting_ai.jobs.digits as dj
from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import ResourceLimitsDict
from handwriting_ai.inference.engine import build_fresh_state_dict
from handwriting_ai.training.train_config import TrainConfig, TrainingResult

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None

pytestmark = pytest.mark.usefixtures("digits_redis")


@pytest.fixture(autouse=True)
def _mock_resources() -> None:
    """Configure resource detection for Windows/non-container environments."""

    def _fake_limits() -> ResourceLimitsDict:
        return {
            "cpu_cores": 4,
            "memory_bytes": 4 * 1024 * 1024 * 1024,
            "optimal_threads": 2,
            "optimal_workers": 0,
            "max_batch_size": 64,
        }

    _test_hooks.detect_resource_limits = _fake_limits


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
            "device": str(cfg["device"]),
            "optim": str(cfg["optim"]),
            "scheduler": str(cfg["scheduler"]),
            "augment": bool(cfg["augment"]),
        },
    }


def test_process_train_job_with_no_publisher() -> None:
    def _quick_training_raises(cfg: TrainConfig) -> TrainingResult:
        # Raise to test that error is raised even without publisher
        raise RuntimeError("training failed")

    _test_hooks.run_training = _quick_training_raises

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

    # Should raise even without publisher
    with pytest.raises(RuntimeError, match="training failed"):
        dj._decode_and_process_train_job(payload)
