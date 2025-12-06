from __future__ import annotations

import secrets
from datetime import UTC, datetime
from pathlib import Path

import pytest

import handwriting_ai.jobs.digits as dj
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import build_fresh_state_dict
from handwriting_ai.training.train_config import (
    TrainConfig,
    TrainingResult,
    TrainingResultMetadata,
)

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None

pytestmark = pytest.mark.usefixtures("digits_redis")


def _make_settings(tmp: Path) -> Settings:
    app: AppConfig = {
        "data_root": tmp / "data",
        "artifacts_root": tmp / "artifacts",
        "logs_root": tmp / "logs",
        "threads": 0,
        "port": 8081,
    }
    dig: DigitsConfig = {
        "model_dir": tmp / "models",
        "active_model": "m",
        "tta": False,
        "uncertain_threshold": 0.5,
        "max_image_mb": 1,
        "max_image_side_px": 64,
        "predict_timeout_seconds": 1,
        "visualize_max_kb": 16,
        "retention_keep_runs": 1,
    }
    sec: SecurityConfig = {"api_key": ""}
    return {"app": app, "digits": dig, "security": sec}


def _training_with_artifacts(cfg: TrainConfig) -> TrainingResult:
    """Create training result for tests."""
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
        "val_acc": 0.9,
        "metadata": meta,
    }


def test_process_train_job_missing_data_bank_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cover digits.py:279 - missing data bank API credentials."""

    monkeypatch.setattr(dj, "_run_training", _training_with_artifacts, raising=True)

    settings = _make_settings(tmp_path)
    monkeypatch.setattr(dj, "_load_settings", staticmethod(lambda: settings))

    # Clear the data bank credentials to trigger line 279
    monkeypatch.delenv("APP__DATA_BANK_API_URL", raising=False)
    monkeypatch.delenv("APP__DATA_BANK_API_KEY", raising=False)

    payload: dict[str, UnknownJson] = {
        "type": "digits.train.v1",
        "request_id": "r-creds",
        "user_id": 1,
        "model_id": "m-creds",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.001,
        "seed": 1,
        "augment": False,
        "notes": None,
    }

    with pytest.raises(RuntimeError, match="missing data bank API credentials"):
        dj._decode_and_process_train_job(payload)
