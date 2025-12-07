from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pytest
from PIL import Image
from platform_core.job_events import JobEventV1, decode_job_event, is_progress
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis, FakeRedisPublishError

import handwriting_ai.jobs.digits as dj
import handwriting_ai.training.mnist_train as mt
from handwriting_ai.training.resources import ResourceLimits
from handwriting_ai.training.train_config import (
    TrainConfig,
    TrainingResult,
    default_train_config,
)


class MnistRawWriter(Protocol):
    def __call__(self, root: Path, n: int = 8) -> None: ...


UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None

pytestmark = pytest.mark.usefixtures("digits_redis")


@pytest.fixture(autouse=True)
def _mock_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock resource detection for Windows/non-container environments."""
    limits = ResourceLimits(
        cpu_cores=4,
        memory_bytes=4 * 1024 * 1024 * 1024,
        optimal_threads=2,
        optimal_workers=0,
        max_batch_size=64,
    )
    import handwriting_ai.training.runtime as rt

    monkeypatch.setattr(rt, "detect_resource_limits", lambda: limits, raising=False)


class _TinyBase:
    def __init__(self, n: int = 4) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img = Image.new("L", (28, 28), 0)
        for y in range(10, 18):
            for x in range(12, 16):
                img.putpixel((x, y), 255)
        return img, idx % 10


def test_process_train_job_emits_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    write_mnist_raw: MnistRawWriter,
    digits_redis: FakeRedis,
) -> None:
    # Fake training that runs real train_with_config on tiny data to drive progress
    def _realish(cfg: TrainConfig) -> TrainingResult:
        train_base = _TinyBase(4)
        test_base = _TinyBase(2)
        # ensure output dir is within tmp
        cfg2 = default_train_config(
            data_root=tmp_path / "data",
            out_dir=tmp_path / "out",
            model_id=cfg["model_id"],
            epochs=2,
            batch_size=2,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            seed=cfg["seed"],
            device=cfg["device"],
            optim=cfg["optim"],
            scheduler="none",
            step_size=1,
            gamma=cfg["gamma"],
            min_lr=cfg["min_lr"],
            patience=0,
            min_delta=cfg["min_delta"],
            threads=0,
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
        )
        # Ensure MNIST raw files exist for calibration
        write_mnist_raw(cfg2["data_root"], n=8)
        return mt.train_with_config(cfg2, (train_base, test_base))

    monkeypatch.setattr(dj, "_run_training", _realish)

    payload: dict[str, UnknownJson] = {
        "type": "digits.train.v1",
        "request_id": "r1",
        "user_id": 9,
        "model_id": "m1",
        "epochs": 2,
        "batch_size": 2,
        "lr": 0.001,
        "seed": 3,
        "augment": False,
        "notes": None,
    }

    dj._decode_and_process_train_job(payload)

    events: list[JobEventV1] = [
        decode_job_event(record.payload) for record in digits_redis.published
    ]
    event_types = {ev["type"] for ev in events}
    assert "digits.job.started.v1" in event_types
    assert "digits.job.completed.v1" in event_types

    payload_types: set[str] = set()
    for ev in events:
        if is_progress(ev):
            progress_payload: JSONValue | None = ev.get("payload")
            if isinstance(progress_payload, str):
                decoded_payload = load_json_str(progress_payload)
                if isinstance(decoded_payload, dict):
                    payload_type = decoded_payload.get("type")
                    if isinstance(payload_type, str):
                        payload_types.add(payload_type)

    assert "digits.metrics.epoch.v1" in payload_types
    assert "digits.metrics.artifact.v1" in payload_types


def test_process_train_job_emitters_with_bad_publisher_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Use a Redis stub that raises to exercise error paths inside the emitters
    fr = FakeRedisPublishError()

    def _redis_for_kv(_: str) -> FakeRedisPublishError:
        return fr

    monkeypatch.setattr(dj, "redis_for_kv", _redis_for_kv, raising=True)

    def _realish(cfg: TrainConfig) -> TrainingResult:
        train_base = _TinyBase(2)
        test_base = _TinyBase(1)
        cfg2 = default_train_config(
            data_root=tmp_path / "data",
            out_dir=tmp_path / "out",
            model_id=cfg["model_id"],
            epochs=1,
            batch_size=1,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            seed=cfg["seed"],
            device=cfg["device"],
            optim=cfg["optim"],
            scheduler="none",
            step_size=1,
            gamma=cfg["gamma"],
            min_lr=cfg["min_lr"],
            patience=0,
            min_delta=cfg["min_delta"],
            threads=0,
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
        )
        return mt.train_with_config(cfg2, (train_base, test_base))

    monkeypatch.setattr(dj, "_run_training", _realish)

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
