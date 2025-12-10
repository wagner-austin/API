from __future__ import annotations

import pytest

from platform_core.json_utils import JSONTypeError
from platform_core.trainer_metrics_events import (
    TrainerMetricsEventV1,
    decode_trainer_metrics_event,
    encode_trainer_metrics_event,
    is_completed_metrics,
    is_config,
    is_progress_metrics,
    make_completed_metrics_event,
    make_config_event,
    make_progress_metrics_event,
)


def _roundtrip(event: TrainerMetricsEventV1) -> TrainerMetricsEventV1:
    encoded = encode_trainer_metrics_event(event)
    return decode_trainer_metrics_event(encoded)


def test_roundtrip_config_event_minimal() -> None:
    event = make_config_event(
        job_id="run-1",
        user_id=5,
        model_family="gpt2",
        model_size="small",
        total_epochs=10,
        queue="primary",
    )
    decoded = _roundtrip(event)
    assert decoded == event
    assert is_config(decoded)
    assert not is_progress_metrics(decoded)
    assert not is_completed_metrics(decoded)


def test_roundtrip_config_event_full() -> None:
    event = make_config_event(
        job_id="run-2",
        user_id=7,
        model_family="gpt2",
        model_size="medium",
        total_epochs=20,
        queue="primary",
        cpu_cores=8,
        memory_mb=16384,
        optimal_threads=4,
        optimal_workers=2,
        batch_size=32,
        learning_rate=0.001,
    )
    decoded = _roundtrip(event)
    assert decoded == event
    assert is_config(decoded)
    # Access through event (already typed as TrainerConfigV1)
    assert event["cpu_cores"] == 8
    assert event["memory_mb"] == 16384
    assert event["optimal_threads"] == 4
    assert event["optimal_workers"] == 2
    assert event["batch_size"] == 32
    assert event["learning_rate"] == 0.001


def test_roundtrip_progress_metrics_event_minimal() -> None:
    event = make_progress_metrics_event(
        job_id="run-3",
        user_id=11,
        epoch=3,
        total_epochs=10,
        step=150,
        train_loss=1.234,
        train_ppl=3.435,
        grad_norm=0.567,
        samples_per_sec=45.2,
    )
    decoded = _roundtrip(event)
    assert decoded == event
    assert is_progress_metrics(decoded)
    assert not is_config(decoded)
    assert not is_completed_metrics(decoded)
    assert decoded["train_loss"] == 1.234
    assert decoded["train_ppl"] == 3.435
    assert decoded["grad_norm"] == 0.567
    assert decoded["samples_per_sec"] == 45.2
    assert "val_loss" not in decoded
    assert "val_ppl" not in decoded


def test_roundtrip_progress_metrics_event_with_validation() -> None:
    event = make_progress_metrics_event(
        job_id="run-3b",
        user_id=11,
        epoch=3,
        total_epochs=10,
        step=150,
        train_loss=1.234,
        train_ppl=3.435,
        grad_norm=0.567,
        samples_per_sec=45.2,
        val_loss=1.456,
        val_ppl=4.29,
    )
    decoded = _roundtrip(event)
    assert decoded == event
    assert is_progress_metrics(decoded)
    assert decoded["train_loss"] == 1.234
    assert decoded["train_ppl"] == 3.435
    assert decoded["val_loss"] == 1.456
    assert decoded["val_ppl"] == 4.29


def test_roundtrip_completed_metrics_event() -> None:
    event = make_completed_metrics_event(
        job_id="run-4",
        user_id=13,
        test_loss=0.456,
        test_ppl=1.578,
        artifact_path="/path/to/model",
    )
    decoded = _roundtrip(event)
    assert decoded == event
    assert is_completed_metrics(decoded)
    assert not is_config(decoded)
    assert not is_progress_metrics(decoded)
    assert decoded["test_loss"] == 0.456
    assert decoded["test_ppl"] == 1.578


def test_decode_config_event_with_int_learning_rate() -> None:
    payload = (
        '{"type": "trainer.metrics.config.v1", "job_id": "r", "user_id": 1, '
        '"model_family": "gpt2", "model_size": "s", "total_epochs": 5, '
        '"queue": "q", "learning_rate": 1}'
    )
    decoded = decode_trainer_metrics_event(payload)
    assert is_config(decoded)
    assert decoded["learning_rate"] == 1.0


def test_decode_progress_metrics_with_int_values() -> None:
    payload = (
        '{"type": "trainer.metrics.progress.v1", "job_id": "r", "user_id": 1, '
        '"epoch": 1, "total_epochs": 5, "step": 10, "train_loss": 2, "train_ppl": 7, '
        '"grad_norm": 1, "samples_per_sec": 50}'
    )
    decoded = decode_trainer_metrics_event(payload)
    assert is_progress_metrics(decoded)
    assert decoded["train_loss"] == 2.0
    assert decoded["train_ppl"] == 7.0
    assert decoded["grad_norm"] == 1.0
    assert decoded["samples_per_sec"] == 50.0


def test_decode_progress_metrics_with_optional_val_metrics() -> None:
    payload = (
        '{"type": "trainer.metrics.progress.v1", "job_id": "r", "user_id": 1, '
        '"epoch": 1, "total_epochs": 5, "step": 10, "train_loss": 2.0, "train_ppl": 7.39, '
        '"grad_norm": 0.5, "samples_per_sec": 45.0, "val_loss": 3, "val_ppl": 20}'
    )
    decoded = decode_trainer_metrics_event(payload)
    assert is_progress_metrics(decoded)
    assert decoded["train_ppl"] == 7.39
    assert decoded["val_loss"] == 3.0
    assert decoded["val_ppl"] == 20.0


def test_decode_completed_metrics_with_int_values() -> None:
    payload = (
        '{"type": "trainer.metrics.completed.v1", "job_id": "r", "user_id": 1, '
        '"test_loss": 1, "test_ppl": 2, "artifact_path": "/a"}'
    )
    decoded = decode_trainer_metrics_event(payload)
    assert is_completed_metrics(decoded)
    assert decoded["test_loss"] == 1.0
    assert decoded["test_ppl"] == 2.0


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        ("[]", "Expected JSON object, got list"),
        ('{"type": 1}', "Field 'type' must be a string"),
        (
            '{"type": "unknown.v1", "job_id": "j", "user_id": 1}',
            "Unknown trainer metrics event type: 'unknown.v1'",
        ),
        (
            '{"type": "trainer.metrics.config.v1"}',
            "Missing required field 'job_id'",
        ),
        (
            '{"type": "trainer.metrics.config.v1", "job_id": 1, "user_id": 1}',
            "Field 'job_id' must be a string",
        ),
        (
            '{"type": "trainer.metrics.config.v1", "job_id": "j", "user_id": 1}',
            "Missing required field 'model_family'",
        ),
        (
            '{"type": "trainer.metrics.progress.v1", "job_id": "j", "user_id": 1}',
            "Missing required field 'epoch'",
        ),
        (
            '{"type": "trainer.metrics.completed.v1", "job_id": "j", "user_id": 1}',
            "Missing required field 'test_loss'",
        ),
        (
            '{"type": "trainer.metrics.completed.v1", "job_id": "j", "user_id": 1, '
            '"test_loss": 1, "test_ppl": "x", "artifact_path": "/a"}',
            "Field 'test_ppl' must be a number",
        ),
    ],
)
def test_decode_trainer_metrics_event_invalid(payload: str, expected_message: str) -> None:
    with pytest.raises(JSONTypeError) as excinfo:
        decode_trainer_metrics_event(payload)
    assert expected_message in str(excinfo.value)
