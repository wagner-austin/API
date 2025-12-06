from __future__ import annotations

import logging

from platform_core.job_events import encode_job_event, make_failed_event
from platform_core.trainer_metrics_events import (
    encode_trainer_metrics_event,
    make_completed_metrics_event,
    make_config_event,
    make_progress_metrics_event,
)
from platform_discord.trainer.handler import (
    decode_trainer_event,
    is_completed,
    is_job_failed,
    is_progress,
)


def test_decode_config_roundtrip() -> None:
    ev = make_config_event(
        job_id="r1",
        user_id=123,
        model_family="gpt2",
        model_size="small",
        total_epochs=5,
        queue="training",
        batch_size=4,
        learning_rate=5e-4,
    )
    out = decode_trainer_event(encode_trainer_metrics_event(ev))
    if out is None:
        raise AssertionError("expected decoded event")
    assert out["type"] == "trainer.metrics.config.v1"
    assert out["job_id"] == "r1"


def test_decode_progress_roundtrip() -> None:
    ev = make_progress_metrics_event(
        job_id="r1",
        user_id=123,
        epoch=1,
        total_epochs=5,
        step=10,
        loss=1.23,
    )
    out = decode_trainer_event(encode_trainer_metrics_event(ev))
    if out is None:
        raise AssertionError("expected decoded event")
    assert is_progress(out)
    assert out["epoch"] == 1
    assert out["loss"] == 1.23


def test_decode_completed_roundtrip() -> None:
    ev = make_completed_metrics_event(
        job_id="r1",
        user_id=123,
        loss=0.5,
        perplexity=2.0,
        artifact_path="/data/artifacts/models/run1",
    )
    out = decode_trainer_event(encode_trainer_metrics_event(ev))
    if out is None:
        raise AssertionError("expected decoded event")
    assert is_completed(out)
    assert out["artifact_path"].endswith("run1")


def test_decode_failed_roundtrip() -> None:
    ev = make_failed_event(
        domain="trainer",
        job_id="r1",
        user_id=123,
        error_kind="system",
        message="oom",
    )
    out = decode_trainer_event(encode_job_event(ev))
    if out is None:
        raise AssertionError("expected decoded event")
    assert is_job_failed(out)
    assert out["error_kind"] == "system"


logger = logging.getLogger(__name__)
