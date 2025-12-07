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


def test_decode_config_minimal_fields() -> None:
    ev = make_config_event(
        job_id="r",
        user_id=1,
        model_family="gpt2",
        model_size="small",
        total_epochs=1,
        queue="training",
    )
    out = decode_trainer_event(encode_trainer_metrics_event(ev))
    if out is None:
        raise AssertionError("expected decoded event")
    assert out["type"] == "trainer.metrics.config.v1"


def test_decode_progress_float_variants() -> None:
    ev = make_progress_metrics_event(
        job_id="r",
        user_id=1,
        epoch=1,
        total_epochs=2,
        step=10,
        train_loss=1.0,
        train_ppl=2.72,
        grad_norm=0.1,
        samples_per_sec=50.0,
    )
    out = decode_trainer_event(encode_trainer_metrics_event(ev))
    if out is None:
        raise AssertionError("expected decoded event")
    assert is_progress(out)
    assert type(out["train_loss"]) is float


def test_decode_completed_fields() -> None:
    ev = make_completed_metrics_event(
        job_id="r",
        user_id=1,
        test_loss=1.0,
        test_ppl=2.0,
        artifact_path="/x",
    )
    out = decode_trainer_event(encode_trainer_metrics_event(ev))
    if out is None:
        raise AssertionError("expected decoded event")
    assert is_completed(out)
    assert out["artifact_path"] == "/x"


def test_decode_failed_system_error() -> None:
    ev = make_failed_event(
        domain="trainer",
        job_id="r",
        user_id=1,
        error_kind="system",
        message="stop",
    )
    out = decode_trainer_event(encode_job_event(ev))
    if out is None:
        raise AssertionError("expected decoded event")
    assert is_job_failed(out)
    assert out["error_kind"] == "system"


def test_decode_unknown_type_returns_none() -> None:
    from platform_core.json_utils import dump_json_str

    u = dump_json_str({"type": "trainer.train.unknown", "foo": 1})
    assert decode_trainer_event(u) is None


logger = logging.getLogger(__name__)
