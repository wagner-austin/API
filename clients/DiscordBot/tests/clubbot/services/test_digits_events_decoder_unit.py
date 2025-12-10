from __future__ import annotations

import logging

from platform_core.digits_metrics_events import decode_digits_event
from platform_core.json_utils import JSONValue, dump_json_str


def test_decode_started_includes_augment_and_batch() -> None:
    payload: JSONValue = {
        "type": "digits.metrics.config.v1",
        "job_id": "r",
        "user_id": 1,
        "model_id": "m",
        "total_epochs": 2,
        "queue": "digits",
        # Optional rich context
        "cpu_cores": 2,
        "optimal_threads": 2,
        "memory_mb": 953,
        "optimal_workers": 0,
        "max_batch_size": 64,
        "device": "cpu",
        # Training config and augmentation
        "batch_size": 64,
        "augment": True,
        "aug_rotate": 10.0,
        "aug_translate": 0.1,
        "noise_prob": 0.2,
        "dots_prob": 0.1,
    }
    ev = decode_digits_event(dump_json_str(payload))
    assert ev["type"] == "digits.metrics.config.v1"
    # Ensure augmentation details are preserved
    assert ev.get("batch_size") == 64
    assert ev.get("augment") is True
    assert ev.get("aug_rotate") == 10.0
    assert ev.get("aug_translate") == 0.1
    assert ev.get("noise_prob") == 0.2
    assert ev.get("dots_prob") == 0.1


def test_decode_started_omits_unknown_optionals() -> None:
    payload: JSONValue = {
        "type": "digits.metrics.config.v1",
        "job_id": "r",
        "user_id": 1,
        "model_id": "m",
        "total_epochs": 1,
        "queue": "digits",
    }
    ev = decode_digits_event(dump_json_str(payload))
    # None of these should be present when not sent by producer
    for k in (
        "batch_size",
        "augment",
        "aug_rotate",
        "aug_translate",
        "noise_prob",
        "dots_prob",
        "device",
        "cpu_cores",
        "memory_mb",
        "optimal_threads",
        "optimal_workers",
        "max_batch_size",
    ):
        assert k not in ev


def test_decode_started_learning_rate_as_int() -> None:
    """Test that learning_rate as int is converted to float."""
    payload: JSONValue = {
        "type": "digits.metrics.config.v1",
        "job_id": "r",
        "user_id": 1,
        "model_id": "m",
        "total_epochs": 1,
        "queue": "digits",
        "learning_rate": 1,  # int instead of float
    }
    ev = decode_digits_event(dump_json_str(payload))
    assert ev.get("learning_rate") == 1.0  # Should be converted to float
    assert type(ev.get("learning_rate")) is float


logger = logging.getLogger(__name__)
