from __future__ import annotations

import pytest
from platform_core.config import _test_hooks as config_test_hooks
from platform_core.json_utils import JSONTypeError, JSONValue
from platform_core.testing import make_fake_env

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import MemoryGuardConfigDict, ResourceLimitsDict
from handwriting_ai.jobs import digits


class _FakeJob:
    """Fake RQ job object with origin attribute."""

    def __init__(self, origin: str) -> None:
        self.origin = origin
        self.id: str | None = "fake-job-id"


def test_get_env_returns_default_when_missing() -> None:
    # Configure fake env to return None for NOT_SET
    env = make_fake_env({})  # Empty env - no NOT_SET key
    config_test_hooks.get_env = env
    assert digits._get_env("NOT_SET", "fallback") == "fallback"


def test_decode_payload_rejects_invalid_types() -> None:
    bad_payload: dict[str, JSONValue] = {"type": "wrong"}
    with pytest.raises(JSONTypeError):
        _ = digits._decode_payload(bad_payload)

    bad_bool_payload: dict[str, JSONValue] = {
        "type": "digits.train.v1",
        "request_id": "r",
        "user_id": True,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
        "augment": False,
        "notes": None,
    }
    with pytest.raises(JSONTypeError):
        _ = digits._decode_payload(bad_bool_payload)


def test_build_config_event_includes_limits() -> None:
    from platform_core.digits_metrics_events import DigitsConfigV1, is_config

    def _fake_limits() -> ResourceLimitsDict:
        return {
            "cpu_cores": 4,
            "memory_bytes": 1024 * 1024 * 512,
            "optimal_threads": 2,
            "optimal_workers": 1,
            "max_batch_size": 8,
        }

    _test_hooks.detect_resource_limits = _fake_limits

    payload: digits.DigitsTrainJobV1 = {
        "type": "digits.train.v1",
        "request_id": "r",
        "user_id": 1,
        "model_id": "m",
        "epochs": 2,
        "batch_size": 4,
        "lr": 0.01,
        "seed": 1,
        "augment": False,
        "notes": None,
    }
    cfg = digits._build_cfg(payload)
    evt = digits._build_config_event(payload, cfg, "q")
    assert is_config(evt)
    config_evt: DigitsConfigV1 = evt
    assert config_evt["queue"] == "q"
    assert config_evt["memory_mb"] == 512


def test_summarize_training_exception_cases() -> None:
    def _fake_mg_cfg() -> MemoryGuardConfigDict:
        return {"enabled": True, "threshold_percent": 50.0, "required_consecutive": 1}

    _test_hooks.get_memory_guard_config = _fake_mg_cfg
    msg = digits._summarize_training_exception(RuntimeError("memory_pressure_guard_triggered"))
    assert "memory pressure" in msg

    err = digits._summarize_training_exception(RuntimeError("artifact upload failed"))
    assert "Artifact upload failed" in err

    generic = digits._summarize_training_exception(ValueError("x"))
    assert "ValueError" in generic
