from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

import handwriting_ai.jobs.digits as dj
from handwriting_ai.training.resources import ResourceLimits
from handwriting_ai.training.train_config import (
    TrainConfig,
    TrainingResult,
)

pytestmark = pytest.mark.usefixtures("digits_redis")


class _StubJobCtx:
    def __init__(self) -> None:
        self.failed_calls = 0

    def publish_started(self) -> None:
        return None

    def publish_progress(
        self, progress: int, message: str | None = None, payload: JSONValue | None = None
    ) -> None:
        return None

    def publish_completed(self, result_id: str, result_bytes: int) -> None:
        return None

    def publish_failed(self, error_kind: str, message: str) -> None:
        self.failed_calls += 1


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


def test_process_train_job_invalid_payload_fields_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_ctx = _StubJobCtx()

    def _ctx(*args: JSONValue, **kwargs: JSONValue) -> _StubJobCtx:
        return job_ctx

    monkeypatch.setattr(dj, "make_job_context", _ctx, raising=True)

    payload: dict[str, JSONValue] = {
        "type": "digits.train.v1",
        "request_id": "r1",
        "user_id": True,  # bool is explicitly rejected
        "model_id": "m1",
        "epochs": "one",
        "batch_size": 4,
        "lr": 0.001,
        "seed": 3,
        "augment": False,
        "notes": None,
    }
    with pytest.raises(JSONTypeError):
        dj._decode_and_process_train_job(payload)


def test_process_train_job_training_error_reraises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job_ctx = _StubJobCtx()

    def _ctx(*args: JSONValue, **kwargs: JSONValue) -> _StubJobCtx:
        return job_ctx

    monkeypatch.setattr(dj, "make_job_context", _ctx, raising=True)

    def _raise_run(_: TrainConfig) -> TrainingResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(dj, "_run_training", _raise_run)
    payload: dict[str, JSONValue] = {
        "type": "digits.train.v1",
        "request_id": "r1",
        "user_id": 2,
        "model_id": "m1",
        "epochs": 1,
        "batch_size": 4,
        "lr": 0.001,
        "seed": 3,
        "augment": False,
        "notes": None,
    }
    with pytest.raises(RuntimeError, match="boom"):
        dj._decode_and_process_train_job(payload)

    assert job_ctx.failed_calls == 1


def test_decode_int_and_float_field_edges() -> None:
    # _decode_int_field
    assert dj._decode_int_field({"x": 7}, "x") == 7
    assert dj._decode_int_field({"x": "8"}, "x") == 8
    with pytest.raises(JSONTypeError):
        dj._decode_int_field({"x": True}, "x")
    with pytest.raises(JSONTypeError):
        dj._decode_int_field({"x": None}, "x")
    # _decode_float_field
    assert dj._decode_float_field({"x": 1.5}, "x") == 1.5
    assert dj._decode_float_field({"x": 3}, "x") == 3.0
    assert dj._decode_float_field({"x": "4.5"}, "x") == 4.5
    with pytest.raises(JSONTypeError):
        dj._decode_float_field({"x": False}, "x")
    with pytest.raises(JSONTypeError):
        dj._decode_float_field({"x": None}, "x")


def test_summarize_training_exception_memory_pressure_guard() -> None:
    exc = RuntimeError("memory_pressure_guard_triggered")
    result = dj._summarize_training_exception(exc)
    assert "Training aborted" in result


def test_summarize_training_exception_artifact_upload_failed() -> None:
    exc = RuntimeError("artifact upload failed: connection timeout")
    result = dj._summarize_training_exception(exc)
    assert result == "Artifact upload failed: upstream API error. See worker logs for details."


def test_summarize_training_exception_generic() -> None:
    exc = ValueError("invalid config")
    result = dj._summarize_training_exception(exc)
    assert result == "ValueError: invalid config"
