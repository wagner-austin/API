from __future__ import annotations

import pytest

from handwriting_ai.training.calibration import measure as m


def test_reset_calibration_state_handles_call() -> None:
    # Should not raise
    m._reset_calibration_state(reset_interop=True)


def test_configure_interop_threads_once_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force configured flag
    monkeypatch.setattr(m, "_INTEROP_CONFIGURED", True)
    # Ensure no error on repeated configuration
    m._configure_interop_threads_once(None)


def test_reset_calibration_state_calls_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"c": False}

    class _Cuda:
        def empty_cache(self) -> None:
            called["c"] = True

    class _Torch:
        def __init__(self) -> None:
            self.cuda = _Cuda()

    monkeypatch.setattr(m, "torch", _Torch(), raising=False)
    m._reset_calibration_state(reset_interop=False)
    assert called["c"] is True


def test_configure_interop_threads_once_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"ok": False, "n": 0}

    class _Torch2:
        def set_num_interop_threads(self, n: int) -> None:
            called["ok"] = True
            called["n"] = n

    monkeypatch.setattr(m, "_INTEROP_CONFIGURED", False)
    monkeypatch.setattr(m, "torch", _Torch2(), raising=False)
    m._configure_interop_threads_once(None)
    assert called["ok"] is True and isinstance(called["n"], int)


def test_get_calibration_model_and_optimizer_cached() -> None:
    # First calls populate cache
    model1 = m._get_calibration_model()
    opt1 = m._get_calibration_optimizer(model1)
    # Subsequent calls should reuse
    model2 = m._get_calibration_model()
    opt2 = m._get_calibration_optimizer(model1)
    assert model1 is model2
    assert opt1 is opt2
