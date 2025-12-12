from __future__ import annotations

from handwriting_ai import _test_hooks
from handwriting_ai.training.calibration import measure as m


def test_reset_calibration_state_handles_call() -> None:
    # Should not raise
    m._reset_calibration_state(reset_interop=True)


def test_configure_interop_threads_once_idempotent() -> None:
    # Force configured flag via hook
    _test_hooks.interop_configured_setter(True)
    # Ensure no error on repeated configuration
    m._configure_interop_threads_once(None)
    # Clean up
    _test_hooks.interop_configured_setter(False)


def test_reset_calibration_state_calls_cuda() -> None:
    called = {"c": False}

    def _fake_empty_cache() -> None:
        called["c"] = True

    _test_hooks.torch_cuda_empty_cache = _fake_empty_cache
    m._reset_calibration_state(reset_interop=False)
    assert called["c"] is True


def test_configure_interop_threads_once_calls() -> None:
    called = {"ok": False, "n": 0}

    def _fake_set_interop(nthreads: int) -> None:
        called["ok"] = True
        called["n"] = nthreads

    # Reset flag to force actual configuration via hook
    _test_hooks.interop_configured_setter(False)
    _test_hooks.torch_set_interop_threads = _fake_set_interop
    m._configure_interop_threads_once(None)
    assert called["ok"] is True and isinstance(called["n"], int)
    # Clean up
    _test_hooks.interop_configured_setter(False)


def test_get_calibration_model_and_optimizer_cached() -> None:
    # First calls populate cache
    model1 = m._get_calibration_model()
    opt1 = m._get_calibration_optimizer(model1)
    # Subsequent calls should reuse
    model2 = m._get_calibration_model()
    opt2 = m._get_calibration_optimizer(model1)
    assert model1 is model2
    assert opt1 is opt2
