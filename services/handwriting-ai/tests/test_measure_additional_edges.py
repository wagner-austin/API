from __future__ import annotations

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import MultiprocessingChildProtocol
from handwriting_ai.training.calibration import measure as m


def test_reset_calibration_state_no_cuda_attr() -> None:
    # Set up hook to do nothing for torch_cuda_empty_cache
    def _noop() -> None:
        pass

    _test_hooks.torch_cuda_empty_cache = _noop
    # Should not raise when cuda empty_cache is a no-op
    m._reset_calibration_state(reset_interop=False)


def test_configure_interop_threads_once_no_attr() -> None:
    # Reset interop configured state via hook
    _test_hooks.interop_configured_setter(False)

    # Set up hook to do nothing for set_interop_threads (simulating no attr)
    def _noop(nthreads: int) -> None:
        _ = nthreads

    _test_hooks.torch_set_interop_threads = _noop
    m._configure_interop_threads_once(None)
    # After call, flag is set even if attribute absent
    assert _test_hooks.interop_configured_getter() is True


def test_join_active_children_immediate_break() -> None:
    def _empty_children() -> list[MultiprocessingChildProtocol]:
        return []

    _test_hooks.mp_active_children = _empty_children
    # Should exit loop immediately without error
    m._join_active_children()
