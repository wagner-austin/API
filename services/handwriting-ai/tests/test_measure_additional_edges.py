from __future__ import annotations

import pytest

from handwriting_ai.training.calibration import measure as m


def test_reset_calibration_state_no_cuda_attr(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TorchNoCuda:
        pass

    monkeypatch.setattr(m, "torch", _TorchNoCuda(), raising=False)
    # Should not raise when cuda attribute is missing
    m._reset_calibration_state(reset_interop=False)


def test_configure_interop_threads_once_no_attr(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TorchNoInterop:
        pass

    monkeypatch.setattr(m, "_INTEROP_CONFIGURED", False)
    monkeypatch.setattr(m, "torch", _TorchNoInterop(), raising=False)
    m._configure_interop_threads_once(None)
    # After call, flag is set even if attribute absent
    assert m._INTEROP_CONFIGURED is True


def test_join_active_children_immediate_break(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MP:
        def active_children(self) -> list[int]:
            return []

    monkeypatch.setattr(m, "_mp", _MP(), raising=False)
    # Should exit loop immediately without error
    m._join_active_children()
