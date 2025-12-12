from __future__ import annotations

from handwriting_ai import _test_hooks
from handwriting_ai.training.calibration import measure as meas


def test_resolve_worker_context_returns_none_for_zero_workers() -> None:
    """Test that _resolve_worker_context returns None for num_workers=0."""
    result = meas._resolve_worker_context(0)
    assert result is None


def test_resolve_worker_context_returns_spawn_on_windows() -> None:
    """Test that Windows uses spawn."""
    _test_hooks.os_name = "nt"
    result = meas._resolve_worker_context(1)
    assert result == "spawn"


def test_resolve_worker_context_prefers_forkserver() -> None:
    """POSIX branch prefers forkserver over spawn when available."""

    def _fake_get_all_start_methods() -> list[str]:
        return ["forkserver", "spawn"]

    _test_hooks.os_name = "posix"
    _test_hooks.mp_get_all_start_methods = _fake_get_all_start_methods

    result = meas._resolve_worker_context(1)
    assert result == "forkserver"


def test_resolve_worker_context_falls_back_to_spawn() -> None:
    """POSIX branch falls back to spawn when forkserver not available."""

    def _fake_get_all_start_methods() -> list[str]:
        return ["spawn"]

    _test_hooks.os_name = "posix"
    _test_hooks.mp_get_all_start_methods = _fake_get_all_start_methods

    result = meas._resolve_worker_context(1)
    assert result == "spawn"


def test_resolve_worker_context_no_supported_method() -> None:
    """POSIX branch returns None when no forkserver/spawn methods are available."""

    def _fake_get_all_start_methods() -> list[str]:
        return []

    _test_hooks.os_name = "posix"
    _test_hooks.mp_get_all_start_methods = _fake_get_all_start_methods

    result = meas._resolve_worker_context(1)
    assert result is None
