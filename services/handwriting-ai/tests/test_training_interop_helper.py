from __future__ import annotations

import pytest

from handwriting_ai import _test_hooks
from handwriting_ai.training.mnist_train import _configure_interop_threads


def test_configure_interop_threads_ok() -> None:
    called: dict[str, int | None] = {"v": None}

    def _ok(nthreads: int) -> None:
        called["v"] = int(nthreads)

    _test_hooks.torch_has_set_num_interop_threads = lambda: True
    _test_hooks.torch_set_interop_threads = _ok
    _configure_interop_threads(2)
    assert called["v"] == 2


def test_configure_interop_threads_raises() -> None:
    def _boom(nthreads: int) -> None:
        raise RuntimeError("boom")

    _test_hooks.torch_has_set_num_interop_threads = lambda: True
    _test_hooks.torch_set_interop_threads = _boom
    # Should raise after logging
    with pytest.raises(RuntimeError, match="boom"):
        _configure_interop_threads(1)


def test_configure_interop_threads_skips_on_none() -> None:
    # No-op when interop_threads is None
    _configure_interop_threads(None)


def test_configure_interop_threads_skips_without_attr() -> None:
    # Simulate torch not having set_num_interop_threads
    _test_hooks.torch_has_set_num_interop_threads = lambda: False
    _configure_interop_threads(2)
