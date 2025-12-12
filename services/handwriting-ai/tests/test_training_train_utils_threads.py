from __future__ import annotations

from handwriting_ai import _test_hooks
from handwriting_ai.training.mnist_train import _configure_threads


class _Cfg:
    def __init__(self, threads: int) -> None:
        self._threads = threads

    def __getitem__(self, key: str) -> int:
        return self._threads


def test_configure_threads_handles_runtimeerror() -> None:
    def _raise(nthreads: int) -> None:
        raise RuntimeError("nope")

    _test_hooks.torch_set_interop_threads = _raise
    _configure_threads(_Cfg(threads=2))
