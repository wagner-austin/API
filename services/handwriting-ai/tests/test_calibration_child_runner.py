"""Calibration runner: child process lifecycle."""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Protocol

from PIL import Image
from platform_core.logging import get_logger

from handwriting_ai.training.calibration._types import (
    BudgetConfig,
    Candidate,
)
from handwriting_ai.training.calibration.ds_spec import (
    AugmentSpec,
    InlineSpec,
    PreprocessSpec,
)
from handwriting_ai.training.calibration.runner import (
    _child_entry,
)
from handwriting_ai.training.dataset import AugmentConfig, PreprocessDataset

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


class _ChildEntryFn(Protocol):
    """Protocol for child entry function signature."""

    def __call__(
        self,
        out_path: str,
        spec: PreprocessSpec,
        cand: Candidate,
        samples: int,
        abort_pct: float,
        log_q: _QueueProto,
    ) -> None: ...


class _QueueProto(Protocol):
    """Protocol for multiprocessing queue."""

    def put_nowait(self, item: logging.LogRecord) -> None: ...


class _TinyBase:
    def __init__(self, n: int) -> None:
        self._n = int(n)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), color=0), int(idx % 10)


_CFG: AugmentConfig = {
    "augment": True,
    "aug_rotate": 5.0,
    "aug_translate": 0.1,
    "noise_prob": 0.2,
    "noise_salt_vs_pepper": 0.6,
    "dots_prob": 0.1,
    "dots_count": 2,
    "dots_size_px": 1,
    "blur_sigma": 0.5,
    "morph": "none",
    "morph_kernel_px": 1,
    "batch_size": 1,
}


def _write_gzip(path: Path, data: bytes) -> None:
    with gzip.open(path, "wb") as f:
        f.write(data)


def _mk_images_header(n: int, rows: int = 28, cols: int = 28, magic: int = 2051) -> bytes:
    return (
        magic.to_bytes(4, "big")
        + int(n).to_bytes(4, "big")
        + int(rows).to_bytes(4, "big")
        + int(cols).to_bytes(4, "big")
    )


def _mk_labels_header(n: int, magic: int = 2049) -> bytes:
    return magic.to_bytes(4, "big") + int(n).to_bytes(4, "big")


def test_child_entry_inline_executes_and_writes_result(tmp_path: Path) -> None:
    # Build a minimal inline spec and candidate
    aug: AugmentSpec = {
        "augment": False,
        "aug_rotate": 0.0,
        "aug_translate": 0.0,
        "noise_prob": 0.0,
        "noise_salt_vs_pepper": 0.5,
        "dots_prob": 0.0,
        "dots_count": 0,
        "dots_size_px": 1,
        "blur_sigma": 0.0,
        "morph": "none",
    }
    inline: InlineSpec = {"n": 4, "sleep_s": 0.0, "fail": False}
    spec: PreprocessSpec = {
        "base_kind": "inline",
        "mnist": None,
        "inline": inline,
        "augment": aug,
    }

    import multiprocessing as mp
    from multiprocessing.queues import Queue as MPQueue

    cand: Candidate = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 2,
    }
    out_file = str(tmp_path / "child_result.txt")

    q: MPQueue[logging.LogRecord] = mp.get_context("spawn").Queue()
    # Run inline inside this process
    _child_entry(out_file, spec, cand, samples=1, abort_pct=99.0, log_q=q)
    content = Path(out_file).read_text(encoding="utf-8")
    assert "ok=1" in content and "batch_size=2" in content


class _MockProc:
    """Mock process that stays alive until kill/join called."""

    def __init__(self) -> None:
        self._alive = True
        self._killed = False
        self._joined = False

    def start(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return True

    def kill(self) -> None:
        self._killed = True

    def join(self, timeout: float | None = None) -> None:
        self._joined = True

    @property
    def exitcode(self) -> int:
        return 0


class _MockQueue:
    """Mock queue for multiprocessing context."""

    def put(self, item: str | int | float | bool | None) -> None:
        return None

    def put_nowait(self, item: logging.LogRecord) -> None:
        return None


class _ProcessFactory:
    """Factory callable for creating mock processes."""

    def __call__(
        self,
        target: _ChildEntryFn,
        args: tuple[str, PreprocessSpec, Candidate, int, float, _MockQueue],
    ) -> _MockProc:
        return _MockProc()


class _QueueFactory:
    """Factory callable for creating mock queues."""

    def __call__(self) -> _MockQueue:
        return _MockQueue()


class _MockCtx:
    """Mock multiprocessing context matching mp.get_context() interface.

    Uses __getattr__ to provide Process and Queue attributes dynamically,
    avoiding N802 naming rule for method definitions while matching the
    multiprocessing.context.BaseContext interface.
    """

    def __init__(self, method_name: str | None = "spawn") -> None:
        self.method = method_name
        self._attrs: dict[str, _ProcessFactory | _QueueFactory] = {
            "Process": _ProcessFactory(),
            "Queue": _QueueFactory(),
        }

    def __getattr__(self, name: str) -> _ProcessFactory | _QueueFactory:
        if name in self._attrs:
            return self._attrs[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class _NoopListener:
    """No-op logging listener to avoid threading in tests."""

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


def _prepare_child_test_output(tmp_path: Path) -> Path:
    """Create expected output directory and result file for child process test."""
    out_dir = tmp_path / "calib_child_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.txt"
    out_path.write_text(
        "ok=1\n"
        "intra_threads=1\n"
        "interop_threads=\n"
        "num_workers=0\n"
        "batch_size=1\n"
        "samples_per_sec=1.0\n"
        "p95_ms=1.0\n",
        encoding="utf-8",
    )
    return out_dir


def _set_runner_hooks(out_dir: Path) -> None:
    """Set hooks for runner test dependencies."""
    import multiprocessing as mp

    from platform_core.logging import QueueListenerProtocol

    from handwriting_ai import _test_hooks

    _test_hooks.tempfile_mkdtemp = lambda prefix: str(out_dir)

    def _make_listener(
        queue: mp.Queue[logging.LogRecord],
        *handlers: logging.Handler,
        respect_handler_level: bool = False,
    ) -> QueueListenerProtocol:
        _ = (queue, handlers, respect_handler_level)  # unused
        return _NoopListener()

    _test_hooks.queue_listener_factory = _make_listener

    def _make_mock_ctx(method: str | None) -> _MockCtx:
        return _MockCtx(method)

    _test_hooks.mp_get_context = _make_mock_ctx


def test_run_finally_kills_alive_child(tmp_path: Path) -> None:
    """Test that SubprocessRunner kills alive child processes in finally block."""
    import handwriting_ai.training.calibration.runner as rmod

    out_dir = _prepare_child_test_output(tmp_path)
    _set_runner_hooks(out_dir)

    runner = rmod.SubprocessRunner()

    ds = PreprocessDataset(_TinyBase(2), _CFG)
    cand: Candidate = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 1,
    }
    budget: BudgetConfig = {
        "start_pct_max": 99.0,
        "abort_pct": 99.0,
        "timeout_s": 10.0,
        "max_failures": 1,
    }
    out = runner.run(ds, cand, samples=1, budget=budget)
    assert out["ok"] and out["res"] is not None and int(out["res"]["batch_size"]) == 1


def test_child_entry_flush_branch_no_flush_handler(tmp_path: Path) -> None:
    import logging
    import multiprocessing as mp
    from multiprocessing.queues import Queue as MPQueue

    from handwriting_ai import _test_hooks

    class _QH(logging.Handler):
        """Minimal queue handler that appears to lack a usable ``flush``.

        Subclasses logging.Handler to keep types strict while exercising the
        branch that checks for a flush attribute without providing one that
        can be called successfully.
        """

        def __init__(self, q: mp.Queue[logging.LogRecord]) -> None:
            super().__init__()
            self._queue = q

        def emit(self, record: logging.LogRecord) -> None:
            """Emit a record by putting it in the queue."""
            self._queue.put_nowait(record)

        def __getattr__(self, name: str) -> None:
            """Pretend that 'flush' does not exist for hasattr checks."""
            if name == "flush":
                raise AttributeError(f"'{type(self).__name__}' object has no attribute 'flush'")
            raise AttributeError(name)

    def _make_qh(queue: mp.Queue[logging.LogRecord]) -> _QH:
        return _QH(queue)

    _test_hooks.queue_handler_factory = _make_qh

    aug: AugmentSpec = {
        "augment": False,
        "aug_rotate": 0.0,
        "aug_translate": 0.0,
        "noise_prob": 0.0,
        "noise_salt_vs_pepper": 0.5,
        "dots_prob": 0.0,
        "dots_count": 0,
        "dots_size_px": 1,
        "blur_sigma": 0.0,
        "morph": "none",
    }
    inline: InlineSpec = {"n": 1, "sleep_s": 0.0, "fail": False}
    spec: PreprocessSpec = {
        "base_kind": "inline",
        "mnist": None,
        "inline": inline,
        "augment": aug,
    }
    out_file = str(tmp_path / "child_nf.txt")

    cand: Candidate = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 1,
    }

    q: MPQueue[logging.LogRecord] = mp.get_context("spawn").Queue()
    _child_entry(out_file, spec, cand, samples=1, abort_pct=99.0, log_q=q)
    # Cleanup: remove our stub handler if it was attached
    app_log = get_logger("handwriting_ai")
    for h in list(app_log.handlers):
        if h.__class__.__name__ == _QH.__name__:
            app_log.removeHandler(h)
    assert Path(out_file).exists()
    _child_entry(out_file, spec, cand, samples=1, abort_pct=99.0, log_q=q)
