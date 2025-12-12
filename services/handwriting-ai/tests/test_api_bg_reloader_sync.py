from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import EventProtocol, ThreadProtocol
from handwriting_ai.api.main import (
    _debug_invoke_reloader_start,
    _debug_invoke_reloader_stop,
    create_app,
)
from handwriting_ai.config import (
    Settings,
)
from handwriting_ai.inference.engine import InferenceEngine


class _SyncEvent:
    """Synchronous stand-in for threading.Event used to control one loop iteration.

    - First call to wait() flips the flag so the background loop exits after one pass.
    """

    def __init__(self) -> None:
        self._is_set: bool = False
        self._first_wait: bool = True

    def is_set(self) -> bool:
        return self._is_set

    def set(self) -> None:
        self._is_set = True

    def wait(self, timeout: float | None = None) -> bool:
        # Flip the flag after the first wait to stop the loop promptly.
        if self._first_wait:
            self._first_wait = False
            self._is_set = True
        return True


class _SyncThread:
    """Synchronous stand-in for threading.Thread to execute target inline for coverage."""

    def __init__(
        self, *, target: Callable[[], None], daemon: bool, name: str | None = None
    ) -> None:
        self._target: Callable[[], None] = target
        self.daemon: bool = daemon
        self._name: str | None = name

    def start(self) -> None:
        # Execute inline in the current thread to ensure coverage collection.
        self._target()

    def join(self, timeout: float | None = None) -> None:
        return None


class _ReloaderEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.reload_calls: int = 0

    def reload_if_changed(self) -> bool:
        self.reload_calls += 1
        return True


def _settings() -> Settings:
    return {
        "app": {
            "data_root": Path("/tmp/data"),
            "artifacts_root": Path("/tmp/artifacts"),
            "logs_root": Path("/tmp/logs"),
            "threads": 0,
            "port": 8081,
        },
        "digits": {
            "model_dir": Path("/tmp/models"),
            "active_model": "mnist_resnet18_v1",
            "tta": False,
            "uncertain_threshold": 0.70,
            "max_image_mb": 2,
            "max_image_side_px": 1024,
            "predict_timeout_seconds": 5,
            "visualize_max_kb": 16,
            "retention_keep_runs": 3,
        },
        "security": {"api_key": ""},
    }


def _sync_event_factory() -> EventProtocol:
    return _SyncEvent()


def _sync_thread_factory(
    *,
    target: Callable[[], None],
    daemon: bool = True,
    name: str | None = None,
) -> ThreadProtocol:
    return _SyncThread(target=target, daemon=daemon, name=name)


def test_bg_reloader_loop_runs_once_with_sync_thread() -> None:
    s = _settings()
    eng = _ReloaderEngine(s)

    # Set hooks to use synchronous Event and Thread
    _test_hooks.event_factory = _sync_event_factory
    _test_hooks.thread_factory = _sync_thread_factory

    app = create_app(s, engine_provider=lambda: eng, reload_interval_seconds=0.01)

    _debug_invoke_reloader_start(app)
    _debug_invoke_reloader_stop(app)

    assert eng.reload_calls >= 1
