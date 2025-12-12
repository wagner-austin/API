from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import ThreadProtocol
from handwriting_ai.api.main import (
    _debug_invoke_reloader_start,
    _debug_invoke_reloader_stop,
    create_app,
)
from handwriting_ai.config import (
    AppConfig,
    DigitsConfig,
    SecurityConfig,
    Settings,
)


def test_reloader_stop_without_startup() -> None:
    app_cfg: AppConfig = {
        "data_root": Path("/tmp/data"),
        "artifacts_root": Path("/tmp/artifacts"),
        "logs_root": Path("/tmp/logs"),
        "threads": 0,
        "port": 8081,
    }
    dig: DigitsConfig = {
        "model_dir": Path("/tmp/models"),
        "active_model": "mnist_resnet18_v1",
        "tta": False,
        "uncertain_threshold": 0.70,
        "max_image_mb": 2,
        "max_image_side_px": 1024,
        "predict_timeout_seconds": 5,
        "visualize_max_kb": 16,
        "retention_keep_runs": 3,
    }
    sec: SecurityConfig = {"api_key": ""}
    s: Settings = {"app": app_cfg, "digits": dig, "security": sec}
    app = create_app(s, reload_interval_seconds=0.05)
    # Invoke stop before any start to hit both-None path
    _debug_invoke_reloader_stop(app)


def test_reloader_debug_noop_when_disabled() -> None:
    app_cfg: AppConfig = {
        "data_root": Path("/tmp/data"),
        "artifacts_root": Path("/tmp/artifacts"),
        "logs_root": Path("/tmp/logs"),
        "threads": 0,
        "port": 8081,
    }
    dig: DigitsConfig = {
        "model_dir": Path("/tmp/models"),
        "active_model": "mnist_resnet18_v1",
        "tta": False,
        "uncertain_threshold": 0.70,
        "max_image_mb": 2,
        "max_image_side_px": 1024,
        "predict_timeout_seconds": 5,
        "visualize_max_kb": 16,
        "retention_keep_runs": 3,
    }
    sec: SecurityConfig = {"api_key": ""}
    s: Settings = {"app": app_cfg, "digits": dig, "security": sec}
    # Disabled interval means no handlers registered; debug helpers should no-op
    app = create_app(s, reload_interval_seconds=0.0)
    _debug_invoke_reloader_start(app)
    _debug_invoke_reloader_stop(app)


def _boom_thread_factory(
    *,
    target: Callable[[], None],
    daemon: bool = True,
    name: str | None = None,
) -> ThreadProtocol:
    """Thread factory that always fails."""
    _ = (target, daemon, name)  # unused
    raise RuntimeError("no thread")


def test_reloader_stop_with_event_but_no_thread() -> None:
    app_cfg: AppConfig = {
        "data_root": Path("/tmp/data"),
        "artifacts_root": Path("/tmp/artifacts"),
        "logs_root": Path("/tmp/logs"),
        "threads": 0,
        "port": 8081,
    }
    dig_cfg: DigitsConfig = {
        "model_dir": Path("/tmp/models"),
        "active_model": "mnist_resnet18_v1",
        "tta": False,
        "uncertain_threshold": 0.70,
        "max_image_mb": 2,
        "max_image_side_px": 1024,
        "predict_timeout_seconds": 5,
        "visualize_max_kb": 16,
        "retention_keep_runs": 3,
    }
    sec_cfg: SecurityConfig = {"api_key": ""}
    s: Settings = {"app": app_cfg, "digits": dig_cfg, "security": sec_cfg}
    app = create_app(s, reload_interval_seconds=0.05)

    # Force thread creation to fail after event is created in the start handler
    _test_hooks.thread_factory = _boom_thread_factory
    with pytest.raises(RuntimeError):
        _debug_invoke_reloader_start(app)
    # Now stop should see stop_evt set while thread is None
    _debug_invoke_reloader_stop(app)
