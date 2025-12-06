from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from handwriting_ai.api.app import create_app
from handwriting_ai.config import (
    Settings,
)


def _count_reloader_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name == "model-reloader")


def test_background_reloader_starts_and_stops_cleanly() -> None:
    s: Settings = {
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
    app = create_app(s, reload_interval_seconds=0.05)
    with TestClient(app):
        # Thread should be running shortly after startup
        for _ in range(20):
            if _count_reloader_threads() > 0:
                break
            time.sleep(0.01)
        assert _count_reloader_threads() > 0
    # On shutdown, thread should stop promptly
    for _ in range(50):
        if _count_reloader_threads() == 0:
            break
        time.sleep(0.02)
    assert _count_reloader_threads() == 0
