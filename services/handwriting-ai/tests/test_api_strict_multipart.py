from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from handwriting_ai.api.main import create_app
from handwriting_ai.config import (
    Settings,
)


def _mk_png_bytes() -> bytes:
    img = Image.new("L", (8, 8), 255)
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def test_rejects_extra_form_field() -> None:
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
    app = create_app(s)
    client = TestClient(app)
    files = {"file": ("img.png", _mk_png_bytes(), "image/png")}
    data = {"note": "extra"}
    r = client.post("/v1/read", files=files, data=data)
    assert r.status_code == 400 and '"code":"malformed_multipart"' in r.text


def test_rejects_multiple_file_parts() -> None:
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
    app = create_app(s)
    client = TestClient(app)
    files_list = [
        ("file", ("a.png", _mk_png_bytes(), "image/png")),
        ("file", ("b.png", _mk_png_bytes(), "image/png")),
    ]
    r = client.post("/v1/read", files=files_list)
    assert r.status_code == 400 and '"code":"malformed_multipart"' in r.text
