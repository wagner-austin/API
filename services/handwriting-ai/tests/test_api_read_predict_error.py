from __future__ import annotations

import io
from concurrent.futures import Future
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from torch import Tensor

from handwriting_ai.api.main import create_app
from handwriting_ai.config import (
    Settings,
)
from handwriting_ai.inference.engine import InferenceEngine
from handwriting_ai.inference.manifest import ModelManifest
from handwriting_ai.inference.types import PredictOutput
from handwriting_ai.preprocess import preprocess_signature

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


def _png_bytes() -> bytes:
    img = Image.new("L", (28, 28), 0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _ErrEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        # Provide a manifest so service_not_ready is not triggered
        self._manifest = ModelManifest(
            schema_version="v1.1",
            model_id="m",
            arch="resnet18",
            n_classes=10,
            version="1.0.0",
            created_at=datetime.now(UTC),
            preprocess_hash=preprocess_signature(),
            val_acc=0.0,
            temperature=1.0,
        )

    def submit_predict(self, preprocessed: Tensor) -> Future[PredictOutput]:
        f: Future[PredictOutput] = Future()
        f.set_exception(RuntimeError("boom"))
        return f


def test_predict_internal_error_500() -> None:
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
    app = create_app(s, engine_provider=lambda: _ErrEngine(s))
    client = TestClient(app, raise_server_exceptions=False)
    files = {"file": ("img.png", _png_bytes(), "image/png")}
    r = client.post("/v1/read", files=files)
    assert r.status_code == 500
    body_obj: UnknownJson = r.json()
    if type(body_obj) is not dict:
        raise AssertionError("expected dict")
    body: dict[str, UnknownJson] = body_obj
    assert body.get("code") == "INTERNAL_ERROR"
    assert type(body.get("request_id")) is str
