from __future__ import annotations

import concurrent.futures
import io
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from platform_core.errors import ErrorCode
from platform_core.json_utils import JSONValue
from platform_core.logging import JsonFormatter, get_logger
from torch import Tensor

from handwriting_ai.api.main import create_app
from handwriting_ai.config import Settings
from handwriting_ai.inference.engine import InferenceEngine
from handwriting_ai.inference.manifest import ModelManifest
from handwriting_ai.inference.types import PredictOutput
from handwriting_ai.preprocess import preprocess_signature


def _mk_settings(tmp: Path, *, api_key: str = "") -> Settings:
    return {
        "app": {"threads": 0, "port": 8081},
        "digits": {
            "model_dir": tmp / "models",
            "active_model": "m",
            "tta": False,
            "uncertain_threshold": 0.5,
            "max_image_mb": 1,
            "max_image_side_px": 1024,
            "predict_timeout_seconds": 1,
            "visualize_max_kb": 64,
            "retention_keep_runs": 1,
        },
        "security": {"api_key": api_key, "api_key_enabled": api_key != ""},
    }


def _png_bytes() -> bytes:
    img = Image.new("L", (28, 28), color=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_read_returns_service_unavailable_when_model_missing(tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path)
    # Create engine explicitly without loading any model
    # (do not call try_load_active() to ensure model remains unloaded)
    engine = InferenceEngine(settings)
    app = create_app(settings, engine_provider=lambda: engine, enforce_api_key=False)
    client = TestClient(app)
    files = {"file": ("d.png", _png_bytes(), "image/png")}
    resp = client.post("/v1/read", files=files)
    assert resp.status_code == 503
    body: JSONValue = resp.json()
    if type(body) is not dict:
        raise AssertionError("expected dict")
    assert body["code"] == ErrorCode.SERVICE_UNAVAILABLE


def test_structured_logs_include_request_and_latency(tmp_path: Path) -> None:
    s = _mk_settings(tmp_path)

    class _T(InferenceEngine):
        def __init__(self, settings: Settings) -> None:
            super().__init__(settings)
            self._manifest = ModelManifest(
                schema_version="v1",
                model_id="test_model",
                arch="resnet18",
                n_classes=10,
                version="1.0.0",
                created_at=datetime.now(UTC),
                preprocess_hash=preprocess_signature(),
                val_acc=0.0,
                temperature=1.0,
            )

        def submit_predict(self, preprocessed: Tensor) -> concurrent.futures.Future[PredictOutput]:
            fut: concurrent.futures.Future[PredictOutput] = concurrent.futures.Future()
            probs = tuple(0.1 for _ in range(10))
            fut.set_result(
                PredictOutput(digit=0, confidence=0.5, probs=probs, model_id="test_model")
            )
            return fut

    app = create_app(s, engine_provider=lambda: _T(s), enforce_api_key=False)
    client = TestClient(app)

    logger = get_logger("handwriting_ai")
    buf = io.StringIO()
    import logging

    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter(static_fields={}, extra_field_names=[]))
    logger.addHandler(handler)
    try:
        files = {"file": ("img.png", _png_bytes(), "image/png")}
        _ = client.post("/v1/read", files=files)
    finally:
        logger.removeHandler(handler)
    body = buf.getvalue()
    normalized = body.replace(" ", "")
    assert '"message":"read_finished"' in normalized
    assert '"request_id":' in normalized
    assert '"latency_ms":' in normalized
    assert '"digit":0' in normalized
    assert '"uncertain":false' in normalized


def test_optional_reloader_not_started_when_disabled(tmp_path: Path) -> None:
    from handwriting_ai import _test_hooks
    from handwriting_ai._test_hooks import ThreadProtocol, ThreadTargetProtocol

    s = _mk_settings(tmp_path)
    started = {"count": 0}

    def _fake_engine() -> InferenceEngine:
        return InferenceEngine(s)

    class _DummyThread:
        def __init__(self) -> None:
            started["count"] += 1

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            return None

    def _fake_thread(
        *, target: ThreadTargetProtocol, daemon: bool = True, name: str | None = None
    ) -> ThreadProtocol:
        _ = (target, daemon, name)
        return _DummyThread()

    _test_hooks.thread_factory = _fake_thread

    _ = create_app(
        s, engine_provider=_fake_engine, reload_interval_seconds=0, enforce_api_key=False
    )
    assert started["count"] == 0


def test_seed_startup_skips_when_missing_files(tmp_path: Path) -> None:
    s = _mk_settings(tmp_path)
    app = create_app(s, enforce_api_key=False)
    # Using TestClient triggers startup handlers; absence of seed files should not raise
    with TestClient(app):
        pass
