from __future__ import annotations

import io
import logging
import tempfile
from concurrent.futures import Future
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from _pytest.capture import CaptureFixture
from fastapi.testclient import TestClient
from PIL import Image
from platform_core.json_utils import dump_json_str
from torch import Tensor

from handwriting_ai.api.app import create_app
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import InferenceEngine, build_fresh_state_dict
from handwriting_ai.inference.manifest import ModelManifest
from handwriting_ai.inference.types import PredictOutput
from handwriting_ai.preprocess import preprocess_signature


def _mk_png_bytes() -> bytes:
    img = Image.new("L", (64, 64), 255)
    for y in range(20, 44):
        for x in range(28, 36):
            img.putpixel((x, y), 0)
    b = BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def _make_test_settings(tmp_path: Path) -> Settings:
    app: AppConfig = {"threads": 0, "port": 8081}
    dig: DigitsConfig = {
        "model_dir": tmp_path / "models",
        "active_model": "test_model",
        "tta": False,
        "uncertain_threshold": 0.70,
        "max_image_mb": 2,
        "max_image_side_px": 1024,
        "predict_timeout_seconds": 5,
        "visualize_max_kb": 16,
        "retention_keep_runs": 3,
    }
    sec: SecurityConfig = {"api_key": ""}
    return {"app": app, "digits": dig, "security": sec}


def test_routes_health_ready_version_and_read(tmp_path: Path) -> None:
    s = _make_test_settings(tmp_path)
    app = create_app(s)
    client = TestClient(app)

    r1 = client.get("/healthz")
    assert r1.status_code == 200
    assert '"status":"ok"' in r1.text

    r2 = client.get("/readyz")
    assert r2.status_code == 503  # degraded returns 503
    assert "status" in r2.text

    # Unsupported media type
    r5 = client.post("/v1/read", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r5.status_code == 415


def test_read_positive_with_engine_override(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    s = _make_test_settings(tmp_path)
    app = create_app(s)

    # Deterministic engine that skips real model
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

        def submit_predict(self, preprocessed: Tensor) -> Future[PredictOutput]:
            f: Future[PredictOutput] = Future()
            probs = tuple(0.1 for _ in range(10))
            f.set_result(PredictOutput(digit=0, confidence=0.5, probs=probs, model_id="test_model"))
            return f

    test_engine = _T(s)
    # Provide a custom engine provider via create_app
    app = create_app(s, engine_provider=lambda: test_engine)

    client = TestClient(app)
    files = {"file": ("img.png", _mk_png_bytes(), "image/png")}
    r = client.post("/v1/read", files=files)
    assert r.status_code == 200
    assert '"digit":' in r.text
    # Response includes latency_ms, and middleware adds X-Request-ID header
    assert '"latency_ms"' in r.text
    hs = str(r.headers)
    assert "x-request-id" in hs.lower()


def test_limits_too_large_and_dimensions(tmp_path: Path) -> None:
    # Set very small max bytes to trigger 413 and small max side px to trigger 400
    app_cfg: AppConfig = {"threads": 0, "port": 8081}
    dig: DigitsConfig = {
        "model_dir": tmp_path / "models",
        "active_model": "test_model",
        "tta": False,
        "uncertain_threshold": 0.70,
        "max_image_mb": 1,
        "max_image_side_px": 128,
        "predict_timeout_seconds": 5,
        "visualize_max_kb": 16,
        "retention_keep_runs": 3,
    }
    sec: SecurityConfig = {"api_key": ""}
    s: Settings = {"app": app_cfg, "digits": dig, "security": sec}
    app = create_app(s)
    client = TestClient(app)

    # Too large: craft bytes > 1MB
    big = b"0" * (1 * 1024 * 1024 + 1)
    r1 = client.post("/v1/read", files={"file": ("big.png", big, "image/png")})
    assert r1.status_code == 413

    # Over dimension
    img = Image.new("L", (2048, 2048), 255)
    b = BytesIO()
    img.save(b, format="PNG")
    r2 = client.post("/v1/read", files={"file": ("large.png", b.getvalue(), "image/png")})
    assert r2.status_code == 400


def test_readyz_stays_not_ready_when_manifest_mismatch() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "models"
        active = "bad_model"
        active_dir = model_dir / active
        active_dir.mkdir(parents=True, exist_ok=True)
        # Write a manifest with mismatched preprocess hash
        manifest = {
            "schema_version": "v1.1",
            "model_id": active,
            "arch": "resnet18",
            "n_classes": 10,
            "version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "preprocess_hash": "v1/different",
            "val_acc": 0.0,
            "temperature": 1.0,
        }
        (active_dir / "manifest.json").write_text(dump_json_str(manifest), encoding="utf-8")
        # Touch a placeholder model file (engine should not attempt to load due to mismatch)
        (active_dir / "model.pt").write_bytes(b"\x80\x04\x95")

        dig: DigitsConfig = {
            "model_dir": model_dir,
            "active_model": active,
            "tta": False,
            "uncertain_threshold": 0.70,
            "max_image_mb": 2,
            "max_image_side_px": 1024,
            "predict_timeout_seconds": 5,
            "visualize_max_kb": 16,
            "retention_keep_runs": 3,
        }
        app_cfg: AppConfig = {"threads": 0, "port": 8081}
        sec_cfg: SecurityConfig = {"api_key": ""}
        s: Settings = {"app": app_cfg, "digits": dig, "security": sec_cfg}
        app = create_app(s)
        client = TestClient(app)
        r = client.get("/readyz")
        assert r.status_code == 503  # degraded returns 503
        assert '"status":"degraded"' in r.text


def test_models_active_and_read_with_artifact() -> None:
    # Build a minimal artifact (manifest + state_dict) and assert readiness and read path.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "digits_models"
        active = "mnist_resnet18_v1"
        active_dir = model_dir / active
        active_dir.mkdir(parents=True, exist_ok=True)

        # Create a compatible model state_dict via engine helper (avoids untyped imports)
        n_classes = 10
        sd = build_fresh_state_dict("resnet18", n_classes)
        import torch

        torch.save(sd, (active_dir / "model.pt").as_posix())

        # Write manifest matching preprocess signature
        man = {
            "schema_version": "v1.1",
            "model_id": active,
            "arch": "resnet18",
            "n_classes": n_classes,
            "version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "preprocess_hash": preprocess_signature(),
            "val_acc": 0.99,
            "temperature": 1.5,
        }
        (active_dir / "manifest.json").write_text(dump_json_str(man), encoding="utf-8")

        dig: DigitsConfig = {
            "model_dir": model_dir,
            "active_model": active,
            "tta": False,
            "uncertain_threshold": 0.70,
            "max_image_mb": 2,
            "max_image_side_px": 1024,
            "predict_timeout_seconds": 5,
            "visualize_max_kb": 16,
            "retention_keep_runs": 3,
        }
        app_cfg: AppConfig = {"threads": 0, "port": 8081}
        sec_cfg: SecurityConfig = {"api_key": ""}
        s: Settings = {"app": app_cfg, "digits": dig, "security": sec_cfg}
        app = create_app(s)
        client = TestClient(app)

        # Ready becomes ready
        r0 = client.get("/readyz")
        assert r0.status_code == 200 and '"status":"ready"' in r0.text

        # Model info carries core manifest fields
        r1 = client.get("/v1/models/active")
        assert r1.status_code == 200
        body = r1.text
        assert '"model_loaded":true' in body and f'"model_id":"{active}"' in body
        assert '"temperature":' in body and '"val_acc":' in body
        assert '"arch":"resnet18"' in body
        assert '"n_classes":10' in body
        assert '"version":"1.0.0"' in body
        assert '"created_at":' in body and '"schema_version":' in body

        # Read works end-to-end
        files = {"file": ("img.png", _mk_png_bytes(), "image/png")}
        r2 = client.post("/v1/read", files=files)
        assert r2.status_code == 200 and '"digit":' in r2.text


def test_structured_logs_on_read(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    s = _make_test_settings(tmp_path)

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

        def submit_predict(self, preprocessed: Tensor) -> Future[PredictOutput]:
            f: Future[PredictOutput] = Future()
            probs = tuple(0.1 for _ in range(10))
            f.set_result(PredictOutput(digit=0, confidence=0.5, probs=probs, model_id="test_model"))
            return f

    app = create_app(s, engine_provider=lambda: _T(s))
    client = TestClient(app)
    # Attach a dedicated handler to capture formatted JSON logs deterministically
    from platform_core.logging import JsonFormatter, get_logger

    logger = get_logger("handwriting_ai")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter(static_fields={}, extra_field_names=[]))
    logger.addHandler(handler)
    try:
        files = {"file": ("img.png", _mk_png_bytes(), "image/png")}
        _ = client.post("/v1/read", files=files)
    finally:
        logger.removeHandler(handler)
    body = buf.getvalue()
    assert '"message": "read_finished"' in body
    assert '"request_id":' in body
    assert '"latency_ms": ' in body
