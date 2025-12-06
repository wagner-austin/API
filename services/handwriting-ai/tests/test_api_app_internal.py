from __future__ import annotations

import io
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as _FutTimeout
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from platform_core.errors import AppError, ErrorCode, HandwritingErrorCode
from starlette.datastructures import FormData
from torch import Tensor

from handwriting_ai.api.app import (
    _debug_invoke_reloader_start,
    _debug_invoke_reloader_stop,
    create_app,
)
from handwriting_ai.api.routes.read import (
    _raise_if_too_large,
    _strict_validate_multipart,
    _validate_image_dimensions,
)
from handwriting_ai.config import (
    AppConfig,
    DigitsConfig,
    Limits,
    SecurityConfig,
    Settings,
)
from handwriting_ai.inference.engine import InferenceEngine
from handwriting_ai.inference.manifest import ModelManifest
from handwriting_ai.inference.types import PredictOutput
from handwriting_ai.preprocess import preprocess_signature

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


class _ReloaderEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.reload_calls: int = 0

    def reload_if_changed(self) -> bool:
        self.reload_calls += 1
        return True


def _base_settings(tmp_path: Path) -> Settings:
    app_cfg: AppConfig = {
        "data_root": tmp_path / "data",
        "artifacts_root": tmp_path / "artifacts",
        "logs_root": tmp_path / "logs",
        "threads": 0,
        "port": 8081,
    }
    dig_cfg: DigitsConfig = {
        "model_dir": tmp_path / "models",
        "active_model": "mnist_resnet18_v1",
        "tta": False,
        "uncertain_threshold": 0.70,
        "max_image_mb": 2,
        "max_image_side_px": 1024,
        "predict_timeout_seconds": 5,
        "visualize_max_kb": 16,
        "retention_keep_runs": 3,
    }
    sec_cfg: SecurityConfig = {"api_key": "k"}
    return {"app": app_cfg, "digits": dig_cfg, "security": sec_cfg}


def test_optional_reloader_start_and_stop(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    eng = _ReloaderEngine(settings)
    app = create_app(settings, engine_provider=lambda: eng, reload_interval_seconds=0.01)

    _debug_invoke_reloader_start(app)
    time.sleep(0.03)
    _debug_invoke_reloader_stop(app)

    assert eng.reload_calls >= 1


def test_raise_if_too_large_raises() -> None:
    limits: Limits = {"max_bytes": 4 * 1024, "max_side_px": 1024}
    raw = b"x" * (limits["max_bytes"] + 1)
    with pytest.raises(AppError):
        _raise_if_too_large(raw, limits)


def test_strict_multipart_rejects_extra_fields() -> None:
    form = FormData([("file", "x"), ("other", "v")])
    with pytest.raises(AppError):
        _strict_validate_multipart(form)


def test_validate_image_dimensions_raises(tmp_path: Path) -> None:
    img_path = tmp_path / "big.png"
    img = Image.new("L", (2048, 2048), color=0)
    img.save(img_path)
    img2 = Image.open(img_path)
    limits: Limits = {"max_bytes": 10 * 1024 * 1024, "max_side_px": 1024}

    with pytest.raises(AppError):
        _validate_image_dimensions(img2, limits)


def test_create_app_integration_healthz_readyz_version(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    app = create_app(settings, engine_provider=lambda: _ReloaderEngine(settings))
    client = TestClient(app)

    r_h = client.get("/healthz")
    r_r = client.get("/readyz")

    assert r_h.status_code == 200 and '"status":"ok"' in r_h.text.replace(" ", "")
    assert r_r.status_code == 503 and '"status":' in r_r.text  # degraded returns 503


def _png_bytes() -> bytes:
    img = Image.new("L", (28, 28), 0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _TimeoutFuture(Future[PredictOutput]):
    def result(self, timeout: float | None = None) -> PredictOutput:
        _ = timeout
        raise _FutTimeout()


class _TimeoutEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
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
        _ = preprocessed
        return _TimeoutFuture()


def test_read_timeout_maps_to_timeout_error(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    app = create_app(settings, engine_provider=lambda: _TimeoutEngine(settings))
    client = TestClient(app, raise_server_exceptions=False)
    files = {"file": ("img.png", _png_bytes(), "image/png")}
    resp = client.post("/v1/read", files=files, headers={"X-Api-Key": "k"})
    assert resp.status_code == 504  # TIMEOUT default status
    body_obj: UnknownJson = resp.json()
    if type(body_obj) is not dict:
        raise AssertionError("expected dict")
    body: dict[str, UnknownJson] = body_obj
    assert body.get("code") == ErrorCode.TIMEOUT


def test_read_rejects_when_content_length_exceeds_limit(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    app = create_app(settings)
    client = TestClient(app)
    files = {"file": ("img.png", _png_bytes(), "image/png")}
    # Forge a Content-Length larger than allowed to exercise header-based check.
    headers = {"Content-Length": str(10 * 1024 * 1024)}
    headers["X-Api-Key"] = "k"
    resp = client.post("/v1/read", files=files, headers=headers)
    assert resp.status_code == 413  # PAYLOAD_TOO_LARGE default status


def test_app_error_handler_maps_app_error(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    app = create_app(settings)

    async def _boom_app() -> None:
        raise AppError(HandwritingErrorCode.invalid_image, "fail", http_status=400)

    app.add_api_route("/boom-app", _boom_app, methods=["GET"])

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/boom-app")
    assert resp.status_code == 400  # invalid_image default status
    body_obj: UnknownJson = resp.json()
    if type(body_obj) is not dict:
        raise AssertionError("expected dict")
    body: dict[str, UnknownJson] = body_obj
    assert body.get("code") == HandwritingErrorCode.invalid_image


def test_unexpected_handler_maps_generic_exception(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    app = create_app(settings)

    async def _boom_generic() -> None:
        raise RuntimeError("boom")

    app.add_api_route("/boom-generic", _boom_generic, methods=["GET"])

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/boom-generic")
    assert resp.status_code == 500
    body_obj: UnknownJson = resp.json()
    if type(body_obj) is not dict:
        raise AssertionError("expected dict")
    body: dict[str, UnknownJson] = body_obj
    assert body.get("code") == ErrorCode.INTERNAL_ERROR
