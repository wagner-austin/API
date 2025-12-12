from __future__ import annotations

import concurrent.futures as cf
import io
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from datetime import datetime
from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image
from platform_core.errors import AppError, ErrorCode, HandwritingErrorCode
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str
from torch import Tensor

from handwriting_ai import _test_hooks
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
    limits_from_settings,
)
from handwriting_ai.inference.engine import InferenceEngine, LoadStateResult, TorchModel
from handwriting_ai.inference.manifest import ModelManifest
from handwriting_ai.inference.types import PredictOutput


def _make_future(result_fn: Callable[[float | None], PredictOutput]) -> Future[PredictOutput]:
    """Create a Future that calls result_fn when result() is called.

    Uses a real Future but executes the callable immediately when result() is called.
    This allows testing timeout behavior and exceptions.
    """

    class _DeferredFuture(Future[PredictOutput]):
        def __init__(self, fn: Callable[[float | None], PredictOutput]) -> None:
            super().__init__()
            self._fn = fn

        def result(self, timeout: float | None = None) -> PredictOutput:
            return self._fn(timeout)

    return _DeferredFuture(result_fn)


@runtime_checkable
class _RespProto(Protocol):
    @property
    def status_code(self) -> int: ...

    @property
    def text(self) -> str: ...


def _resp_obj(r: _RespProto) -> dict[str, JSONValue]:
    obj: JSONValue = load_json_str(r.text)
    if not isinstance(obj, dict):
        raise AssertionError("response JSON is not an object")
    return obj


class _DummyModel:
    def eval(self) -> Self:
        return self

    def __call__(self, x: Tensor) -> Tensor:
        return x

    def load_state_dict(self, sd: dict[str, Tensor]) -> LoadStateResult:
        return LoadStateResult((), ())

    def train(self, mode: bool = True) -> Self:
        return self

    def state_dict(self) -> dict[str, Tensor]:
        return {}

    def parameters(self) -> Sequence[nn.Parameter]:
        return []


def _mk_settings(tmp: Path, *, api_key: str = "") -> Settings:
    app_cfg: AppConfig = {
        "data_root": tmp,
        "artifacts_root": tmp,
        "logs_root": tmp,
        "threads": 1,
        "port": 8081,
    }
    digits_cfg: DigitsConfig = {
        "model_dir": tmp / "models",
        "active_model": "active",
        "tta": False,
        "uncertain_threshold": 0.5,
        "max_image_mb": 1,
        "max_image_side_px": 1024,
        "predict_timeout_seconds": 1,
        "visualize_max_kb": 64,
        "retention_keep_runs": 1,
    }
    sec_cfg: SecurityConfig = {"api_key": api_key}
    return {"app": app_cfg, "digits": digits_cfg, "security": sec_cfg}


def _png_bytes(w: int = 28, h: int = 28, color: int = 0) -> bytes:
    img = Image.new("L", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_basic_routes_and_models(tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path)
    engine = InferenceEngine(settings)
    app = create_app(settings=settings, engine_provider=lambda: engine)
    client = TestClient(app)

    # readyz not ready (no model/manifest) - returns 503 when degraded
    r1 = client.get("/readyz")
    body1 = _resp_obj(r1)
    assert r1.status_code == 503 and body1["status"] == "degraded"

    # Attach manifest and stub model to become ready
    man = ModelManifest(
        schema_version="v1.1",
        model_id="m1",
        arch="resnet18",
        n_classes=10,
        version="1.0",
        created_at=datetime.utcnow(),
        preprocess_hash="v1/grayscale+otsu+lcc+deskew{angle_conf}+center+resize28+mnistnorm",
        val_acc=0.9,
        temperature=1.0,
    )
    # Private attributes set only for readiness and metadata
    engine._manifest = man
    model: TorchModel = _DummyModel()
    engine._model = model

    r2 = client.get("/readyz")
    body2 = _resp_obj(r2)
    assert r2.status_code == 200 and body2["status"] == "ready"

    # /v1/models/active reflects manifest
    r3 = client.get("/v1/models/active")
    body3 = _resp_obj(r3)
    assert r3.status_code == 200 and body3["model_loaded"] is True
    assert body3["model_id"] == "m1"

    # Health endpoint
    assert client.get("/healthz").status_code == 200


def test_read_success_uncertain_and_visual(tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path)
    engine = InferenceEngine(settings)

    probs_tuple = tuple(0.1 for _ in range(10))
    out: PredictOutput = {"digit": 3, "confidence": 0.4, "probs": probs_tuple, "model_id": "m"}

    def _result_fn(_t: float | None) -> PredictOutput:
        return out

    fut: Future[PredictOutput] = _make_future(_result_fn)

    def _submit_predict_stub(preprocessed: Tensor) -> Future[PredictOutput]:
        _ = preprocessed  # Unused
        return fut

    _test_hooks.submit_predict_override = _submit_predict_stub

    app = create_app(settings=settings, engine_provider=lambda: engine)
    client = TestClient(app)
    files = {"file": ("a.png", _png_bytes(), "image/png")}
    r = client.post("/v1/read?visualize=true", files=files)
    assert r.status_code == 200
    body = _resp_obj(r)
    assert body["digit"] == 3 and body["uncertain"] is True
    assert isinstance(body["visual_png_b64"], str) and len(body["visual_png_b64"]) > 0


def test_read_timeout_and_not_ready(tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path)
    engine = InferenceEngine(settings)

    # Timeout path
    def _raise_timeout(_: float | None) -> PredictOutput:
        raise cf.TimeoutError()

    fut_timeout: Future[PredictOutput] = _make_future(_raise_timeout)

    def _submit_timeout(preprocessed: Tensor) -> Future[PredictOutput]:
        _ = preprocessed  # Unused
        return fut_timeout

    _test_hooks.submit_predict_override = _submit_timeout
    app1 = create_app(settings=settings, engine_provider=lambda: engine)
    c1 = TestClient(app1)
    r1 = c1.post("/v1/read", files={"file": ("a.png", _png_bytes(), "image/png")})
    r1_body = _resp_obj(r1)
    assert r1.status_code == 504 and r1_body["code"] == ErrorCode.TIMEOUT

    # Not ready path
    def _raise_not_ready(_: float | None) -> PredictOutput:
        raise RuntimeError("Model not loaded")

    fut_not_ready: Future[PredictOutput] = _make_future(_raise_not_ready)

    def _submit_not_ready(preprocessed: Tensor) -> Future[PredictOutput]:
        _ = preprocessed  # Unused
        return fut_not_ready

    _test_hooks.submit_predict_override = _submit_not_ready
    app2 = create_app(settings=settings, engine_provider=lambda: engine)
    c2 = TestClient(app2)
    r2 = c2.post("/v1/read", files={"file": ("a.png", _png_bytes(), "image/png")})
    r2_body = _resp_obj(r2)
    assert r2.status_code == 503 and r2_body["code"] == ErrorCode.SERVICE_UNAVAILABLE


def test_read_validation_paths(tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path)
    engine = InferenceEngine(settings)
    # Successful future (not used when validation fails)
    probs_val = tuple(0.1 for _ in range(10))
    out: PredictOutput = {"digit": 1, "confidence": 0.9, "probs": probs_val, "model_id": "m"}

    def _return_out(_: float | None) -> PredictOutput:
        return out

    def _submit_success(preprocessed: Tensor) -> Future[PredictOutput]:
        _ = preprocessed  # Unused
        return _make_future(_return_out)

    _test_hooks.submit_predict_override = _submit_success
    app = create_app(settings=settings, engine_provider=lambda: engine)
    client = TestClient(app)

    # Unsupported content type
    r0 = client.post("/v1/read", files={"file": ("x.txt", b"abc", "text/plain")})
    assert r0.status_code == 415

    # Extra unexpected multipart field
    files_extra = {"file": ("a.png", _png_bytes(), "image/png")}
    data_extra = {"junk": "x"}
    r1 = client.post("/v1/read", files=files_extra, data=data_extra)
    r1_body = _resp_obj(r1)
    assert r1.status_code == 400
    assert r1_body["code"] == HandwritingErrorCode.malformed_multipart.value

    # Multiple file parts not allowed
    files_multi = [
        ("file", ("a.png", _png_bytes(), "image/png")),
        ("file", ("b.png", _png_bytes(), "image/png")),
    ]
    r2 = client.post("/v1/read", files=files_multi)
    r2_body = _resp_obj(r2)
    assert r2.status_code == 400
    assert r2_body["code"] == HandwritingErrorCode.malformed_multipart.value

    # File body too large
    too_big = b"0" * (limits_from_settings(settings)["max_bytes"] + 1)
    r3 = client.post("/v1/read", files={"file": ("a.png", too_big, "image/png")})
    r3_body = _resp_obj(r3)
    assert r3.status_code == 413 and r3_body["code"] == ErrorCode.PAYLOAD_TOO_LARGE

    # Invalid image bytes
    r4 = client.post("/v1/read", files={"file": ("a.png", b"not-an-image", "image/png")})
    r4_body = _resp_obj(r4)
    assert r4.status_code == 400 and r4_body["code"] == HandwritingErrorCode.invalid_image.value

    # Image dimensions too large
    small_settings: Settings = {
        "app": settings["app"],
        "digits": {
            "model_dir": settings["digits"]["model_dir"],
            "active_model": settings["digits"]["active_model"],
            "tta": False,
            "uncertain_threshold": 0.5,
            "max_image_mb": 1,
            "max_image_side_px": 8,
            "predict_timeout_seconds": 1,
            "visualize_max_kb": 64,
            "retention_keep_runs": 1,
        },
        "security": settings["security"],
    }
    app2 = create_app(settings=small_settings, engine_provider=lambda: engine)
    c2 = TestClient(app2)
    r5 = c2.post("/v1/read", files={"file": ("a.png", _png_bytes(16, 16), "image/png")})
    r5_body = _resp_obj(r5)
    assert r5.status_code == 400 and r5_body["code"] == HandwritingErrorCode.bad_dimensions.value


def test_admin_upload_activate_and_errors(tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path)
    engine = InferenceEngine(settings)
    # Active model id matches upload so engine.try_load_active may be called
    d = settings["digits"]
    settings2: Settings = {
        "app": settings["app"],
        "digits": {
            "model_dir": d["model_dir"],
            "active_model": "m2",
            "tta": d["tta"],
            "uncertain_threshold": d["uncertain_threshold"],
            "max_image_mb": d["max_image_mb"],
            "max_image_side_px": d["max_image_side_px"],
            "predict_timeout_seconds": d["predict_timeout_seconds"],
            "visualize_max_kb": d["visualize_max_kb"],
            "retention_keep_runs": d["retention_keep_runs"],
        },
        "security": settings["security"],
    }
    app = create_app(settings=settings2, engine_provider=lambda: engine)
    client = TestClient(app)

    # Valid manifest JSON
    from handwriting_ai.preprocess import preprocess_signature

    man = {
        "schema_version": "v1.1",
        "model_id": "m2",
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0",
        "created_at": datetime.utcnow().isoformat(),
        "preprocess_hash": preprocess_signature(),
        "val_acc": 0.9,
        "temperature": 1.0,
    }
    man_bytes = dump_json_str(man).encode("utf-8")

    # Set hooks for loader and validator to succeed
    def _fake_load(path: Path) -> dict[str, Tensor]:
        _ = path  # Unused
        return {"fc.weight": torch.zeros(1), "fc.bias": torch.zeros(1)}

    def _fake_validate(sd: dict[str, Tensor], arch: str, n_classes: int) -> None:
        _ = (sd, arch, n_classes)  # Unused

    _test_hooks.load_state_dict_file = _fake_load
    _test_hooks.validate_state_dict = _fake_validate

    files = {
        "manifest": ("manifest.json", man_bytes, "application/json"),
        "model": ("model.pt", b"weights", "application/octet-stream"),
    }
    data = {"model_id": "m2", "activate": "true"}
    r_ok = client.post("/v1/admin/models/upload", files=files, data=data)
    ok_body = _resp_obj(r_ok)
    assert r_ok.status_code == 200 and ok_body["ok"] is True

    # Invalid manifest JSON
    files_bad = {
        "manifest": ("manifest.json", b"not-json", "application/json"),
        "model": ("model.pt", b"weights", "application/octet-stream"),
    }
    r_bad = client.post("/v1/admin/models/upload", files=files_bad, data=data)
    bad_body = _resp_obj(r_bad)
    assert r_bad.status_code == 400
    assert bad_body["code"] == HandwritingErrorCode.preprocessing_failed.value

    # Activate False with empty model bytes → invalid_model
    files_empty = {
        "manifest": ("manifest.json", man_bytes, "application/json"),
        "model": ("model.pt", b"", "application/octet-stream"),
    }
    data_empty = {"model_id": "m2", "activate": "false"}
    r_empty = client.post("/v1/admin/models/upload", files=files_empty, data=data_empty)
    empty_body = _resp_obj(r_empty)
    assert r_empty.status_code == 400
    assert empty_body["code"] == HandwritingErrorCode.invalid_model.value


def test_exception_handlers_and_api_key(tmp_path: Path) -> None:
    # Require API key
    settings = _mk_settings(tmp_path, api_key="secret")
    engine = InferenceEngine(settings)
    probs_zeros = tuple(0.0 for _ in range(10))
    out: PredictOutput = {"digit": 0, "confidence": 1.0, "probs": probs_zeros, "model_id": "m"}

    def _return_out2(_: float | None) -> PredictOutput:
        return out

    def _submit_success2(preprocessed: Tensor) -> Future[PredictOutput]:
        _ = preprocessed  # Unused
        return _make_future(_return_out2)

    _test_hooks.submit_predict_override = _submit_success2
    app = create_app(settings=settings, engine_provider=lambda: engine)

    def _boom_app() -> None:
        raise AppError(HandwritingErrorCode.invalid_image, "bad", 400)

    def _boom_generic() -> None:
        raise RuntimeError("fail")

    app.add_api_route("/boom-app", _boom_app, methods=["GET"])
    app.add_api_route("/boom-generic", _boom_generic, methods=["GET"])

    client = TestClient(app, raise_server_exceptions=False)
    files = {"file": ("a.png", _png_bytes(), "image/png")}
    # Unauthenticated should fail
    r_unauth = client.post("/v1/read", files=files)
    assert r_unauth.status_code == 401
    # Authenticated works
    r_auth = client.post("/v1/read", files=files, headers={"X-API-Key": "secret"})
    assert r_auth.status_code == 200

    e1 = client.get("/boom-app")
    e1_body = _resp_obj(e1)
    assert e1.status_code == 400 and e1_body["code"] == HandwritingErrorCode.invalid_image.value
    e2 = client.get("/boom-generic")
    e2_body = _resp_obj(e2)
    assert e2.status_code == 500 and e2_body["code"] == ErrorCode.INTERNAL_ERROR


def test_optional_reloader_handlers(tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path)
    engine = InferenceEngine(settings)
    app = create_app(
        settings=settings, engine_provider=lambda: engine, reload_interval_seconds=0.05
    )
    # Manually invoke start/stop handlers
    _debug_invoke_reloader_start(app)
    time.sleep(0.12)
    _debug_invoke_reloader_stop(app)
    # We cannot introspect private state; ensure the app still responds
    client = TestClient(app)
    assert client.get("/healthz").status_code == 200

    # Interval <= 0: handlers not registered, calls should be no-op
    app2 = create_app(
        settings=settings, engine_provider=lambda: engine, reload_interval_seconds=0.0
    )
    _debug_invoke_reloader_start(app2)
    _debug_invoke_reloader_stop(app2)
