from __future__ import annotations

import base64
import io
import time
from concurrent.futures import Future
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from platform_core.errors import HandwritingErrorCode, handwriting_status_for
from platform_core.json_utils import JSONValue
from torch import Tensor, nn

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import PreprocessOptionsDict, PreprocessOutputDict
from handwriting_ai.api import main as app_mod
from handwriting_ai.api.main import create_app
from handwriting_ai.config import Settings
from handwriting_ai.inference.engine import InferenceEngine, LoadStateResult, TorchModel
from handwriting_ai.inference.manifest import ModelManifest
from handwriting_ai.inference.types import PredictOutput


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("L", (28, 28), color=0).save(buf, format="PNG")
    return buf.getvalue()


def _settings(tmp: Path) -> Settings:
    return {
        "app": {"threads": 0, "port": 8081},
        "digits": {
            "model_dir": tmp / "models",
            "active_model": "m",
            "tta": False,
            "uncertain_threshold": 0.5,
            "max_image_mb": 2,
            "max_image_side_px": 1024,
            "predict_timeout_seconds": 2,
            "visualize_max_kb": 16,
            "retention_keep_runs": 1,
        },
        "security": {"api_key": "", "api_key_enabled": False},
    }


class _ReadyEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._manifest = ModelManifest(
            schema_version="v1",
            model_id="m",
            arch="resnet18",
            n_classes=10,
            version="1.0.0",
            created_at=datetime.now(UTC),
            preprocess_hash="sig",
            val_acc=0.1,
            temperature=1.0,
        )
        self._model: TorchModel | None = self
        self.reloads = 0

    def __call__(self, x: Tensor) -> Tensor:
        return x

    def load_state_dict(self, sd: dict[str, Tensor]) -> LoadStateResult:
        return LoadStateResult((), ())

    def train(self, mode: bool = True) -> _ReadyEngine:
        return self

    def eval(self) -> _ReadyEngine:
        return self

    def state_dict(self) -> dict[str, Tensor]:
        return {}

    def parameters(self) -> tuple[nn.Parameter, ...]:
        return ()

    def submit_predict(self, preprocessed: Tensor) -> Future[PredictOutput]:
        fut: Future[PredictOutput] = Future()
        probs = tuple(0.1 for _ in range(10))
        fut.set_result(
            PredictOutput(digit=0, confidence=0.9, probs=probs, model_id="m"),
        )
        return fut

    def reload_if_changed(self) -> bool:
        self.reloads += 1
        return True


def test_read_happy_path_logs_and_returns_prediction(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    def _fake_preprocess(img: Image.Image, opts: PreprocessOptionsDict) -> PreprocessOutputDict:
        _ = (img, opts)  # unused
        png = base64.b64decode(base64.b64encode(b"x"))
        return {"tensor": Tensor(), "visual_png": png}

    _test_hooks.run_preprocess = _fake_preprocess
    engine = _ReadyEngine(settings)
    app = create_app(settings, engine_provider=lambda: engine, enforce_api_key=False)
    client = TestClient(app)
    files = {"file": ("d.png", _png_bytes(), "image/png")}
    resp = client.post("/v1/read", files=files)
    assert resp.status_code == 200
    data: JSONValue = resp.json()
    if type(data) is not dict:
        raise AssertionError("expected dict")
    assert data["digit"] == 0
    assert data["uncertain"] is False


def test_optional_reloader_handlers_can_start_and_stop() -> None:
    app = FastAPI()
    eng = _ReadyEngine(_settings(Path(".")))

    # Speed up loop and ensure thread records reloads
    interval = 0.05
    app_mod._setup_optional_reloader(app, eng, interval)
    app_mod._debug_invoke_reloader_start(app)
    time.sleep(0.12)
    app_mod._debug_invoke_reloader_stop(app)
    assert eng.reloads >= 1


def test_readyz_not_ready_and_health(tmp_path: Path) -> None:
    class _NotReadyEngine(_ReadyEngine):
        @property
        def manifest(self) -> ModelManifest | None:
            return None

        @property
        def ready(self) -> bool:
            return False

    settings = _settings(tmp_path)
    app = create_app(
        settings,
        engine_provider=lambda: _NotReadyEngine(settings),
        enforce_api_key=False,
    )
    client = TestClient(app)
    r_health = client.get("/healthz")
    assert r_health.status_code == 200
    ready_raw: JSONValue = client.get("/readyz").json()
    if type(ready_raw) is not dict:
        raise AssertionError("expected dict")
    assert ready_raw["status"] == "degraded"
    assert ready_raw["reason"] == "model not loaded"


def test_read_validation_errors(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(
        settings,
        engine_provider=lambda: _ReadyEngine(settings),
        enforce_api_key=False,
    )
    client = TestClient(app)
    files = {"file": ("d.txt", b"abc", "text/plain")}
    r = client.post("/v1/read", files=files)
    assert r.status_code == 415  # UNSUPPORTED_MEDIA_TYPE default status

    big_headers = {"Content-Length": str(settings["digits"]["max_image_mb"] * 1024 * 1024 + 10)}
    files_png = {"file": ("d.png", _png_bytes(), "image/png")}
    r2 = client.post("/v1/read", files=files_png, headers=big_headers)
    assert r2.status_code == 413  # PAYLOAD_TOO_LARGE default status

    large_img = Image.new("L", (2048, 2048), color=0)
    buf = io.BytesIO()
    large_img.save(buf, format="PNG")
    files_large = {"file": ("big.png", buf.getvalue(), "image/png")}
    small_settings = _settings(tmp_path)
    small_settings["digits"]["max_image_side_px"] = 10
    client_small = TestClient(
        create_app(
            small_settings,
            engine_provider=lambda: _ReadyEngine(small_settings),
            enforce_api_key=False,
        )
    )
    r3 = client_small.post("/v1/read", files=files_large)
    assert r3.status_code == handwriting_status_for(HandwritingErrorCode.bad_dimensions)


def test_read_runtime_error_not_loaded(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    class _ErrEngine(_ReadyEngine):
        def submit_predict(self, preprocessed: Tensor) -> Future[PredictOutput]:
            class _Fut(Future[PredictOutput]):
                def result(self, timeout: float | None = None) -> PredictOutput:
                    raise RuntimeError("Model not loaded")

            return _Fut()

    client = TestClient(
        create_app(settings, engine_provider=lambda: _ErrEngine(settings), enforce_api_key=False)
    )
    files = {"file": ("d.png", _png_bytes(), "image/png")}
    resp = client.post("/v1/read", files=files)
    assert resp.status_code == 503  # SERVICE_UNAVAILABLE default status


def test_read_timeout_cancels_future(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    class _TimeoutFuture(Future[PredictOutput]):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_called = False

        def result(self, timeout: float | None = None) -> PredictOutput:
            raise TimeoutError()

        def cancel(self) -> bool:
            self.cancel_called = True
            return True

    class _TimeoutEngine(_ReadyEngine):
        def submit_predict(self, preprocessed: Tensor) -> Future[PredictOutput]:
            return _TimeoutFuture()

    client = TestClient(
        create_app(
            settings,
            engine_provider=lambda: _TimeoutEngine(settings),
            enforce_api_key=False,
        )
    )
    files = {"file": ("d.png", _png_bytes(), "image/png")}
    resp = client.post("/v1/read", files=files)
    assert resp.status_code == 504  # TIMEOUT default status


def test_admin_upload_branches(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings["security"] = {"api_key": "k", "api_key_enabled": True}

    _test_hooks.preprocess_signature = lambda: "sig"

    app = create_app(settings, engine_provider=lambda: _ReadyEngine(settings), enforce_api_key=None)
    client = TestClient(app)

    def _post(files: dict[str, tuple[str, bytes, str]], data: dict[str, str]) -> int:
        r = client.post(
            "/v1/admin/models/upload",
            headers={"X-Api-Key": "k"},
            files=files,
            data=data,
        )
        return r.status_code

    manifest_valid = (
        b'{"schema_version":"v1","model_id":"m","arch":"resnet18","n_classes":10,'
        b'"version":"1.0.0","created_at":"2025-01-01T00:00:00Z","preprocess_hash":"sig",'
        b'"val_acc":0.1,"temperature":1.0}'
    )

    files_invalid_json = {
        "manifest": ("manifest.json", b"{bad", "application/json"),
        "model": ("m.pt", b"bytes", "application/octet-stream"),
    }
    status_invalid = _post(files_invalid_json, {"model_id": "m", "activate": "false"})
    assert status_invalid == handwriting_status_for(HandwritingErrorCode.preprocessing_failed)

    manifest_mismatch = manifest_valid.replace(b"sig", b"bad")
    files_sig_bad = {
        "manifest": ("manifest.json", manifest_mismatch, "application/json"),
        "model": ("m.pt", b"bytes", "application/octet-stream"),
    }
    assert _post(files_sig_bad, {"model_id": "m", "activate": "false"}) == handwriting_status_for(
        HandwritingErrorCode.preprocessing_failed
    )

    manifest_model_mismatch = manifest_valid.replace(b'"model_id":"m"', b'"model_id":"x"')
    files_id_bad = {
        "manifest": ("manifest.json", manifest_model_mismatch, "application/json"),
        "model": ("m.pt", b"bytes", "application/octet-stream"),
    }
    assert _post(files_id_bad, {"model_id": "m", "activate": "false"}) == handwriting_status_for(
        HandwritingErrorCode.preprocessing_failed
    )

    def _load_fail(path: Path) -> dict[str, Tensor]:
        _ = path  # Unused - just raise error
        raise ValueError("bad model")

    _test_hooks.load_state_dict_file = _load_fail
    files_activate = {
        "manifest": ("manifest.json", manifest_valid, "application/json"),
        "model": ("m.pt", b"bytes", "application/octet-stream"),
    }
    assert _post(files_activate, {"model_id": "m", "activate": "true"}) == handwriting_status_for(
        HandwritingErrorCode.invalid_model
    )

    files_empty = {
        "manifest": ("manifest.json", manifest_valid, "application/json"),
        "model": ("m.pt", b"", "application/octet-stream"),
    }
    assert _post(files_empty, {"model_id": "m", "activate": "false"}) == handwriting_status_for(
        HandwritingErrorCode.invalid_model
    )
