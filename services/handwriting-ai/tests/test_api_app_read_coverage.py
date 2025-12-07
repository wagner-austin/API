from __future__ import annotations

import io
from concurrent.futures import Future
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from platform_core.json_utils import JSONValue, load_json_bytes
from torch import Tensor

from handwriting_ai.api.app import create_app
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import InferenceEngine
from handwriting_ai.inference.manifest import ModelManifest
from handwriting_ai.inference.types import PredictOutput
from handwriting_ai.preprocess import preprocess_signature


def _settings(tmp: Path, *, api_key_enabled: bool) -> Settings:
    app_cfg: AppConfig = {
        "data_root": tmp / "data",
        "artifacts_root": tmp / "artifacts",
        "logs_root": tmp / "logs",
        "threads": 0,
        "port": 8081,
    }
    # Keep small limits for fast tests
    dig_cfg: DigitsConfig = {
        "model_dir": tmp / "models",
        "seed_root": tmp / "seed",  # Isolated seed path (empty = no seeding)
        "active_model": "m",
        "tta": False,
        "uncertain_threshold": 0.2,
        "max_image_mb": 1,
        "max_image_side_px": 1024,
        "predict_timeout_seconds": 2,
        "visualize_max_kb": 16,
        "retention_keep_runs": 1,
        "allowed_hosts": frozenset(),
    }
    sec_cfg: SecurityConfig = {
        "api_key": "k" if api_key_enabled else "",
        "api_key_enabled": api_key_enabled,
    }
    return {"app": app_cfg, "digits": dig_cfg, "security": sec_cfg}


class _StubPredictEngine(InferenceEngine):
    """Engine that bypasses model execution and returns a fixed prediction."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        man: ModelManifest = {
            "schema_version": "v1",
            "model_id": "m",
            "arch": "resnet18",
            "n_classes": 10,
            "version": "1.0.0",
            "created_at": datetime.now(UTC),
            "preprocess_hash": preprocess_signature(),
            "val_acc": 0.5,
            "temperature": 1.0,
        }
        self._manifest = man

    def submit_predict(self, preprocessed: Tensor) -> Future[PredictOutput]:
        fut: Future[PredictOutput] = Future()
        out: PredictOutput = {
            "digit": 3,
            "confidence": 0.9,
            "probs": tuple(0.1 for _ in range(10)),
            "model_id": "m",
        }
        fut.set_result(out)
        return fut


def _png_bytes(size: tuple[int, int] = (28, 28)) -> bytes:
    img = Image.new("L", size, color=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_ready_and_model_active_none(tmp_path: Path) -> None:
    s = _settings(tmp_path, api_key_enabled=False)
    # Engine without manifest to exercise not_ready and model_active-none branches
    engine = InferenceEngine(s)
    app = create_app(s, engine_provider=lambda: engine, enforce_api_key=False)
    with TestClient(app) as client:
        resp_h = client.get("/healthz")
        body_h: JSONValue = load_json_bytes(resp_h.content)
        assert resp_h.status_code == 200
        assert isinstance(body_h, dict) and body_h.get("status") == "ok"

        resp_r = client.get("/readyz")
        body_r: JSONValue = load_json_bytes(resp_r.content)
        assert resp_r.status_code == 503  # degraded returns 503
        if type(body_r) is not dict:
            raise AssertionError("expected dict")
        assert body_r.get("status") == "degraded"
        # Also covers /v1/models/active when manifest is None
        resp_m = client.get("/v1/models/active")
        body_m: JSONValue = load_json_bytes(resp_m.content)
        assert resp_m.status_code == 200
        if type(body_m) is not dict:
            raise AssertionError("expected dict")
        assert body_m.get("model_loaded") is False and body_m.get("model_id") is None


def test_read_happy_and_error_branches(tmp_path: Path) -> None:
    s = _settings(tmp_path, api_key_enabled=False)
    engine = _StubPredictEngine(s)
    app = create_app(s, engine_provider=lambda: engine, enforce_api_key=False)
    with TestClient(app) as client:
        # Happy path (PNG)
        good = {"file": ("img.png", _png_bytes(), "image/png")}
        ok = client.post("/v1/read", files=good)
        assert ok.status_code == 200
        ok_body: JSONValue = load_json_bytes(ok.content)
        if type(ok_body) is not dict:
            raise AssertionError("expected dict")
        assert ok_body.get("digit") == 3 and ok_body.get("model_id") == "m"

        # Unsupported content type
        bad_ct = {"file": ("x.txt", b"abc", "text/plain")}
        resp_ct = client.post("/v1/read", files=bad_ct)
        assert resp_ct.status_code == 415

        # Content-Length too large (set max to 0 via header)
        too_large = {"file": ("img.png", _png_bytes(), "image/png")}
        resp_len = client.post("/v1/read", files=too_large, headers={"Content-Length": "9999999"})
        # Header is larger than configured limit so reject
        assert resp_len.status_code == 413

        # Invalid image bytes
        bad_img = {"file": ("img.png", b"not a real png", "image/png")}
        resp_img = client.post("/v1/read", files=bad_img)
        assert resp_img.status_code == 400

        # Bad dimensions
        huge = {"file": ("img.png", _png_bytes((2048, 2048)), "image/png")}
        resp_dim = client.post("/v1/read", files=huge)
        assert resp_dim.status_code == 400


def test_admin_upload_error_branches_and_seed_startup(tmp_path: Path) -> None:
    # Build settings with API disabled for simple POSTs
    s = _settings(tmp_path, api_key_enabled=False)
    app = create_app(s, enforce_api_key=False)
    # Trigger startup event to exercise seeding handler (no seed present -> noop path)
    with TestClient(app) as client:
        # Invalid JSON manifest
        files_bad_manifest = {
            "manifest": ("manifest.json", b"{", "application/json"),
            "model": ("model.pt", b"data", "application/octet-stream"),
        }
        resp_bad_manifest = client.post(
            "/v1/admin/models/upload",
            files=files_bad_manifest,
            data={"model_id": "m", "activate": "false"},
        )
        assert resp_bad_manifest.status_code == 400

        # Preprocess signature mismatch
        manifest_mismatch = (
            b'{"schema_version":"v1","model_id":"m","arch":"resnet18","n_classes":10,'
            b'"version":"1.0.0","created_at":"2020-01-01T00:00:00Z","preprocess_hash":"wrong",'
            b'"val_acc":0.1,"temperature":1.0}'
        )
        files_sig = {
            "manifest": ("manifest.json", manifest_mismatch, "application/json"),
            "model": ("model.pt", b"data", "application/octet-stream"),
        }
        resp_sig = client.post(
            "/v1/admin/models/upload", files=files_sig, data={"model_id": "m", "activate": "false"}
        )
        assert resp_sig.status_code == 400

        # Model id mismatch
        manifest_body = (
            b'{"schema_version":"v1","model_id":"wrong","arch":"resnet18","n_classes":10,'
            b'"version":"1.0.0","created_at":"2020-01-01T00:00:00Z",'
            b'"preprocess_hash":"' + preprocess_signature().encode("utf-8") + b'",'
            b'"val_acc":0.1,"temperature":1.0}'
        )
        files_id = {
            "manifest": ("manifest.json", manifest_body, "application/json"),
            "model": ("model.pt", b"data", "application/octet-stream"),
        }
        resp_id = client.post(
            "/v1/admin/models/upload", files=files_id, data={"model_id": "m", "activate": "false"}
        )
        assert resp_id.status_code == 400

        # Activate=True with invalid model bytes -> invalid_model
        manifest_ok = (
            b'{"schema_version":"v1","model_id":"m","arch":"resnet18","n_classes":10,'
            b'"version":"1.0.0","created_at":"2020-01-01T00:00:00Z",'
            b'"preprocess_hash":"' + preprocess_signature().encode("utf-8") + b'",'
            b'"val_acc":0.1,"temperature":1.0}'
        )
        files_bad_model = {
            "manifest": ("manifest.json", manifest_ok, "application/json"),
            "model": ("model.pt", b"not a torch state dict", "application/octet-stream"),
        }
        resp_bad_model = client.post(
            "/v1/admin/models/upload",
            files=files_bad_model,
            data={"model_id": "m", "activate": "true"},
        )
        assert resp_bad_model.status_code == 400
