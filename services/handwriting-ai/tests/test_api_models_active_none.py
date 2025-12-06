from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str

from handwriting_ai.api.app import create_app
from handwriting_ai.config import Settings, ensure_settings
from handwriting_ai.inference.engine import InferenceEngine


def _mk_settings(tmp_dir: Path) -> Settings:
    base: Settings = {
        "app": {
            "data_root": tmp_dir,
            "artifacts_root": tmp_dir,
            "logs_root": tmp_dir,
            "threads": 1,
            "port": 8080,
        },
        "digits": {
            "model_dir": tmp_dir,
            "active_model": "active",
            "tta": False,
            "uncertain_threshold": 0.5,
            "max_image_mb": 1,
            "max_image_side_px": 64,
            "predict_timeout_seconds": 1,
            "visualize_max_kb": 64,
            "retention_keep_runs": 1,
        },
        "security": {"api_key": "", "api_key_enabled": False},
    }
    return ensure_settings(base, create_dirs=True)


def test_models_active_without_manifest(tmp_path: Path) -> None:
    settings = _mk_settings(tmp_path)
    engine = InferenceEngine(settings)
    app = create_app(settings=settings, engine_provider=lambda: engine)
    client = TestClient(app)
    resp = client.get("/v1/models/active")
    assert resp.status_code == 200
    obj_raw = load_json_str(resp.text)
    if type(obj_raw) is not dict:
        raise AssertionError("expected dict")
    obj: dict[str, JSONValue] = obj_raw
    assert obj.get("model_loaded") is False
    assert obj.get("model_id") is None
