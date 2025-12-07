from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis

from handwriting_ai.api.app import create_app
from handwriting_ai.config import Settings, ensure_settings


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


def test_readyz_degraded_no_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    import handwriting_ai.api.routes.health as health_mod

    fr = FakeRedis()

    def _rf(url: str) -> FakeRedis:
        return fr

    monkeypatch.setattr(health_mod, "redis_for_kv", _rf)
    app = create_app(settings=_mk_settings(tmp_path))
    client = TestClient(app)
    r = client.get("/readyz")
    assert r.status_code == 503
    obj_raw = load_json_str(r.text)
    if type(obj_raw) is not dict:
        raise AssertionError("expected dict")
    obj: dict[str, JSONValue] = obj_raw
    assert obj.get("status") == "degraded"
    fr.assert_only_called({"ping", "scard", "close"})
