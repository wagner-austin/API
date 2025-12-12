from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from platform_core.errors import AppError, HandwritingErrorCode

from handwriting_ai.api.main import create_app
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings


def _mk_app(tmp_path: Path) -> FastAPI:
    app_cfg: AppConfig = {"threads": 0, "port": 8081}
    dig_cfg: DigitsConfig = {
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
    sec_cfg: SecurityConfig = {"api_key": ""}
    s: Settings = {"app": app_cfg, "digits": dig_cfg, "security": sec_cfg}
    app = create_app(s)

    # Add dynamic routes to trigger handlers
    def _r1() -> None:
        raise AppError(HandwritingErrorCode.invalid_image, "bad", 400)

    app.add_api_route("/raise-app-error", _r1, methods=["GET"])

    def _r2() -> None:
        raise RuntimeError("boom")

    app.add_api_route("/raise-exception", _r2, methods=["GET"])

    return app


def test_app_error_handler_shapes_body(tmp_path: Path) -> None:
    app = _mk_app(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/raise-app-error")
    assert r.status_code == 400
    assert '"code":"invalid_image"' in r.text and '"request_id"' in r.text


def test_unexpected_handler_shapes_body(tmp_path: Path) -> None:
    app = _mk_app(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/raise-exception")
    assert r.status_code == 500
    assert '"code":"INTERNAL_ERROR"' in r.text and '"request_id"' in r.text
