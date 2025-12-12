from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from platform_core.errors import ErrorCode

from handwriting_ai.api.main import create_app
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings


def _make_test_settings(tmp_path: Path) -> Settings:
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
    return {"app": app_cfg, "digits": dig_cfg, "security": sec_cfg}


def test_exception_normalize_middleware_catches_exception_group(tmp_path: Path) -> None:
    """ExceptionNormalizeMiddleware catches BaseExceptionGroup and converts to AppError."""
    s = _make_test_settings(tmp_path)
    app = create_app(s)

    async def raise_exception_group() -> None:
        # Create and raise a BaseExceptionGroup
        raise ExceptionGroup("test group", [ValueError("inner error")])

    app.add_api_route("/raise-group", raise_exception_group, methods=["GET"])

    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/raise-group")

    # The middleware converts ExceptionGroup to AppError with INTERNAL_ERROR code
    assert r.status_code == 500
    body: dict[str, str | int | None] = r.json()
    assert body["code"] == ErrorCode.INTERNAL_ERROR


def test_exception_normalize_middleware_passes_through_normal_exceptions(
    tmp_path: Path,
) -> None:
    """Test that normal exceptions are not caught by ExceptionNormalizeMiddleware."""
    s = _make_test_settings(tmp_path)
    app = create_app(s)

    async def raise_runtime_error() -> None:
        raise RuntimeError("normal error")

    app.add_api_route("/raise-normal", raise_runtime_error, methods=["GET"])

    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/raise-normal")

    # Normal exceptions go through the regular handler
    assert r.status_code == 500
    body: dict[str, str | int | None] = r.json()
    assert body["code"] == ErrorCode.INTERNAL_ERROR
