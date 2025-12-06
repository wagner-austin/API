from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from platform_core.request_context import install_request_id_middleware, request_id_var

from handwriting_ai.api.app import create_app
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings


def test_request_id_header_roundtrip() -> None:
    app_cfg: AppConfig = {
        "data_root": Path("/tmp/data"),
        "artifacts_root": Path("/tmp/artifacts"),
        "logs_root": Path("/tmp/logs"),
        "threads": 0,
        "port": 8081,
    }
    dig_cfg: DigitsConfig = {
        "model_dir": Path("/tmp/models"),
        "active_model": "mnist_resnet18_v1",
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
    client = TestClient(app)
    r = client.get("/healthz", headers={"X-Request-ID": "req-123"})
    # Middleware should propagate request id to response.
    # Use string form to avoid type ambiguity.
    hs = str(r.headers)
    assert "x-request-id" in hs.lower() and "req-123" in hs


def test_request_id_header_generated_when_missing() -> None:
    """Middleware should generate a request id when client does not send one."""
    app_cfg: AppConfig = {
        "data_root": Path("/tmp/data"),
        "artifacts_root": Path("/tmp/artifacts"),
        "logs_root": Path("/tmp/logs"),
        "threads": 0,
        "port": 8081,
    }
    dig_cfg: DigitsConfig = {
        "model_dir": Path("/tmp/models"),
        "active_model": "mnist_resnet18_v1",
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
    client = TestClient(app)
    r = client.get("/healthz")
    hs = {k.lower(): v for k, v in r.headers.items()}
    assert "x-request-id" in hs and isinstance(hs["x-request-id"], str) and hs["x-request-id"]


def test_request_id_middleware_resets_context_on_exception() -> None:
    app = FastAPI()
    install_request_id_middleware(app)
    seen: list[str | None] = []

    async def boom() -> None:
        seen.append(request_id_var.get())
        raise RuntimeError("boom")

    app.add_api_route("/boom", boom, methods=["GET"])

    client = TestClient(app, raise_server_exceptions=False)

    r1 = client.get("/boom", headers={"X-Request-ID": "req-abc"})
    assert r1.status_code == 500
    assert seen[0] == "req-abc"

    r2 = client.get("/boom", headers={"X-Request-ID": "req-def"})
    assert r2.status_code == 500
    assert seen[1] == "req-def"
