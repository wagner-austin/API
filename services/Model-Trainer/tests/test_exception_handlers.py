from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from platform_core.fastapi import install_exception_handlers_fastapi


def test_unhandled_exception_handler_path() -> None:
    app = FastAPI()
    install_exception_handlers_fastapi(app, logger_name="test", request_id_var=None)

    def boom() -> None:
        raise RuntimeError("oops")

    app.add_api_route("/boom", boom, methods=["GET"])
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/boom")
    assert response.status_code == 500
    assert '"code":"INTERNAL_ERROR"' in response.text
