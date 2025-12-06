from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from platform_core.errors import AppError, ErrorCode
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.json_utils import JSONValue, load_json_str
from platform_core.request_context import install_request_id_middleware


def test_error_handler_includes_request_id() -> None:
    app = FastAPI()
    install_exception_handlers_fastapi(app, logger_name="test")
    install_request_id_middleware(app)

    def boom() -> None:
        raise AppError(ErrorCode.INVALID_INPUT, "nope")

    app.add_api_route("/boom", boom, methods=["GET"])  # avoid decorator type inference pitfalls

    client = TestClient(app)
    r = client.get("/boom", headers={"X-Request-ID": "abc"})
    assert r.status_code == 400

    obj_raw = load_json_str(r.text)
    assert isinstance(obj_raw, dict) and "code" in obj_raw
    obj: dict[str, JSONValue] = obj_raw
    code_o: JSONValue = obj.get("code")
    rid_o: JSONValue = obj.get("request_id")
    assert isinstance(code_o, str) and code_o == "INVALID_INPUT"
    assert isinstance(rid_o, str) and rid_o == "abc"
