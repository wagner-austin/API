from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from platform_core.http_client import HttpxResponse
from platform_core.json_utils import JSONValue, dump_json_str
from platform_core.model_trainer_client import (
    HTTPModelTrainerClient,
    ModelTrainerAPIError,
    RunStatus,
    TrainResponse,
    _extract_message,
    _load_json_dict,
    _parse_last_heartbeat,
    _parse_status_response,
    _require_optional_str,
    _require_str_field,
    build_async_client,
)


class _FakeResponse:
    def __init__(self, status: int, json_body: JSONValue | None = None, text: str | None = None):
        self.status_code = int(status)
        self._json = json_body
        if text is not None:
            self.text = text
        elif json_body is None:
            self.text = ""
        else:
            self.text = dump_json_str(json_body)
        self.headers: Mapping[str, str] = {}
        self.content: bytes | bytearray = self.text.encode("utf-8")

    def json(self) -> JSONValue:
        if self._json is None:
            raise ValueError("No JSON body")
        return self._json


class _FakeClient:
    def __init__(self, post_resp: _FakeResponse, get_resp: _FakeResponse):
        self._post_resp = post_resp
        self._get_resp = get_resp
        self.post_headers: dict[str, str] | None = None
        self.post_body: JSONValue | None = None
        self.get_headers: dict[str, str] | None = None

    async def aclose(self) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> HttpxResponse:
        self.post_headers = dict(headers)
        self.post_body = json
        _ = files  # unused in these tests
        return self._post_resp

    async def get(self, url: str, *, headers: Mapping[str, str]) -> HttpxResponse:
        self.get_headers = dict(headers)
        return self._get_resp


@pytest.mark.asyncio
async def test_train_and_status_success() -> None:
    post_resp = _FakeResponse(200, {"run_id": "r1", "job_id": "j1"})
    get_resp = _FakeResponse(
        200,
        {"run_id": "r1", "status": "running", "last_heartbeat_ts": "12.5", "message": None},
    )
    client = _FakeClient(post_resp, get_resp)
    http = HTTPModelTrainerClient(base_url="https://api", api_key="k", client=client)
    out = await http.train(
        user_id=1,
        model_family="gpt2",
        model_size="small",
        max_seq_len=128,
        num_epochs=1,
        batch_size=2,
        learning_rate=5e-4,
        corpus_path="/data/corpus",
        tokenizer_id="tok1",
        request_id="req1",
    )
    assert out["run_id"] == "r1" and out["job_id"] == "j1"
    st = await http.status(run_id="r1", request_id="req2")
    assert st["status"] == "running" and st["last_heartbeat_ts"] == 12.5
    assert client.post_headers is not None and client.get_headers is not None
    assert client.post_headers.get("X-Api-Key") == "k"
    await http.aclose()


@pytest.mark.asyncio
async def test_train_error_and_message_extraction() -> None:
    post_resp = _FakeResponse(400, {"message": "bad request"})
    get_resp = _FakeResponse(200, {"run_id": "r1", "status": "queued", "last_heartbeat_ts": None})
    http = HTTPModelTrainerClient(
        base_url="https://api", api_key=None, client=_FakeClient(post_resp, get_resp)
    )
    with pytest.raises(ModelTrainerAPIError) as exc:
        await http.train(
            user_id=1,
            model_family="gpt2",
            model_size="small",
            max_seq_len=128,
            num_epochs=1,
            batch_size=2,
            learning_rate=5e-4,
            corpus_path="/data/corpus",
            tokenizer_id="tok1",
            request_id="req1",
        )
    assert exc.value.status == 400 and "bad request" in str(exc.value)
    await http.aclose()


@pytest.mark.asyncio
async def test_invalid_response_bodies_raise() -> None:
    post_resp = _FakeResponse(200, ["not", "a", "dict"])
    get_resp = _FakeResponse(200, {"run_id": "r1", "status": "done", "last_heartbeat_ts": "bad"})
    http = HTTPModelTrainerClient(
        base_url="https://api", api_key=None, client=_FakeClient(post_resp, get_resp)
    )
    with pytest.raises(ModelTrainerAPIError):
        await http.train(
            user_id=1,
            model_family="gpt2",
            model_size="small",
            max_seq_len=128,
            num_epochs=1,
            batch_size=2,
            learning_rate=5e-4,
            corpus_path="/data/corpus",
            tokenizer_id="tok1",
            request_id="req1",
        )
    with pytest.raises(ModelTrainerAPIError):
        await http.status(run_id="r1", request_id="req2")
    await http.aclose()


def test_extract_message_fallbacks() -> None:
    assert _extract_message(_FakeResponse(400, {"message": ""})) == "HTTP 400"
    assert _extract_message(_FakeResponse(404, {"detail": "nope"})) == "nope"
    assert _extract_message(_FakeResponse(503, text="not json", json_body=None)) == "HTTP 503"
    assert _extract_message(_FakeResponse(500, {"error": "boom"})) == "boom"
    assert _extract_message(_FakeResponse(418, ["not", "dict"])) == "HTTP 418"


def test_load_json_dict_invalid_json_raises_api_error() -> None:
    resp = _FakeResponse(200, json_body=None, text="not-json")
    # Simulate response object for _extract_message to ensure fallback path
    msg = _extract_message(resp)
    assert msg == "HTTP 200"
    with pytest.raises(ModelTrainerAPIError):
        _ = _load_json_dict("not-json")


def test_headers_omit_api_key_when_blank() -> None:
    dummy_client = _FakeClient(_FakeResponse(200, {"ok": True}), _FakeResponse(200, {"ok": True}))
    client = HTTPModelTrainerClient(base_url="https://api", api_key="  ", client=dummy_client)
    headers = client._headers("req")
    assert "X-Api-Key" not in headers
    assert headers["X-Request-ID"] == "req"


def test_status_error_path_raises_api_error() -> None:
    post_resp = _FakeResponse(200, {"run_id": "r1", "job_id": "j1"})
    error_resp = _FakeResponse(404, {"detail": "missing"})
    http = HTTPModelTrainerClient(
        base_url="https://api", api_key=None, client=_FakeClient(post_resp, error_resp)
    )
    with pytest.raises(ModelTrainerAPIError) as exc:
        asyncio.run(http.status(run_id="r1", request_id="req2"))
    assert exc.value.status == 404 and "missing" in str(exc.value)


def test_parse_last_heartbeat_accepts_numeric_and_rejects_invalid() -> None:
    ok = _parse_last_heartbeat(1.0)
    assert ok == 1.0
    with pytest.raises(ModelTrainerAPIError):
        _parse_last_heartbeat({"bad": "type"})
    with pytest.raises(ModelTrainerAPIError):
        _parse_last_heartbeat("not-a-number")


def test_train_and_status_getitem_key_errors() -> None:
    tr = TrainResponse("r", "j")
    with pytest.raises(KeyError):
        _ = tr["unknown"]
    status = RunStatus(run_id="r", status="s", last_heartbeat_ts=None, message=None)
    assert status["run_id"] == "r"
    assert status["status"] == "s"
    assert status["message"] is None
    with pytest.raises(KeyError):
        _ = status["other"]


def test_parse_status_response_paths() -> None:
    obj_ok: dict[str, JSONValue] = {
        "run_id": "r1",
        "status": "ok",
        "last_heartbeat_ts": "1.5",
        "message": None,
    }
    res = _parse_status_response(obj_ok)
    assert res.status == "ok" and res.last_heartbeat_ts == 1.5 and res.message is None
    obj_bad: dict[str, JSONValue] = {"run_id": "r1", "status": "ok", "message": 123}
    with pytest.raises(ModelTrainerAPIError):
        _parse_status_response(obj_bad)


def test_build_async_client_uses_httpx_and_closes() -> None:
    client = build_async_client(timeout_seconds=0.1)
    asyncio.run(client.aclose())


def test_require_helpers_cover_return_and_error_paths() -> None:
    assert _require_str_field({"run_id": "r"}, "run_id") == "r"
    with pytest.raises(ModelTrainerAPIError):
        _require_str_field({"run_id": 5}, "run_id")
    assert _require_optional_str({"message": "ok"}, "message") == "ok"
    with pytest.raises(ModelTrainerAPIError):
        _require_optional_str({"message": 123}, "message")
