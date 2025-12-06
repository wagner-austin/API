from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping

import pytest
from platform_core.http_client import HttpxResponse
from platform_core.json_utils import JSONValue, dump_json_str
from platform_core.model_trainer_client import (
    HTTPModelTrainerClient,
    ModelTrainerAPIError,
    _extract_message,
)


class _FakeResponse:
    """Protocol-compliant fake response for testing."""

    def __init__(
        self, status: int, json_body: JSONValue | None = None, text: str | None = None
    ) -> None:
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
    """Protocol-compliant fake async HTTP client for testing."""

    def __init__(self, post_resp: _FakeResponse, get_resp: _FakeResponse) -> None:
        self._post_resp = post_resp
        self._get_resp = get_resp

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
        _ = url, headers, json, files
        return self._post_resp

    async def get(self, url: str, *, headers: Mapping[str, str]) -> HttpxResponse:
        _ = url, headers
        return self._get_resp


def test_client_error_paths_and_status_conversion() -> None:
    base = "https://example/api"
    # Train: 400 with message in JSON
    train_body: JSONValue = {"message": "bad"}
    post_resp = _FakeResponse(400, train_body)
    # Status: 200 with heartbeat as string
    status_body: JSONValue = {"run_id": "r1", "status": "running", "last_heartbeat_ts": "12.5"}
    get_resp = _FakeResponse(200, status_body)

    async def _run() -> None:
        client = HTTPModelTrainerClient(
            base_url=base, api_key="k", client=_FakeClient(post_resp, get_resp)
        )
        with pytest.raises(ModelTrainerAPIError) as exc:
            await client.train(
                user_id=1,
                model_family="gpt2",
                model_size="small",
                max_seq_len=128,
                num_epochs=1,
                batch_size=2,
                learning_rate=5e-4,
                corpus_path="/data/corpus",
                tokenizer_id="tok1",
                request_id="req",
            )
        assert exc.value.status == 400
        st = await client.status(run_id="r1", request_id="req2")
        assert st.last_heartbeat_ts == 12.5
        await client.aclose()

    asyncio.run(_run())


def test_client_train_invalid_body_and_http_error_fallback() -> None:
    base = "https://example/api"
    post_resp = _FakeResponse(500, text="oops")
    # GET returns list (invalid response body)
    get_body: JSONValue = [1, 2]
    get_resp = _FakeResponse(200, get_body)

    async def _run() -> None:
        client = HTTPModelTrainerClient(
            base_url=base, api_key="k", client=_FakeClient(post_resp, get_resp)
        )

        with pytest.raises(ModelTrainerAPIError) as e1:
            await client.train(
                user_id=1,
                model_family="gpt2",
                model_size="small",
                max_seq_len=128,
                num_epochs=1,
                batch_size=2,
                learning_rate=5e-4,
                corpus_path="/data/corpus",
                tokenizer_id="tok1",
                request_id="req",
            )
        assert "HTTP 500" in str(e1.value)

        with pytest.raises(ModelTrainerAPIError) as e2:
            await client.status(run_id="x", request_id="req2")
        assert "Invalid response body" in str(e2.value)
        await client.aclose()

    asyncio.run(_run())


def test_client_success_without_api_key_and_status_detail_message() -> None:
    base = "https://example/api"
    # Train success
    train_body: JSONValue = {"run_id": "r2", "job_id": "j2"}
    post_resp = _FakeResponse(200, train_body)
    # Status 404 with detail
    status_body: JSONValue = {"detail": "oops"}
    get_resp = _FakeResponse(404, status_body)

    async def _run() -> None:
        client = HTTPModelTrainerClient(
            base_url=base,
            api_key=None,
            client=_FakeClient(post_resp, get_resp),
        )
        out = await client.train(
            user_id=2,
            model_family="gpt2",
            model_size="small",
            max_seq_len=128,
            num_epochs=1,
            batch_size=2,
            learning_rate=5e-4,
            corpus_path="/data/corpus",
            tokenizer_id="tok1",
            request_id="req3",
        )
        assert out.run_id == "r2" and out.job_id == "j2"

        with pytest.raises(ModelTrainerAPIError) as exc2:
            await client.status(run_id=out.run_id, request_id="req4")
        assert "oops" in str(exc2.value)
        await client.aclose()

    asyncio.run(_run())


logging.getLogger(__name__)


def test_extract_message_blank_and_non_string_fields() -> None:
    # Blank message string falls back to HTTP status text
    blank_body: JSONValue = {"message": ""}
    resp_blank = _FakeResponse(400, blank_body)
    assert _extract_message(resp_blank) == "HTTP 400"

    # Non-string message also falls back to HTTP status text
    non_str_body: JSONValue = {"message": 123}
    resp_non_str = _FakeResponse(400, non_str_body)
    assert _extract_message(resp_non_str) == "HTTP 400"

    # Response returning non-dict from json() should fall back to HTTP status text
    class _ListResponse:
        def __init__(self, text: str, body: JSONValue, status: int) -> None:
            self.text = text
            self.status_code = status
            self._body = body
            self.headers: Mapping[str, str] = {}
            self.content: bytes | bytearray = text.encode("utf-8")

        def json(self) -> JSONValue:
            return self._body

    list_body: JSONValue = ["x"]
    fake = _ListResponse(text="{ not_really_json }", body=list_body, status=500)
    assert _extract_message(fake) == "HTTP 500"
