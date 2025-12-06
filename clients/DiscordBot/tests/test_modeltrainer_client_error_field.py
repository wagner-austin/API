from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping

import pytest
from platform_core.http_client import HttpxResponse
from platform_core.json_utils import JSONValue, dump_json_str
from platform_core.model_trainer_client import HTTPModelTrainerClient, ModelTrainerAPIError


class _FakeResponse:
    """Protocol-compliant fake response for testing."""

    def __init__(self, status: int, json_body: JSONValue) -> None:
        self.status_code = int(status)
        self._json = json_body
        self.text = dump_json_str(json_body)
        self.headers: Mapping[str, str] = {}
        self.content: bytes | bytearray = self.text.encode("utf-8")

    def json(self) -> JSONValue:
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


def test_client_error_field_is_used_in_message() -> None:
    base = "https://example/api"
    # POST returns 503 with error field
    post_body: JSONValue = {"error": "service down"}
    post_resp = _FakeResponse(503, post_body)
    # GET returns not found
    get_body: JSONValue = {"message": "not found"}
    get_resp = _FakeResponse(404, get_body)

    async def _run() -> None:
        client = HTTPModelTrainerClient(
            base_url=base,
            api_key="k",
            client=_FakeClient(post_resp, get_resp),
        )
        with pytest.raises(ModelTrainerAPIError) as exc:
            await client.train(
                user_id=1,
                model_family="gpt2",
                model_size="small",
                max_seq_len=32,
                num_epochs=1,
                batch_size=1,
                learning_rate=5e-4,
                corpus_path="/data/corpus",
                tokenizer_id="tok",
                request_id="r",
            )
        assert exc.value.status == 503
        assert "service down" in str(exc.value)
        await client.aclose()

    asyncio.run(_run())


logging.getLogger(__name__)
