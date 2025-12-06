from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping

from platform_core.http_client import HttpxResponse
from platform_core.json_utils import JSONValue, dump_json_str
from platform_core.model_trainer_client import HTTPModelTrainerClient


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

    def __init__(self, train_resp: _FakeResponse, status_resp: _FakeResponse) -> None:
        self._train_resp = train_resp
        self._status_resp = status_resp

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
        return self._train_resp

    async def get(self, url: str, *, headers: Mapping[str, str]) -> HttpxResponse:
        _ = url, headers
        return self._status_resp


def test_client_train_and_status_roundtrip() -> None:
    base = "https://example/api"
    # Train endpoint response
    train_body: JSONValue = {"run_id": "r123", "job_id": "j456"}
    train_resp = _FakeResponse(200, train_body)
    # Status endpoint response
    status_body: JSONValue = {
        "run_id": "r123",
        "status": "queued",
        "last_heartbeat_ts": None,
        "message": None,
    }
    status_resp = _FakeResponse(200, status_body)

    async def _run() -> None:
        client = HTTPModelTrainerClient(
            base_url=base, api_key="k", client=_FakeClient(train_resp, status_resp)
        )
        out = await client.train(
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
        assert out.run_id == "r123" and out.job_id == "j456"
        st = await client.status(run_id=out.run_id, request_id="req2")
        assert st.status == "queued" and st.run_id == "r123"
        await client.aclose()

    asyncio.run(_run())


logger = logging.getLogger(__name__)
