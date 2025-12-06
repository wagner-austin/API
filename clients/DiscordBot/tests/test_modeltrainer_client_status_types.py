from __future__ import annotations

import asyncio

import pytest
from platform_core.json_utils import JSONValue
from platform_core.model_trainer_client import HTTPModelTrainerClient, ModelTrainerAPIError
from tests.support.httpx_fakes import FakeHttpxAsyncClient, FakeResponse, Request


def test_modeltrainer_status_type_narrowing_to_none() -> None:
    base = "https://example/api"

    def _handler(request: Request) -> FakeResponse:
        if request.method.upper() == "GET" and str(request.url) == f"{base}/runs/rx":
            payload: JSONValue = {
                "run_id": "rx",
                "status": "done",
                "last_heartbeat_ts": [],
                "message": 123,
            }
            return FakeResponse(
                status_code=200, text="", headers={}, content=b"", json_value=payload
            )
        if request.method.upper() == "POST" and str(request.url) == f"{base}/runs/train":
            return FakeResponse(
                status_code=200,
                text="",
                headers={},
                content=b"",
                json_value={"run_id": "rx", "job_id": "jx"},
            )
        return FakeResponse(
            status_code=404,
            text="not found",
            headers={},
            content=b"",
            json_value={"message": "not found"},
        )

    async def _run() -> None:
        client = HTTPModelTrainerClient(
            base_url=base,
            api_key="k",
            client=FakeHttpxAsyncClient(_handler),
        )
        _ = await client.train(
            user_id=7,
            model_family="gpt2",
            model_size="small",
            max_seq_len=64,
            num_epochs=1,
            batch_size=1,
            learning_rate=5e-4,
            corpus_path="/data/corpus",
            tokenizer_id="tok",
            request_id="r",
        )
        with pytest.raises(ModelTrainerAPIError):
            _ = await client.status(run_id="rx", request_id="r2")
        await client.aclose()

    asyncio.run(_run())
