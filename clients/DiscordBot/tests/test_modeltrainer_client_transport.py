from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest
from platform_core.http_client import HttpxResponse
from platform_core.json_utils import JSONValue
from platform_core.model_trainer_client import HTTPModelTrainerClient


class TransportError(Exception):
    """Transport error for testing connection failures."""


class _FakeClient:
    """Fake client that raises transport errors."""

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
        raise TransportError("conn")

    async def get(self, url: str, *, headers: Mapping[str, str]) -> HttpxResponse:
        raise TransportError("conn")


def test_transport_errors_propagate() -> None:
    async def _run() -> None:
        client = HTTPModelTrainerClient(
            base_url="https://example/api",
            api_key="k",
            client=_FakeClient(),
        )
        with pytest.raises(TransportError):
            await client.train(
                user_id=1,
                model_family="gpt2",
                model_size="small",
                max_seq_len=16,
                num_epochs=1,
                batch_size=1,
                learning_rate=5e-4,
                corpus_path="/data",
                tokenizer_id="tok",
                request_id="r",
            )
        with pytest.raises(TransportError):
            await client.status(run_id="x", request_id="r2")
        await client.aclose()

    asyncio.run(_run())
