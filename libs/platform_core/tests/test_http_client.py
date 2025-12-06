from __future__ import annotations

import asyncio

from platform_core.http_client import build_async_client, build_client


def test_build_async_client_basic() -> None:
    client = build_async_client(timeout_seconds=1.0)
    asyncio.run(client.aclose())


def test_build_async_client_with_transport() -> None:
    class _FakeTransport:
        async def aclose(self) -> None:
            return None

    client = build_async_client(timeout_seconds=1.0, transport=_FakeTransport())
    asyncio.run(client.aclose())


def test_build_client_basic() -> None:
    client = build_client(timeout_seconds=1.0)
    client.close()


def test_build_client_with_transport() -> None:
    class _FakeTransport:
        def close(self) -> None:
            return None

    client = build_client(timeout_seconds=1.0, transport=_FakeTransport())
    client.close()
