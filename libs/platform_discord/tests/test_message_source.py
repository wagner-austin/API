from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from platform_discord import message_source as mod
from platform_discord.testing import (
    FakeAsyncClient,
    FakeAsyncClientHook,
    FakePubSub,
    _production_path_is_dir,
    fake_async_client_from_url,
    fake_path_is_dir_false,
    fake_path_is_dir_true,
    hooks,
)


@pytest.fixture()
def fake_client() -> FakeAsyncClient:
    """Create a fake async client for testing."""
    return FakeAsyncClient()


@pytest.fixture()
def _use_fake_async_client(fake_client: FakeAsyncClient) -> Generator[None, None, None]:
    """Set up fake async client via hooks."""
    hooks.async_client_from_url = FakeAsyncClientHook(fake_client)
    yield


@pytest.mark.asyncio
async def test_message_source_subscribe_get_close(
    _use_fake_async_client: None,
    fake_client: FakeAsyncClient,
) -> None:
    src = mod.RedisPubSubSource("redis://x")
    assert await src.get() is None
    await src.close()
    await src.subscribe("chan")

    out1 = await src.get()
    assert out1 is None

    fake_client._pubsub.enqueue_message("message", 123)
    out2 = await src.get()
    assert out2 is None

    fake_client._pubsub.enqueue_message("message", "hello")
    out3 = await src.get()
    assert out3 == "hello"

    await src.close()


@pytest.mark.asyncio
async def test_pubsub_subscribe_with_no_channels() -> None:
    """Cover subscribe() with empty channels."""
    pubsub = FakePubSub()
    await pubsub.subscribe()  # No channels - hits early return
    assert pubsub._channel is None


@pytest.mark.asyncio
async def test_fake_async_client_from_url_returns_working_client() -> None:
    """Cover fake_async_client_from_url function."""
    # Get client and use it via message source (production path)
    hooks.async_client_from_url = fake_async_client_from_url
    src = mod.RedisPubSubSource("redis://fake")
    await src.subscribe("test-chan")
    # Verify message passing works
    result = await src.get()
    assert result is None  # Queue is empty initially
    await src.close()


def test_fake_path_is_dir_true_returns_true() -> None:
    """Cover fake_path_is_dir_true function."""
    result = fake_path_is_dir_true(Path("/any/path"))
    assert result is True


def test_fake_path_is_dir_false_returns_false() -> None:
    """Cover fake_path_is_dir_false function."""
    result = fake_path_is_dir_false(Path("/any/path"))
    assert result is False


def test_production_path_is_dir_returns_true_for_existing_dir(tmp_path: Path) -> None:
    """Cover _production_path_is_dir with real directory."""
    result = _production_path_is_dir(tmp_path)
    assert result is True


def test_production_path_is_dir_returns_false_for_file(tmp_path: Path) -> None:
    """Cover _production_path_is_dir with file path."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("test", encoding="utf-8")
    result = _production_path_is_dir(file_path)
    assert result is False
