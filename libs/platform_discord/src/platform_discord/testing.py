"""Public test utilities for platform_discord.

Provides Protocol types and hooks for testing Discord components against real code paths.

Usage:
    from platform_discord.testing import (
        hooks,
        reset_hooks,
        FakeDiscordModule,
        FakeEmbed,
        FakeColor,
    )

    # Set up fake discord module for testing
    hooks.load_discord_module = lambda: FakeDiscordModule()

    # Reset to production after test
    reset_hooks()
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Protocol

from platform_workers.redis import (
    PubSubMessage,
    RedisAsyncProto,
    RedisPubSubProto,
)
from platform_workers.redis import (
    _redis_async_from_url as _real_redis_async_from_url,
)

from .embed_helpers import (
    _DiscordColorCtor,
    _DiscordColorValue,
    _DiscordEmbedClient,
    _DiscordEmbedCtor,
    _DiscordFieldProxy,
    _DiscordFooterProxy,
    _DiscordModule,
)
from .embed_helpers import (
    _load_discord_module as _real_load_discord_module,
)

# ---------------------------------------------------------------------------
# Fake Discord Module Implementation
# ---------------------------------------------------------------------------


class FakeColorValue:
    """Fake discord Color value implementing _DiscordColorValue protocol."""

    __slots__ = ("_value",)

    def __init__(self, value: int) -> None:
        self._value = int(value)

    @property
    def value(self) -> int:
        return self._value


class FakeFieldProxy:
    """Fake discord field proxy implementing _DiscordFieldProxy protocol."""

    __slots__ = ("_inline", "_name", "_value")

    def __init__(self, name: str, value: str, inline: bool) -> None:
        self._name = name
        self._value = value
        self._inline = inline

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def value(self) -> str | None:
        return self._value

    @property
    def inline(self) -> bool:
        return self._inline


class FakeFooterProxy:
    """Fake discord footer proxy implementing _DiscordFooterProxy protocol."""

    __slots__ = ("_text",)

    def __init__(self) -> None:
        self._text: str | None = None

    @property
    def text(self) -> str | None:
        return self._text


class FakeEmbed:
    """Fake discord.Embed implementing _DiscordEmbedClient protocol."""

    __slots__ = ("_color", "_description", "_fields", "_footer", "_title")

    def __init__(
        self,
        *,
        title: str | None = None,
        description: str | None = None,
        color: _DiscordColorValue | None = None,
    ) -> None:
        self._title = title
        self._description = description
        self._color = color
        self._footer = FakeFooterProxy()
        self._fields: list[_DiscordFieldProxy] = []

    @property
    def title(self) -> str | None:
        return self._title

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def color(self) -> _DiscordColorValue | None:
        return self._color

    @property
    def footer(self) -> _DiscordFooterProxy:
        return self._footer

    @property
    def fields(self) -> list[_DiscordFieldProxy]:
        return self._fields

    def add_field(self, *, name: str, value: str, inline: bool = True) -> _DiscordEmbedClient:
        self._fields.append(FakeFieldProxy(name, value, inline))
        return self

    def set_footer(self, *, text: str) -> _DiscordEmbedClient:
        self._footer._text = text
        return self


class FakeColor:
    """Fake discord.Color constructor implementing _DiscordColorCtor protocol."""

    def __call__(self, value: int) -> _DiscordColorValue:
        return FakeColorValue(value)


class _FakeEmbedCtor:
    """Fake embed constructor implementing _DiscordEmbedCtor protocol."""

    def __call__(
        self,
        *,
        title: str | None = None,
        description: str | None = None,
        color: _DiscordColorValue | None = None,
    ) -> _DiscordEmbedClient:
        return FakeEmbed(title=title, description=description, color=color)


class FakeDiscordModule:
    """Fake discord module implementing _DiscordModule protocol."""

    Embed: _DiscordEmbedCtor = _FakeEmbedCtor()
    Color: _DiscordColorCtor = FakeColor()


# ---------------------------------------------------------------------------
# Fake Async Redis for PubSub Testing
# ---------------------------------------------------------------------------


class FakePubSub:
    """Fake Redis PubSub implementing RedisPubSubProto."""

    def __init__(self) -> None:
        self._channel: str | None = None
        self._queue: list[PubSubMessage] = []

    async def subscribe(self, *channels: str) -> None:
        if channels:
            self._channel = channels[0]

    async def get_message(
        self,
        *,
        ignore_subscribe_messages: bool = True,
        timeout: float = 1.0,
    ) -> PubSubMessage | None:
        del ignore_subscribe_messages, timeout  # unused in fake
        if self._queue:
            return self._queue.pop(0)
        return None

    async def close(self) -> None:
        self._channel = None
        self._queue.clear()

    def enqueue_message(self, msg_type: str, data: str | int | None) -> None:
        """Add a message to the queue for testing."""
        msg: PubSubMessage = {"type": msg_type, "data": data, "channel": "", "pattern": None}
        self._queue.append(msg)


class FakeAsyncClient:
    """Fake async Redis client implementing RedisAsyncProto."""

    def __init__(self) -> None:
        self._pubsub = FakePubSub()

    def pubsub(self) -> RedisPubSubProto:
        return self._pubsub


# ---------------------------------------------------------------------------
# Hook Protocol Types
# ---------------------------------------------------------------------------


class LoadDiscordModuleCallable(Protocol):
    """Protocol for loading discord module."""

    def __call__(self) -> _DiscordModule:
        """Load and return the discord module."""
        ...


class AsyncClientFromUrlCallable(Protocol):
    """Protocol for creating async client from URL."""

    def __call__(self, url: str) -> RedisAsyncProto:
        """Create async client from URL."""
        ...


class PathCheckCallable(Protocol):
    """Protocol for checking if a path is a directory."""

    def __call__(self, path: Path) -> bool:
        """Return True if path is a directory."""
        ...


# ---------------------------------------------------------------------------
# Production Implementations
# ---------------------------------------------------------------------------


def _production_load_discord_module() -> _DiscordModule:
    """Production implementation that loads real discord."""
    return _real_load_discord_module()


def _production_async_client_from_url(url: str) -> RedisAsyncProto:
    """Production implementation that creates real async redis client."""
    return _real_redis_async_from_url(url)


def _production_path_is_dir(path: Path) -> bool:
    """Production implementation that uses real Path.is_dir()."""
    return path.is_dir()


# ---------------------------------------------------------------------------
# Fake Implementations for Testing
# ---------------------------------------------------------------------------


def fake_load_discord_module() -> _DiscordModule:
    """Fake implementation that returns FakeDiscordModule.

    Use this in tests:
        hooks.load_discord_module = fake_load_discord_module
    """
    return FakeDiscordModule()


def fake_async_client_from_url(url: str) -> RedisAsyncProto:
    """Fake implementation that returns FakeAsyncClient.

    Use this in tests:
        hooks.async_client_from_url = fake_async_client_from_url
    """
    del url  # unused in fake
    return FakeAsyncClient()


class FakeAsyncClientHook:
    """Callable hook that returns a specific FakeAsyncClient.

    Use this in tests when you need to access the client:
        client = FakeAsyncClient()
        hooks.async_client_from_url = FakeAsyncClientHook(client)
    """

    __slots__ = ("_client",)

    def __init__(self, client: FakeAsyncClient) -> None:
        self._client = client

    def __call__(self, url: str) -> RedisAsyncProto:
        del url  # unused in fake
        return self._client


def fake_path_is_dir_true(path: Path) -> bool:
    """Fake implementation that always returns True.

    Use this in tests:
        hooks.path_is_dir = fake_path_is_dir_true
    """
    del path  # unused in fake
    return True


def fake_path_is_dir_false(path: Path) -> bool:
    """Fake implementation that always returns False.

    Use this in tests:
        hooks.path_is_dir = fake_path_is_dir_false
    """
    del path  # unused in fake
    return False


# ---------------------------------------------------------------------------
# Hooks Container
# ---------------------------------------------------------------------------


class _Hooks:
    """Mutable container for test hooks.

    Production code calls these hooks directly. Tests override them with fakes.
    """

    load_discord_module: LoadDiscordModuleCallable
    async_client_from_url: AsyncClientFromUrlCallable
    path_is_dir: PathCheckCallable


# Global hooks instance
hooks: Final[_Hooks] = _Hooks()


def set_production_hooks() -> None:
    """Set all hooks to production implementations."""
    hooks.load_discord_module = _production_load_discord_module
    hooks.async_client_from_url = _production_async_client_from_url
    hooks.path_is_dir = _production_path_is_dir


def reset_hooks() -> None:
    """Reset hooks to production implementations."""
    set_production_hooks()


# Initialize with production hooks by default
set_production_hooks()


__all__ = [
    "AsyncClientFromUrlCallable",
    "FakeAsyncClient",
    "FakeAsyncClientHook",
    "FakeColor",
    "FakeColorValue",
    "FakeDiscordModule",
    "FakeEmbed",
    "FakeFieldProxy",
    "FakeFooterProxy",
    "FakePubSub",
    "LoadDiscordModuleCallable",
    "PathCheckCallable",
    "fake_async_client_from_url",
    "fake_load_discord_module",
    "fake_path_is_dir_false",
    "fake_path_is_dir_true",
    "hooks",
    "reset_hooks",
    "set_production_hooks",
]
