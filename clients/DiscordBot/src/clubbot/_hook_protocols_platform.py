"""Protocol types for clubbot's platform-level hooks.

These cover the dependencies the bot shares with the rest of the monorepo --
settings, HTTP clients, the RQ harness, the guard entry point, logging and the
service container. The bot's own service and Discord-facing protocols are in
:mod:`clubbot._hook_protocols_services`; the implementations behind all of them
are in :mod:`clubbot._hook_defaults` and the bindings in
:mod:`clubbot._test_hooks`.
"""

from __future__ import annotations

import urllib.parse as _url
from typing import Protocol

from platform_core.config import DiscordbotSettings
from platform_core.http_client import HttpxAsyncClient, HttpxClient, Timeout
from platform_core.logging import LogFormat, LogLevel
from platform_discord.protocols import (
    BotProto,
)
from platform_workers.rq_harness import (
    RQClientQueue,
    RQRetryLike,
    _RedisBytesClient,
)


class LoadSettingsProtocol(Protocol):
    """Protocol for settings loader function."""

    def __call__(self) -> DiscordbotSettings:
        """Load and return settings."""
        ...


class BuildClientProtocol(Protocol):
    """Protocol for sync HTTP client builder."""

    def __call__(self, timeout: float) -> HttpxClient:
        """Build and return a sync HTTP client."""
        ...


class BuildAsyncClientProtocol(Protocol):
    """Protocol for async HTTP client builder."""

    def __call__(self, timeout: float) -> HttpxAsyncClient:
        """Build and return an async HTTP client."""
        ...


class RqBytesClientFactoryProtocol(Protocol):
    """Protocol for RQ bytes client connection factory."""

    def __call__(self, url: str) -> _RedisBytesClient:
        """Create bytes client for RQ from URL."""
        ...


class RqQueueProtocol(Protocol):
    """Protocol for RQ queue factory."""

    def __call__(self, name: str, *, connection: _RedisBytesClient) -> RQClientQueue:
        """Create RQ queue."""
        ...


class RqRetryProtocol(Protocol):
    """Protocol for RQ retry factory."""

    def __call__(self, *, max_retries: int, intervals: list[int]) -> RQRetryLike:
        """Create RQ retry configuration."""
        ...


class TimeoutCtorProtocol(Protocol):
    """Protocol for httpx.Timeout constructor."""

    def __call__(self, timeout: float) -> Timeout:
        """Create a Timeout instance."""
        ...


class HttpxClientCtorProtocol(Protocol):
    """Protocol for httpx.Client constructor."""

    def __call__(self, *, timeout: Timeout) -> HttpxClient:
        """Create an HttpxClient instance."""
        ...


class HttpxModuleProtocol(Protocol):
    """Protocol for httpx module interface used by transcript api_client."""

    Timeout: TimeoutCtorProtocol
    Client: HttpxClientCtorProtocol


class LoadHttpxModuleProtocol(Protocol):
    """Protocol for the httpx module loader."""

    def __call__(self) -> HttpxModuleProtocol:
        """Import httpx and return it behind its protocol."""
        ...


class DigitsEnqueuerLike(Protocol):
    """Protocol for digits enqueuer interface.

    Defines the interface that RQDigitsEnqueuer and test fakes implement.
    """

    def enqueue_train(
        self,
        *,
        request_id: str,
        user_id: int,
        model_id: str,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
        augment: bool,
        notes: str | None = None,
    ) -> str:
        """Enqueue a training job and return the job ID."""
        ...


class BuildDigitsEnqueuerProtocol(Protocol):
    """Protocol for digits enqueuer builder."""

    def __call__(self, redis_url: str) -> DigitsEnqueuerLike | None:
        """Build a digits enqueuer from redis URL.

        Returns DigitsEnqueuerLike instance or None if redis_url is empty.
        """
        ...


class SetupLoggingProtocol(Protocol):
    """Protocol for setup_logging function."""

    def __call__(
        self,
        *,
        level: LogLevel,
        service_name: str,
        format_mode: LogFormat,
        instance_id: str | None = None,
        extra_fields: list[str] | None = None,
    ) -> None:
        """Set up logging."""
        ...


class ServiceContainerProtocol(Protocol):
    """Protocol for ServiceContainer - minimal interface for settings access."""

    cfg: DiscordbotSettings


class BotOrchestratorProtocol(Protocol):
    """Protocol for BotOrchestrator."""

    def run(self) -> None:
        """Run the bot."""
        ...


class CreateBotOrchestratorProtocol(Protocol):
    """Protocol for BotOrchestrator constructor."""

    def __call__(self, container: ServiceContainerProtocol) -> BotOrchestratorProtocol:
        """Create a BotOrchestrator from container."""
        ...


class CreateServiceContainerProtocol(Protocol):
    """Protocol for ServiceContainer.from_env factory."""

    def __call__(self) -> ServiceContainerProtocol:
        """Create ServiceContainer from environment."""
        ...


class BotLikeProtocol(Protocol):
    """Protocol for discord.py Bot-like objects."""

    def run(self, token: str) -> None:
        """Run the bot with the given token."""
        ...


class FetchedUserLike(Protocol):
    """Protocol for objects that may or may not satisfy UserProto.

    This is intentionally broader than UserProto to allow testing defensive code
    that checks isinstance(user_obj, UserProto).
    """

    @property
    def id(self) -> int:
        """User ID."""
        ...


class BotFetchUserProtocol(Protocol):
    """Protocol for bot.fetch_user hook."""

    async def __call__(self, bot: BotProto, user_id: int) -> FetchedUserLike:
        """Fetch a user by ID."""
        ...


class BuildBotProtocol(Protocol):
    """Protocol for building a Bot instance."""

    def __call__(self) -> BotLikeProtocol:
        """Build and return a Bot instance."""
        ...


class UrlSplitProtocol(Protocol):
    """Protocol for urllib.parse.urlsplit function."""

    def __call__(self, url: str) -> _url.SplitResult:
        """Split a URL into its components."""
        ...


class TranscriptResultLike(Protocol):
    """Protocol for TranscriptResult-like objects."""

    url: str
    video_id: str
    text: str
