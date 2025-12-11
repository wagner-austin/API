"""Test hooks for clubbot - allows injecting test dependencies.

This module provides hooks for dependency injection in tests. Production code
sets hooks to real implementations at startup; tests set them to fakes.

Hooks are module-level callables that production code calls directly. Tests
assign fake implementations before running the code under test.

Usage in production code:
    from clubbot import _test_hooks
    settings = _test_hooks.load_settings()

Usage in tests:
    from clubbot import _test_hooks
    from tests.support.settings import build_settings
    _test_hooks.load_settings = lambda: build_settings()
"""

from __future__ import annotations

import urllib.parse as _url
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

import discord
from discord.abc import Snowflake as DiscordSnowflake
from discord.app_commands import AppCommand, CommandTree
from platform_core.config import DiscordbotSettings
from platform_core.config import load_discordbot_settings as _real_load_discordbot_settings
from platform_core.http_client import HttpxAsyncClient, HttpxClient, Timeout
from platform_core.http_client import build_async_client as _real_build_async_client
from platform_core.http_client import build_client as _real_build_client
from platform_core.logging import LogFormat, LogLevel
from platform_discord.embed_helpers import EmbedProto
from platform_discord.handwriting.runtime import DigitsRuntime, RequestAction
from platform_discord.protocols import (
    BotProto,
    FileProto,
    FollowupProto,
    InteractionProto,
    MessageProto,
    ResponseProto,
    UserProto,
    _DiscordUser,
)
from platform_workers.rq_harness import (
    RQClientQueue,
    RQRetryLike,
    _RedisBytesClient,
)
from platform_workers.rq_harness import redis_raw_for_rq as _real_redis_raw_for_rq
from platform_workers.rq_harness import rq_queue as _real_rq_queue
from platform_workers.rq_harness import rq_retry as _real_rq_retry

from clubbot.services.registry import ServiceDef

# =============================================================================
# Protocol definitions for hookable dependencies
# =============================================================================


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


class GuardFindMonorepoRootProtocol(Protocol):
    """Protocol for finding monorepo root."""

    def __call__(self, start: Path) -> Path:
        """Find the monorepo root from a starting path."""
        ...


class GuardLoadOrchestratorProtocol(Protocol):
    """Protocol for loading guard orchestrator."""

    def __call__(self, monorepo_root: Path) -> GuardRunForProjectProtocol:
        """Load the orchestrator module and return run_for_project."""
        ...


class GuardRunForProjectProtocol(Protocol):
    """Protocol for run_for_project function from monorepo_guards."""

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int:
        """Run guards for a project."""
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


# =============================================================================
# Bot Fetch User Hook (for base.py defensive code testing)
# =============================================================================


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


async def _default_bot_fetch_user(bot: BotProto, user_id: int) -> FetchedUserLike:
    """Production implementation - calls bot.fetch_user directly."""
    return await bot.fetch_user(user_id)


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


# =============================================================================
# QR Service Hooks
# =============================================================================


class QRResultLike(Protocol):
    """Protocol for QRResult-like objects."""

    image_png: bytes
    url: str


class QRServiceLike(Protocol):
    """Protocol for QRService-like objects."""

    def generate_qr(self, url: str) -> QRResultLike:
        """Generate a QR code for the given URL."""
        ...


class QRServiceFactoryProtocol(Protocol):
    """Protocol for QR service factory."""

    def __call__(self, cfg: DiscordbotSettings) -> QRServiceLike:
        """Create a QR service from config."""
        ...


def _default_qr_service_factory(cfg: DiscordbotSettings) -> QRServiceLike:
    """Production implementation - creates QRService."""
    from clubbot.services.qr.client import QRService

    return QRService(cfg)


# Hook for creating QR service. Tests override for testing.
qr_service_factory: QRServiceFactoryProtocol = _default_qr_service_factory


# =============================================================================
# Digits Notifier Hooks
# =============================================================================


class OnCompletedProtocol(Protocol):
    """Protocol for on_completed function from platform_discord.handwriting.runtime."""

    def __call__(
        self,
        runtime: DigitsRuntime,
        *,
        user_id: int,
        request_id: str,
        model_id: str,
        run_id: str | None,
        val_acc: float,
    ) -> RequestAction | None:
        """Handle completed event. Returns RequestAction or None."""
        ...


def _default_on_completed(
    runtime: DigitsRuntime,
    *,
    user_id: int,
    request_id: str,
    model_id: str,
    run_id: str | None,
    val_acc: float,
) -> RequestAction | None:
    """Production implementation - calls platform_discord on_completed."""
    from platform_discord.handwriting.runtime import on_completed as _real_on_completed

    return _real_on_completed(
        runtime,
        user_id=user_id,
        request_id=request_id,
        model_id=model_id,
        run_id=run_id,
        val_acc=val_acc,
    )


# Hook for on_completed. Tests override to inject None-returning implementation.
on_completed: OnCompletedProtocol = _default_on_completed


# =============================================================================
# Transcript Service Hooks
# =============================================================================


class TranscriptPayloadDict(Protocol):
    """Protocol for transcript payload dictionaries."""

    def __getitem__(self, key: str) -> str: ...


class ValidateYoutubeUrlForClientProtocol(Protocol):
    """Protocol for validate_youtube_url in transcript client."""

    def __call__(self, url: str) -> str:
        """Validate and canonicalize a YouTube URL."""
        ...


class ExtractVideoIdProtocol(Protocol):
    """Protocol for extract_video_id function."""

    def __call__(self, url: str) -> str:
        """Extract video ID from a YouTube URL."""
        ...


class CaptionsProtocol(Protocol):
    """Protocol for captions function in transcript api_client."""

    def __call__(
        self,
        client: dict[str, float | str],
        *,
        url: str,
        preferred_langs: list[str],
    ) -> dict[str, str]:
        """Fetch captions for a video."""
        ...


def _default_validate_youtube_url_for_client(url: str) -> str:
    """Production implementation - validates YouTube URL for transcript client."""
    from clubbot.utils.youtube import validate_youtube_url as _real_validate

    return _real_validate(url)


def _default_extract_video_id(url: str) -> str:
    """Production implementation - extracts video ID."""
    from clubbot.utils.youtube import extract_video_id as _real_extract

    return _real_extract(url)


def _default_captions(
    client: dict[str, float | str],
    *,
    url: str,
    preferred_langs: list[str],
) -> dict[str, str]:
    """Production implementation - calls api_client.captions."""
    from clubbot.services.transcript.api_client import TranscriptApiClient
    from clubbot.services.transcript.api_client import captions as _real_captions

    typed_client: TranscriptApiClient = {
        "base_url": str(client.get("base_url", "")),
        "timeout_seconds": float(client.get("timeout_seconds", 30.0)),
    }
    result = _real_captions(typed_client, url=url, preferred_langs=preferred_langs)
    return {"url": result["url"], "video_id": result["video_id"], "text": result["text"]}


# Hook for validate_youtube_url in transcript client. Tests override for testing.
validate_youtube_url_for_client: ValidateYoutubeUrlForClientProtocol = (
    _default_validate_youtube_url_for_client
)

# Hook for extract_video_id. Tests override for testing.
extract_video_id: ExtractVideoIdProtocol = _default_extract_video_id

# Hook for captions function. Tests override with fakes.
captions: CaptionsProtocol = _default_captions


# =============================================================================
# Bot Tree Sync Protocol (for orchestrator command syncing)
# =============================================================================


class GuildLikeProto(Protocol):
    """Protocol for guild-like objects."""

    @property
    def id(self) -> int:
        """Get the guild ID."""
        ...

    @property
    def name(self) -> str:
        """Get the guild name."""
        ...


class TreeSyncProtocol(Protocol):
    """Protocol for bot.tree.sync function."""

    async def __call__(self, *, guild: GuildLikeProto | None = None) -> list[str]:
        """Sync commands to Discord."""
        ...


class SnowflakeLike(Protocol):
    """Protocol for snowflake-like objects (used by discord.py for IDs)."""

    @property
    def id(self) -> int:
        """Get the snowflake ID."""
        ...


class AppCommandLike(Protocol):
    """Protocol for app command-like objects returned by sync."""

    @property
    def name(self) -> str:
        """Get the command name."""
        ...


class SnowflakeProto(Protocol):
    """Protocol for Discord snowflake-like objects."""

    @property
    def id(self) -> int:
        """Snowflake ID."""
        ...


class BotTreeProto(Protocol):
    """Protocol for bot command tree-like objects.

    Matches the sync() method signature of discord.app_commands.CommandTree.
    """

    async def sync(self, *, guild: DiscordSnowflake | None = None) -> list[AppCommand]:
        """Sync commands to Discord."""
        ...


class TreeSyncFactoryProtocol(Protocol):
    """Protocol for tree sync factory - allows testing of command sync.

    Uses BotTreeProto to accept any object with a sync() method,
    enabling both production CommandTree and test fakes.
    """

    async def __call__(
        self, tree: BotTreeProto, guild: DiscordSnowflake | None = None
    ) -> list[AppCommand]:
        """Sync commands using the bot tree."""
        ...


# =============================================================================
# Discord Exception Types (for orchestrator error handling)
# =============================================================================


class DiscordExceptionTypesProtocol(Protocol):
    """Protocol for Discord exception types factory.

    Returns a tuple of exception types that the orchestrator catches
    when sending error responses to interactions.
    """

    def __call__(self) -> tuple[type[Exception], type[Exception], type[Exception]]:
        """Return (HTTPException, ForbiddenError, NotFoundError) exception types."""
        ...


def _default_discord_exception_types() -> tuple[type[Exception], type[Exception], type[Exception]]:
    """Production implementation - returns actual platform_discord exceptions."""
    from platform_discord.exceptions import DForbiddenError, DHTTPExceptionError, DNotFoundError

    return (DHTTPExceptionError, DForbiddenError, DNotFoundError)


# =============================================================================
# App Command Error Handler Hook (for testing orchestrator error handler)
# =============================================================================


class AppCommandErrorHandlerProtocol(Protocol):
    """Protocol for app command error handler.

    Defines the interface for handling application command errors in the orchestrator.
    Accepts both discord.Interaction (production) and InteractionProto (tests).
    """

    async def __call__(
        self, interaction: discord.Interaction | InteractionProto, error: Exception
    ) -> None:
        """Handle an application command error."""
        ...


async def _default_app_command_error_handler(
    interaction: discord.Interaction | InteractionProto, error: Exception
) -> None:
    """Production implementation of app command error handler."""
    from monorepo_guards._types import UnknownJson
    from platform_core.logging import get_logger

    logger = get_logger(__name__)
    original_obj: UnknownJson = getattr(error, "original", None)
    original = original_obj if isinstance(original_obj, Exception) else error
    logger.exception("Unhandled application command error: %s", original)
    # Get exception types from hook (allows tests to inject custom types)
    http_exc, forbidden_exc, notfound_exc = discord_exception_types()
    try:
        if interaction.response.is_done():
            await interaction.followup.send(
                "An error occurred. Please try again later.", ephemeral=True
            )
        else:
            await interaction.response.send_message(
                "An error occurred. Please try again later.", ephemeral=True
            )
    except (http_exc, forbidden_exc, notfound_exc):
        logger.exception("Failed to send error response to interaction")
        raise


# =============================================================================
# Orchestrator Sync Hook (for testing sync_commands exception paths)
# =============================================================================


class OrchestratorSyncGlobalProtocol(Protocol):
    """Protocol for orchestrator _sync_global method."""

    async def __call__(self) -> bool:
        """Perform global command sync. Returns True if sync was performed."""
        ...


# Global hook for orchestrator _sync_global. When set, orchestrator calls this
# instead of its own _sync_global method. Tests use this to inject exceptions.
orchestrator_sync_global_override: OrchestratorSyncGlobalProtocol | None = None


# =============================================================================
# Orchestrator Build Bot Hook (for testing run() method)
# =============================================================================


class BotRunnerProtocol(Protocol):
    """Protocol for objects that can run like a bot."""

    def run(self, token: str) -> None:
        """Run the bot with the given token."""
        ...


class OrchestratorBuildBotProtocol(Protocol):
    """Protocol for orchestrator build_bot override."""

    def __call__(self) -> BotRunnerProtocol:
        """Build and return a bot-like object."""
        ...


class OrchestratorBuildBotHookProtocol(Protocol):
    """Protocol for orchestrator build_bot hook.

    Takes the orchestrator instance and returns a bot-like object.
    Production calls orchestrator.build_bot(); tests return fakes.
    """

    def __call__(self, orchestrator: OrchestratorLike) -> BotRunnerProtocol:
        """Build and return a bot-like object."""
        ...


class OrchestratorLike(Protocol):
    """Protocol for BotOrchestrator-like objects."""

    def build_bot(self) -> BotRunnerProtocol:
        """Build and return a bot."""
        ...


def _default_orchestrator_build_bot(orchestrator: OrchestratorLike) -> BotRunnerProtocol:
    """Production implementation - calls orchestrator.build_bot()."""
    return orchestrator.build_bot()


# Hook for orchestrator build_bot. Production calls build_bot(); tests override.
orchestrator_build_bot: OrchestratorBuildBotHookProtocol = _default_orchestrator_build_bot

# Legacy: Global hook for orchestrator build_bot (for backwards compatibility).
# Deprecated - use orchestrator_build_bot instead.
orchestrator_build_bot_override: OrchestratorBuildBotProtocol | None = None


# =============================================================================
# Orchestrator Service Registry Hook
# =============================================================================


class GetServiceRegistryProtocol(Protocol):
    """Protocol for getting service registry."""

    def __call__(self) -> dict[str, ServiceDef]:
        """Return the service registry."""
        ...


def _default_get_service_registry() -> dict[str, ServiceDef]:
    """Production implementation - returns actual SERVICE_REGISTRY."""
    from clubbot.services.registry import SERVICE_REGISTRY

    return SERVICE_REGISTRY


# Hook for getting SERVICE_REGISTRY. Tests override to return custom registries.
get_service_registry: GetServiceRegistryProtocol = _default_get_service_registry


# =============================================================================
# Orchestrator Bot.get_cog Hook
# =============================================================================


class CogLike(Protocol):
    """Protocol for cog-like objects."""

    def ensure_subscriber_started(self) -> None:
        """Ensure the subscriber is started."""
        ...


class GetCogProtocol(Protocol):
    """Protocol for bot.get_cog method."""

    def __call__(self, name: str) -> CogLike | None:
        """Get a cog by name."""
        ...


# Override for bot.get_cog. When set, orchestrator.start_background_subscribers uses this.
orchestrator_get_cog_override: GetCogProtocol | None = None


# =============================================================================
# Orchestrator Add Listener Hook
# =============================================================================


class AddListenerProtocol(Protocol):
    """Protocol for bot.add_listener method."""

    def __call__(self, func: Callable[[], Awaitable[None]], name: str | None = None) -> None:
        """Add a listener function."""
        ...


# Override for bot.add_listener. When set, orchestrator.register_listeners uses this.
orchestrator_add_listener_override: AddListenerProtocol | None = None


# =============================================================================
# Trainer Cog Protocols
# =============================================================================


class TrainerEventSubscriberLike(Protocol):
    """Protocol for TrainerEventSubscriber interface."""

    def start(self) -> None:
        """Start the subscriber."""
        ...


class TrainerEventSubscriberFactoryProtocol(Protocol):
    """Protocol for creating TrainerEventSubscriber."""

    def __call__(
        self,
        *,
        bot: BotProto,
        redis_url: str,
        events_channel: str,
    ) -> TrainerEventSubscriberLike:
        """Create a TrainerEventSubscriber."""
        ...


class TrainResponseLike(Protocol):
    """Protocol for train response objects - supports subscript access by string key."""

    def __getitem__(self, key: str) -> str:
        """Get value by key (run_id, job_id)."""
        ...


class TrainerApiClientLike(Protocol):
    """Protocol for HTTP model trainer API client interface."""

    async def train(
        self,
        *,
        user_id: int,
        model_family: str,
        model_size: str,
        max_seq_len: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        corpus_path: str,
        tokenizer_id: str,
        request_id: str,
    ) -> TrainResponseLike:
        """Start a training job."""
        ...

    async def aclose(self) -> None:
        """Close the client."""
        ...


class TrainerApiClientFactoryProtocol(Protocol):
    """Protocol for creating HTTP model trainer API client."""

    def __call__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: int,
    ) -> TrainerApiClientLike:
        """Create a model trainer client."""
        ...


# =============================================================================
# Digits Cog Protocols
# =============================================================================


class DigitsEventSubscriberLike(Protocol):
    """Protocol for DigitsEventSubscriber interface."""

    def start(self) -> None:
        """Start the subscriber."""
        ...

    async def stop(self) -> None:
        """Stop the subscriber."""
        ...


class DigitsEventSubscriberFactoryProtocol(Protocol):
    """Protocol for creating DigitsEventSubscriber."""

    def __call__(
        self,
        *,
        bot: BotProto,
        redis_url: str,
    ) -> DigitsEventSubscriberLike:
        """Create a DigitsEventSubscriber."""
        ...


# =============================================================================
# Discord Protocol Wrappers
# =============================================================================


# Use the actual protocol types from platform_discord for compatibility
InteractionProtoLike = InteractionProto
UserProtoLike = UserProto


# Use the actual protocol types from platform_discord for compatibility
ResponseProtoLike = ResponseProto
EmbedLike = EmbedProto
FileLike = FileProto
MessageLike = MessageProto
FollowupProtoLike = FollowupProto


class DiscordInteractionLike(Protocol):
    """Protocol for discord.Interaction-like objects (before wrapping).

    Uses _DiscordUser (only requires id property) to be compatible with
    Discord's User | Member return type.
    """

    @property
    def user(self) -> _DiscordUser:
        """Get the user."""
        ...


class WrapInteractionProtocol(Protocol):
    """Protocol for wrap_interaction function."""

    def __call__(self, interaction: DiscordInteractionLike) -> InteractionProtoLike:
        """Wrap a discord.Interaction into InteractionProto."""
        ...


# =============================================================================
# Default (production) implementations
# =============================================================================


def _default_load_settings() -> DiscordbotSettings:
    """Production implementation - loads settings from environment."""
    return _real_load_discordbot_settings()


def _default_build_client(timeout: float) -> HttpxClient:
    """Production implementation - builds real sync HTTP client."""
    return _real_build_client(timeout)


def _default_build_async_client(timeout: float) -> HttpxAsyncClient:
    """Production implementation - builds real async HTTP client."""
    return _real_build_async_client(timeout)


def _default_redis_raw_for_rq(url: str) -> _RedisBytesClient:
    """Production implementation - creates real RQ Redis connection."""
    return _real_redis_raw_for_rq(url)


def _default_rq_queue(name: str, *, connection: _RedisBytesClient) -> RQClientQueue:
    """Production implementation - creates real RQ queue."""
    return _real_rq_queue(name, connection=connection)


def _default_rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
    """Production implementation - creates real RQ retry."""
    return _real_rq_retry(max_retries=max_retries, intervals=intervals)


def _default_guard_find_monorepo_root(start: Path) -> Path:
    """Production implementation - finds monorepo root by climbing directories."""
    current = start
    while True:
        if (current / "libs").is_dir():
            return current
        if current.parent == current:
            raise RuntimeError("monorepo root with 'libs' directory not found")
        current = current.parent


def _default_guard_load_orchestrator(monorepo_root: Path) -> GuardRunForProjectProtocol:
    """Production implementation - loads the orchestrator module."""
    import sys

    libs_path = monorepo_root / "libs"
    guards_src = libs_path / "monorepo_guards" / "src"
    sys.path.insert(0, str(guards_src))
    sys.path.insert(0, str(libs_path))
    mod = __import__("monorepo_guards.orchestrator", fromlist=["run_for_project"])
    run_for_project: GuardRunForProjectProtocol = mod.run_for_project
    return run_for_project


def _default_build_digits_enqueuer(redis_url: str) -> DigitsEnqueuerLike | None:
    """Production implementation - builds real RQDigitsEnqueuer.

    This is a forward reference to avoid circular imports.
    The actual implementation is in clubbot.container._build_digits_enqueuer.
    """
    from platform_core.queues import DIGITS_QUEUE

    from clubbot.services.jobs.digits_enqueuer import RQDigitsEnqueuer

    url = (redis_url or "").strip()
    if not url:
        return None
    enqueuer: DigitsEnqueuerLike = RQDigitsEnqueuer(
        redis_url=url,
        queue_name=DIGITS_QUEUE,
        job_timeout_s=25200,
        result_ttl_s=86400,
        failure_ttl_s=604800,
        retry_max=2,
        retry_intervals_s=(60, 300),
    )
    return enqueuer


def _default_load_httpx_module() -> HttpxModuleProtocol:
    """Production implementation - loads real httpx module."""
    httpx_mod = __import__("httpx")
    result: HttpxModuleProtocol = httpx_mod
    return result


def _default_setup_logging(
    *,
    level: LogLevel,
    service_name: str,
    format_mode: LogFormat,
    instance_id: str | None = None,
    extra_fields: list[str] | None = None,
) -> None:
    """Production implementation - sets up logging via platform_core."""
    from platform_core.logging import setup_logging as _real_setup_logging

    _real_setup_logging(
        level=level,
        service_name=service_name,
        format_mode=format_mode,
        instance_id=instance_id,
        extra_fields=extra_fields,
    )


def _default_create_service_container() -> ServiceContainerProtocol:
    """Production implementation - creates ServiceContainer from env."""
    from clubbot.container import ServiceContainer

    result: ServiceContainerProtocol = ServiceContainer.from_env()
    return result


def _default_create_bot_orchestrator(
    container: ServiceContainerProtocol,
) -> BotOrchestratorProtocol:
    """Production implementation - creates BotOrchestrator."""
    from clubbot.container import ServiceContainer
    from clubbot.orchestrator import BotOrchestrator

    # BotOrchestrator expects ServiceContainer; protocol is compatible at runtime
    if not isinstance(container, ServiceContainer):
        raise TypeError("Expected ServiceContainer instance")
    return BotOrchestrator(container)


def _default_urlsplit(url: str) -> _url.SplitResult:
    """Production implementation - uses stdlib urlsplit."""
    return _url.urlsplit(url)


def _default_trainer_event_subscriber_factory(
    *,
    bot: BotProto,
    redis_url: str,
    events_channel: str,
) -> TrainerEventSubscriberLike:
    """Production implementation - creates real TrainerEventSubscriber."""
    from clubbot.services.jobs.trainer_notifier import TrainerEventSubscriber

    result: TrainerEventSubscriberLike = TrainerEventSubscriber(
        bot=bot,
        redis_url=redis_url,
        events_channel=events_channel,
    )
    return result


def _default_trainer_api_client_factory(
    *,
    base_url: str,
    api_key: str | None,
    timeout_seconds: int,
) -> TrainerApiClientLike:
    """Production implementation - creates real HTTPModelTrainerClient."""
    from platform_core.model_trainer_client import HTTPModelTrainerClient

    result: TrainerApiClientLike = HTTPModelTrainerClient(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    return result


def _default_digits_event_subscriber_factory(
    *,
    bot: BotProto,
    redis_url: str,
) -> DigitsEventSubscriberLike:
    """Production implementation - creates real DigitsEventSubscriber."""
    from clubbot.services.jobs.digits_notifier import DigitsEventSubscriber

    result: DigitsEventSubscriberLike = DigitsEventSubscriber(
        bot=bot,
        redis_url=redis_url,
    )
    return result


def _default_wrap_interaction(interaction: DiscordInteractionLike) -> InteractionProtoLike:
    """Production implementation - wraps discord.Interaction."""
    from platform_discord.protocols import wrap_interaction as _real_wrap_interaction

    result: InteractionProtoLike = _real_wrap_interaction(interaction)
    return result


async def _default_tree_sync(
    tree: BotTreeProto, guild: DiscordSnowflake | None = None
) -> list[AppCommand]:
    """Production implementation - calls tree.sync() directly."""
    return await tree.sync(guild=guild)


# =============================================================================
# Module-level hooks
# =============================================================================

# Hook for settings loading. Tests override to return test settings.
load_settings: LoadSettingsProtocol = _default_load_settings

# Hook for sync HTTP client builder. Tests override with fake client factory.
build_client: BuildClientProtocol = _default_build_client

# Hook for async HTTP client builder. Tests override with fake client factory.
build_async_client: BuildAsyncClientProtocol = _default_build_async_client

# Hook for RQ bytes client connection. Tests override with fake bytes client.
redis_raw_for_rq: RqBytesClientFactoryProtocol = _default_redis_raw_for_rq

# Hook for RQ queue factory. Tests override with FakeQueue.
rq_queue: RqQueueProtocol = _default_rq_queue

# Hook for RQ retry factory. Tests override with FakeRetry.
rq_retry: RqRetryProtocol = _default_rq_retry

# Hook for guard find_monorepo_root. Tests override to return fake paths.
guard_find_monorepo_root: GuardFindMonorepoRootProtocol = _default_guard_find_monorepo_root

# Hook for guard load_orchestrator. Tests override to return fake orchestrators.
guard_load_orchestrator: GuardLoadOrchestratorProtocol = _default_guard_load_orchestrator

# Hook for loading httpx module (used by transcript api_client).
load_httpx_module: Callable[[], HttpxModuleProtocol] = _default_load_httpx_module

# Hook for building digits enqueuer. Tests override with fake enqueuer builders.
build_digits_enqueuer: BuildDigitsEnqueuerProtocol = _default_build_digits_enqueuer

# Hook for setup_logging. Tests override to skip actual logging setup.
setup_logging: SetupLoggingProtocol = _default_setup_logging

# Hook for creating ServiceContainer from env. Tests override with fakes.
create_service_container: CreateServiceContainerProtocol = _default_create_service_container

# Hook for creating BotOrchestrator. Tests override with fakes.
create_bot_orchestrator: CreateBotOrchestratorProtocol = _default_create_bot_orchestrator

# Hook for urlsplit. Tests override to simulate parse errors.
urlsplit: UrlSplitProtocol = _default_urlsplit

# Hook for TrainerEventSubscriber factory. Tests override with fakes.
trainer_event_subscriber_factory: TrainerEventSubscriberFactoryProtocol = (
    _default_trainer_event_subscriber_factory
)

# Hook for HTTPModelTrainerClient factory. Tests override with fakes.
trainer_api_client_factory: TrainerApiClientFactoryProtocol = _default_trainer_api_client_factory

# Hook for DigitsEventSubscriber factory. Tests override with fakes.
digits_event_subscriber_factory: DigitsEventSubscriberFactoryProtocol = (
    _default_digits_event_subscriber_factory
)

# Hook for wrap_interaction. Tests override to return fake interactions.
wrap_interaction: WrapInteractionProtocol = _default_wrap_interaction

# Hook for tree sync. Tests override to observe sync calls.
tree_sync: TreeSyncFactoryProtocol = _default_tree_sync

# Hook for Discord exception types. Tests override to use custom exception types.
discord_exception_types: DiscordExceptionTypesProtocol = _default_discord_exception_types

# Hook for app command error handler. Tests override with protocol-typed fakes.
app_command_error_handler: AppCommandErrorHandlerProtocol = _default_app_command_error_handler


# =============================================================================
# Transcript Cog Hooks
# =============================================================================


class ValidateYoutubeUrlProtocol(Protocol):
    """Protocol for YouTube URL validation function."""

    def __call__(self, url: str) -> str:
        """Validate a YouTube URL and return the canonical form.

        Raises:
            AppError: If URL is invalid.
        """
        ...


def _default_validate_youtube_url(url: str) -> str:
    """Production implementation - validates YouTube URL."""
    from clubbot.utils.youtube import validate_youtube_url as _real_validate

    return _real_validate(url)


# Hook for YouTube URL validation. Tests override to skip validation.
validate_youtube_url: ValidateYoutubeUrlProtocol = _default_validate_youtube_url

# Hook for bot.fetch_user. Tests override to inject non-UserProto for defensive testing.
bot_fetch_user: BotFetchUserProtocol = _default_bot_fetch_user


class _SyncCallable(Protocol):
    """Protocol for sync functions that can be run in a thread."""

    def __call__(self, url: str) -> TranscriptResultLike:
        """Call the function with a URL."""
        ...


class AsyncioToThreadProtocol(Protocol):
    """Protocol for asyncio.to_thread wrapper.

    Note: This protocol is typed for TranscriptResult to satisfy guards.
    Production code uses asyncio.to_thread directly.
    """

    async def __call__(self, func: _SyncCallable, url: str) -> TranscriptResultLike:
        """Run a function in a thread pool."""
        ...


async def _default_asyncio_to_thread(func: _SyncCallable, url: str) -> TranscriptResultLike:
    """Production implementation - uses asyncio.to_thread."""
    import asyncio as _asyncio

    return await _asyncio.to_thread(func, url)


# Hook for asyncio.to_thread. Tests override for synchronous testing.
asyncio_to_thread: AsyncioToThreadProtocol = _default_asyncio_to_thread


__all__ = [
    "AddListenerProtocol",
    "AppCommand",
    "AppCommandErrorHandlerProtocol",
    "AppCommandLike",
    # Transcript cog protocols
    "AsyncioToThreadProtocol",
    # Protocols
    "BotFetchUserProtocol",
    "BotLikeProtocol",
    "BotOrchestratorProtocol",
    "BotProto",
    "BotRunnerProtocol",
    "BotTreeProto",
    "BuildAsyncClientProtocol",
    "BuildClientProtocol",
    "BuildDigitsEnqueuerProtocol",
    # Transcript service protocols
    "CaptionsProtocol",
    "CogLike",
    "CommandTree",
    "CreateBotOrchestratorProtocol",
    "CreateServiceContainerProtocol",
    "DigitsEnqueuerLike",
    "DigitsEventSubscriberFactoryProtocol",
    "DigitsEventSubscriberLike",
    # Digits notifier protocols
    "DigitsRuntime",
    "DiscordExceptionTypesProtocol",
    "DiscordInteractionLike",
    "EmbedLike",
    "ExtractVideoIdProtocol",
    "FetchedUserLike",
    "FileLike",
    "FollowupProtoLike",
    "GetCogProtocol",
    "GetServiceRegistryProtocol",
    "GuardFindMonorepoRootProtocol",
    "GuardLoadOrchestratorProtocol",
    "GuardRunForProjectProtocol",
    "GuildLikeProto",
    "HttpxModuleProtocol",
    "InteractionProtoLike",
    "LoadSettingsProtocol",
    "MessageLike",
    "OnCompletedProtocol",
    "OrchestratorBuildBotHookProtocol",
    "OrchestratorBuildBotProtocol",
    "OrchestratorLike",
    "OrchestratorSyncGlobalProtocol",
    # QR service protocols
    "QRResultLike",
    "QRServiceFactoryProtocol",
    "QRServiceLike",
    "RequestAction",
    "ResponseProtoLike",
    "RqBytesClientFactoryProtocol",
    "RqQueueProtocol",
    "RqRetryProtocol",
    "ServiceContainerProtocol",
    "ServiceDef",
    "SetupLoggingProtocol",
    "SnowflakeLike",
    "SnowflakeProto",
    "TimeoutCtorProtocol",
    "TrainResponseLike",
    "TrainerApiClientFactoryProtocol",
    "TrainerApiClientLike",
    "TrainerEventSubscriberFactoryProtocol",
    "TrainerEventSubscriberLike",
    "TranscriptPayloadDict",
    "TranscriptResultLike",
    "TreeSyncFactoryProtocol",
    "TreeSyncProtocol",
    "UrlSplitProtocol",
    "UserProtoLike",
    "ValidateYoutubeUrlForClientProtocol",
    "ValidateYoutubeUrlProtocol",
    "WrapInteractionProtocol",
    "_SyncCallable",
    "_default_app_command_error_handler",
    # Transcript cog defaults
    "_default_asyncio_to_thread",
    # Default implementations
    "_default_build_async_client",
    "_default_build_client",
    "_default_build_digits_enqueuer",
    # Transcript service defaults
    "_default_captions",
    "_default_create_bot_orchestrator",
    "_default_create_service_container",
    "_default_digits_event_subscriber_factory",
    "_default_discord_exception_types",
    "_default_extract_video_id",
    "_default_get_service_registry",
    "_default_guard_find_monorepo_root",
    "_default_guard_load_orchestrator",
    "_default_load_httpx_module",
    "_default_load_settings",
    # Digits notifier defaults
    "_default_on_completed",
    "_default_orchestrator_build_bot",
    # QR service defaults
    "_default_qr_service_factory",
    "_default_redis_raw_for_rq",
    "_default_rq_queue",
    "_default_rq_retry",
    "_default_setup_logging",
    "_default_trainer_api_client_factory",
    "_default_trainer_event_subscriber_factory",
    "_default_tree_sync",
    "_default_urlsplit",
    "_default_validate_youtube_url",
    "_default_validate_youtube_url_for_client",
    "_default_wrap_interaction",
    "app_command_error_handler",
    # Transcript cog hooks
    "asyncio_to_thread",
    "bot_fetch_user",
    # Module-level hooks
    "build_async_client",
    "build_client",
    "build_digits_enqueuer",
    # Transcript service hooks
    "captions",
    "create_bot_orchestrator",
    "create_service_container",
    "digits_event_subscriber_factory",
    "discord_exception_types",
    "extract_video_id",
    "get_service_registry",
    "guard_find_monorepo_root",
    "guard_load_orchestrator",
    "load_httpx_module",
    "load_settings",
    # Digits notifier hooks
    "on_completed",
    "orchestrator_add_listener_override",
    "orchestrator_build_bot",
    "orchestrator_build_bot_override",
    "orchestrator_get_cog_override",
    "orchestrator_sync_global_override",
    # QR service hooks
    "qr_service_factory",
    "redis_raw_for_rq",
    "rq_queue",
    "rq_retry",
    "setup_logging",
    "trainer_api_client_factory",
    "trainer_event_subscriber_factory",
    "tree_sync",
    "urlsplit",
    "validate_youtube_url",
    "validate_youtube_url_for_client",
    "wrap_interaction",
]
