"""Protocol types for clubbot's service and Discord-facing hooks.

These cover the QR, transcript, trainer and digits services plus the Discord
plumbing the orchestrator and cogs reach through -- command tree sync, the
exception classes, interaction wrapping. The platform-level protocols they
build on are in :mod:`clubbot._hook_protocols_platform`.
"""

from __future__ import annotations

from typing import Protocol

import discord
from discord.abc import Snowflake as DiscordSnowflake
from discord.app_commands import AppCommand
from platform_core.config import DiscordbotSettings
from platform_discord.protocols import (
    BotProto,
    InteractionProto,
    _DiscordUser,
)

from clubbot._hook_protocols_platform import (
    TranscriptResultLike,
)
from clubbot.services.registry import ServiceDef


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


class DiscordExceptionTypesProtocol(Protocol):
    """Protocol for Discord exception types factory.

    Returns a tuple of exception types that the orchestrator catches
    when sending error responses to interactions.
    """

    def __call__(self) -> tuple[type[Exception], type[Exception], type[Exception]]:
        """Return (HTTPException, ForbiddenError, NotFoundError) exception types."""
        ...


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


class BotRunnerProtocol(Protocol):
    """Protocol for objects that can run like a bot."""

    def run(self, token: str) -> None:
        """Run the bot with the given token."""
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


class GetServiceRegistryProtocol(Protocol):
    """Protocol for getting service registry."""

    def __call__(self) -> dict[str, ServiceDef]:
        """Return the service registry."""
        ...


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

    def __call__(self, interaction: DiscordInteractionLike) -> InteractionProto:
        """Wrap a discord.Interaction into InteractionProto."""
        ...


class ValidateYoutubeUrlProtocol(Protocol):
    """Protocol for YouTube URL validation function."""

    def __call__(self, url: str) -> str:
        """Validate a YouTube URL and return the canonical form.

        Raises:
            AppError: If URL is invalid.
        """
        ...


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
