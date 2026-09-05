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

The bindings live here and nowhere else, because a test rebinds a hook by
assigning to this module's attribute -- a binding in a submodule would not be
seen by that assignment. The protocol types are in
:mod:`clubbot._hook_protocols_platform` and :mod:`clubbot._hook_protocols_services`
and the production implementations in :mod:`clubbot._hook_defaults`, all
re-exported here so callers still have one import to make.
"""

from __future__ import annotations

import discord
from discord.app_commands import AppCommand, CommandTree
from platform_discord.protocols import BotProto, InteractionProto

from clubbot._hook_defaults import (
    _default_asyncio_to_thread,
    _default_bot_fetch_user,
    _default_build_async_client,
    _default_build_client,
    _default_build_digits_enqueuer,
    _default_captions,
    _default_create_bot_orchestrator,
    _default_create_service_container,
    _default_digits_event_subscriber_factory,
    _default_discord_exception_types,
    _default_extract_video_id,
    _default_get_service_registry,
    _default_load_httpx_module,
    _default_load_settings,
    _default_orchestrator_build_bot,
    _default_qr_service_factory,
    _default_redis_raw_for_rq,
    _default_rq_queue,
    _default_rq_retry,
    _default_setup_logging,
    _default_trainer_api_client_factory,
    _default_trainer_event_subscriber_factory,
    _default_tree_sync,
    _default_validate_youtube_url,
    _default_validate_youtube_url_for_client,
    _default_wrap_interaction,
)
from clubbot._hook_protocols_platform import (
    BotFetchUserProtocol,
    BotLikeProtocol,
    BotOrchestratorProtocol,
    BuildAsyncClientProtocol,
    BuildClientProtocol,
    BuildDigitsEnqueuerProtocol,
    CreateBotOrchestratorProtocol,
    CreateServiceContainerProtocol,
    DigitsEnqueuerLike,
    FetchedUserLike,
    HttpxModuleProtocol,
    LoadHttpxModuleProtocol,
    LoadSettingsProtocol,
    RqBytesClientFactoryProtocol,
    RqQueueProtocol,
    RqRetryProtocol,
    ServiceContainerProtocol,
    SetupLoggingProtocol,
    TimeoutCtorProtocol,
    TranscriptResultLike,
)
from clubbot._hook_protocols_services import (
    AppCommandErrorHandlerProtocol,
    AppCommandLike,
    AsyncioToThreadProtocol,
    BotRunnerProtocol,
    BotTreeProto,
    CaptionsProtocol,
    DigitsEventSubscriberFactoryProtocol,
    DigitsEventSubscriberLike,
    DiscordExceptionTypesProtocol,
    DiscordInteractionLike,
    ExtractVideoIdProtocol,
    GetServiceRegistryProtocol,
    GuildLikeProto,
    OrchestratorBuildBotHookProtocol,
    OrchestratorLike,
    QRResultLike,
    QRServiceFactoryProtocol,
    QRServiceLike,
    SnowflakeLike,
    SnowflakeProto,
    TrainerApiClientFactoryProtocol,
    TrainerApiClientLike,
    TrainerEventSubscriberFactoryProtocol,
    TrainerEventSubscriberLike,
    TrainResponseLike,
    TranscriptPayloadDict,
    TreeSyncFactoryProtocol,
    TreeSyncProtocol,
    ValidateYoutubeUrlForClientProtocol,
    ValidateYoutubeUrlProtocol,
    WrapInteractionProtocol,
    _SyncCallable,
)
from clubbot.services.registry import ServiceDef

# Hook for creating QR service. Tests override for testing.
qr_service_factory: QRServiceFactoryProtocol = _default_qr_service_factory

# Hook for validate_youtube_url in transcript client. Tests override for testing.
validate_youtube_url_for_client: ValidateYoutubeUrlForClientProtocol = (
    _default_validate_youtube_url_for_client
)

# Hook for extract_video_id. Tests override for testing.
extract_video_id: ExtractVideoIdProtocol = _default_extract_video_id

# Hook for captions function. Tests override with fakes.
captions: CaptionsProtocol = _default_captions

# Hook for orchestrator build_bot. Production calls build_bot(); tests override.
orchestrator_build_bot: OrchestratorBuildBotHookProtocol = _default_orchestrator_build_bot

# Hook for getting SERVICE_REGISTRY. Tests override to return custom registries.
get_service_registry: GetServiceRegistryProtocol = _default_get_service_registry

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
# Hook for guard load_orchestrator. Tests override to return fake orchestrators.
# Hook for loading httpx module (used by transcript api_client).
load_httpx_module: LoadHttpxModuleProtocol = _default_load_httpx_module

# Hook for building digits enqueuer. Tests override with fake enqueuer builders.
build_digits_enqueuer: BuildDigitsEnqueuerProtocol = _default_build_digits_enqueuer

# Hook for setup_logging. Tests override to skip actual logging setup.
setup_logging: SetupLoggingProtocol = _default_setup_logging

# Hook for creating ServiceContainer from env. Tests override with fakes.
create_service_container: CreateServiceContainerProtocol = _default_create_service_container

# Hook for creating BotOrchestrator. Tests override with fakes.
create_bot_orchestrator: CreateBotOrchestratorProtocol = _default_create_bot_orchestrator


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


async def _default_app_command_error_handler(
    interaction: discord.Interaction | InteractionProto, error: Exception
) -> None:
    """Production implementation of app command error handler.

    This is the one default that reads a hook rather than only being bound to
    one, so it lives beside the bindings: reading discord_exception_types as a
    global of this module is what lets a test's rebinding of it take effect.
    """
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


# Hook for app command error handler. Tests override with protocol-typed fakes.
app_command_error_handler: AppCommandErrorHandlerProtocol = _default_app_command_error_handler

# Hook for YouTube URL validation. Tests override to skip validation.
validate_youtube_url: ValidateYoutubeUrlProtocol = _default_validate_youtube_url

# Hook for bot.fetch_user. Tests override to inject non-UserProto for defensive testing.
bot_fetch_user: BotFetchUserProtocol = _default_bot_fetch_user

# Hook for asyncio.to_thread. Tests override for synchronous testing.
asyncio_to_thread: AsyncioToThreadProtocol = _default_asyncio_to_thread

__all__ = [
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
    "CommandTree",
    "CreateBotOrchestratorProtocol",
    "CreateServiceContainerProtocol",
    "DigitsEnqueuerLike",
    "DigitsEventSubscriberFactoryProtocol",
    "DigitsEventSubscriberLike",
    "DiscordExceptionTypesProtocol",
    "DiscordInteractionLike",
    "ExtractVideoIdProtocol",
    "FetchedUserLike",
    "GetServiceRegistryProtocol",
    "GuildLikeProto",
    "HttpxModuleProtocol",
    "LoadHttpxModuleProtocol",
    "LoadSettingsProtocol",
    "OrchestratorBuildBotHookProtocol",
    "OrchestratorLike",
    # QR service protocols
    "QRResultLike",
    "QRServiceFactoryProtocol",
    "QRServiceLike",
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
    "ValidateYoutubeUrlForClientProtocol",
    "ValidateYoutubeUrlProtocol",
    "WrapInteractionProtocol",
    "_SyncCallable",
    "_default_app_command_error_handler",
    # Transcript cog defaults
    "_default_asyncio_to_thread",
    # Default implementations
    "_default_bot_fetch_user",
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
    "_default_load_httpx_module",
    "_default_load_settings",
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
    "load_httpx_module",
    "load_settings",
    "orchestrator_build_bot",
    # QR service hooks
    "qr_service_factory",
    "redis_raw_for_rq",
    "rq_queue",
    "rq_retry",
    "setup_logging",
    "trainer_api_client_factory",
    "trainer_event_subscriber_factory",
    "tree_sync",
    "validate_youtube_url",
    "validate_youtube_url_for_client",
    "wrap_interaction",
]
