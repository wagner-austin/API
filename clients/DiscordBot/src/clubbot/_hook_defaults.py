"""Production implementations behind every clubbot hook.

:mod:`clubbot._test_hooks` binds each hook to the function here that does the
real work, so the hook is callable without any test having run. Tests rebind
the hook on the facade module and restore it afterwards; nothing in this
module reads a hook, which is what keeps that rebinding effective.
"""

from __future__ import annotations

import urllib.parse as _url

from discord.abc import Snowflake as DiscordSnowflake
from discord.app_commands import AppCommand
from platform_core.config.discordbot import Settings
from platform_core.config.discordbot import load_settings as _real_load_discordbot_settings
from platform_core.http_client import HttpxAsyncClient, HttpxClient
from platform_core.http_client import build_async_client as _real_build_async_client
from platform_core.http_client import build_client as _real_build_client
from platform_core.logging import LogFormat, LogLevel
from platform_discord.protocols import (
    BotProto,
    InteractionProto,
)
from platform_workers.rq_harness import (
    RQClientQueue,
    RQRetryLike,
    _RedisBytesClient,
)
from platform_workers.rq_harness import redis_raw_for_rq as _real_redis_raw_for_rq
from platform_workers.rq_harness import rq_queue as _real_rq_queue
from platform_workers.rq_harness import rq_retry as _real_rq_retry

from clubbot._hook_protocols_platform import (
    BotOrchestratorProtocol,
    DigitsEnqueuerLike,
    FetchedUserLike,
    HttpxModuleProtocol,
    ServiceContainerProtocol,
    TranscriptResultLike,
)
from clubbot._hook_protocols_services import (
    BotRunnerProtocol,
    BotTreeProto,
    DigitsEventSubscriberLike,
    DiscordInteractionLike,
    OrchestratorLike,
    QRServiceLike,
    TrainerApiClientLike,
    TrainerEventSubscriberLike,
    _SyncCallable,
)
from clubbot.services.registry import ServiceDef


async def _default_bot_fetch_user(bot: BotProto, user_id: int) -> FetchedUserLike:
    """Production implementation - calls bot.fetch_user directly."""
    return await bot.fetch_user(user_id)


def _default_qr_service_factory(cfg: Settings) -> QRServiceLike:
    """Production implementation - creates QRService."""
    from clubbot.services.qr.client import QRService

    return QRService(cfg)


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


def _default_discord_exception_types() -> tuple[type[Exception], type[Exception], type[Exception]]:
    """Production implementation - returns actual platform_discord exceptions."""
    from platform_discord.exceptions import DForbiddenError, DHTTPExceptionError, DNotFoundError

    return (DHTTPExceptionError, DForbiddenError, DNotFoundError)


def _default_orchestrator_build_bot(orchestrator: OrchestratorLike) -> BotRunnerProtocol:
    """Production implementation - calls orchestrator.build_bot()."""
    return orchestrator.build_bot()


def _default_get_service_registry() -> dict[str, ServiceDef]:
    """Production implementation - returns actual SERVICE_REGISTRY."""
    from clubbot.services.registry import SERVICE_REGISTRY

    return SERVICE_REGISTRY


def _default_load_settings() -> Settings:
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


def _default_wrap_interaction(interaction: DiscordInteractionLike) -> InteractionProto:
    """Production implementation - wraps discord.Interaction."""
    from platform_discord.protocols import wrap_interaction as _real_wrap_interaction

    result: InteractionProto = _real_wrap_interaction(interaction)
    return result


async def _default_tree_sync(
    tree: BotTreeProto, guild: DiscordSnowflake | None = None
) -> list[AppCommand]:
    """Production implementation - calls tree.sync() directly."""
    return await tree.sync(guild=guild)


def _default_validate_youtube_url(url: str) -> str:
    """Production implementation - validates YouTube URL."""
    from clubbot.utils.youtube import validate_youtube_url as _real_validate

    return _real_validate(url)


async def _default_asyncio_to_thread(func: _SyncCallable, url: str) -> TranscriptResultLike:
    """Production implementation - uses asyncio.to_thread."""
    import asyncio as _asyncio

    return await _asyncio.to_thread(func, url)
