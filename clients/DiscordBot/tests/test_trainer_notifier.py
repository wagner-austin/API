from __future__ import annotations

import asyncio
import logging

from platform_core.job_events import JobFailedV1, make_failed_event
from platform_core.trainer_metrics_events import (
    TrainerCompletedMetricsV1,
    TrainerConfigV1,
    TrainerProgressMetricsV1,
    make_completed_metrics_event,
    make_config_event,
    make_progress_metrics_event,
)
from platform_discord.embed_helpers import EmbedProto
from platform_discord.protocols import FileProto, MessageProto, UserProto
from platform_discord.trainer.handler import TrainerEventV1
from tests.support.discord_fakes import FakeMessage

from clubbot.services.jobs.trainer_notifier import TrainerEventSubscriber


class _TrackingUser:
    """Fake user that tracks sent embeds for test verification."""

    def __init__(self) -> None:
        self.sent_embeds: list[EmbedProto] = []

    @property
    def id(self) -> int:
        return 12345

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto:
        if embed is not None:
            self.sent_embeds.append(embed)
        return FakeMessage()


class _TrackingBot:
    """Fake bot that tracks fetch_user calls for test verification."""

    def __init__(self) -> None:
        self.user = _TrackingUser()
        self.last_user_id: int | None = None

    async def fetch_user(self, user_id: int, /) -> UserProto:
        self.last_user_id = user_id
        return self.user


def _config() -> TrainerConfigV1:
    return make_config_event(
        job_id="r",
        user_id=1,
        model_family="gpt2",
        model_size="small",
        total_epochs=1,
        queue="training",
    )


def _progress() -> TrainerProgressMetricsV1:
    return make_progress_metrics_event(
        job_id="r",
        user_id=1,
        epoch=1,
        total_epochs=1,
        step=10,
        loss=1.0,
    )


def _completed() -> TrainerCompletedMetricsV1:
    return make_completed_metrics_event(
        job_id="r",
        user_id=1,
        loss=0.5,
        perplexity=2.0,
        artifact_path="/x",
    )


def _failed() -> JobFailedV1:
    return make_failed_event(
        domain="trainer",
        job_id="r",
        user_id=1,
        error_kind="system",
        message="boom",
    )


def test_notifier_handles_all_events() -> None:
    bot = _TrackingBot()
    sub = TrainerEventSubscriber(bot=bot, redis_url="redis://example")

    async def _run() -> None:
        config_ev: TrainerEventV1 = _config()
        progress_ev: TrainerEventV1 = _progress()
        completed_ev: TrainerEventV1 = _completed()
        failed_ev: TrainerEventV1 = _failed()
        await sub._handle_event(config_ev)
        await sub._handle_event(progress_ev)
        await sub._handle_event(completed_ev)
        await sub._handle_event(failed_ev)

    asyncio.run(_run())
    assert bot.last_user_id == 1
    # Verify at least one message was sent (embeds are in the sent list)
    assert bot.user.sent_embeds


logger = logging.getLogger(__name__)
