"""Trainer event subscriber and Discord notifications (strict, typed, DRY)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from platform_discord.bot_subscriber import BotEventSubscriber
from platform_discord.protocols import BotProto
from platform_discord.subscriber import MessageSource
from platform_discord.trainer.handler import (
    TrainerEventV1,
    decode_trainer_event,
    handle_trainer_event,
)
from platform_discord.trainer.runtime import (
    RequestAction,
    TrainerRuntime,
    new_runtime,
)

_EVENT_TASK_NAME: Final[str] = "trainer-event-subscriber"
_DEFAULT_TRAINER_EVENTS_CHANNEL: Final[str] = "trainer:events"


class TrainerEventSubscriber(BotEventSubscriber[TrainerEventV1]):
    """Event subscriber for trainer events with Discord DM notifications.

    Inherits lifecycle management and DM notifications from BotEventSubscriber.
    Uses notify_simple since trainer notifications don't need edit-in-place.
    """

    __slots__ = ("_runtime",)

    def __init__(
        self,
        *,
        bot: BotProto,
        redis_url: str,
        events_channel: str | None = None,
        source_factory: Callable[[str], MessageSource] | None = None,
    ) -> None:
        super().__init__(
            bot,
            redis_url=redis_url,
            events_channel=events_channel or _DEFAULT_TRAINER_EVENTS_CHANNEL,
            task_name=_EVENT_TASK_NAME,
            decode=decode_trainer_event,
            source_factory=source_factory,
        )
        self._runtime: TrainerRuntime = new_runtime()

    async def _handle_event(self, ev: TrainerEventV1) -> None:
        act: RequestAction | None = handle_trainer_event(self._runtime, ev)
        if act is not None:
            await self._maybe_notify(act)

    async def _maybe_notify(self, act: RequestAction) -> None:
        embed = act["embed"]
        if embed is None:
            return
        await self.notify(act["user_id"], act["request_id"], embed)


__all__ = ["TrainerEventSubscriber"]
