"""Turkic event subscriber and Discord notifications (strict, typed, DRY)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, Protocol

from platform_core.job_events import default_events_channel
from platform_discord.bot_subscriber import BotEventSubscriber
from platform_discord.protocols import BotProto
from platform_discord.subscriber import MessageSource
from platform_discord.turkic import (
    RequestAction,
    TurkicEventV1,
    TurkicRuntime,
    decode_turkic_event,
    new_runtime,
)
from platform_discord.turkic.handler import (
    handle_turkic_event as _real_handle_turkic_event,
)


class _HandleTurkicEventProtocol(Protocol):
    """Protocol for handle_turkic_event function."""

    def __call__(self, runtime: TurkicRuntime, event: TurkicEventV1) -> RequestAction | None:
        """Handle a turkic event and return action or None."""
        ...


def _default_handle_turkic_event(
    runtime: TurkicRuntime, event: TurkicEventV1
) -> RequestAction | None:
    """Production implementation - calls platform_discord handler."""
    return _real_handle_turkic_event(runtime, event)


# Hook for handle_turkic_event. Tests override to return None.
handle_turkic_event: _HandleTurkicEventProtocol = _default_handle_turkic_event

_EVENT_TASK_NAME: Final[str] = "turkic-event-subscriber"


class TurkicEventSubscriber(BotEventSubscriber[TurkicEventV1]):
    """Event subscriber for turkic events with Discord DM notifications.

    Inherits lifecycle management and DM notifications from BotEventSubscriber.
    Uses edit-in-place DM notifications by job_id.
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
            events_channel=events_channel or default_events_channel("turkic"),
            task_name=_EVENT_TASK_NAME,
            decode=decode_turkic_event,
            source_factory=source_factory,
        )
        self._runtime: TurkicRuntime = new_runtime()

    async def _handle_event(self, ev: TurkicEventV1) -> None:
        act: RequestAction | None = handle_turkic_event(self._runtime, ev)
        if act is not None:
            await self._maybe_notify(act)

    async def _maybe_notify(self, act: RequestAction) -> None:
        embed = act["embed"]
        if embed is None:
            return
        await self.notify(act["user_id"], act["job_id"], embed)


__all__ = [
    "TurkicEventSubscriber",
    "_HandleTurkicEventProtocol",
    "_default_handle_turkic_event",
    "handle_turkic_event",
]
