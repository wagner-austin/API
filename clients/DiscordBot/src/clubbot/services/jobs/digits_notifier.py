"""Digits training event subscriber and Discord notifications (strict, typed, DRY)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, Protocol

from platform_core.digits_metrics_events import DEFAULT_DIGITS_EVENTS_CHANNEL
from platform_discord.bot_subscriber import BotEventSubscriber
from platform_discord.handwriting import (
    DigitsEventV1,
    DigitsRuntime,
    RequestAction,
    decode_digits_event_safe,
    new_runtime,
)
from platform_discord.handwriting.handler import (
    handle_digits_event as _real_handle_digits_event,
)
from platform_discord.protocols import BotProto
from platform_discord.subscriber import MessageSource


class _HandleDigitsEventProtocol(Protocol):
    """Protocol for handle_digits_event function."""

    def __call__(self, runtime: DigitsRuntime, event: DigitsEventV1) -> RequestAction | None:
        """Handle a digits event and return action or None."""
        ...


def _default_handle_digits_event(
    runtime: DigitsRuntime, event: DigitsEventV1
) -> RequestAction | None:
    """Production implementation - calls platform_discord handler."""
    return _real_handle_digits_event(runtime, event)


# Hook for handle_digits_event. Tests override to return None.
handle_digits_event: _HandleDigitsEventProtocol = _default_handle_digits_event

_EVENT_TASK_NAME: Final[str] = "digits-event-subscriber"


class DigitsEventSubscriber(BotEventSubscriber[DigitsEventV1]):
    """Event subscriber for digits training events with Discord DM notifications.

    Inherits lifecycle management and DM notifications from BotEventSubscriber.
    Uses notify since digits notifications support edit-in-place.
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
            events_channel=events_channel or DEFAULT_DIGITS_EVENTS_CHANNEL,
            task_name=_EVENT_TASK_NAME,
            decode=decode_digits_event_safe,
            source_factory=source_factory,
        )
        self._runtime: DigitsRuntime = new_runtime()

    async def _handle_event(self, ev: DigitsEventV1) -> None:
        act: RequestAction | None = handle_digits_event(self._runtime, ev)
        if act is not None:
            await self._maybe_notify(act)

    async def _maybe_notify(self, act: RequestAction) -> None:
        embed = act["embed"]
        if embed is None:
            return
        await self.notify(act["user_id"], act["request_id"], embed)


__all__ = [
    "DigitsEventSubscriber",
    "_HandleDigitsEventProtocol",
    "_default_handle_digits_event",
    "handle_digits_event",
]
