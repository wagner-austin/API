"""Turkic event subscriber and Discord notifications (strict, typed, DRY)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, TypeGuard

from platform_core.job_events import (
    JobCompletedV1,
    JobEventV1,
    JobFailedV1,
    JobProgressV1,
    JobStartedV1,
    decode_job_event,
    default_events_channel,
)
from platform_discord.bot_subscriber import BotEventSubscriber
from platform_discord.protocols import BotProto
from platform_discord.subscriber import MessageSource
from platform_discord.turkic.runtime import (
    RequestAction,
    TurkicRuntime,
    new_runtime,
)
from platform_discord.turkic.runtime import (
    on_completed as _on_completed,
)
from platform_discord.turkic.runtime import (
    on_failed as _on_failed,
)
from platform_discord.turkic.runtime import (
    on_progress as _on_progress,
)
from platform_discord.turkic.runtime import (
    on_started as _on_started,
)

_EVENT_TASK_NAME: Final[str] = "turkic-event-subscriber"


def _is_started(ev: JobEventV1) -> TypeGuard[JobStartedV1]:
    return ev["type"] == "turkic.job.started.v1"


def _is_progress(ev: JobEventV1) -> TypeGuard[JobProgressV1]:
    return ev["type"] == "turkic.job.progress.v1"


def _is_completed(ev: JobEventV1) -> TypeGuard[JobCompletedV1]:
    return ev["type"] == "turkic.job.completed.v1"


def _is_failed(ev: JobEventV1) -> TypeGuard[JobFailedV1]:
    return ev["type"] == "turkic.job.failed.v1"


class TurkicEventSubscriber(BotEventSubscriber[JobEventV1]):
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
        def _decode(message: str) -> JobEventV1 | None:
            event = decode_job_event(message)
            if event["domain"] != "turkic":
                return None
            return event

        super().__init__(
            bot,
            redis_url=redis_url,
            events_channel=events_channel or default_events_channel("turkic"),
            task_name=_EVENT_TASK_NAME,
            decode=_decode,
            source_factory=source_factory,
        )
        self._runtime: TurkicRuntime = new_runtime()

    async def _handle_event(self, ev: JobEventV1) -> None:
        if _is_started(ev):
            act: RequestAction = _on_started(
                self._runtime,
                user_id=ev["user_id"],
                job_id=ev["job_id"],
                queue=ev["queue"],
            )
            await self._maybe_notify(act)
            return
        if _is_progress(ev):
            progress_ev = ev
            act2: RequestAction = _on_progress(
                self._runtime,
                user_id=progress_ev["user_id"],
                job_id=progress_ev["job_id"],
                progress=progress_ev["progress"],
                message=progress_ev.get("message"),
            )
            await self._maybe_notify(act2)
            return
        if _is_completed(ev):
            completed_ev = ev
            act3: RequestAction = _on_completed(
                self._runtime,
                user_id=completed_ev["user_id"],
                job_id=completed_ev["job_id"],
                result_id=completed_ev["result_id"],
                result_bytes=completed_ev["result_bytes"],
            )
            await self._maybe_notify(act3)
            return
        if _is_failed(ev):
            failed_ev = ev
            act4: RequestAction = _on_failed(
                self._runtime,
                user_id=failed_ev["user_id"],
                job_id=failed_ev["job_id"],
                error_kind=failed_ev["error_kind"],
                message=failed_ev["message"],
                status="failed",
            )
            await self._maybe_notify(act4)

    async def _maybe_notify(self, act: RequestAction) -> None:
        embed = act["embed"]
        if embed is None:
            return
        await self.notify(act["user_id"], act["job_id"], embed)


__all__ = ["TurkicEventSubscriber"]
