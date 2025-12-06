"""Base class for Discord bot event subscribers with task lifecycle management.

This module provides a generic base class for bot event subscribers that:
- Manage task lifecycle (start/stop) via TaskRunner
- Handle bot integration (fetch_user, send DM)
- Support message caching for edit-in-place notifications

Subclasses must implement _handle_event() to process decoded events.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Final, Generic, TypedDict, TypeVar

from platform_core.logging import get_logger

from .embed_helpers import EmbedProto
from .message_source import RedisPubSubSource
from .protocols import BotProto, MessageProto
from .subscriber import MessageSource, RedisEventSubscriber
from .task_runner import Closable, Runnable, TaskRunner

E = TypeVar("E")

Decoder = Callable[[str], E | None]
Handler = Callable[[E], Awaitable[None]]


class _BuildResult(TypedDict):
    """TypedDict for TaskRunner build result."""

    runnable: Runnable
    closable: Closable


class BotEventSubscriber(Generic[E]):
    """Base class for bot event subscribers with task lifecycle management.

    Type parameter E is the decoded event type.

    Handles:
    - Task creation and cancellation via TaskRunner
    - Redis pubsub source management
    - User DM notifications with optional edit-in-place
    - Message caching by request ID

    Subclasses must implement:
    - _handle_event(event: E) -> None
    """

    __slots__ = (
        "_bot",
        "_decode",
        "_events_channel",
        "_logger",
        "_messages",
        "_redis_url",
        "_runner",
        "_source",
        "_source_factory",
        "_subscriber",
        "_task_name",
    )

    def __init__(
        self,
        bot: BotProto,
        *,
        redis_url: str,
        events_channel: str,
        task_name: str,
        decode: Decoder[E],
        source_factory: Callable[[str], MessageSource] | None = None,
    ) -> None:
        """Initialize the subscriber.

        Args:
            bot: Discord bot for user fetching.
            redis_url: Redis connection URL.
            events_channel: Redis pubsub channel to subscribe to.
            task_name: Name for the asyncio task.
            decode: Function to decode raw messages into events.
            source_factory: Optional factory for message source (for testing).
        """
        self._bot: Final[BotProto] = bot
        self._redis_url: Final[str] = redis_url
        self._events_channel: Final[str] = events_channel
        self._task_name: Final[str] = task_name
        self._decode: Final[Decoder[E]] = decode
        self._source_factory: Final[Callable[[str], MessageSource]] = (
            source_factory if source_factory is not None else _default_source_factory
        )
        self._logger = get_logger(__name__)
        self._messages: dict[str, MessageProto] = {}
        self._subscriber: RedisEventSubscriber[E] | None = None
        self._source: MessageSource | None = None

        def _build() -> _BuildResult:
            source = self._source_factory(self._redis_url)
            subscriber: RedisEventSubscriber[E] = RedisEventSubscriber(
                channel=self._events_channel,
                source=source,
                decode=self._decode,
                handle=self._handle_event,
            )
            self._subscriber = subscriber
            self._source = source
            return {"runnable": subscriber, "closable": source}

        def _on_err(exc: BaseException) -> None:
            self._logger.error("Subscriber task %s failed: %s", self._task_name, str(exc))

        self._runner: Final[TaskRunner] = TaskRunner(
            build=_build, name=self._task_name, on_error=_on_err
        )

    @property
    def bot(self) -> BotProto:
        """The Discord bot instance."""
        return self._bot

    @property
    def events_channel(self) -> str:
        """The Redis pubsub channel being subscribed to."""
        return self._events_channel

    @property
    def task_name(self) -> str:
        """The name of the asyncio task."""
        return self._task_name

    @property
    def is_running(self) -> bool:
        """Whether the subscriber task is currently running."""
        return self._runner._task is not None

    def start(self) -> None:
        """Start the subscriber task. Idempotent."""
        self._runner.start()

    async def stop(self) -> None:
        """Stop the subscriber task. Re-raises task exceptions."""
        try:
            await self._runner.stop()
        finally:
            src = self._source
            if src is not None:
                await src.close()
            self._source = None
            self._subscriber = None

    async def _handle_event(self, event: E) -> None:
        """Handle a decoded event. Subclasses must implement.

        Args:
            event: The decoded event to handle.

        Raises:
            NotImplementedError: If subclass doesn't override.
        """
        raise NotImplementedError("Subclasses must implement _handle_event")

    async def notify(
        self,
        user_id: int,
        request_id: str,
        embed: EmbedProto,
        *,
        edit_existing: bool = True,
    ) -> None:
        """Send or edit a DM to a user with an embed.

        If edit_existing is True and a message for request_id exists in cache,
        the existing message will be edited. Otherwise a new message is sent.

        Args:
            user_id: Discord user ID to notify.
            request_id: Request ID for message caching.
            embed: Embed to send.
            edit_existing: If True, edit existing message for request_id.
        """
        user_obj = await self._bot.fetch_user(user_id)

        if edit_existing:
            existing = self._messages.get(request_id)
            if existing is not None:
                await existing.edit(embed=embed)
                return

        msg = await user_obj.send(embed=embed)
        self._messages[request_id] = msg

    def clear_message_cache(self) -> None:
        """Clear the message cache."""
        self._messages.clear()

    def get_cached_message(self, request_id: str) -> MessageProto | None:
        """Get a cached message by request ID.

        Args:
            request_id: The request ID to look up.

        Returns:
            The cached MessageProto, or None if not found.
        """
        return self._messages.get(request_id)

    async def _run(self) -> None:
        """Run one iteration for testing."""
        await self._runner.run_once()


def _default_source_factory(url: str) -> MessageSource:
    """Default factory that creates RedisPubSubSource."""
    return RedisPubSubSource(url)


__all__ = ["BotEventSubscriber", "Decoder"]
