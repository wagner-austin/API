"""Tests for BotEventSubscriber base class."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypedDict

import pytest

from platform_discord.bot_subscriber import BotEventSubscriber, _default_source_factory
from platform_discord.embed_helpers import EmbedData, EmbedFieldData, EmbedProto
from platform_discord.protocols import BotProto, FileProto, MessageProto, UserProto
from platform_discord.subscriber import MessageSource


class _TestEvent(TypedDict):
    """Test event type."""

    type: str
    user_id: int
    request_id: str


class _FakeSource(MessageSource):
    """Fake message source for testing."""

    def __init__(self) -> None:
        self.messages: list[str | None] = []
        self.subscribed: str | None = None
        self.closed: bool = False

    async def subscribe(self, channel: str) -> None:
        self.subscribed = channel

    async def get(self) -> str | None:
        await asyncio.sleep(0)
        if self.messages:
            return self.messages.pop(0)
        return None

    async def close(self) -> None:
        self.closed = True


class _FakeMessage:
    """Protocol-compliant fake for MessageProto."""

    def __init__(self, msg_id: int) -> None:
        self._id = msg_id
        self.edit_calls: list[EmbedProto | None] = []

    @property
    def id(self) -> int:
        return self._id

    async def edit(
        self, *, content: str | None = None, embed: EmbedProto | None = None
    ) -> MessageProto:
        _ = content
        self.edit_calls.append(embed)
        return self


class _FakeUser:
    """Protocol-compliant fake for UserProto."""

    def __init__(self, user_id: int) -> None:
        self._id = user_id
        self.send_calls: list[tuple[str | None, EmbedProto | None]] = []
        self._next_message_id = 1

    @property
    def id(self) -> int:
        return self._id

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto:
        _ = file
        self.send_calls.append((content, embed))
        msg = _FakeMessage(self._next_message_id)
        self._next_message_id += 1
        return msg


class _FakeBot:
    """Protocol-compliant fake for BotProto."""

    def __init__(self) -> None:
        self._users: dict[int, _FakeUser] = {}
        self.fetch_calls: list[int] = []

    def set_user(self, user_id: int, user: _FakeUser) -> None:
        self._users[user_id] = user

    async def fetch_user(self, user_id: int, /) -> UserProto:
        self.fetch_calls.append(user_id)
        if user_id in self._users:
            return self._users[user_id]
        user = _FakeUser(user_id)
        self._users[user_id] = user
        return user


class _FakeEmbed:
    """Protocol-compliant fake for EmbedProto."""

    __slots__ = ("_color", "_description", "_fields", "_footer_text", "_title")

    def __init__(
        self,
        *,
        title: str | None = None,
        description: str | None = None,
        color: int | None = None,
    ) -> None:
        self._title = title
        self._description = description
        self._color = color
        self._footer_text: str | None = None
        self._fields: list[EmbedFieldData] = []

    @property
    def title(self) -> str | None:
        return self._title

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def color_value(self) -> int | None:
        return self._color

    @property
    def footer_text(self) -> str | None:
        return self._footer_text

    @property
    def field_count(self) -> int:
        return len(self._fields)

    def add_field(self, *, name: str, value: str, inline: bool = True) -> None:
        field: EmbedFieldData = {"name": name, "value": value, "inline": inline}
        self._fields.append(field)

    def set_footer(self, *, text: str) -> None:
        self._footer_text = text

    def get_field(self, name: str) -> EmbedFieldData | None:
        for field in self._fields:
            if field["name"] == name:
                return field
        return None

    def has_field(self, name: str) -> bool:
        return self.get_field(name) is not None

    def get_all_fields(self) -> list[EmbedFieldData]:
        return list(self._fields)

    def get_field_value(self, name: str) -> str | None:
        field = self.get_field(name)
        return field["value"] if field is not None else None

    def to_dict(self) -> EmbedData:
        result: EmbedData = {}
        if self._title is not None:
            result["title"] = self._title
        if self._description is not None:
            result["description"] = self._description
        if self._color is not None:
            result["color"] = self._color
        if self._fields:
            result["fields"] = list(self._fields)
        if self._footer_text:
            result["footer"] = {"text": self._footer_text, "icon_url": None}
        return result


_SourceFactory = Callable[[str], MessageSource]


class _ConcreteSubscriber(BotEventSubscriber[_TestEvent]):
    """Concrete implementation for testing."""

    def __init__(
        self,
        bot: BotProto,
        *,
        redis_url: str,
        events_channel: str,
        source_factory: _SourceFactory | None = None,
    ) -> None:
        super().__init__(
            bot,
            redis_url=redis_url,
            events_channel=events_channel,
            task_name="test-subscriber",
            decode=_decode_test_event,
            source_factory=source_factory,
        )
        self.handled_events: list[_TestEvent] = []

    async def _handle_event(self, event: _TestEvent) -> None:
        self.handled_events.append(event)


def _decode_test_event(payload: str) -> _TestEvent | None:
    """Decode test event from JSON-like string."""
    if '"type":"test"' not in payload:
        return None
    return {"type": "test", "user_id": 123, "request_id": "req-1"}


@pytest.mark.asyncio
async def test_properties() -> None:
    """Test that properties return correct values."""
    bot = _FakeBot()
    sub = _ConcreteSubscriber(
        bot,
        redis_url="redis://localhost",
        events_channel="test:events",
    )
    assert sub.bot is bot
    assert sub.events_channel == "test:events"
    assert sub.task_name == "test-subscriber"
    assert sub.is_running is False


@pytest.mark.asyncio
async def test_start_and_stop() -> None:
    """Test start creates task and stop cancels it."""
    source = _FakeSource()
    source.messages = [None]  # Single message to process
    bot = _FakeBot()
    sub = _ConcreteSubscriber(
        bot,
        redis_url="redis://localhost",
        events_channel="test:events",
        source_factory=lambda _: source,
    )
    sub.start()
    assert sub.is_running is True
    await asyncio.sleep(0)
    await sub.stop()
    assert sub.is_running is False
    assert source.closed is True


@pytest.mark.asyncio
async def test_start_idempotent() -> None:
    """Test start is idempotent."""
    source = _FakeSource()
    source.messages = [None]
    bot = _FakeBot()
    sub = _ConcreteSubscriber(
        bot,
        redis_url="redis://localhost",
        events_channel="test:events",
        source_factory=lambda _: source,
    )
    sub.start()
    sub.start()  # Should be no-op
    await asyncio.sleep(0)
    await sub.stop()
    assert source.closed is True


@pytest.mark.asyncio
async def test_handle_event_raises_not_implemented() -> None:
    """Test that base class _handle_event raises NotImplementedError."""
    bot = _FakeBot()

    # Create instance directly without concrete implementation
    base_sub: BotEventSubscriber[_TestEvent] = BotEventSubscriber(
        bot,
        redis_url="redis://localhost",
        events_channel="test:events",
        task_name="base-test",
        decode=_decode_test_event,
    )
    with pytest.raises(NotImplementedError, match="Subclasses must implement"):
        await base_sub._handle_event({"type": "test", "user_id": 1, "request_id": "r"})


@pytest.mark.asyncio
async def test_notify_sends_new_message() -> None:
    """Test notify sends new DM when no cached message."""
    bot = _FakeBot()
    user = _FakeUser(123)
    bot.set_user(123, user)
    sub = _ConcreteSubscriber(
        bot,
        redis_url="redis://localhost",
        events_channel="test:events",
    )
    embed = _FakeEmbed(title="Test")
    await sub.notify(123, "req-1", embed)
    assert len(user.send_calls) == 1
    # Message should be cached
    cached = sub.get_cached_message("req-1")
    if cached is None:
        pytest.fail("expected cached message")


@pytest.mark.asyncio
async def test_notify_edits_existing_message() -> None:
    """Test notify edits existing message when cached."""
    bot = _FakeBot()
    user = _FakeUser(123)
    bot.set_user(123, user)
    sub = _ConcreteSubscriber(
        bot,
        redis_url="redis://localhost",
        events_channel="test:events",
    )
    embed1 = _FakeEmbed(title="First")
    embed2 = _FakeEmbed(title="Second")
    # Send first message
    await sub.notify(123, "req-1", embed1)
    assert len(user.send_calls) == 1
    # Edit with second message
    await sub.notify(123, "req-1", embed2)
    # Should not send new message
    assert len(user.send_calls) == 1
    # Should have edited
    cached = sub.get_cached_message("req-1")
    if cached is None:
        pytest.fail("expected cached message")
    if type(cached) is not _FakeMessage:
        pytest.fail("expected _FakeMessage instance")
    assert len(cached.edit_calls) == 1


@pytest.mark.asyncio
async def test_notify_skip_edit_when_disabled() -> None:
    """Test notify sends new message when edit_existing=False."""
    bot = _FakeBot()
    user = _FakeUser(123)
    bot.set_user(123, user)
    sub = _ConcreteSubscriber(
        bot,
        redis_url="redis://localhost",
        events_channel="test:events",
    )
    embed1 = _FakeEmbed(title="First")
    embed2 = _FakeEmbed(title="Second")
    await sub.notify(123, "req-1", embed1)
    await sub.notify(123, "req-1", embed2, edit_existing=False)
    # Should have sent two messages
    assert len(user.send_calls) == 2


@pytest.mark.asyncio
async def test_clear_message_cache() -> None:
    """Test clear_message_cache removes all cached messages."""
    bot = _FakeBot()
    user = _FakeUser(123)
    bot.set_user(123, user)
    sub = _ConcreteSubscriber(
        bot,
        redis_url="redis://localhost",
        events_channel="test:events",
    )
    embed = _FakeEmbed(title="Test")
    await sub.notify(123, "req-1", embed)
    await sub.notify(123, "req-2", embed, edit_existing=False)
    if sub.get_cached_message("req-1") is None:
        pytest.fail("expected cached message for req-1")
    if sub.get_cached_message("req-2") is None:
        pytest.fail("expected cached message for req-2")
    sub.clear_message_cache()
    assert sub.get_cached_message("req-1") is None
    assert sub.get_cached_message("req-2") is None


@pytest.mark.asyncio
async def test_get_cached_message_returns_none_for_unknown() -> None:
    """Test get_cached_message returns None for unknown request ID."""
    bot = _FakeBot()
    sub = _ConcreteSubscriber(
        bot,
        redis_url="redis://localhost",
        events_channel="test:events",
    )
    assert sub.get_cached_message("unknown") is None


@pytest.mark.asyncio
async def test_run_processes_events() -> None:
    """Test _run processes events via run_once."""
    source = _FakeSource()
    source.messages = ['{"type":"test"}']
    bot = _FakeBot()
    sub = _ConcreteSubscriber(
        bot,
        redis_url="redis://localhost",
        events_channel="test:events",
        source_factory=lambda _: source,
    )
    await sub._run()
    assert len(sub.handled_events) == 1
    assert sub.handled_events[0]["type"] == "test"


@pytest.mark.asyncio
async def test_default_source_factory() -> None:
    """Test _default_source_factory creates RedisPubSubSource."""
    from platform_discord.message_source import RedisPubSubSource

    source = _default_source_factory("redis://localhost")
    assert type(source) is RedisPubSubSource


@pytest.mark.asyncio
async def test_stop_without_source() -> None:
    """Test stop handles case when source is None."""
    bot = _FakeBot()
    sub = _ConcreteSubscriber(
        bot,
        redis_url="redis://localhost",
        events_channel="test:events",
    )
    # Stop without ever starting
    await sub.stop()
    # Should not raise


@pytest.mark.asyncio
async def test_stop_closes_source_in_finally() -> None:
    """Test stop closes source even if runner.stop raises."""
    source = _FakeSource()
    source.messages = []
    bot = _FakeBot()
    sub = _ConcreteSubscriber(
        bot,
        redis_url="redis://localhost",
        events_channel="test:events",
        source_factory=lambda _: source,
    )
    sub.start()
    await asyncio.sleep(0)
    # Force an error in the task
    task = sub._runner._task
    if task is not None:
        task.cancel()
    await asyncio.sleep(0)
    await sub.stop()
    assert source.closed is True


@pytest.mark.asyncio
async def test_bot_property() -> None:
    """Test bot property returns the bot instance."""
    bot = _FakeBot()
    sub = _ConcreteSubscriber(
        bot,
        redis_url="redis://localhost",
        events_channel="test:events",
    )
    assert sub.bot == bot


class _FailingSource(MessageSource):
    """Source that raises an error after subscribing."""

    def __init__(self) -> None:
        self.closed = False

    async def subscribe(self, channel: str) -> None:
        pass

    async def get(self) -> str | None:
        raise RuntimeError("Simulated failure")

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_on_error_callback_called_on_task_failure() -> None:
    """Test that _on_err callback is called when task fails."""
    source = _FailingSource()
    bot = _FakeBot()
    sub = _ConcreteSubscriber(
        bot,
        redis_url="redis://localhost",
        events_channel="test:events",
        source_factory=lambda _: source,
    )
    sub.start()
    # Wait for task to fail
    await asyncio.sleep(0.05)
    # The error callback should have been called and logged
    # Stop will re-raise the exception
    with pytest.raises(RuntimeError, match="Simulated failure"):
        await sub.stop()
    assert source.closed is True
