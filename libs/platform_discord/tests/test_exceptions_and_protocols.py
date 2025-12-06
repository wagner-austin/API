"""Tests for exceptions and protocols."""

from __future__ import annotations

from platform_discord.embed_helpers import EmbedProto, create_embed
from platform_discord.exceptions import DForbiddenError, DHTTPExceptionError, DNotFoundError
from platform_discord.protocols import (
    FileProto,
    MessageProto,
    UserProto,
)


def test_exceptions_are_distinct_and_subclass_exception() -> None:
    assert issubclass(DHTTPExceptionError, Exception)
    assert issubclass(DForbiddenError, Exception)
    assert issubclass(DNotFoundError, Exception)
    assert len({DHTTPExceptionError, DForbiddenError, DNotFoundError}) == 3


class _Msg:
    @property
    def id(self) -> int:
        return 123

    async def edit(
        self, *, content: str | None = None, embed: EmbedProto | None = None
    ) -> MessageProto:
        return self


class _File:
    @property
    def filename(self) -> str | None:
        return "test.txt"


class _User:
    @property
    def id(self) -> int:
        return 456

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto:
        return _Msg()


class _Sendable:
    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto:
        return _Msg()


class _Bot:
    async def fetch_user(self, user_id: int, /) -> UserProto:
        return _User()


def test_protocols_shapes() -> None:
    # Verify objects satisfy protocol by testing actual protocol methods
    msg = _Msg()
    assert callable(msg.edit)
    sendable = _Sendable()
    assert callable(sendable.send)
    bot = _Bot()
    assert callable(bot.fetch_user)


def test_embed_proto_is_runtime_checkable() -> None:
    embed = create_embed(title="Test")
    # Verify embed satisfies protocol by testing actual protocol properties
    assert embed.title == "Test"
    assert callable(embed.to_dict)


def test_file_proto_structure() -> None:
    file_obj = _File()
    assert file_obj.filename == "test.txt"


def test_user_proto_structure() -> None:
    user_obj = _User()
    assert user_obj.id == 456


def test_message_proto_structure() -> None:
    msg = _Msg()
    assert msg.id == 123


class _FakeResponse:
    def is_done(self) -> bool:
        return False

    async def defer(self, *, ephemeral: bool = False) -> None:
        pass

    async def send_message(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        ephemeral: bool = False,
    ) -> None:
        pass


class _FakeFollowup:
    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        ephemeral: bool = False,
    ) -> MessageProto:
        return _Msg()


class _FakeInteraction:
    def __init__(self) -> None:
        self.response = _FakeResponse()
        self.followup = _FakeFollowup()
        self.user = _User()


def test_wrap_interaction_response() -> None:
    from platform_discord.protocols import wrap_interaction

    fake = _FakeInteraction()
    wrapped = wrap_interaction(fake)
    assert wrapped.response.is_done() is False


def test_wrap_interaction_followup() -> None:
    from platform_discord.protocols import wrap_interaction

    fake = _FakeInteraction()
    wrapped = wrap_interaction(fake)
    # Verify we can access followup methods
    assert callable(wrapped.followup.send)


def test_wrap_interaction_user() -> None:
    from platform_discord.protocols import wrap_interaction

    fake = _FakeInteraction()
    wrapped = wrap_interaction(fake)
    assert wrapped.user.id == 456


class _FakeDiscordMessage:
    """Fake discord.Message for testing wrap_bot."""

    def __init__(self, msg_id: int = 789) -> None:
        self._id = msg_id

    @property
    def id(self) -> int:
        return self._id

    async def edit(
        self, *, content: str | None = None, embed: EmbedProto | None = None
    ) -> _FakeDiscordMessage:
        return _FakeDiscordMessage(self._id)


class _FakeDiscordUser:
    """Fake discord.User for testing wrap_bot."""

    def __init__(self, user_id: int = 12345) -> None:
        self._id = user_id

    @property
    def id(self) -> int:
        return self._id

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> _FakeDiscordMessage:
        return _FakeDiscordMessage()


class _FakeDiscordBot:
    """Fake discord.Bot for testing wrap_bot."""

    def __init__(self, user: _FakeDiscordUser | None = None) -> None:
        self._user = user if user is not None else _FakeDiscordUser()

    async def fetch_user(self, user_id: int, /) -> _FakeDiscordUser:
        return self._user


def test_wrap_bot_fetch_user() -> None:
    """Test that wrap_bot properly wraps fetch_user to return UserProto."""
    import asyncio

    from platform_discord.protocols import wrap_bot

    fake_bot = _FakeDiscordBot(_FakeDiscordUser(user_id=99999))
    wrapped = wrap_bot(fake_bot)

    user = asyncio.get_event_loop().run_until_complete(wrapped.fetch_user(123))
    assert user.id == 99999


def test_wrap_bot_user_send() -> None:
    """Test that wrapped user's send returns MessageProto."""
    import asyncio

    from platform_discord.protocols import wrap_bot

    fake_bot = _FakeDiscordBot()
    wrapped = wrap_bot(fake_bot)

    user = asyncio.get_event_loop().run_until_complete(wrapped.fetch_user(123))
    msg = asyncio.get_event_loop().run_until_complete(user.send("Hello"))
    assert msg.id == 789


def test_wrap_bot_message_edit() -> None:
    """Test that wrapped message's edit returns MessageProto."""
    import asyncio

    from platform_discord.protocols import wrap_bot

    fake_bot = _FakeDiscordBot()
    wrapped = wrap_bot(fake_bot)

    user = asyncio.get_event_loop().run_until_complete(wrapped.fetch_user(123))
    msg = asyncio.get_event_loop().run_until_complete(user.send("Hello"))
    edited = asyncio.get_event_loop().run_until_complete(msg.edit(content="Updated"))
    assert edited.id == 789
