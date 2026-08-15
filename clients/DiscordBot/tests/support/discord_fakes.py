"""Protocol-compliant fake implementations for Discord testing.

These fakes implement the protocols defined in platform_discord.protocols
without importing discord.py directly, avoiding Any types. They answer calls
and hold no assertions; the doubles that record what was sent to them are in
tests.support.discord_recorders.
"""

from __future__ import annotations

from typing import NoReturn

import discord
from platform_discord.embed_helpers import EmbedData, EmbedFieldData, EmbedProto
from platform_discord.protocols import (
    FileProto,
    FollowupProto,
    MessageProto,
    ResponseProto,
    UserProto,
)


class FakeEmbed:
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


class FakeFile:
    """Protocol-compliant fake for FileProto."""

    __slots__ = ("_filename",)

    def __init__(self, filename: str | None = None) -> None:
        self._filename = filename

    @property
    def filename(self) -> str | None:
        return self._filename


class FakeMessage:
    """Protocol-compliant fake for a Discord message."""

    @property
    def id(self) -> int:
        return 12345

    async def edit(
        self, *, content: str | None = None, embed: EmbedProto | None = None
    ) -> MessageProto:
        return self


class FakeUser:
    """Protocol-compliant fake implementing UserProto."""

    def __init__(self, *, user_id: int = 67890) -> None:
        self._id = user_id
        self.sent: list[tuple[str | None, EmbedProto | None, FileProto | None]] = []

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
        self.sent.append((content, embed, file))
        return FakeMessage()


class FakeResponse:
    """Protocol-compliant fake implementing ResponseProto."""

    def __init__(self, *, done: bool = False) -> None:
        self._done = done
        self.sent: list[tuple[str | None, EmbedProto | None, bool]] = []

    def is_done(self) -> bool:
        return self._done

    async def send_message(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        ephemeral: bool = False,
    ) -> None:
        self.sent.append((content, embed, ephemeral))

    async def defer(self, *, ephemeral: bool = False) -> None:
        self._done = True


class FakeResponseRaises:
    """Fake response that raises on send_message."""

    def __init__(self, *, done: bool = False) -> None:
        self._done = done

    def is_done(self) -> bool:
        return self._done

    async def send_message(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        ephemeral: bool = False,
    ) -> None:
        raise RuntimeError("send_message failed")

    async def defer(self, *, ephemeral: bool = False) -> None:
        self._done = True


class FakeFollowup:
    """Protocol-compliant fake implementing FollowupProto."""

    def __init__(self) -> None:
        self.sent: list[tuple[str | None, EmbedProto | None, FileProto | None, bool]] = []

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
        ephemeral: bool = False,
    ) -> MessageProto:
        self.sent.append((content, embed, file, ephemeral))
        return FakeMessage()


class FakeFollowupRaises:
    """Fake followup that raises on send."""

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
        ephemeral: bool = False,
    ) -> NoReturn:
        _ = (content, embed, file, ephemeral)
        raise RuntimeError("followup.send failed")


class RaisingFollowup:
    """Followup that raises a provided exception type on send."""

    def __init__(self, exc_type: type[BaseException]) -> None:
        self._exc_type = exc_type

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
        ephemeral: bool = False,
    ) -> MessageProto:
        _ = (content, embed, file, ephemeral)
        raise self._exc_type()


class FakeInteraction:
    """Protocol-compliant fake implementing InteractionProto.

    This implements the minimal interface used by BaseCog methods.
    """

    def __init__(
        self,
        *,
        response: ResponseProto | None = None,
        followup: FollowupProto | None = None,
        user: UserProto | None = None,
    ) -> None:
        resp: ResponseProto = FakeResponse() if response is None else response
        follow: FollowupProto = FakeFollowup() if followup is None else followup
        usr: UserProto = FakeUser() if user is None else user
        self._response = resp
        self._followup = follow
        self._user = usr

    @property
    def response(self) -> ResponseProto:
        return self._response

    @property
    def followup(self) -> FollowupProto:
        return self._followup

    @property
    def user(self) -> UserProto:
        return self._user


class FakeBot:
    """Protocol-compliant fake implementing BotProto."""

    def __init__(self, user: UserProto | None = None, *, application_id: int | None = None) -> None:
        self._user: UserProto = user if user is not None else FakeUser()
        self.application_id = application_id

    async def fetch_user(self, user_id: int, /) -> UserProto:
        return self._user


class FakeBotRaises:
    """Fake bot that raises on fetch_user."""

    async def fetch_user(self, user_id: int, /) -> NoReturn:
        raise RuntimeError("fetch_user failed")


class NoIdUser:
    """User-like object with missing id attribute (id -> None)."""

    @property
    def id(self) -> int | None:
        return None

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto:
        _ = (content, embed, file)
        return FakeMessage()


class StrAppIdBot:
    """BotProto with application_id as a string to test negative path."""

    def __init__(self, app_id: str = "1234567890") -> None:
        self.application_id = app_id

    async def fetch_user(self, user_id: int, /) -> UserProto:
        return FakeUser(user_id=user_id)


class NoneAppIdBot:
    """BotProto with application_id = None for env fallback tests."""

    def __init__(self) -> None:
        self.application_id: int | None = None

    async def fetch_user(self, user_id: int, /) -> UserProto:
        return FakeUser(user_id=user_id)


class FakeAttachment(discord.Attachment):
    """Lightweight attachment fake with bytes payload."""

    def __init__(self, *, filename: str, content_type: str, size: int, data: bytes) -> None:
        self.filename = filename
        self.content_type = content_type
        self.size = size
        self._data = data

    async def read(self, *, use_cached: bool = False) -> bytes:
        _ = use_cached
        return self._data


class FakeMessageSource:
    """Fake message source that completes immediately for lifecycle tests."""

    __slots__ = ("_index", "_messages", "closed")

    def __init__(self, *, messages: list[str] | None = None) -> None:
        self.closed = False
        self._messages: list[str] = messages if messages is not None else []
        self._index = 0

    async def subscribe(self, channel: str) -> None:
        _ = channel

    async def get(self) -> str | None:
        if self._index >= len(self._messages):
            return None
        msg = self._messages[self._index]
        self._index += 1
        return msg

    async def close(self) -> None:
        self.closed = True
