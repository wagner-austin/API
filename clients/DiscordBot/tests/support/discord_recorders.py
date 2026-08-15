"""Recording and tracking doubles for Discord testing.

Each of these keeps a list of what it was asked to send, edit or log, so a
test can assert on the calls rather than on a return value. The inert fakes
they are built from are in tests.support.discord_fakes.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_discord.embed_helpers import EmbedProto
from platform_discord.protocols import (
    FileProto,
    MessageProto,
    UserProto,
)

from tests.support.discord_fakes import FakeMessage, FakeUser, NoIdUser


class RecordedSend(TypedDict):
    """Record of a message sent via response or followup."""

    where: Literal["response", "followup"]
    content: str | None
    embed: EmbedProto | None
    file: FileProto | None
    ephemeral: bool


class RecordingResponse:
    """Response that records sends for assertions."""

    def __init__(self, sent: list[RecordedSend], *, done: bool = False) -> None:
        self._done = done
        self._sent = sent

    def is_done(self) -> bool:
        return self._done

    async def send_message(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        ephemeral: bool = False,
    ) -> None:
        self._done = True
        self._sent.append(
            {
                "where": "response",
                "content": content,
                "embed": embed,
                "file": None,
                "ephemeral": ephemeral,
            }
        )

    async def defer(self, *, ephemeral: bool = False) -> None:
        self._done = True
        self._sent.append(
            {
                "where": "response",
                "content": None,
                "embed": None,
                "file": None,
                "ephemeral": ephemeral,
            }
        )


class RecordingFollowup:
    """Followup that records sends for assertions."""

    def __init__(self, sent: list[RecordedSend]) -> None:
        self._sent = sent

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
        ephemeral: bool = False,
    ) -> MessageProto:
        self._sent.append(
            {
                "where": "followup",
                "content": content,
                "embed": embed,
                "file": file,
                "ephemeral": ephemeral,
            }
        )
        return FakeMessage()


class RecordingInteraction:
    """Interaction that records all response/followup sends."""

    def __init__(
        self,
        *,
        user: UserProto | None = None,
        response_done: bool = False,
    ) -> None:
        usr: UserProto = user if user is not None else FakeUser()
        self._user = usr
        self.sent: list[RecordedSend] = []
        self.response = RecordingResponse(self.sent, done=response_done)
        self.followup = RecordingFollowup(self.sent)

    @property
    def user(self) -> UserProto:
        return self._user


class NoIdUserInteraction:
    """Interaction with a NoIdUser for testing user.id=None error paths.

    This is a separate class because NoIdUser.id returns int|None which doesn't
    match UserProto.id -> int, so we can't use RecordingInteraction.
    """

    def __init__(self, *, response_done: bool = False) -> None:
        self._user = NoIdUser()
        self.sent: list[RecordedSend] = []
        self.response = RecordingResponse(self.sent, done=response_done)
        self.followup = RecordingFollowup(self.sent)

    @property
    def user(self) -> NoIdUser:
        return self._user


class TrackingMessage:
    """Message that records embed edits into its parent TrackingUser."""

    def __init__(self, owner: TrackingUser) -> None:
        self._owner = owner

    @property
    def id(self) -> int:
        return 12345

    async def edit(
        self, *, content: str | None = None, embed: EmbedProto | None = None
    ) -> MessageProto:
        _ = content
        if embed is not None:
            self._owner.embeds.append(embed)
        return self


class TrackingUser:
    """User that records sent and edited embeds in a list."""

    def __init__(self, *, user_id: int = 67890) -> None:
        self._id = user_id
        self.embeds: list[EmbedProto | None] = []

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
        _ = (content, file)
        self.embeds.append(embed)
        return TrackingMessage(self)


class TrackingBot:
    """Bot that returns a provided TrackingUser from fetch_user."""

    def __init__(self, user: TrackingUser) -> None:
        self._user = user

    async def fetch_user(self, user_id: int, /) -> UserProto:
        _ = user_id
        return self._user


class RecordingLogger:
    """Logger that records all calls for test verification."""

    def __init__(self) -> None:
        self.debug_calls: list[tuple[str, tuple[str, ...], dict[str, str] | None]] = []
        self.info_calls: list[tuple[str, tuple[str, ...], dict[str, str] | None]] = []
        self.warning_calls: list[tuple[str, tuple[str, ...], dict[str, str] | None]] = []
        self.exception_calls: list[tuple[str, tuple[str, ...], dict[str, str] | None]] = []
        self.extra: dict[str, str] | None = None

    def debug(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
        self.debug_calls.append((msg, args, extra))

    def info(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
        self.info_calls.append((msg, args, extra))

    def warning(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
        self.warning_calls.append((msg, args, extra))

    def exception(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
        self.exception_calls.append((msg, args, extra))


class RecordingLoggerWithExtra(RecordingLogger):
    """Logger with extra dict for request_id propagation tests."""

    def __init__(self, extra: dict[str, str]) -> None:
        super().__init__()
        self.extra = extra
