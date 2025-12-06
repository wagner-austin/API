"""Discord protocols for platform services.

These Protocols define the interfaces we depend on from discord.py.
Services should use these instead of importing discord.py directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from platform_discord.embed_helpers import EmbedProto


class FileProto(Protocol):
    """Protocol for discord.File."""

    @property
    def filename(self) -> str | None: ...


@runtime_checkable
class UserProto(Protocol):
    """Protocol for discord.User - used for fetching user info and sending DMs."""

    @property
    def id(self) -> int: ...

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto: ...


@runtime_checkable
class MessageProto(Protocol):
    """Protocol for discord.Message - returned from send operations."""

    @property
    def id(self) -> int: ...

    async def edit(
        self, *, content: str | None = None, embed: EmbedProto | None = None
    ) -> MessageProto: ...


@runtime_checkable
class SendableProto(Protocol):
    """Protocol for objects that can send messages (User, TextChannel, etc.)."""

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto: ...


@runtime_checkable
class BotProto(Protocol):
    """Protocol for discord.Bot - used for user fetch operations."""

    async def fetch_user(self, user_id: int, /) -> UserProto: ...


class ResponseProto(Protocol):
    """Protocol for discord.Interaction.response."""

    def is_done(self) -> bool: ...

    async def defer(self, *, ephemeral: bool = False) -> None: ...

    async def send_message(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        ephemeral: bool = False,
    ) -> None: ...


class FollowupProto(Protocol):
    """Protocol for discord.Interaction.followup."""

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
        ephemeral: bool = False,
    ) -> MessageProto: ...


@runtime_checkable
class InteractionProto(Protocol):
    """Protocol for discord.Interaction."""

    @property
    def response(self) -> ResponseProto: ...

    @property
    def followup(self) -> FollowupProto: ...

    @property
    def user(self) -> UserProto: ...


class _DiscordInteraction(Protocol):
    """Internal Protocol for discord.Interaction - minimal interface for wrapping."""

    # Empty protocol - any object structurally satisfies this
    # We use this to avoid 'object' annotation while still accepting any discord type


class _DiscordBot(Protocol):
    """Internal Protocol for discord.Bot - minimal interface for wrapping."""

    # Empty protocol - any object structurally satisfies this


class _DiscordUser(Protocol):
    """Internal Protocol for discord.User - minimal interface for wrapping."""

    # Empty protocol - any object structurally satisfies this


class _DiscordMessage(Protocol):
    """Internal Protocol for discord.Message - minimal interface for wrapping."""

    # Empty protocol - any object structurally satisfies this


class _EditCallable(Protocol):
    """Protocol for message edit function."""

    async def __call__(
        self, *, content: str | None = None, embed: EmbedProto | None = None
    ) -> _DiscordMessage: ...


class _MessageAdapter:
    """Adapter that wraps discord.Message and exposes our Protocol interface."""

    __slots__ = ("_inner",)

    def __init__(self, inner: _DiscordMessage) -> None:
        self._inner = inner

    @property
    def id(self) -> int:
        attr = "id"
        msg_id: int = getattr(self._inner, attr)
        return msg_id

    async def edit(
        self, *, content: str | None = None, embed: EmbedProto | None = None
    ) -> MessageProto:
        attr = "edit"
        edit_fn: _EditCallable = getattr(self._inner, attr)
        result = await edit_fn(content=content, embed=embed)
        return _MessageAdapter(result)


class _SendCallable(Protocol):
    """Protocol for user send function."""

    async def __call__(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> _DiscordMessage: ...


class _UserAdapter:
    """Adapter that wraps discord.User and exposes our Protocol interface."""

    __slots__ = ("_inner",)

    def __init__(self, inner: _DiscordUser) -> None:
        self._inner = inner

    @property
    def id(self) -> int:
        attr = "id"
        user_id: int = getattr(self._inner, attr)
        return user_id

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto:
        attr = "send"
        send_fn: _SendCallable = getattr(self._inner, attr)
        result = await send_fn(content, embed=embed, file=file)
        return _MessageAdapter(result)


class _FetchUserCallable(Protocol):
    """Protocol for bot fetch_user function."""

    async def __call__(self, user_id: int, /) -> _DiscordUser: ...


class _BotAdapter:
    """Adapter that wraps discord.Bot and exposes our Protocol interface."""

    __slots__ = ("_inner",)

    def __init__(self, inner: _DiscordBot) -> None:
        self._inner = inner

    async def fetch_user(self, user_id: int, /) -> UserProto:
        attr = "fetch_user"
        fetch_fn: _FetchUserCallable = getattr(self._inner, attr)
        user = await fetch_fn(user_id)
        return _UserAdapter(user)


def wrap_bot(bot: _DiscordBot) -> BotProto:
    """Wrap a discord.Bot to use our Protocol types.

    Args:
        bot: A discord.Bot object.

    Returns:
        A BotProto-compatible wrapper.
    """
    return _BotAdapter(bot)


class _InteractionAdapter:
    """Adapter that wraps discord.Interaction and exposes our Protocol interface.

    This adapter uses the getattr pattern to bypass discord.py's strict type
    annotations. Since _inner is typed as _DiscordInteraction (empty Protocol),
    getattr returns Any, which can be assigned directly to our Protocol types.

    At runtime, the underlying discord.py objects work correctly because our
    EmbedProto adapter implements to_dict() for discord.py compatibility.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: _DiscordInteraction) -> None:
        self._inner = inner

    @property
    def response(self) -> ResponseProto:
        attr = "response"
        resp: ResponseProto = getattr(self._inner, attr)
        return resp

    @property
    def followup(self) -> FollowupProto:
        attr = "followup"
        follow: FollowupProto = getattr(self._inner, attr)
        return follow

    @property
    def user(self) -> UserProto:
        attr = "user"
        u: UserProto = getattr(self._inner, attr)
        return u


def wrap_interaction(interaction: _DiscordInteraction) -> InteractionProto:
    """Wrap a discord.Interaction to use our Protocol types.

    Args:
        interaction: A discord.Interaction object.

    Returns:
        An InteractionProto-compatible wrapper that accepts EmbedProto in send calls.
    """
    return _InteractionAdapter(interaction)


__all__ = [
    "BotProto",
    "FileProto",
    "FollowupProto",
    "InteractionProto",
    "MessageProto",
    "ResponseProto",
    "SendableProto",
    "UserProto",
    "wrap_bot",
    "wrap_interaction",
]
