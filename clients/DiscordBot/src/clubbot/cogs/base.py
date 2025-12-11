from __future__ import annotations

import asyncio
import uuid
from typing import Protocol, runtime_checkable

import discord
from discord.ext import commands
from monorepo_guards._types import UnknownJson
from platform_core.logging import get_logger
from platform_discord.exceptions import DForbiddenError as _DForbiddenError
from platform_discord.exceptions import DHTTPExceptionError as _DHTTPExceptionError
from platform_discord.exceptions import DNotFoundError as _DNotFoundError
from platform_discord.protocols import BotProto as _BotProto
from platform_discord.protocols import InteractionProto as _InteractionProto
from platform_discord.protocols import UserProto as _UserProto
from platform_discord.rate_limiter import RateLimiter

from clubbot import _test_hooks

# Use InteractionProto for methods that only use response/followup/user
_InteractionType = _InteractionProto

_DM_EXC: tuple[type[Exception], ...] = (
    _DHTTPExceptionError,
    _DForbiddenError,
    _DNotFoundError,
    RuntimeError,
)


@runtime_checkable
class BotForSetup(_BotProto, Protocol):
    async def add_cog(self, cog: commands.Cog) -> None: ...


class _Logger(Protocol):
    def debug(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None: ...
    def info(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None: ...
    def warning(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None: ...
    def exception(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None: ...


class _ExtraLogger:
    __slots__ = ("_extra", "_logger")

    def __init__(self, logger: _Logger, extra: dict[str, str]) -> None:
        self._logger = logger
        self._extra = extra

    @property
    def extra(self) -> dict[str, str]:
        return self._extra

    def debug(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
        merged = self._extra if extra is None else {**self._extra, **extra}
        self._logger.debug(msg, *args, extra=merged)

    def info(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
        merged = self._extra if extra is None else {**self._extra, **extra}
        self._logger.info(msg, *args, extra=merged)

    def warning(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
        merged = self._extra if extra is None else {**self._extra, **extra}
        self._logger.warning(msg, *args, extra=merged)

    def exception(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
        merged = self._extra if extra is None else {**self._extra, **extra}
        self._logger.exception(msg, *args, extra=merged)


class _HasIntId(Protocol):
    @property
    def id(self) -> int | None: ...


class BaseCog(commands.Cog):
    """Shared base for cogs to provide request-scoped logging and helpers."""

    def __init__(self) -> None:
        super().__init__()
        self.logger = get_logger(self.__class__.__module__)
        # Bot is injected by discord.py at runtime when the cog is added.
        # Keep loosely typed; concrete cogs can narrow.
        self.bot: _BotProto | None = None

    @staticmethod
    def new_request_id() -> str:
        # Short, unique request id for correlating logs
        return uuid.uuid4().hex[:8]

    def request_logger(self, request_id: str) -> _ExtraLogger:
        return _ExtraLogger(self.logger, {"request_id": request_id})

    async def _safe_defer(self, interaction: _InteractionType, *, ephemeral: bool) -> bool:
        if interaction.response.is_done():
            return True
        task = asyncio.create_task(interaction.response.defer(ephemeral=ephemeral))
        # Yield control to allow defer to run immediately
        await asyncio.sleep(0)
        if not task.done():
            # Defer call did not fail synchronously; treat as successful ack
            return True
        exc = task.exception()
        if exc is None:
            return True
        code_obj: UnknownJson = getattr(exc, "code", None)
        return bool(isinstance(code_obj, int) and code_obj == 40060)

    async def handle_user_error(
        self,
        interaction: _InteractionType,
        log: _Logger,
        message: str,
    ) -> None:
        log.debug("User error: %s", message)
        content = message
        try:
            if interaction.response.is_done():
                await interaction.followup.send(content, ephemeral=True)
            else:
                await interaction.response.send_message(content, ephemeral=True)
        except _DM_EXC as e:
            try:
                await interaction.followup.send(content, ephemeral=True)
            except _DM_EXC:
                self.logger.warning("Failed to send user error response: %s", e)
                return

    async def handle_exception(
        self,
        interaction: _InteractionType,
        log: _Logger,
        exc: Exception,
    ) -> None:
        log.exception("Unhandled exception: %s", str(exc))
        # Include request id for traceability in user message if available
        req_id: str | None = None
        extra_obj: UnknownJson = getattr(log, "extra", None)
        if isinstance(extra_obj, dict):
            rid = extra_obj.get("request_id")
            if isinstance(rid, str):
                req_id = rid

        base_msg = "An error occurred. Please try again later."
        content = base_msg + (f" (req={req_id})" if req_id else "")

        try:
            if interaction.response.is_done():
                await interaction.followup.send(content, ephemeral=True)
            else:
                await interaction.response.send_message(content, ephemeral=True)
        except _DM_EXC as e:
            self.logger.warning("Failed to send exception response: %s", e)
            return

    async def notify_user(self, user_id: int, message: str) -> None:
        try:
            bot = self.bot
            if bot is None:
                return
            user_obj = await _test_hooks.bot_fetch_user(bot, user_id)
            if isinstance(user_obj, _UserProto):
                await user_obj.send(message)
                return
            raise _DHTTPExceptionError("Bot user object lacks send()")
        except _DM_EXC as e:
            get_logger(__name__).warning("Failed to DM user=%s: %s", user_id, e)
            return

    async def dm_file(self, user_id: int, content: str, file: discord.File) -> None:
        try:
            bot = self.bot
            if bot is None:
                return
            user_obj = await _test_hooks.bot_fetch_user(bot, user_id)
            if isinstance(user_obj, _UserProto):
                await user_obj.send(content, file=file)
                return
            raise _DHTTPExceptionError("Bot user object lacks send()")
        except _DM_EXC as e:
            get_logger(__name__).warning("Failed to DM file to user=%s: %s", user_id, e)
            return

    @staticmethod
    def decode_int_attr(obj: _HasIntId | None, name: str) -> int | None:
        if obj is None or name != "id":
            return None
        value = obj.id
        return value if isinstance(value, int) else None

    async def check_rate_limit(
        self,
        interaction: _InteractionType,
        *,
        rate_limiter: RateLimiter,
        user_id: int,
        command: str,
        log: _Logger,
        public_responses: bool,
    ) -> bool:
        allowed, wait_seconds = rate_limiter.allow(user_id, command)
        if allowed:
            return True
        await interaction.followup.send(
            f"Please wait {int(wait_seconds)} seconds before using this command again",
            ephemeral=not public_responses,
        )
        log.info("Rate limited user=%s command=%s", str(user_id), command)
        return False


__all__ = ["BaseCog", "BotForSetup", "_BotProto", "_Logger"]
