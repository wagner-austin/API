from __future__ import annotations

import logging
from typing import Protocol

import discord
import pytest
from discord.ext import commands
from platform_discord.protocols import wrap_bot
from tests.conftest import _build_settings

from clubbot.cogs.qr import QRCog
from clubbot.config import DiscordbotSettings
from clubbot.services.qr.client import QRService


class _AllowedContexts(Protocol):
    """Protocol for command allowed contexts."""

    guild: bool
    dm_channel: bool
    private_channel: bool


class _AllowedInstalls(Protocol):
    """Protocol for command allowed installs."""

    user: bool
    guild: bool


class _AppCommandLike(Protocol):
    """Protocol for app command that has name and metadata attributes."""

    name: str


class _CommandTree(Protocol):
    """Protocol for command tree."""

    def get_commands(self) -> list[_AppCommandLike]: ...


def _make_cfg() -> DiscordbotSettings:
    return _build_settings(qr_default_border=2, qr_api_url="http://localhost:8080")


def _get_tree(bot: commands.Bot) -> _CommandTree:
    """Get command tree from bot using Protocol type."""
    attr_name = "tree"
    tree: _CommandTree = object.__getattribute__(bot, attr_name)
    return tree


def _find_command_metadata(
    bot: commands.Bot, name: str
) -> tuple[bool, str, _AllowedContexts | None, _AllowedInstalls | None]:
    """Find command by name and return its metadata via getattr chain.

    Returns (found, name, allowed_contexts, allowed_installs).
    """
    tree = _get_tree(bot)
    cmds_list = tree.get_commands()
    for item in cmds_list:
        cmd_name = item.name
        if cmd_name == name:
            contexts: _AllowedContexts | None = object.__getattribute__(item, "allowed_contexts")
            installs: _AllowedInstalls | None = object.__getattribute__(item, "allowed_installs")
            return (True, cmd_name, contexts, installs)
    return (False, "", None, None)


@pytest.mark.asyncio
async def test_qrcode_metadata_allows_dms_and_user_installs() -> None:
    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)
    cfg = _make_cfg()
    service = QRService(cfg)

    await bot.add_cog(QRCog(wrap_bot(bot), cfg, service))

    found, cmd_name, ctxs, inst = _find_command_metadata(bot, "qrcode")
    assert found, "qrcode command not registered on app command tree"
    assert cmd_name == "qrcode"

    # Validate contexts include DM/guild via allowed_contexts (discord.py 2.4)
    if ctxs is None:
        raise AssertionError("expected ctxs")
    assert ctxs.guild is True
    assert ctxs.dm_channel is True
    assert ctxs.private_channel is True

    # Validate installs include user and guild via allowed_installs
    if inst is None:
        raise AssertionError("expected inst")
    assert inst.user is True
    assert inst.guild is True


logger = logging.getLogger(__name__)
