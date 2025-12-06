from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable
from typing import Protocol

import discord
from discord.ext import commands
from monorepo_guards._types import UnknownJson
from platform_core.logging import get_logger
from platform_discord.exceptions import DForbiddenError as _DForbiddenError
from platform_discord.exceptions import DHTTPExceptionError as _DHTTPExceptionError
from platform_discord.exceptions import DNotFoundError as _DNotFoundError

from .container import ServiceContainer


class GuildLike(Protocol):
    """Protocol for objects that have guild ID and name attributes."""

    id: int
    name: str


class BotOrchestrator:
    """Coordinates bot lifecycle: build, wire, listen, and run.

    - Builds the `commands.Bot` instance
    - Wires cogs via the ServiceContainer (pre-login; no lazy loading)
    - Registers event listeners
    - Validates token preflight and runs the bot
    - Handles command sync policy with per-guild fallback
    """

    def __init__(self, container: ServiceContainer) -> None:
        self.container = container
        self.bot: commands.Bot | None = None
        self.logger = get_logger(__name__)
        # Sync bookkeeping to avoid hammering the commands endpoints on reconnects
        self._has_synced_once: bool = False
        self._last_present_ids: set[int] = set()
        # Track a separate one-time global sync for DM availability
        self._has_synced_global: bool = False
        # Exposed listeners for tests (set in register_listeners)
        self._on_ready_listener: Callable[[], Awaitable[None]] | None = None
        self._on_guild_join_listener: Callable[[GuildLike], Awaitable[None]] | None = None

    def build_bot(self) -> commands.Bot:
        intents = discord.Intents.default()
        # Enable message content if you see warnings about privileged intent.
        # This requires the Message Content Intent to be enabled in the Developer Portal.
        intents.message_content = True

        # discord.py manages app commands via bot.tree; override setup_hook for lifecycle wiring.
        container = self.container
        register_listeners = self.register_listeners

        class _Bot(commands.Bot):
            async def setup_hook(self) -> None:
                await container.wire_bot_async(self)
                register_listeners()
                start_subscribers = self_orchestrator.start_background_subscribers
                start_subscribers()

        self_orchestrator = self
        self.bot = _Bot(command_prefix="!", intents=intents)
        return self.bot

    async def sync_commands(self) -> None:
        assert self.bot is not None
        logger = self.logger
        task: asyncio.Task[bool] = asyncio.create_task(self._sync_global())
        await asyncio.wait({task})
        exc = task.exception()
        if exc is None:
            did_global: bool = bool(task.result())
            if not did_global:
                logger.info("Command sync is up-to-date; no changes applied")
            return
        exc_class_name = exc.__class__.__name__ if isinstance(exc, BaseException) else ""
        if "Forbidden" in exc_class_name:
            logger.warning(
                "Missing access when syncing commands; skipping global fallback: %s", exc
            )
            return
        if "HTTP" in exc_class_name:
            raise exc
        raise RuntimeError("Command sync failed")

    # Per-guild sync removed: we rely on global commands only.

    async def _sync_global(self) -> bool:
        assert self.bot is not None
        cfg = self.container.cfg
        logger = self.logger
        if not cfg["discord"]["commands_sync_global"]:
            return False
        if self._has_synced_global:
            logger.info("Global command sync already performed in this process; skipping")
            return False
        await self.bot.tree.sync()
        self._has_synced_global = True
        logger.info("Performed global command sync (DMs enabled; propagation may take time)")
        return True

    def register_listeners(self) -> None:
        assert self.bot is not None

        async def on_ready() -> None:
            logger = get_logger(__name__)
            assert self.bot is not None
            logger.info(
                "Logged in as %s (ID: %s)",
                self.bot.user,
                self.bot.user and self.bot.user.id,
            )
            if await self._sync_global():
                logger.info("Startup global command sync completed")
            else:
                logger.info("Startup global command sync skipped")

        async def on_connect() -> None:
            get_logger(__name__).info("Gateway connected")

        async def on_resumed() -> None:
            get_logger(__name__).info("Gateway resumed session")

        async def on_guild_join(guild: GuildLike) -> None:
            logger = get_logger(__name__)
            logger.info("Joined guild %s (%s)", guild.id, guild.name)
            # Global commands are sufficient; no per-guild sync needed.

        async def on_application_command_error(
            interaction: discord.Interaction, error: Exception
        ) -> None:
            # The cog-level handlers already take care of most user errors;
            # this is a final catch-all.
            logger = get_logger(__name__)
            original_obj: UnknownJson = getattr(error, "original", None)
            original = original_obj if isinstance(original_obj, Exception) else error
            logger.exception("Unhandled application command error: %s", original)
            try:
                if interaction.response.is_done():
                    await interaction.followup.send(
                        "An error occurred. Please try again later.", ephemeral=True
                    )
                else:
                    await interaction.response.send_message(
                        "An error occurred. Please try again later.", ephemeral=True
                    )
            except (_DHTTPExceptionError, _DForbiddenError, _DNotFoundError):
                logger.exception("Failed to send error response to interaction")
                raise

        # Register listeners (no decorators to keep type-checkers happy)
        self.bot.add_listener(on_ready)
        self.bot.add_listener(on_connect)
        self.bot.add_listener(on_resumed)
        self.bot.add_listener(on_guild_join)
        self.bot.add_listener(on_application_command_error)
        # Expose for tests (avoid accessing internal listener tables)
        self._on_ready_listener = on_ready
        self._on_guild_join_listener = on_guild_join

    def start_background_subscribers(self) -> None:
        assert self.bot is not None
        # Registry-driven startup: map service ids to cog names
        from .services.registry import SERVICE_REGISTRY

        id_to_cog: dict[str, str] = {
            "digits": "DigitsCog",
            "trainer": "TrainerCog",
            # transcript currently API-only in this bot; no subscriber class
        }
        for sid, cog_name in id_to_cog.items():
            if sid not in SERVICE_REGISTRY:
                continue
            cog = self.bot.get_cog(cog_name)
            if cog is not None and hasattr(cog, "ensure_subscriber_started"):
                cog.ensure_subscriber_started()

    def _preflight_token_check(self) -> None:
        token = self.container.cfg["discord"]["token"]
        if token.startswith("Bot "):
            raise RuntimeError(
                "DISCORD_TOKEN should be the raw token string, without the 'Bot ' prefix."
            )

    def run(self) -> None:
        # Build
        bot = self.build_bot()
        # Cog wiring and listener registration handled in setup_hook
        # Validate and run
        self._preflight_token_check()
        bot.run(self.container.cfg["discord"]["token"])

    # Internal helpers
    def _token_matches_app_id(self, token: str, app_id: str) -> bool:
        first = token.split(".")[0]
        padding = "=" * (-len(first) % 4)
        decoded = base64.b64decode(first + padding).decode("utf-8", errors="strict")
        return decoded == app_id
