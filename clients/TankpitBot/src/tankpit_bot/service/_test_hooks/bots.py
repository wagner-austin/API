"""Bot-construction seam: what a session runs when it starts.

Kept apart from the serving and process seams because it is the only
one that reaches into the bot itself rather than into the operating
system or the HTTP layer.
"""

from __future__ import annotations

from typing import Protocol

from tankpit_bot.service.session_runner import BotFactoryProtocol
from tankpit_bot.stream.types import StreamConfigDict


class BotFactoryBuilderProtocol(Protocol):
    """Builds a :class:`BotFactoryProtocol` from session-level configuration."""

    def __call__(
        self,
        target_url: str,
        *,
        headless: bool,
        prefer_account: bool,
        stream_config: StreamConfigDict | None,
    ) -> BotFactoryProtocol:
        """Return a bot factory bound to the requested session config.

        Args:
            target_url: URL the bot navigates to on session start.
            headless: Whether the launched Chromium runs headless.
            prefer_account: Whether the bot uses account credentials
                instead of guest login.
            stream_config: Display-capture parameters, or ``None``
                when the session is not streamed.

        Returns:
            A callable that :class:`SessionRunner` invokes once per
            session with a shared bridge + bus.
        """
        ...


def _real_build_bot_factory(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
    stream_config: StreamConfigDict | None,
) -> BotFactoryProtocol:
    """Production bot factory — constructs a real :class:`Bot` per session.

    Args:
        target_url: URL the bot navigates to on session start.
        headless: Whether the launched Chromium runs headless.
        prefer_account: Whether the bot uses account credentials
            instead of guest login.
        stream_config: Display-capture parameters, or ``None`` when
            the session is not streamed. The bot's run then owns the
            Xvfb + ffmpeg pair for the session's lifetime.

    Returns:
        A :class:`BotFactoryProtocol` callable that
        :class:`SessionRunner` invokes once per session.
    """
    from tankpit_bot.bot.base import Bot
    from tankpit_bot.bus.mode_bridge import ModeBridgeProtocol
    from tankpit_bot.bus.status_bus import StatusBusProtocol
    from tankpit_bot.service.session_runner import RunnableBotProtocol

    def factory(
        *,
        mode_bridge: ModeBridgeProtocol,
        status_bus: StatusBusProtocol,
    ) -> RunnableBotProtocol:
        return Bot(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
            mode_bridge=mode_bridge,
            status_bus=status_bus,
            stream_config=stream_config,
        )

    return factory


#: Bot-factory builder hook — production constructs a real
#: :class:`Bot`; tests replace with a fake that produces a
#: :class:`RunnableBotProtocol` stub.
build_bot_factory: BotFactoryBuilderProtocol = _real_build_bot_factory
