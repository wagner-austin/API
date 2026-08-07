"""Shared site, bot, and runner doubles for the service-main tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

from tankpit_bot.bus.frame_bus import (
    FrameBusProtocol,
)
from tankpit_bot.bus.mode_bridge import (
    ModeBridgeProtocol,
)
from tankpit_bot.bus.status_bus import (
    StatusBusProtocol,
)
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service.session_runner import (
    BotFactoryProtocol,
    RunnableBotProtocol,
)


class _RecordingSite:
    """SiteRunnerProtocol stand-in that captures its lifecycle calls."""

    def __init__(self) -> None:
        """Initialise the call counters."""
        self.start_calls = 0
        self.cleanup_calls = 0

    async def start(self) -> None:
        """Record one ``start`` invocation."""
        self.start_calls += 1

    async def cleanup(self) -> None:
        """Record one ``cleanup`` invocation."""
        self.cleanup_calls += 1


class _CancellingSite:
    """SiteRunnerProtocol stand-in that cancels the caller on ``start``.

    Raises :class:`asyncio.CancelledError` synchronously from ``start``
    so :func:`run_service_forever`'s ``finally`` runs the site's
    ``cleanup`` before the exception propagates. Simpler and stricter-
    typed than reaching for ``asyncio.current_task().cancel``, which
    leaks ``Any`` through mypy's strict rules.
    """

    def __init__(self) -> None:
        """Initialise the call counters."""
        self.start_calls = 0
        self.cleanup_calls = 0

    async def start(self) -> None:
        """Cancel the calling task by raising :class:`asyncio.CancelledError`."""
        self.start_calls += 1
        raise asyncio.CancelledError

    async def cleanup(self) -> None:
        """Record one ``cleanup`` invocation."""
        self.cleanup_calls += 1


class _RecordingBot:
    """Runnable bot stand-in for the default-bot-factory test."""

    def __init__(self) -> None:
        """Initialise the call log."""
        self.runs: list[tuple[int, Path]] = []

    def run(
        self,
        *,
        session_seconds: int,
        session_kills: int = 0,
        stop_file_path: Path,
    ) -> None:
        """Record one ``run`` invocation."""
        self.runs.append((session_seconds, stop_file_path))


def _make_recording_bot_factory(
    recording_bot: _RecordingBot,
) -> service_hooks.BotFactoryBuilderProtocol:
    """Return a builder that ignores its args and hands back ``recording_bot``.

    Args:
        recording_bot: Bot the produced factory will return per call.

    Returns:
        A :class:`BotFactoryBuilderProtocol`-compatible callable.
    """

    def builder(target_url: str, *, headless: bool, prefer_account: bool) -> BotFactoryProtocol:
        _ = (target_url, headless, prefer_account)

        def factory(
            *,
            mode_bridge: ModeBridgeProtocol,
            status_bus: StatusBusProtocol,
            frame_bus: FrameBusProtocol,
        ) -> RunnableBotProtocol:
            _ = (mode_bridge, status_bus, frame_bus)
            return recording_bot

        return factory

    return builder


class _IdleProbeRunner:
    """``is_running`` stub whose answer the test flips at will."""

    def __init__(self, *, running: bool = False) -> None:
        """Start with the given running answer."""
        self.running = running

    def start(self, *, session_seconds: int = 0, session_kills: int = 0) -> None:
        """Unused — the idle monitor never starts sessions."""
        raise AssertionError("exit_when_idle must never call start()")

    def request_stop(self) -> None:
        """Unused — the idle monitor never stops sessions."""
        raise AssertionError("exit_when_idle must never call request_stop()")

    def is_running(self) -> bool:
        return self.running
