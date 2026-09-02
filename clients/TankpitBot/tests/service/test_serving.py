"""Tests for the shared serve loop both HTTP surfaces run under.

Lifted out of ``test_service_main.py`` 2026-09-01 alongside the
function itself, when the fleet manager grew a drain and needed the
same start / wait / cleanup shape the bot service already had.
"""

from __future__ import annotations

import asyncio

import pytest

from tankpit_bot.service.serving import run_until_stopped
from tests.service._service_main_harness import _CancellingSite, _RecordingSite


class TestRunUntilStopped:
    """The blocking serve loop that binds a site to a stop signal."""

    async def test_starts_the_site_and_waits_on_the_stop_event(self) -> None:
        """Site is started, teardown waits until the stop event fires."""
        site = _RecordingSite()
        stop_event = asyncio.Event()

        async def flip_after_first_tick() -> None:
            await asyncio.sleep(0)
            stop_event.set()

        await asyncio.gather(
            run_until_stopped(site, stop_event, name="Test surface"),
            flip_after_first_tick(),
        )

        assert site.start_calls == 1
        assert site.cleanup_calls == 1

    async def test_cleanup_runs_even_when_stop_event_is_pre_set(self) -> None:
        """A stop event set before the wait still tears the site down."""
        site = _RecordingSite()
        stop_event = asyncio.Event()
        stop_event.set()

        await run_until_stopped(site, stop_event, name="Test surface")

        assert site.start_calls == 1
        assert site.cleanup_calls == 1

    async def test_cleanup_runs_when_start_raises(self) -> None:
        """A site that fails to start is still torn down before propagating.

        The ``AppRunner`` may have opened the socket before ``start``
        raised, so the ``finally`` is what stops a failed boot from
        leaking a bound port.
        """
        site = _CancellingSite()
        stop_event = asyncio.Event()

        with pytest.raises(asyncio.CancelledError):
            await run_until_stopped(site, stop_event, name="Test surface")

        assert site.start_calls == 1
        assert site.cleanup_calls == 1
