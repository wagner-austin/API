"""Shared stubs for the service test modules.

The recording session runner, the no-op shutdown, and the stub
response the streaming tests write into. The pytest fixtures that
build them live in ``tests/service/conftest.py`` -- a fixture cannot
travel by import without becoming an unused-name violation.
"""

from __future__ import annotations

import threading

from tankpit_bot.service.session_runner import SessionAlreadyRunningError


class _RecordingRunner:
    """SessionRunner stand-in that records lifecycle calls."""

    def __init__(
        self,
        *,
        starts_reject: bool = False,
        already_running: bool = False,
        on_start: threading.Event | None = None,
    ) -> None:
        """Configure the fake runner's behaviour.

        Args:
            starts_reject: When True, ``start`` raises
                :class:`SessionAlreadyRunningError`. Simulates the
                race between two concurrent ``POST /start`` calls
                after the pre-check but before the state lock.
            already_running: When True, ``is_running`` returns True —
                the ``POST /start`` pre-check trips before ``start``
                is even called.
            on_start: Optional threading.Event set by ``start`` so the
                calling test can wait for the executor thread to run.
        """
        self.start_calls: int = 0
        self.stop_calls: int = 0
        self.last_session_seconds: int = -1
        self.last_session_kills: int = -1
        self._starts_reject = starts_reject
        self._already_running = already_running
        self._on_start = on_start

    def is_running(self) -> bool:
        return self._already_running

    def start(self, *, session_seconds: int = 0, session_kills: int = 0) -> None:
        self.start_calls += 1
        self.last_session_seconds = session_seconds
        self.last_session_kills = session_kills
        if self._on_start is not None:
            self._on_start.set()
        if self._starts_reject:
            raise SessionAlreadyRunningError("simulated race")

    def request_stop(self) -> None:
        self.stop_calls += 1


def _noop_shutdown() -> None:
    """Placeholder ``on_shutdown`` for routes that never fire it."""


class _StubResponse:
    """Minimal ``aiohttp.web.StreamResponse`` stand-in for drain tests."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []

    async def write(self, data: bytes) -> None:
        self.writes.append(data)
