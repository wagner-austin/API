"""Fixtures every capture-helper test shares."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from tankpit_bot.stream import _test_hooks as stream_hooks


@pytest.fixture(autouse=True)
def _socket_root(tmp_path: Path) -> Generator[Path, None, None]:
    """Root the sound server's socket directory in the test's own tree.

    Yields:
        The directory installed as the socket root.
    """
    root = tmp_path / "sockets"
    stream_hooks.socket_root = lambda: root
    yield root
    stream_hooks.socket_root = stream_hooks._real_socket_root


@pytest.fixture(autouse=True)
def _reap() -> Generator[list[stream_hooks.CaptureProcessProtocol], None, None]:
    """Kill every child a test's spawner left running, and wait for its end.

    The wait has no timeout: after ``kill`` the child will end, and on a
    loaded Windows host its rundown has outlasted a fixed ten seconds
    (board task 06fc3195; ``scripts/killed_wait_rules.py`` records the
    measurement and keeps the bound from coming back).

    Yields:
        The list the test's spawner should append processes to.
    """
    spawned: list[stream_hooks.CaptureProcessProtocol] = []
    yield spawned
    for process in spawned:
        if process.poll() is None:
            process.kill()
            process.wait()
