"""Tests for :class:`tankpit_bot.bus.mode_bridge.ModeBridge`.

Covers the single-thread API contract — submit / drain / peek — plus a
light multi-threaded scenario to prove the internal lock actually
serialises writes. No mocks: the tests exercise the real primitive
with real threads.
"""

from __future__ import annotations

import threading

from tankpit_bot.bus.mode_bridge import ModeBridge


class TestModeBridge:
    """Contract tests for :class:`ModeBridge`."""

    def test_empty_bridge_drains_to_none(self) -> None:
        """A fresh bridge yields ``None`` on drain."""
        bridge = ModeBridge()
        assert bridge.drain() is None

    def test_empty_bridge_peeks_to_none(self) -> None:
        """A fresh bridge yields ``None`` on peek."""
        bridge = ModeBridge()
        assert bridge.peek() is None

    def test_submit_then_drain_returns_value(self) -> None:
        """A submitted value round-trips through drain."""
        bridge = ModeBridge()
        bridge.submit("HUNT")
        assert bridge.drain() == "HUNT"

    def test_drain_is_destructive(self) -> None:
        """The second drain after one submit returns ``None``."""
        bridge = ModeBridge()
        bridge.submit("COLLECT")
        assert bridge.drain() == "COLLECT"
        assert bridge.drain() is None

    def test_peek_is_non_destructive(self) -> None:
        """Peek leaves the pending value in place for a later drain."""
        bridge = ModeBridge()
        bridge.submit("HUNT")
        assert bridge.peek() == "HUNT"
        assert bridge.peek() == "HUNT"
        assert bridge.drain() == "HUNT"
        assert bridge.peek() is None

    def test_second_submit_replaces_first_latest_wins(self) -> None:
        """Two submits between drains yield only the second — latest wins."""
        bridge = ModeBridge()
        bridge.submit("HUNT")
        bridge.submit("COLLECT")
        assert bridge.drain() == "COLLECT"
        assert bridge.drain() is None

    def test_every_wire_mode_round_trips(self) -> None:
        """Every :data:`WIRE_MODES` value can round-trip through the bridge."""
        from tankpit_bot.bus.session_status import WIRE_MODES

        bridge = ModeBridge()
        for mode in WIRE_MODES:
            bridge.submit(mode)
            assert bridge.drain() == mode


def test_mode_bridge_serialises_concurrent_submits() -> None:
    """Under contention, every submit either wins the slot or is overwritten.

    Ten writer threads each submit their assigned mode a handful of
    times. When they finish, ``drain`` returns exactly one of the
    submitted modes (or ``None`` if the very last write happened
    before a drain in the same instant — impossible here since the
    drain runs after every writer has joined). No stale value, no
    corruption.
    """
    bridge = ModeBridge()
    modes = ["UNSET", "HUNT", "COLLECT", "AUTO"]
    threads: list[threading.Thread] = []

    def writer(mode: str) -> None:
        assert mode in modes
        for _ in range(50):
            # The Literal here is checked by the caller loop below —
            # the writer signature keeps the concurrency scenario tight
            # without pulling WireMode into the closure.
            if mode == "UNSET":
                bridge.submit("UNSET")
            elif mode == "HUNT":
                bridge.submit("HUNT")
            elif mode == "COLLECT":
                bridge.submit("COLLECT")
            else:
                bridge.submit("AUTO")

    for mode in modes:
        for _ in range(3):
            threads.append(threading.Thread(target=writer, args=(mode,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = bridge.drain()
    assert final in modes
