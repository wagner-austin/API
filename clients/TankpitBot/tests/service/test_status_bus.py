"""Tests for :class:`tankpit_bot.service.status_bus.StatusBus`.

Covers the subscribe / unsubscribe / publish contract, the
cache-on-publish behaviour that makes fresh subscribers see the
current state, the latest-wins semantic on slow subscribers, and the
close-unblocks-waiter contract that keeps SSE tear-down clean. Includes
one multi-threaded scenario where the publisher and subscriber run on
different threads, mirroring the real service topology.
"""

from __future__ import annotations

import threading
import time

from tankpit_bot.service.status_bus import StatusBus, StatusSubscriber
from tankpit_bot.service.types import SessionStatusDict, idle_session_status, make_live_stats


def _make_status(tick_timestamp_ms: int, kills: int = 0) -> SessionStatusDict:
    """Build a distinguishable status frame for round-trip assertions.

    Args:
        tick_timestamp_ms: Snapshot capture wall-clock.
        kills: Kill count for the embedded stats dict.

    Returns:
        A populated :class:`SessionStatusDict`.
    """
    if kills == 0 and tick_timestamp_ms == 0:
        return idle_session_status(tick_timestamp_ms=0)
    return SessionStatusDict(
        running=True,
        manual_mode="HUNT",
        active_mode="HUNT",
        active_mode_state="ACQUIRE",
        session_started_ms=1000,
        tick_timestamp_ms=tick_timestamp_ms,
        stats=make_live_stats(kills=kills, hits=0, misses=0, radars_used=0, teleports=0),
    )


# =============================================================================
# StatusSubscriber
# =============================================================================


class TestStatusSubscriber:
    """Contract tests for :class:`StatusSubscriber`."""

    def test_next_frame_returns_pushed_frame(self) -> None:
        """Push then next_frame yields the frame."""
        sub = StatusSubscriber()
        frame = _make_status(tick_timestamp_ms=1)
        sub.push(frame)
        assert sub.next_frame(timeout=0.5) == frame

    def test_next_frame_is_destructive(self) -> None:
        """Calling next_frame twice after one push returns None the second time."""
        sub = StatusSubscriber()
        frame = _make_status(tick_timestamp_ms=1)
        sub.push(frame)
        assert sub.next_frame(timeout=0.5) == frame
        assert sub.next_frame(timeout=0.05) is None

    def test_push_after_push_is_latest_wins(self) -> None:
        """Two pushes before a consumer read collapse to the second frame."""
        sub = StatusSubscriber()
        first = _make_status(tick_timestamp_ms=1)
        second = _make_status(tick_timestamp_ms=2, kills=3)
        sub.push(first)
        sub.push(second)
        assert sub.next_frame(timeout=0.5) == second

    def test_timeout_returns_none(self) -> None:
        """No push before the timeout returns None."""
        sub = StatusSubscriber()
        assert sub.next_frame(timeout=0.05) is None

    def test_close_unblocks_waiter(self) -> None:
        """A subscriber closed while a consumer is blocked wakes with None."""
        sub = StatusSubscriber()

        result: list[SessionStatusDict | None] = []

        def wait_for_frame() -> None:
            result.append(sub.next_frame(timeout=None))

        waiter = threading.Thread(target=wait_for_frame)
        waiter.start()
        time.sleep(0.05)
        sub.close()
        waiter.join(timeout=1.0)

        assert not waiter.is_alive()
        assert result == [None]
        assert sub.closed is True

    def test_push_after_close_is_dropped(self) -> None:
        """A push after close leaves the subscriber empty."""
        sub = StatusSubscriber()
        sub.close()
        sub.push(_make_status(tick_timestamp_ms=1))
        assert sub.next_frame(timeout=0.05) is None

    def test_close_is_idempotent(self) -> None:
        """Calling close twice has no additional effect."""
        sub = StatusSubscriber()
        sub.close()
        sub.close()
        assert sub.closed is True


# =============================================================================
# StatusBus
# =============================================================================


class TestStatusBus:
    """Contract tests for :class:`StatusBus`."""

    def test_publish_reaches_every_subscriber(self) -> None:
        """A publish fan-outs to every registered subscriber."""
        bus = StatusBus()
        sub_a = bus.subscribe()
        sub_b = bus.subscribe()
        frame = _make_status(tick_timestamp_ms=1)

        bus.publish(frame)

        assert sub_a.next_frame(timeout=0.5) == frame
        assert sub_b.next_frame(timeout=0.5) == frame

    def test_publish_with_no_subscribers_is_a_noop(self) -> None:
        """A publish with an empty subscriber list does not raise."""
        bus = StatusBus()
        bus.publish(_make_status(tick_timestamp_ms=1))
        assert bus.subscriber_count() == 0

    def test_late_subscriber_gets_cached_frame_immediately(self) -> None:
        """A subscriber registered after a publish sees the cached frame."""
        bus = StatusBus()
        first = _make_status(tick_timestamp_ms=1)
        bus.publish(first)

        late = bus.subscribe()

        assert late.next_frame(timeout=0.5) == first

    def test_fresh_subscriber_without_publish_blocks_until_publish(self) -> None:
        """A fresh subscriber with no cached frame times out before publish."""
        bus = StatusBus()
        sub = bus.subscribe()
        assert sub.next_frame(timeout=0.05) is None

    def test_unsubscribe_removes_subscriber_and_closes_it(self) -> None:
        """Unsubscribe stops delivering frames and marks the subscriber closed."""
        bus = StatusBus()
        sub = bus.subscribe()
        assert bus.subscriber_count() == 1

        bus.unsubscribe(sub)

        assert bus.subscriber_count() == 0
        assert sub.closed is True

    def test_unsubscribe_is_idempotent_on_unknown_subscriber(self) -> None:
        """Unsubscribing a subscriber that was never registered is a no-op."""
        bus = StatusBus()
        stray = StatusSubscriber()
        bus.unsubscribe(stray)  # must not raise
        assert stray.closed is True  # still closes to unblock any waiter

    def test_publish_does_not_deliver_to_unsubscribed(self) -> None:
        """After unsubscribe, a publish does not reach the dropped subscriber."""
        bus = StatusBus()
        sub = bus.subscribe()
        bus.unsubscribe(sub)

        bus.publish(_make_status(tick_timestamp_ms=1))

        assert sub.next_frame(timeout=0.05) is None

    def test_late_publish_updates_cached_frame(self) -> None:
        """A subsequent publish replaces the cached frame served on subscribe."""
        bus = StatusBus()
        bus.publish(_make_status(tick_timestamp_ms=1))
        second = _make_status(tick_timestamp_ms=2, kills=5)
        bus.publish(second)

        late = bus.subscribe()

        assert late.next_frame(timeout=0.5) == second


def test_status_bus_publish_reaches_subscriber_across_threads() -> None:
    """A subscriber blocked on next_frame wakes when a publisher on another
    thread pushes a frame — mirrors the tick-loop → SSE topology."""
    bus = StatusBus()
    sub = bus.subscribe()
    received: list[SessionStatusDict | None] = []

    def consume() -> None:
        received.append(sub.next_frame(timeout=1.0))

    consumer = threading.Thread(target=consume)
    consumer.start()

    time.sleep(0.05)
    frame = _make_status(tick_timestamp_ms=7)
    bus.publish(frame)

    consumer.join(timeout=1.0)
    assert not consumer.is_alive()
    assert received == [frame]


def test_subscriber_indefinite_wait_loops_back_after_notify() -> None:
    """A ``next_frame(timeout=None)`` waiter loops back to the while check.

    Exercises the branch on ``timeout is not None and not notified`` when
    ``timeout`` is ``None``: the waiter must skip the timeout-return path
    and loop back to re-check ``self._latest`` / ``self._closed``. The
    subscriber is woken by :meth:`push` on another thread so the loop
    exits with the pushed frame.
    """
    sub = StatusSubscriber()
    received: list[SessionStatusDict | None] = []

    def consume() -> None:
        received.append(sub.next_frame(timeout=None))

    consumer = threading.Thread(target=consume)
    consumer.start()

    time.sleep(0.05)
    frame = _make_status(tick_timestamp_ms=11, kills=1)
    sub.push(frame)

    consumer.join(timeout=1.0)
    assert not consumer.is_alive()
    assert received == [frame]
