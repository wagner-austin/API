"""Tests for :class:`tankpit_bot.bus.frame_bus.FrameBus`.

Covers the subscribe / unsubscribe / publish contract, the
cache-on-publish behaviour that gives fresh ``/video`` viewers an
instant first frame, the latest-wins semantic on slow connections, the
``latest()`` accessor ``GET /frame`` serves from, and the
close-unblocks-waiter contract that keeps MJPEG tear-down clean.
Includes one multi-threaded scenario mirroring the real
screencast-thread → aiohttp-executor topology.
"""

from __future__ import annotations

import threading
import time

from tankpit_bot.bus.frame_bus import FrameBus, FrameSubscriber

# =============================================================================
# FrameSubscriber
# =============================================================================


class TestFrameSubscriber:
    """Contract tests for :class:`FrameSubscriber`."""

    def test_next_frame_returns_pushed_frame(self) -> None:
        """Push then next_frame yields the frame."""
        sub = FrameSubscriber()
        sub.push(b"jpeg-1")
        assert sub.next_frame(timeout=0.5) == b"jpeg-1"

    def test_next_frame_is_destructive(self) -> None:
        """Calling next_frame twice after one push returns None the second time."""
        sub = FrameSubscriber()
        sub.push(b"jpeg-1")
        assert sub.next_frame(timeout=0.5) == b"jpeg-1"
        assert sub.next_frame(timeout=0.05) is None

    def test_push_after_push_is_latest_wins(self) -> None:
        """Two pushes before a consumer read collapse to the second frame."""
        sub = FrameSubscriber()
        sub.push(b"jpeg-1")
        sub.push(b"jpeg-2")
        assert sub.next_frame(timeout=0.5) == b"jpeg-2"

    def test_timeout_returns_none(self) -> None:
        """No push before the timeout returns None."""
        sub = FrameSubscriber()
        assert sub.next_frame(timeout=0.05) is None

    def test_close_unblocks_waiter(self) -> None:
        """A subscriber closed while a consumer is blocked wakes with None."""
        sub = FrameSubscriber()

        result: list[bytes | None] = []

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
        sub = FrameSubscriber()
        sub.close()
        sub.push(b"jpeg-1")
        assert sub.next_frame(timeout=0.05) is None

    def test_close_is_idempotent(self) -> None:
        """Calling close twice has no additional effect."""
        sub = FrameSubscriber()
        sub.close()
        sub.close()
        assert sub.closed is True


# =============================================================================
# FrameBus
# =============================================================================


class TestFrameBus:
    """Contract tests for :class:`FrameBus`."""

    def test_publish_reaches_every_subscriber(self) -> None:
        """A publish fan-outs to every registered subscriber."""
        bus = FrameBus()
        sub_a = bus.subscribe()
        sub_b = bus.subscribe()

        bus.publish(b"jpeg-1")

        assert sub_a.next_frame(timeout=0.5) == b"jpeg-1"
        assert sub_b.next_frame(timeout=0.5) == b"jpeg-1"

    def test_publish_with_no_subscribers_is_a_noop(self) -> None:
        """A publish with an empty subscriber list does not raise."""
        bus = FrameBus()
        bus.publish(b"jpeg-1")
        assert bus.subscriber_count() == 0

    def test_late_subscriber_gets_cached_frame_immediately(self) -> None:
        """A subscriber registered after a publish sees the cached frame."""
        bus = FrameBus()
        bus.publish(b"jpeg-1")

        late = bus.subscribe()

        assert late.next_frame(timeout=0.5) == b"jpeg-1"

    def test_fresh_subscriber_without_publish_blocks_until_publish(self) -> None:
        """A fresh subscriber with no cached frame times out before publish."""
        bus = FrameBus()
        sub = bus.subscribe()
        assert sub.next_frame(timeout=0.05) is None

    def test_unsubscribe_removes_subscriber_and_closes_it(self) -> None:
        """Unsubscribe stops delivering frames and marks the subscriber closed."""
        bus = FrameBus()
        sub = bus.subscribe()
        assert bus.subscriber_count() == 1

        bus.unsubscribe(sub)

        assert bus.subscriber_count() == 0
        assert sub.closed is True

    def test_unsubscribe_is_idempotent_on_unknown_subscriber(self) -> None:
        """Unsubscribing a subscriber that was never registered is a no-op."""
        bus = FrameBus()
        stray = FrameSubscriber()
        bus.unsubscribe(stray)  # must not raise
        assert stray.closed is True  # still closes to unblock any waiter

    def test_publish_does_not_deliver_to_unsubscribed(self) -> None:
        """After unsubscribe, a publish does not reach the dropped subscriber."""
        bus = FrameBus()
        sub = bus.subscribe()
        bus.unsubscribe(sub)

        bus.publish(b"jpeg-1")

        assert sub.next_frame(timeout=0.05) is None

    def test_latest_none_before_any_publish(self) -> None:
        """``latest`` reports None on a virgin bus (``GET /frame`` → 404)."""
        bus = FrameBus()
        assert bus.latest() is None

    def test_latest_returns_most_recent_publish(self) -> None:
        """``latest`` tracks the newest frame across publishes."""
        bus = FrameBus()
        bus.publish(b"jpeg-1")
        bus.publish(b"jpeg-2")
        assert bus.latest() == b"jpeg-2"


def test_frame_bus_publish_reaches_subscriber_across_threads() -> None:
    """A subscriber blocked on next_frame wakes when a publisher on another
    thread pushes a frame — mirrors the screencast → MJPEG topology."""
    bus = FrameBus()
    sub = bus.subscribe()
    received: list[bytes | None] = []

    def consume() -> None:
        received.append(sub.next_frame(timeout=1.0))

    consumer = threading.Thread(target=consume)
    consumer.start()

    time.sleep(0.05)
    bus.publish(b"jpeg-7")

    consumer.join(timeout=1.0)
    assert not consumer.is_alive()
    assert received == [b"jpeg-7"]


def test_subscriber_indefinite_wait_loops_back_after_notify() -> None:
    """A ``next_frame(timeout=None)`` waiter loops back to the while check.

    Exercises the branch on ``timeout is not None and not notified`` when
    ``timeout`` is ``None``: the waiter must skip the timeout-return path
    and loop back to re-check ``self._latest`` / ``self._closed``.
    """
    sub = FrameSubscriber()
    received: list[bytes | None] = []

    def consume() -> None:
        received.append(sub.next_frame(timeout=None))

    consumer = threading.Thread(target=consume)
    consumer.start()

    time.sleep(0.05)
    sub.push(b"jpeg-11")

    consumer.join(timeout=1.0)
    assert not consumer.is_alive()
    assert received == [b"jpeg-11"]


# =============================================================================
# Drop accounting
# =============================================================================


def test_a_slow_consumer_is_counted_as_a_drop_not_as_silence() -> None:
    """THE MEASUREMENT THAT WAS MISSING.

    Latest-wins discards, and until 2026-09-04 nothing counted it. That
    mattered because downstream a dropped frame and a frame that was
    never produced are indistinguishable -- both are just a longer gap
    between arrivals. Every rate taken from the ``/video`` end
    therefore counts survivors, and a still game and a starved
    connection read identically.

    Three frames published, one consumer that takes nothing until the
    end: two were overwritten and the bus must say so.
    """
    bus = FrameBus()
    subscriber = bus.subscribe()

    bus.publish(b"jpeg-1")
    bus.publish(b"jpeg-2")
    bus.publish(b"jpeg-3")

    stats = bus.stats()
    assert stats["published"] == 3
    assert stats["dropped"] == 2
    assert stats["delivered"] == 0
    assert subscriber.next_frame(timeout=0.1) == b"jpeg-3"
    assert bus.stats()["delivered"] == 1


def test_a_consumer_that_keeps_up_drops_nothing() -> None:
    """The control. Without it the counter above could just be a clock.

    Same three frames, consumed between each publish: nothing is
    overwritten, so the drop count must stay at zero. A counter that
    rises no matter what measures nothing.
    """
    bus = FrameBus()
    subscriber = bus.subscribe()

    for marker in (b"jpeg-1", b"jpeg-2", b"jpeg-3"):
        bus.publish(marker)
        assert subscriber.next_frame(timeout=0.1) == marker

    stats = bus.stats()
    assert stats["published"] == 3
    assert stats["dropped"] == 0
    assert stats["delivered"] == 3


def test_a_departed_viewers_losses_survive_its_unsubscribe() -> None:
    """A tab closed because the picture was bad must not erase the why.

    The losses are carried onto the bus as the subscriber is removed,
    so the totals still describe the session after the viewer has gone.
    """
    bus = FrameBus()
    subscriber = bus.subscribe()
    bus.publish(b"jpeg-1")
    bus.publish(b"jpeg-2")

    bus.unsubscribe(subscriber)

    stats = bus.stats()
    assert stats["published"] == 2
    assert stats["dropped"] == 1
    assert stats["subscribers"] == 0


def test_a_bus_nobody_watches_publishes_without_dropping() -> None:
    """Zero subscribers is not a drop.

    The caster runs only on viewer demand, but the cache-on-publish
    path still records production. Counting an unwatched publish as a
    loss would make an idle service look like a broken one.
    """
    bus = FrameBus()

    bus.publish(b"jpeg-1")
    bus.publish(b"jpeg-2")

    stats = bus.stats()
    assert stats["published"] == 2
    assert stats["dropped"] == 0
    assert stats["delivered"] == 0
    assert stats["subscribers"] == 0


def test_frames_pushed_after_close_are_not_counted_as_drops() -> None:
    """A closed subscriber is a torn-down connection, not a loss.

    ``push`` returns early once closed. Counting those as drops would
    attribute every frame published after a viewer left to that
    viewer, which would make a normal disconnect look like starvation.
    """
    subscriber = FrameSubscriber()
    subscriber.push(b"jpeg-1")
    subscriber.close()

    subscriber.push(b"jpeg-2")
    subscriber.push(b"jpeg-3")

    assert subscriber.dropped == 0
