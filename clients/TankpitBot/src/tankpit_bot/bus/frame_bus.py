"""Threadsafe JPEG-frame fan-out for the watch-page video stream.

The ``POST /cast`` route publishes each JPEG the in-page caster sends;
each ``GET /video`` MJPEG connection owns its own
:class:`FrameSubscriber` and waits on the next frame.

Threadsafe is not decoration here, and WHICH threads has changed. The
publisher used to be the Playwright thread -- first a CDP screencast
handler, then a CDP binding relay -- which is the thread the tick loop
runs on, so a busy tick stopped frames reaching this bus at all and the
latest-wins rule below collapsed the resulting burst into one picture
(:mod:`tankpit_bot.browser.live_view` has the measurements). The
publisher is now aiohttp's event loop on the MAIN thread, and the
session runs on an executor thread, so publishing is independent of
what the bot is doing. Subscribers still wait on executor threads.

Semantics match :mod:`tankpit_bot.bus.status_bus` deliberately:

* **Latest-wins per subscriber.** A slow phone connection sees the
  newest frame, never a growing backlog — stale video frames are
  worthless.
* **Cache-on-publish.** A fresh subscriber painted mid-session
  immediately gets the current frame instead of a blank stream.
* **Explicit unsubscribe.** ``GET /video`` handlers must remove their
  subscriber when the connection closes; the tick loop reads
  :meth:`FrameBus.subscriber_count` as the DEMAND signal that decides
  whether the in-page caster runs at all.
"""

from __future__ import annotations

import threading
from typing import Protocol

from typing_extensions import TypedDict


class FrameStatsDict(TypedDict):
    """What the bus produced against what viewers actually got.

    Attributes:
        published: Frames handed to :meth:`FrameBus.publish` since
            boot — the caster's real production, before any loss.
        delivered: Frames subscribers handed on to their consumers.
        dropped: Frames discarded by latest-wins because a newer one
            arrived before the consumer took the previous.
        subscribers: Currently registered subscribers.
    """

    published: int
    delivered: int
    dropped: int
    subscribers: int


class FrameSubscriberProtocol(Protocol):
    """Surface one MJPEG consumer uses to await JPEG frames."""

    def push(self, frame: bytes) -> None:
        """Store ``frame`` and wake any waiter.

        Args:
            frame: Latest JPEG frame bytes.
        """
        ...

    def next_frame(self, timeout: float | None = None) -> bytes | None:
        """Wait for and consume the next JPEG frame.

        Args:
            timeout: Maximum seconds to block; ``None`` blocks
                indefinitely.

        Returns:
            The next frame, or ``None`` on timeout / close.
        """
        ...

    def close(self) -> None:
        """Mark the subscriber closed and unblock any waiter."""
        ...

    @property
    def closed(self) -> bool:
        """Return True when the subscriber has been closed."""
        ...

    @property
    def dropped(self) -> int:
        """Return how many frames were overwritten before consumption."""
        ...

    @property
    def delivered(self) -> int:
        """Return how many frames reached the consumer."""
        ...


class FrameBusProtocol(Protocol):
    """Surface the ``/cast`` publisher and the viewer handlers share."""

    def publish(self, frame: bytes) -> None:
        """Push ``frame`` to every subscriber and cache it.

        Args:
            frame: JPEG frame bytes to broadcast.
        """
        ...

    def subscribe(self) -> FrameSubscriberProtocol:
        """Register a new subscriber and hand it the cached frame.

        Returns:
            A subscriber whose owner is responsible for calling
            :meth:`unsubscribe` (or the subscriber's ``close``) when
            done.
        """
        ...

    def unsubscribe(self, subscriber: FrameSubscriberProtocol) -> None:
        """Drop ``subscriber`` from the fan-out and close it.

        Args:
            subscriber: The subscriber to remove.
        """
        ...

    def subscriber_count(self) -> int:
        """Return the number of currently-registered subscribers.

        Returns:
            The current subscriber count — the tick loop's caster
            demand signal.
        """
        ...

    def latest(self) -> bytes | None:
        """Return the most recently published frame, if any.

        Returns:
            The cached frame, or ``None`` when nothing has been
            published since service boot.
        """
        ...

    def stats(self) -> FrameStatsDict:
        """Return production versus delivery counts.

        Returns:
            The counts, as of this call.
        """
        ...


class FrameSubscriber:
    """One MJPEG connection's slot in the fan-out.

    Guarded by an internal :class:`threading.Condition`: the ``/cast``
    route calls :meth:`push` on the aiohttp event loop; the ``/video``
    handler calls :meth:`next_frame` on an aiohttp executor thread.
    """

    def __init__(self) -> None:
        """Create an empty subscriber with no cached frame."""
        self._cond = threading.Condition()
        self._latest: bytes | None = None
        self._closed = False
        self._dropped = 0
        self._delivered = 0

    def push(self, frame: bytes) -> None:
        """Store ``frame`` and wake any waiting :meth:`next_frame`.

        Frames pushed after :meth:`close` are dropped silently — a
        closed subscriber is a torn-down ``/video`` connection.

        LATEST-WINS MEANS THIS DISCARDS, AND THE DISCARD IS COUNTED
        HERE because here is the only place that can see it. Arriving
        on top of a frame the consumer has not taken yet means that
        frame never reaches the viewer, and downstream the loss is
        invisible: a dropped frame and a frame that was never produced
        both show up as a gap between arrivals. Every rate measured at
        the receiving end therefore counts survivors, not production,
        and cannot tell a still game from a starved connection. The
        counter is what separates them.

        Args:
            frame: Latest JPEG frame bytes.
        """
        with self._cond:
            if self._closed:
                return
            if self._latest is not None:
                self._dropped += 1
            self._latest = frame
            self._cond.notify_all()

    def next_frame(self, timeout: float | None = None) -> bytes | None:
        """Wait for and consume the next JPEG frame.

        Args:
            timeout: Maximum seconds to block. ``None`` blocks
                indefinitely. On timeout, returns ``None`` without
                consuming any cached frame — the ``/video`` handler
                loops with a keepalive resend.

        Returns:
            The next cached frame, or ``None`` if the timeout expired
            or the subscriber was closed while waiting.
        """
        with self._cond:
            while self._latest is None and not self._closed:
                notified = self._cond.wait(timeout=timeout)
                if timeout is not None and not notified:
                    return None
            frame = self._latest
            self._latest = None
            if frame is not None:
                self._delivered += 1
            return frame

    def close(self) -> None:
        """Mark the subscriber closed and unblock any waiter."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        """Return True when :meth:`close` has been called.

        Returns:
            True if the subscriber has been closed.
        """
        with self._cond:
            return self._closed

    @property
    def dropped(self) -> int:
        """Frames this subscriber lost to a newer one.

        Returns:
            How many pushed frames overwrote a frame the consumer had
            not taken yet.
        """
        with self._cond:
            return self._dropped

    @property
    def delivered(self) -> int:
        """Frames this subscriber actually handed to its consumer.

        Returns:
            How many frames :meth:`next_frame` returned.
        """
        with self._cond:
            return self._delivered


class FrameBus:
    """Fan-out of JPEG frames to N ``/video`` subscribers.

    Owned by the service; the ``/cast`` route publishes here, and the
    aiohttp handlers call :meth:`subscribe` / :meth:`unsubscribe` around
    each connection's lifetime. :meth:`subscriber_count` doubles as the
    caster demand signal: zero subscribers → the tick loop stops the
    in-page caster so unwatched sessions pay nothing.
    """

    def __init__(self) -> None:
        """Create an empty bus with no subscribers and no cached frame."""
        self._lock = threading.Lock()
        self._subscribers: list[FrameSubscriberProtocol] = []
        self._latest: bytes | None = None
        self._published = 0
        self._retired_dropped = 0
        self._retired_delivered = 0

    def publish(self, frame: bytes) -> None:
        """Push ``frame`` to every subscriber and cache it for late joiners.

        Args:
            frame: JPEG frame bytes to broadcast.
        """
        with self._lock:
            self._latest = frame
            self._published += 1
            recipients = list(self._subscribers)
        for subscriber in recipients:
            subscriber.push(frame)

    def subscribe(self) -> FrameSubscriberProtocol:
        """Register a new subscriber and hand it the cached frame.

        Returns:
            A new subscriber. The caller is responsible for calling
            :meth:`unsubscribe` (or the subscriber's ``close``) when
            done. Concretely a :class:`FrameSubscriber`; typed as the
            Protocol so consumers depend on the abstract surface.
        """
        subscriber: FrameSubscriberProtocol = FrameSubscriber()
        with self._lock:
            self._subscribers.append(subscriber)
            latest = self._latest
        if latest is not None:
            subscriber.push(latest)
        return subscriber

    def unsubscribe(self, subscriber: FrameSubscriberProtocol) -> None:
        """Drop ``subscriber`` from the fan-out and close it.

        Idempotent: unsubscribing a subscriber that was never
        registered (or was already removed) is a no-op. The subscriber
        is always closed so any waiter unblocks.

        Args:
            subscriber: The subscriber to remove.
        """
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)
                # Carried onto the bus BEFORE the subscriber is
                # forgotten. A viewer's losses are most interesting
                # exactly when that viewer has gone -- a tab closed
                # because the picture was bad takes the evidence with
                # it otherwise.
                self._retired_dropped += subscriber.dropped
                self._retired_delivered += subscriber.delivered
        subscriber.close()

    def subscriber_count(self) -> int:
        """Return the number of currently-registered subscribers.

        Returns:
            The subscriber count — read by the tick loop as the
            caster demand signal and by the idle-exit monitor.
        """
        with self._lock:
            return len(self._subscribers)

    def latest(self) -> bytes | None:
        """Return the most recently published frame, if any.

        Returns:
            The cached frame, or ``None`` when nothing has been
            published since service boot. Served by ``GET /frame``.
        """
        with self._lock:
            return self._latest

    def stats(self) -> FrameStatsDict:
        """Return what this bus made versus what viewers received.

        THE MEASUREMENT NOTHING ELSE CAN TAKE. Every rate observed at
        the far end of the stream counts frames that survived, so a
        still game and a starved connection produce the same numbers
        there. ``published`` is what the caster actually produced;
        ``dropped`` is what latest-wins discarded on the way out. Only
        the pair separates the two explanations.

        Live and retired subscribers are summed together so a viewer
        closing its tab does not erase its own losses.

        Returns:
            The counts, as of this call.
        """
        with self._lock:
            live = list(self._subscribers)
            published = self._published
            dropped = self._retired_dropped
            delivered = self._retired_delivered
        for subscriber in live:
            dropped += subscriber.dropped
            delivered += subscriber.delivered
        return FrameStatsDict(
            published=published,
            delivered=delivered,
            dropped=dropped,
            subscribers=len(live),
        )


__all__ = [
    "FrameBus",
    "FrameBusProtocol",
    "FrameStatsDict",
    "FrameSubscriber",
    "FrameSubscriberProtocol",
]
