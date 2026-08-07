"""Threadsafe JPEG-frame fan-out for the watch-page video stream.

The screencast handler (Playwright thread) publishes each JPEG frame
Chrome pushes; each ``GET /video`` MJPEG connection owns its own
:class:`FrameSubscriber` and waits on the next frame. Semantics match
:mod:`tankpit_bot.bus.status_bus` deliberately:

* **Latest-wins per subscriber.** A slow phone connection sees the
  newest frame, never a growing backlog — stale video frames are
  worthless.
* **Cache-on-publish.** A fresh subscriber painted mid-session
  immediately gets the current frame instead of a blank stream.
* **Explicit unsubscribe.** ``GET /video`` handlers must remove their
  subscriber when the connection closes; the tick loop reads
  :meth:`FrameBus.subscriber_count` as the DEMAND signal that decides
  whether the Chrome screencast runs at all.
"""

from __future__ import annotations

import threading
from typing import Protocol


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


class FrameBusProtocol(Protocol):
    """Surface the screencast publisher and HTTP handlers share."""

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
            The current subscriber count — the tick loop's screencast
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


class FrameSubscriber:
    """One MJPEG connection's slot in the fan-out.

    Guarded by an internal :class:`threading.Condition`: the screencast
    handler calls :meth:`push` on the Playwright thread; the ``/video``
    handler calls :meth:`next_frame` on an aiohttp executor thread.
    """

    def __init__(self) -> None:
        """Create an empty subscriber with no cached frame."""
        self._cond = threading.Condition()
        self._latest: bytes | None = None
        self._closed = False

    def push(self, frame: bytes) -> None:
        """Store ``frame`` and wake any waiting :meth:`next_frame`.

        Frames pushed after :meth:`close` are dropped silently — a
        closed subscriber is a torn-down ``/video`` connection.

        Args:
            frame: Latest JPEG frame bytes.
        """
        with self._cond:
            if self._closed:
                return
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
            if self._latest is None:
                return None
            frame = self._latest
            self._latest = None
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


class FrameBus:
    """Fan-out of JPEG frames to N ``/video`` subscribers.

    Owned by the service; the bot's screencast handler publishes here,
    and the aiohttp handlers call :meth:`subscribe` / :meth:`unsubscribe`
    around each connection's lifetime. :meth:`subscriber_count` doubles
    as the screencast demand signal: zero subscribers → the tick loop
    stops the Chrome screencast so unwatched sessions pay nothing.
    """

    def __init__(self) -> None:
        """Create an empty bus with no subscribers and no cached frame."""
        self._lock = threading.Lock()
        self._subscribers: list[FrameSubscriberProtocol] = []
        self._latest: bytes | None = None

    def publish(self, frame: bytes) -> None:
        """Push ``frame`` to every subscriber and cache it for late joiners.

        Args:
            frame: JPEG frame bytes to broadcast.
        """
        with self._lock:
            self._latest = frame
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
        subscriber.close()

    def subscriber_count(self) -> int:
        """Return the number of currently-registered subscribers.

        Returns:
            The subscriber count — read by the tick loop as the
            screencast demand signal and by the idle-exit monitor.
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


__all__ = [
    "FrameBus",
    "FrameBusProtocol",
    "FrameSubscriber",
    "FrameSubscriberProtocol",
]
