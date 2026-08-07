"""Threadsafe status fan-out for SSE subscribers.

The sync tick loop publishes a :class:`SessionStatusDict` snapshot
after every tick; each SPA subscriber owns its own
:class:`StatusSubscriber` and waits on the next frame. Semantics:

* **Latest-wins per subscriber.** If a subscriber is slow and a new
  frame arrives before it consumes the previous one, the previous
  frame is overwritten. The SPA cares about "current state," not
  every tick.
* **Cache-on-publish.** The most recent frame is cached so a fresh
  subscriber painted mid-session immediately gets the current state
  instead of waiting for the next tick.
* **Explicit unsubscribe.** Subscribers must be removed when the SSE
  stream closes, otherwise the tick loop keeps publishing into a dead
  slot. :meth:`StatusSubscriber.close` unblocks any waiter with
  ``None`` so its handler tears down cleanly.
"""

from __future__ import annotations

import threading
from typing import Protocol

from tankpit_bot.bus.session_status import SessionStatusDict


class StatusSubscriberProtocol(Protocol):
    """Surface one SSE consumer uses to await status frames.

    Kept alongside :class:`StatusSubscriber` so consumers that want DI
    can accept a fake without inheritance.
    """

    def push(self, status: SessionStatusDict) -> None:
        """Store ``status`` and wake any waiter.

        Args:
            status: Latest session status snapshot.
        """
        ...

    def next_frame(self, timeout: float | None = None) -> SessionStatusDict | None:
        """Wait for and consume the next status frame.

        Args:
            timeout: Maximum seconds to block; ``None`` blocks
                indefinitely.

        Returns:
            The next status frame, or ``None`` on timeout / close.
        """
        ...

    def close(self) -> None:
        """Mark the subscriber closed and unblock any waiter."""
        ...

    @property
    def closed(self) -> bool:
        """Return True when the subscriber has been closed."""
        ...


class StatusBusProtocol(Protocol):
    """Surface the tick loop uses to fan out status frames.

    Kept alongside :class:`StatusBus` for the same DI-without-
    inheritance reason as :class:`ModeBridgeProtocol`.
    """

    def publish(self, status: SessionStatusDict) -> None:
        """Push ``status`` to every subscriber and cache it.

        Args:
            status: Snapshot to broadcast.
        """
        ...

    def subscribe(self) -> StatusSubscriberProtocol:
        """Register a new subscriber and hand it the cached frame.

        Returns:
            A subscriber whose owner is responsible for calling
            :meth:`unsubscribe` (or the subscriber's ``close``) when
            done.
        """
        ...

    def unsubscribe(self, subscriber: StatusSubscriberProtocol) -> None:
        """Drop ``subscriber`` from the fan-out and close it.

        Args:
            subscriber: The subscriber to remove.
        """
        ...

    def subscriber_count(self) -> int:
        """Return the number of currently-registered subscribers.

        Returns:
            The current subscriber count.
        """
        ...


class StatusSubscriber:
    """One SSE subscriber's slot in the fan-out.

    Guarded by an internal :class:`threading.Condition`: the tick loop
    calls :meth:`push` on the publish thread; the SSE stream handler
    calls :meth:`next_frame` on its own thread.
    """

    def __init__(self) -> None:
        """Create an empty subscriber with no cached frame."""
        self._cond = threading.Condition()
        self._latest: SessionStatusDict | None = None
        self._closed = False

    def push(self, status: SessionStatusDict) -> None:
        """Store ``status`` and wake any waiting :meth:`next_frame`.

        Called from the tick-loop publish thread. If the subscriber has
        already been closed the frame is dropped silently — a closed
        subscriber is a torn-down SSE connection.

        Args:
            status: Latest session status snapshot.
        """
        with self._cond:
            if self._closed:
                return
            self._latest = status
            self._cond.notify_all()

    def next_frame(self, timeout: float | None = None) -> SessionStatusDict | None:
        """Wait for and consume the next status frame.

        Args:
            timeout: Maximum seconds to block. ``None`` blocks
                indefinitely. On timeout, returns ``None`` without
                consuming any cached frame — the caller can loop with
                a shorter timeout for heartbeats.

        Returns:
            The next cached :class:`SessionStatusDict`, or ``None`` if
            the timeout expired or the subscriber was closed while
            waiting.
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
        """Mark the subscriber closed and unblock any waiter.

        Called by the SSE stream handler when the client disconnects.
        Any thread parked in :meth:`next_frame` wakes and observes the
        closed state so its caller can tear down cleanly. Subsequent
        pushes are dropped.
        """
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


class StatusBus:
    """Fan-out of :class:`SessionStatusDict` frames to N subscribers.

    Owned by the service; the tick loop publishes here after every
    tick, and the aiohttp SSE handler calls :meth:`subscribe` /
    :meth:`unsubscribe` around the client's connection lifetime.
    """

    def __init__(self) -> None:
        """Create an empty bus with no subscribers and no cached frame."""
        self._lock = threading.Lock()
        self._subscribers: list[StatusSubscriberProtocol] = []
        self._latest: SessionStatusDict | None = None

    def publish(self, status: SessionStatusDict) -> None:
        """Push ``status`` to every subscriber and cache it for late joiners.

        Args:
            status: Snapshot to broadcast.
        """
        with self._lock:
            self._latest = status
            recipients = list(self._subscribers)
        for subscriber in recipients:
            subscriber.push(status)

    def subscribe(self) -> StatusSubscriberProtocol:
        """Register a new subscriber and hand it the cached frame.

        A fresh subscriber painted mid-session immediately observes the
        most recent status frame — no waiting for the next tick.

        Returns:
            A new subscriber. The caller is responsible for calling
            :meth:`unsubscribe` (or the subscriber's ``close``) when
            it is done. Concretely this is a :class:`StatusSubscriber`;
            the return type is the Protocol so consumers can depend on
            the abstract surface.
        """
        subscriber: StatusSubscriberProtocol = StatusSubscriber()
        with self._lock:
            self._subscribers.append(subscriber)
            latest = self._latest
        if latest is not None:
            subscriber.push(latest)
        return subscriber

    def unsubscribe(self, subscriber: StatusSubscriberProtocol) -> None:
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
            The subscriber count. Exposed for introspection + tests.
        """
        with self._lock:
            return len(self._subscribers)


__all__ = [
    "StatusBus",
    "StatusBusProtocol",
    "StatusSubscriber",
    "StatusSubscriberProtocol",
]
