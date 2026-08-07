"""Replay page doubles and the clock they advance.

The fake page a replayed probe drives, its keyboard, and the frame
batch source that feeds captured messages back in step with the clock.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import (
    dataclass,
)

from platform_core.json_utils import (
    JSONValue,
)

from tankpit_bot._test_hooks import (
    BufferedMessageSourceProtocol,
    KeyboardProtocol,
    ResponseProtocol,
)
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler


@dataclass
class ReplayClock:
    """Monotonically advancing clock under harness control.

    Mirrors the production wall-clock signature
    (``Callable[[], int]``) so the action-lab hook
    ``action_hooks.get_current_time_ms`` can be pointed at it
    transparently.
    """

    now_ms: int = 0

    def __call__(self) -> int:
        """Return the current controlled timestamp in milliseconds."""
        return self.now_ms

    def advance(self, delta_ms: int) -> None:
        """Move the clock forward by ``delta_ms``.

        Args:
            delta_ms: Milliseconds to advance.
        """
        self.now_ms += delta_ms


class FrameBatchSource:
    """Mutable cursor over a captured-frame stream.

    The cursor walks a recorded payload list one batch at a time. The
    harness uses this instead of a generator/Iterator so the cursor is
    a first-class object with explicit state -- callers can inspect
    progress without consuming the source.
    """

    def __init__(self, payloads: list[str], batch_size: int) -> None:
        """Initialize the frame cursor.

        Args:
            payloads: Ordered base64-encoded received payloads.
            batch_size: Number of payloads returned per ``next_batch``
                call.
        """
        self._payloads = payloads
        self._batch_size = batch_size
        self._cursor = 0

    def next_batch(self) -> list[str]:
        """Pop the next batch.

        Returns:
            A list of up to ``batch_size`` payloads. Empty when the
            source is exhausted.
        """
        if self._cursor >= len(self._payloads):
            return []
        batch = self._payloads[self._cursor : self._cursor + self._batch_size]
        self._cursor += len(batch)
        return batch

    @property
    def consumed(self) -> int:
        """Return the number of payloads handed out so far."""
        return self._cursor


class _ReplayKeyboard:
    """No-op keyboard satisfying :class:`KeyboardProtocol`.

    The action-lab attempt loops never exercise keyboard input, so the
    methods are bare stubs that simply absorb their arguments.
    """

    def press(self, key: str, *, delay: float | None = None) -> None:
        """Absorb a key-press request."""
        _ = (key, delay)

    def type(self, text: str, *, delay: float | None = None) -> None:
        """Absorb a text-type request."""
        _ = (text, delay)


class ClockAdvancingPage:
    """``PageProtocol`` whose ``wait_for_timeout`` advances a clock.

    Lifted from per-test ``_FakePage`` forks that all share the same
    shape: every wait advances a :class:`ReplayClock`, optionally runs
    an ``on_wait`` callback (used by tests that sequence world-state
    providers between waits), and records the wait duration. All other
    PageProtocol methods are no-ops.

    Used by tests that don't need real frame replay (the
    :class:`ReplayPage` harness already covers that) but still need a
    page whose ``wait_for_timeout`` ticks deterministically.
    """

    url = "https://tankpit.com/play"

    def __init__(
        self,
        clock: ReplayClock,
        *,
        on_wait: Callable[[], None] | None = None,
    ) -> None:
        """Initialize with a clock and an optional wait-side-effect.

        Args:
            clock: Clock advanced by every ``wait_for_timeout`` call.
            on_wait: Optional callback invoked after each clock tick.
                Used by tests that sequence world-state snapshots
                between waits (the callback advances the provider).
                Tests that need to wire the callback after the page
                already exists can set ``page.on_wait`` directly.
        """
        self._clock = clock
        self.on_wait = on_wait
        self._keyboard = _ReplayKeyboard()
        self.waits: list[float] = []

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Return the no-op keyboard."""
        return self._keyboard

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """Absorb a navigation request; never returns a real response."""
        _ = (url, referer, timeout, wait_until)
        return None

    def wait_for_timeout(self, timeout: float) -> None:
        """Advance the clock by ``timeout`` ms and run ``on_wait``."""
        self.waits.append(timeout)
        self._clock.advance(int(timeout))
        if self.on_wait is not None:
            self.on_wait()

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Absorb an event-wait request."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Absorb a function-wait request."""
        _ = (expression, timeout)

    def close(
        self,
        *,
        reason: str | None = None,
        run_before_unload: bool | None = None,
    ) -> None:
        """Absorb a close request."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Absorb an evaluate request; never returns a real value."""
        _ = expression
        return None


class ReplayPage:
    """Page substitute that feeds frames each ``wait_for_timeout`` call.

    The replay harness owns the frame stream and the clock. Each time
    the action-lab wait helpers call ``page.wait_for_timeout(ms)``:

    1. The clock advances by ``ms``.
    2. The next batch of recorded frames is appended to the probe's
       ``_cdp_message_buffer``.

    When the frame source is exhausted, subsequent waits still advance
    the clock -- this is how the wait helpers reach their timeout in a
    recorded session that ends before the requested outcome.

    Implements the full :class:`tankpit_bot._test_hooks.PageProtocol`
    surface so the harness can assign ``probe._page = ReplayPage(...)``
    without weakening the production type. The methods action-lab waits
    do not call are simple stubs.
    """

    def __init__(
        self,
        probe: BufferedMessageSourceProtocol,
        frame_source: FrameBatchSource,
        clock: ReplayClock,
    ) -> None:
        """Initialize the replay page.

        Args:
            probe: Probe whose ``_cdp_message_buffer`` receives frames.
            frame_source: Mutable cursor over the recorded frame stream.
            clock: Shared replay clock advanced on every wait.
        """
        self._probe = probe
        self._frame_source = frame_source
        self._clock = clock
        self._keyboard = _ReplayKeyboard()
        self._url = "https://tankpit.com/play"
        self.waits_ms: list[float] = []
        self.frames_fed: int = 0

    @property
    def url(self) -> str:
        """Return the current URL."""
        return self._url

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Return the no-op keyboard satisfying ``KeyboardProtocol``."""
        return self._keyboard

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """Absorb a navigation request without doing any real network IO."""
        _ = (referer, timeout, wait_until)
        self._url = url
        return None

    def wait_for_timeout(self, timeout: float) -> None:
        """Advance the clock and feed the next batch of frames.

        Args:
            timeout: Milliseconds to advance.
        """
        self.waits_ms.append(timeout)
        self._clock.advance(int(timeout))
        batch = self._frame_source.next_batch()
        if not batch:
            return
        self._probe._cdp_message_buffer.extend(batch)
        self.frames_fed += len(batch)

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Absorb an event-wait request without blocking."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Absorb a function-wait request without evaluating anything."""
        _ = (expression, timeout)

    def close(
        self,
        *,
        reason: str | None = None,
        run_before_unload: bool | None = None,
    ) -> None:
        """Absorb a close request."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Return ``None`` for any JS expression -- nothing to evaluate."""
        _ = expression
        return None
