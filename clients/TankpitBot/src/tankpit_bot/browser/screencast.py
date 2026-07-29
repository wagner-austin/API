"""Chrome screencast relay — live game video without a display stack.

``Page.startScreencast`` makes Chrome push a JPEG of every composited
frame over the CDP session the bot already holds; this module acks each
frame (Chrome throttles the stream until the previous frame is acked)
and publishes the decoded bytes into the service's
:class:`~tankpit_bot.service.frame_bus.FrameBus`, where the ``/video``
MJPEG handler fans them out to phones.

This replaces the entire Sunshine/Vibeshine virtual-display pipeline
for tankpit monitoring: no virtual monitor, no desktop capture, and —
decisively — no input injection anywhere, so the host mouse cannot be
touched (2026-07-28 decision: cut tankpit loose from fiesta).

The tick loop toggles the stream on DEMAND: subscribers on the frame
bus → :meth:`ScreencastService.start`; none → :meth:`stop`. Unwatched
sessions pay zero encode cost.
"""

from __future__ import annotations

import base64
from collections.abc import Callable

from platform_core.json_utils import JSONObject, require_int, require_str
from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol

log = get_logger(__name__)

SCREENCAST_QUALITY = 70
"""JPEG quality Chrome encodes each screencast frame at (0-100)."""

SCREENCAST_MAX_DIMENSION = 1024
"""Cap on each frame edge in device-independent pixels.

The game canvas streams fine at this size on a phone screen while
keeping per-frame payloads (base64 over CDP + JPEG over MJPEG) small
enough that the relay never becomes the session's bandwidth story.
"""


class ScreencastService:
    """Owns the ``Page.screencastFrame`` relay for one browser session.

    Single-threaded by construction: :meth:`start` / :meth:`stop` run on
    the Playwright thread (tick loop), and Playwright dispatches the
    frame events on that same thread while it pumps — so ``active``
    needs no lock. Only the ``publish`` callback crosses threads, and
    the frame bus is threadsafe.
    """

    def __init__(self, publish: Callable[[bytes], None]) -> None:
        """Bind the relay to its frame sink.

        Args:
            publish: Threadsafe sink each decoded JPEG frame is pushed
                into — production wires
                :meth:`~tankpit_bot.service.frame_bus.FrameBus.publish`.
        """
        self._publish = publish
        self._cdp: CDPSessionProtocol | None = None
        self.active = False

    def start(self, cdp: CDPSessionProtocol) -> None:
        """Begin the screencast on ``cdp``. Idempotent while active.

        The frame handler is registered once per CDP session — a
        re-start after a stop reuses the existing registration, and a
        NEW session (fresh ``cdp`` object) gets a fresh registration.

        Args:
            cdp: Active CDP session attached to the live tankpit page.
        """
        if self.active:
            return
        if self._cdp is not cdp:
            cdp.on("Page.screencastFrame", self._on_frame)
            self._cdp = cdp
        cdp.send(
            "Page.startScreencast",
            {
                "format": "jpeg",
                "quality": SCREENCAST_QUALITY,
                "maxWidth": SCREENCAST_MAX_DIMENSION,
                "maxHeight": SCREENCAST_MAX_DIMENSION,
                "everyNthFrame": 1,
            },
        )
        self.active = True
        log.info("Screencast started (viewer connected)")

    def stop(self, cdp: CDPSessionProtocol) -> None:
        """Stop the screencast. Idempotent while inactive.

        Args:
            cdp: Active CDP session the stream was started on.
        """
        if not self.active:
            return
        cdp.send("Page.stopScreencast")
        self.active = False
        log.info("Screencast stopped (no viewers)")

    def _on_frame(self, params: JSONObject) -> None:
        """Ack and publish one ``Page.screencastFrame`` event.

        The ack goes FIRST: Chrome suspends the stream until the
        in-flight frame is acked, so acking before the decode keeps the
        pipeline moving even if publish is momentarily slow. Sending
        from inside a CDP event handler is the established pattern here
        (``CDPService._record_frame`` pops sent-frame metadata the same
        way).

        Args:
            params: CDP event parameters carrying the base64 ``data``
                and the ``sessionId`` to ack.

        Raises:
            JSONTypeError: When the event omits ``data`` or
                ``sessionId`` — a CDP drift that must fail loudly, not
                degrade to a frozen stream.
            RuntimeError: When no CDP session is attached — an
                invariant violation, since the handler is only ever
                registered on the session stored in ``_cdp``.
        """
        session_id = require_int(params, "sessionId")
        data = require_str(params, "data")
        cdp = self._cdp
        if cdp is None:
            raise RuntimeError("screencastFrame received with no CDP session attached")
        cdp.send("Page.screencastFrameAck", {"sessionId": session_id})
        self._publish(base64.b64decode(data))


__all__ = [
    "SCREENCAST_MAX_DIMENSION",
    "SCREENCAST_QUALITY",
    "ScreencastService",
]
