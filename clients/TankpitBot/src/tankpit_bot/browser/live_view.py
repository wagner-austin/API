"""Page-push live view — steady-fps game video without the tick thread.

The 2026-07-28 CDP screencast relay was rate-limited by the SAME
Playwright thread the tick loop owns: Chrome only sends the next
screencast frame after the previous one is ACKED, and the acks only
flow while that thread pumps — so every heavy tick operation (map
open, scan, teleport handling) stalled the whole stream for seconds
(user report 2026-07-29: "the stream is a bit laggy... seems to
freeze or drop frames"). This module replaces the relay with capture
INSIDE the game page: an injected interval composites the client's
six stacked canvases ([[rendering-pipeline]] — Background/Tanks/
Action/Map/Overlay at 384x256 plus the 384x48 Menu strip, DPI-scaled)
into one JPEG per frame and hands the data URL to the bot through a
CDP BINDING (``Runtime.addBinding`` → ``window.__botCastDeliver``).

Why a binding and not a loopback HTTP POST: Chrome's Local Network
Access gate intercepts page fetches to 127.0.0.1 behind a permission
no automated browser can grant — the fetch neither resolves nor
rejects, it hangs forever (measured 2026-07-29: caster ticks=36 in
3 s, posts=0, the one-shot probe fetch equally stuck; Playwright
1.57 exposes no ``local-network-access`` permission to grant). The
binding channel has no such gate — and unlike the screencast it has
NO BACKPRESSURE: the page keeps capturing at its configured fps
regardless of what the bot thread is doing; frames queued during a
tick stall burst-deliver afterwards and collapse into the
latest-wins frame bus.

The tick loop drives demand exactly like the screencast did:
subscribers on the frame bus → :meth:`LiveViewService.ensure` every
tick (idempotent in-page, and re-evaluating each tick self-heals the
caster across page navigations, which wipe injected JS — the binding
itself survives navigations); zero subscribers →
:meth:`LiveViewService.stop`.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable

from platform_core.json_utils import JSONObject, require_str
from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.bot.config import resolve_video_fps, resolve_video_quality

log = get_logger(__name__)

BINDING_NAME = "__botCastDeliver"
"""Name of the CDP binding the caster calls with each frame's data URL."""

_DATA_URL_PREFIX = "data:image/jpeg;base64,"

_CASTER_TEMPLATE = """
(() => {
  if (window.__botCast === undefined) {
    window.__botCast = {
      timer: null,
      collect() {
        const found = [];
        for (const c of document.querySelectorAll("canvas")) {
          const r = c.getBoundingClientRect();
          if (c.width > 0 && r.width > 0 && r.height > 0) {
            found.push({ c, r, z: Number(getComputedStyle(c).zIndex) || 0 });
          }
        }
        return found;
      },
      frame() {
        const visible = this.collect();
        if (visible.length === 0) {
          if (!this.emptyLogged) {
            this.emptyLogged = true;
            console.error("BotCastHook no visible canvases");
          }
          return null;
        }
        const minX = Math.min(...visible.map((v) => v.r.left));
        const minY = Math.min(...visible.map((v) => v.r.top));
        const maxX = Math.max(...visible.map((v) => v.r.right));
        const maxY = Math.max(...visible.map((v) => v.r.bottom));
        const scale = Math.max(...visible.map((v) => v.c.width / v.r.width));
        const off =
          window.__botCastCanvas ||
          (window.__botCastCanvas = document.createElement("canvas"));
        const w = Math.round((maxX - minX) * scale);
        const h = Math.round((maxY - minY) * scale);
        if (off.width !== w || off.height !== h) {
          off.width = w;
          off.height = h;
        }
        const g = off.getContext("2d");
        g.fillStyle = "#000";
        g.fillRect(0, 0, w, h);
        visible.sort((a, b) => a.z - b.z);
        for (const v of visible) {
          g.drawImage(
            v.c,
            (v.r.left - minX) * scale,
            (v.r.top - minY) * scale,
            v.r.width * scale,
            v.r.height * scale,
          );
        }
        return off.toDataURL("image/jpeg", __QUALITY__);
      },
      start() {
        if (this.timer !== null) {
          return;
        }
        this.timer = setInterval(() => {
          if (typeof window.__BINDING__ !== "function") {
            return;
          }
          let data = null;
          try {
            data = this.frame();
          } catch (err) {
            if (!this.frameErrorLogged) {
              this.frameErrorLogged = true;
              console.error("BotCastHook frame failed:", String(err));
            }
            return;
          }
          if (data !== null) {
            window.__BINDING__(data);
          }
        }, __INTERVAL_MS__);
      },
      stop() {
        if (this.timer !== null) {
          clearInterval(this.timer);
          this.timer = null;
        }
      },
    };
  }
  window.__botCast.start();
})()
"""

_STOP_EXPRESSION = "(() => { if (window.__botCast !== undefined) { window.__botCast.stop(); } })()"


def build_caster_expression(fps: float, quality: float) -> str:
    """Render the in-page caster snippet for the configured cadence.

    Args:
        fps: Frames per second the page interval targets.
        quality: JPEG quality (0..1) passed to ``toDataURL``.

    Returns:
        A self-contained JS expression that defines ``window.__botCast``
        (once) and starts its interval — idempotent on re-evaluation.

    Raises:
        ValueError: When fps is not positive (the interval math would
            divide by zero) or quality falls outside (0, 1].
    """
    if fps <= 0:
        raise ValueError(f"video fps must be positive, got {fps}")
    if not 0 < quality <= 1:
        raise ValueError(f"video quality must be in (0, 1], got {quality}")
    interval_ms = max(1, round(1000 / fps))
    return (
        _CASTER_TEMPLATE.replace("__QUALITY__", repr(quality))
        .replace("__INTERVAL_MS__", str(interval_ms))
        .replace("__BINDING__", BINDING_NAME)
    )


class LiveViewService:
    """Owns the in-page caster + binding relay for one browser session.

    Single-threaded by construction, like the screencast service it
    replaced: :meth:`ensure` / :meth:`stop` and the binding events all
    run on the Playwright thread, so ``active`` needs no lock. Only
    the ``publish`` callback crosses threads, and the frame bus is
    threadsafe.
    """

    def __init__(self, publish: Callable[[bytes], None]) -> None:
        """Bind the relay to its frame sink.

        Args:
            publish: Threadsafe sink each decoded JPEG frame is pushed
                into — production wires
                :meth:`~tankpit_bot.service.frame_bus.FrameBus.publish`.
        """
        self._publish = publish
        self._expression = build_caster_expression(resolve_video_fps(), resolve_video_quality())
        self._cdp: CDPSessionProtocol | None = None
        self.active = False

    def ensure(self, cdp: CDPSessionProtocol) -> None:
        """(Re)start the in-page caster. Called EVERY demanded tick.

        The first call on a CDP session registers the frame binding
        (CDP bindings survive page navigations). The caster snippet is
        idempotent in-page (an existing interval is kept), and
        re-evaluating each tick is the self-heal for page navigations —
        quit-to-lobby or a re-login wipes injected JS, and the next
        demanded tick simply reinstalls the caster.

        Args:
            cdp: Active CDP session attached to the live tankpit page.
        """
        if self._cdp is not cdp:
            cdp.on("Runtime.bindingCalled", self._on_binding_called)
            cdp.send("Runtime.addBinding", {"name": BINDING_NAME})
            self._cdp = cdp
        cdp.send("Runtime.evaluate", {"expression": self._expression})
        if not self.active:
            self.active = True
            log.info("Live view casting (viewer connected)")

    def stop(self, cdp: CDPSessionProtocol) -> None:
        """Stop the in-page caster. Idempotent while inactive.

        Args:
            cdp: Active CDP session the caster was installed on.
        """
        if not self.active:
            return
        cdp.send("Runtime.evaluate", {"expression": _STOP_EXPRESSION})
        self.active = False
        log.info("Live view stopped (no viewers)")

    def _on_binding_called(self, params: JSONObject) -> None:
        """Decode one caster frame from a ``Runtime.bindingCalled`` event.

        Args:
            params: CDP event parameters carrying the binding ``name``
                and the data-URL ``payload``.

        Raises:
            JSONTypeError: When the event omits ``name`` or
                ``payload`` — CDP drift that must fail loudly.
            ValueError: When the payload is not a JPEG data URL or its
                base64 is corrupt — caster drift, equally loud.
        """
        if require_str(params, "name") != BINDING_NAME:
            return
        payload = require_str(params, "payload")
        if not payload.startswith(_DATA_URL_PREFIX):
            raise ValueError("caster payload is not a JPEG data URL")
        try:
            frame = base64.b64decode(payload[len(_DATA_URL_PREFIX) :], validate=True)
        except binascii.Error as exc:
            raise ValueError("caster payload carries invalid base64") from exc
        self._publish(frame)


__all__ = [
    "BINDING_NAME",
    "LiveViewService",
    "build_caster_expression",
]
