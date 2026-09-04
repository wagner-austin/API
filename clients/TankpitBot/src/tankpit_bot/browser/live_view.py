"""Page-push live view — game video that never touches the tick thread.

Capture happens INSIDE the game page: an injected interval composites
the client's six stacked canvases ([[rendering-pipeline]] —
Background/Tanks/Action/Map/Overlay at 384x256 plus the 384x48 Menu
strip, DPI-scaled) into one JPEG per frame. The page then POSTs that
JPEG to the service's own ``/cast`` route on loopback, where aiohttp
receives it on the MAIN thread and publishes it to the frame bus.

**Two earlier transports failed the same way, and the POST is what
removes the shared cause.**

The 2026-07-28 CDP screencast relay was rate-limited by the SAME
Playwright thread the tick loop owns: Chrome only sends the next
screencast frame after the previous one is ACKED, and the acks only
flow while that thread pumps (user report 2026-07-29: "the stream is a
bit laggy... seems to freeze or drop frames").

Its replacement, a CDP BINDING (``Runtime.addBinding`` →
``window.__botCastDeliver``), removed the ack handshake but not the
thread. Binding events are dispatched by Playwright, on the connection
Playwright owns, driven by the thread running the tick loop. Frames
produced while that thread was busy queued and burst-delivered
afterwards, and the latest-wins frame bus collapsed each burst to ONE
frame. User report 2026-09-03: "it lags whenever the bot opens the map
or shoots off screen or teleports. its like were showing a slideshow" —
and the tick log agreed exactly: 7.84 s on a ``cmd=shoot`` at a distant
target, 2.73 s on ``cmd=map_open``.

Two measurements bound where the binding did and did not survive. During
``page.wait_for_timeout`` — how the tick loop waits (``bot/tick_loop.py``
L 241, L 246) — binding frames arrived at 31.3/s, versus 29.3/s while
the thread was otherwise inside Playwright: waiting does not starve the
stream, because the dispatcher pumps throughout. But a pure-Python busy
stretch blocked delivery completely: 0 frames for 3 s, then 94 in 7 ms.
The heavy operations the user named are exactly that shape.

Why a POST was believed impossible, and why it is not: Chrome's Local
Network Access gate intercepts page fetches to 127.0.0.1 behind a
permission no automated browser can grant THROUGH THE PERMISSIONS API —
the fetch neither resolves nor rejects, it hangs forever (measured
2026-07-29: caster ticks=36 in 3 s, posts=0; Playwright 1.57 exposes no
``local-network-access`` permission to grant). That was recorded as a
law and read as one, and it was cited as the reason this transport could
not be reconsidered.

CORRECTION (2026-09-03): the gate is a Chromium FEATURE, and this
process owns its own launch args. Measured, five POSTs from a real
https://tankpit.com page to a loopback listener in this workspace's own
container:

    default flags                              server received 0/5
    with :data:`~tankpit_bot.sniffer.chrome_launch.LOOPBACK_POST_ARGS`
                                               server received 5/5

So the constraint was "not grantable per-page at runtime", not "not
possible". Nobody had tried turning the feature off. Recorded because a
law stated without its escape hatch stops the next reader from looking.

The POST inherits the binding's one real virtue — NO BACKPRESSURE, the
page captures at its configured fps regardless of the bot — and drops
its defect, because nothing on the delivery path is owned by the tick
thread any more.

The tick loop still drives demand: subscribers on the frame bus →
:meth:`LiveViewService.ensure` every tick (idempotent in-page, and
re-evaluating each tick self-heals the caster across page navigations,
which wipe injected JS); zero subscribers → :meth:`LiveViewService.stop`.

UNCHANGED FRAMES ARE NOT SENT (2026-09-03). The interval samples on a
wall clock, but the tankpit client paints on DIRTY FLAGS: its rAF loop
runs at 60 Hz and draws nothing unless a layer, a tank or an action was
marked dirty, and game state advances on its own ``setTimeout`` tick
rather than per frame. Measured at 12 Hz sampling, roughly 71 per cent
of captures were byte-identical to the one before — the page was
base64-ing and shipping about 800 KB/s to re-send a picture the viewer
already had. The caster now compares each data URL to the previous one
and posts only on a change.

This is safe for a new viewer because
:meth:`~tankpit_bot.bus.frame_bus.FrameBus.subscribe` hands a fresh
subscriber the CACHED frame, so a tile drawn during a still moment gets
a picture immediately rather than waiting for the next repaint. The
MJPEG keepalive re-sends the last frame on its own timer for
intermediaries that idle out an inactive connection.

Dedup does not itself make the picture smoother — it makes every frame
on the wire a real one. The ceiling is the game's own paint rate, and
the two numbers taken for that ceiling both need an asterisk. A CDP
screencast measurement on 2026-07-29 recorded 0.6 fps idle and 2.8 fps
in play, but screencast delivery was ack-gated on the tick thread, so
it reports what got THROUGH, not what was painted. A 2026-09-03 count
of distinct frames over the binding caster found 3.0 to 3.2 per second,
and that count was taken downstream of the latest-wins bus, which is
precisely where a burst collapses to one. Neither is a clean
measurement of the client, and they agree partly because they share the
defect. What the paint ceiling actually is must be re-measured over the
POST transport, with ``tankpit-stream-probe`` against the public
stream.
"""

from __future__ import annotations

from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.bot.config import resolve_video_fps, resolve_video_quality

log = get_logger(__name__)


_CASTER_TEMPLATE = """
(() => {
  if (window.__botCast === undefined) {
    window.__botCast = {
      timer: null,
      lastData: null,
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
          if (data === null || data === this.lastData) {
            return;
          }
          this.lastData = data;
          // POST, not a CDP binding. The binding delivered frames on
          // the connection Playwright owns, which is driven by the
          // thread running the tick loop -- so a heavy tick (map open,
          // teleport, a shot at a distant target) queued every frame
          // produced during it and released them in one burst that the
          // latest-wins bus collapsed to ONE. Seven seconds of play
          // arrived as a single picture.
          //
          // fetch reaches the service's aiohttp loop on the MAIN
          // thread, which the tick loop never occupies, so delivery no
          // longer depends on what the bot is doing.
          //
          // Sent as BYTES, not as the base64 data URL the comparison
          // above uses. `toDataURL` is what makes the cheap
          // string-equality dedup possible, but base64 inflates every
          // frame by a third and would have to be decoded again on the
          // service side. `fetch` on a data: URL is a local decode, so
          // the wire carries the JPEG itself.
          //
          // No `keepalive`: it caps the total inflight body at 64 KB
          // across ALL keepalive requests, and one composited frame is
          // already about 60 KB, so a second concurrent frame would be
          // refused by the browser rather than sent.
          //
          // Re-typed text/plain so the POST stays a CORS SIMPLE
          // request. The bytes are unchanged -- only the label moves.
          // A Blob posted as image/jpeg is not simple, so the browser
          // sends an OPTIONS preflight first, and a preflight the
          // service does not answer means the frame is never sent at
          // all. A simple request needs no preflight and no CORS
          // response headers, because nothing here reads the reply.
          // The route validates the JPEG magic bytes rather than
          // trusting this header, so the label costs nothing.
          fetch(data)
            .then((r) => r.blob())
            .then((blob) =>
              fetch(__CAST_URL__, {
                method: "POST",
                body: new Blob([blob], { type: "text/plain" }),
              }),
            )
            .catch((err) => {
              if (!this.postErrorLogged) {
                this.postErrorLogged = true;
                console.error("BotCastHook post failed:", String(err));
              }
            });
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


def build_caster_expression(fps: float, quality: float, cast_url: str) -> str:
    """Render the in-page caster snippet for the configured cadence.

    Args:
        fps: Frames per second the page interval targets.
        quality: JPEG quality (0..1) passed to ``toDataURL``.
        cast_url: Absolute URL the page POSTs each frame to. Must be
            non-empty: a caster with nowhere to send is a timer burning
            JPEG encodes for nothing.

    Returns:
        A self-contained JS expression that defines ``window.__botCast``
        (once) and starts its interval — idempotent on re-evaluation.

    Raises:
        ValueError: When fps is not positive (the interval math would
            divide by zero), quality falls outside (0, 1], or the cast
            URL is empty.
    """
    if fps <= 0:
        raise ValueError(f"video fps must be positive, got {fps}")
    if not 0 < quality <= 1:
        raise ValueError(f"video quality must be in (0, 1], got {quality}")
    if not cast_url:
        raise ValueError("cast URL must not be empty")
    interval_ms = max(1, round(1000 / fps))
    return (
        _CASTER_TEMPLATE.replace("__QUALITY__", repr(quality))
        .replace("__INTERVAL_MS__", str(interval_ms))
        .replace("__CAST_URL__", dump_json_str(cast_url))
    )


class LiveViewService:
    """Installs and removes the in-page caster for one browser session.

    It no longer RELAYS frames, which is the whole point: the caster
    POSTs them straight to the service's aiohttp loop. This object only
    turns casting on and off, so nothing about frame delivery depends on
    the Playwright thread any more.
    """

    def __init__(self, cast_url: str) -> None:
        """Bind the caster to the endpoint it will post frames to.

        Args:
            cast_url: Absolute URL of the service's frame-intake route.
                The service knows its own port and passes it down; this
                class does not read the environment, so a bot launched
                without a service cannot be given a caster that posts
                into nothing.
        """
        self._expression = build_caster_expression(
            resolve_video_fps(), resolve_video_quality(), cast_url
        )
        self._cdp: CDPSessionProtocol | None = None
        self.active = False

    def ensure(self, cdp: CDPSessionProtocol) -> None:
        """(Re)start the in-page caster. Called EVERY demanded tick.

        The caster snippet is idempotent in-page (an existing interval
        is kept), and re-evaluating each tick is the self-heal for page
        navigations — quit-to-lobby or a re-login wipes injected JS, and
        the next demanded tick simply reinstalls the caster.

        Args:
            cdp: Active CDP session attached to the live tankpit page.
        """
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


__all__ = [
    "LiveViewService",
    "build_caster_expression",
]
