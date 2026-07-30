"""Click-to-flag: a human bug marker with tick-context snapshot.

A human watching ``make run`` sees a bug and clicks the HUD's flag
button. The click travels over a CDP binding (the same
``Runtime.addBinding`` channel the live-view caster uses — page →
loopback fetches hang forever behind Chrome's Local Network Access
gate, bindings do not) and lands here, where it becomes a
``human_flag`` DIAGNOSTIC event on ``runs/<mode>/latest.events.jsonl``
carrying the click's wall-clock timestamp plus a JSON snapshot of the
last :data:`FLAG_RING_SIZE` HUD payloads — what the bot was thinking
in the ticks leading up to the click. ``make analyze`` can then anchor
on the flag instead of the human reconstructing "it was around when it
teleported twice" from memory.

Single-threaded by construction, like
:class:`~tankpit_bot.browser.live_view.LiveViewService`:
:meth:`FlagCaptureService.ensure`, :meth:`FlagCaptureService.record_tick`,
and the binding events all run on the Playwright thread, so the ring
buffer needs no lock.
"""

from __future__ import annotations

from collections import deque

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_str,
)
from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.browser.overlay import OverlayStateDict, encode_overlay_state
from tankpit_bot.runtime_logging import emit_diagnostic

log = get_logger(__name__)

FLAG_BINDING_NAME = "__botFlagDeliver"
"""Name of the CDP binding the HUD flag button calls on click."""

FLAG_RING_SIZE = 8
"""HUD payloads kept as the flag's lead-up context (~8 ticks ≈ 16 s)."""


class FlagCaptureService:
    """Owns the flag binding + recent-tick ring for one browser session."""

    def __init__(self) -> None:
        """Initialize with no CDP session and an empty tick ring."""
        self._cdp: CDPSessionProtocol | None = None
        self._recent: deque[OverlayStateDict] = deque(maxlen=FLAG_RING_SIZE)

    def ensure(self, cdp: CDPSessionProtocol) -> None:
        """Register the flag binding on the session. Called every tick.

        Idempotent per session: the first call registers the binding
        (CDP bindings survive page navigations), later calls with the
        same session are no-ops, and a fresh session after a browser
        restart re-registers.

        Args:
            cdp: Active CDP session attached to the live tankpit page.
        """
        if self._cdp is cdp:
            return
        cdp.on("Runtime.bindingCalled", self._on_binding_called)
        cdp.send("Runtime.addBinding", {"name": FLAG_BINDING_NAME})
        self._cdp = cdp

    def record_tick(self, overlay: OverlayStateDict) -> None:
        """Push one tick's HUD payload into the lead-up ring.

        Args:
            overlay: The payload the HUD rendered this tick.
        """
        self._recent.append(overlay)

    def _on_binding_called(self, params: JSONObject) -> None:
        """Turn one flag click into a ``human_flag`` diagnostic event.

        Args:
            params: CDP event parameters carrying the binding ``name``
                and the click's JSON ``payload``.

        Raises:
            JSONTypeError: When the event or its payload omits a
                required field — HUD/CDP drift that must fail loudly.
        """
        if require_str(params, "name") != FLAG_BINDING_NAME:
            return
        click = narrow_json_to_dict(load_json_str(require_str(params, "payload")))
        flag_seq = require_int(click, "flag_seq")
        clicked_at_ms = require_int(click, "clicked_at_ms")
        recent: list[JSONValue] = [encode_overlay_state(o) for o in self._recent]
        emit_diagnostic(
            diagnostic_kind="human_flag",
            flag_seq=flag_seq,
            clicked_at_ms=clicked_at_ms,
            recent_ticks=dump_json_str(recent),
        )
        log.info(
            "HUMAN FLAG #%d captured (%d lead-up ticks snapshotted)",
            flag_seq,
            len(recent),
        )


__all__ = [
    "FLAG_BINDING_NAME",
    "FLAG_RING_SIZE",
    "FlagCaptureService",
]
