"""Browser-layer dependency injection: the autoscroll enforcer.

Lives here rather than in the process-wide ``_test_hooks`` package for a
layering reason. The enforcer must name :class:`WorldService` — it waits
on the session's own ``self_state`` to prove the tank spawned before it
trusts a toggle ack — and ``_test_hooks`` sits BELOW ``sniffer``, because
``sniffer/world_service.py`` depends on the terrain and clock seams it
provides. Naming ``WorldService`` from there closes an import cycle
through ``state`` (measured 2026-08-07).

``action_lab`` and ``replay`` own package-local ``_test_hooks`` modules
for the same reason, so this is the established shape
([[session-state-deglobalisation]] step 8).
"""

from __future__ import annotations

from typing import Protocol

from tankpit_bot._test_hooks import CDPSessionProtocol, PageWaitProtocol
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.types import CapturedMessage


class AutoscrollEnforcerProtocol(Protocol):
    """Protocol for leaving a session with autoscroll wire-verified OFF."""

    def __call__(
        self,
        page: PageWaitProtocol,
        cdp: CDPSessionProtocol,
        messages: list[CapturedMessage],
        ws: WorldService,
    ) -> None:
        """Leave the session with autoscroll wire-verified OFF.

        Args:
            page: Live game page.
            cdp: Active CDP session carrying the game websocket.
            messages: Capture buffer shared with the CDP service.
            ws: The session's world service; the spawn wait reads it.
        """
        ...


def _real_ensure_autoscroll_off(
    page: PageWaitProtocol,
    cdp: CDPSessionProtocol,
    messages: list[CapturedMessage],
    ws: WorldService,
) -> None:
    """Real implementation -- delegate to the autoscroll module.

    The import is deferred so this module never drags the rest of the
    browser layer in at import time.

    Args:
        page: Live game page.
        cdp: Active CDP session carrying the game websocket.
        messages: Capture buffer shared with the CDP service.
        ws: The session's world service; the spawn wait reads it.
    """
    from tankpit_bot.browser.autoscroll import ensure_autoscroll_off as _impl

    _impl(page, cdp, messages, ws)


#: The autoscroll-off dance. Tests replace this attribute via
#: save-and-restore to skip the live wire handshake.
ensure_autoscroll_off: AutoscrollEnforcerProtocol = _real_ensure_autoscroll_off


__all__ = [
    "AutoscrollEnforcerProtocol",
    "_real_ensure_autoscroll_off",
    "ensure_autoscroll_off",
]
