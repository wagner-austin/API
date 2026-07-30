"""Wire-verified autoscroll-off enforcement at session start.

Autoscroll recenters the viewport when a walk reaches an edge tile
([[viewport-shift-protocol]]); the bot's whole scan/landing accounting
assumes the viewport is FIXED, so a session that starts with the
toggle ON silently skews everything downstream. The setting is
SERVER-PERSISTED per account and history shows it flipping by
accident (the 2026-07-24 key probe left it ON for a day; user ruling
2026-07-29: "we need to make sure autoscroll is always off otherwise
it throws the bot off. its a toggle... sometimes it resets to on").

There is no read-only query on the wire -- the only instrument is the
``a`` key toggle and the server's plaintext ``"A0"``/``"A1"`` ack --
so enforcement is a press-and-verify dance: press once and read the
ack; landing on ``A1`` proves the setting WAS off, so press again and
require the ``A0`` ack back. Either path ends wire-verified OFF. A
missing ack is a hard failure -- an unverified toggle must never be
guessed at (probe precedent, 2026-07-25).

The toggle only works IN-GAME (user ruling 2026-07-29: "you cant
enable or disable autoscroll til the bot is in the game btw") -- a
press on the entry screen acks nothing, which is exactly how the
first live firing failed at 23:08 ("game ready" is still pre-spawn).
The dance therefore waits for the wire to establish ``self_state``
(the tank's position broadcast proves the tank is in the game) before
the first press.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import AutoscrollPageProtocol
from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.protocol.decoders.text import try_decode_plaintext_ack
from tankpit_bot.types.message import CapturedMessage

log = get_logger(__name__)


_TOGGLE_SETTLE_MS = 1500
"""Wait after the key press for the server ack to land in the capture."""

_IN_GAME_POLL_MS = 500
"""Poll interval while waiting for the wire to prove the tank spawned."""

_IN_GAME_WAIT_BUDGET_MS = 30_000
"""Hard budget for the spawn wait; a tank with no position broadcast
after this long is a broken session, not a slow one."""


def _wait_until_in_game(page: AutoscrollPageProtocol) -> None:
    """Block until the wire establishes ``self_state`` (tank spawned).

    Args:
        page: Live game page (its wait pumps the event loop so CDP
            handlers keep filling the world service while we wait).

    Raises:
        RuntimeError: When no position broadcast arrives within the
            budget -- the toggle would silently ack nothing pre-spawn,
            and an unverified toggle must never be guessed at.
    """
    from tankpit_bot.sniffer.world_state import get_world_state

    waited_ms = 0
    while waited_ms < _IN_GAME_WAIT_BUDGET_MS:
        if get_world_state()["self_state"] is not None:
            return
        page.wait_for_timeout(float(_IN_GAME_POLL_MS))
        waited_ms += _IN_GAME_POLL_MS
    raise RuntimeError("tank never spawned within the autoscroll wait budget; toggle unverifiable")


def _read_autoscroll_ack(messages: list[CapturedMessage], start_index: int) -> bool | None:
    """Scan captured frames from ``start_index`` for the autoscroll ack.

    The ack is the server's PLAINTEXT two-byte echo (``"A0"``/``"A1"``,
    un-XORed), read from the raw frame body exactly like the viewport
    probe reads it.

    Args:
        messages: Capture buffer shared with the CDP service.
        start_index: Buffer length at the moment of the key press.

    Returns:
        The acked enabled flag, or ``None`` when no ack has arrived.
    """
    for captured in messages[start_index:]:
        if captured["direction"] != "received":
            continue
        data = decode_base64_safe(captured["payload"])
        if not data:
            continue
        offset = 0
        while offset + 2 < len(data):
            length = data[offset] | (data[offset + 1] << 8)
            offset += 2
            if length == 0 or offset + length > len(data):
                break
            body = data[offset : offset + length]
            offset += length
            ack = try_decode_plaintext_ack(body)
            if ack is not None and ack["msg_type"] == "autoscroll_ack":
                return ack["enabled"]
    return None


def _press_and_read(page: AutoscrollPageProtocol, messages: list[CapturedMessage]) -> bool:
    """Press ``a`` once and return the wire-verified new state.

    Args:
        page: Live game page.
        messages: Capture buffer shared with the CDP service.

    Returns:
        The acked enabled flag after the toggle.

    Raises:
        RuntimeError: When no ack arrives -- the toggle is unverified
            and the session must not continue on a guess.
    """
    start_index = len(messages)
    page.keyboard.press("a")
    page.wait_for_timeout(float(_TOGGLE_SETTLE_MS))
    enabled = _read_autoscroll_ack(messages, start_index)
    if enabled is None:
        raise RuntimeError("no autoscroll ack after the 'a' press; toggle unverified")
    return enabled


def ensure_autoscroll_off(page: AutoscrollPageProtocol, messages: list[CapturedMessage]) -> None:
    """Leave the session with autoscroll wire-verified OFF.

    Args:
        page: Live game page.
        messages: Capture buffer shared with the CDP service.

    Raises:
        RuntimeError: When the tank never spawns, an ack is missing,
            or the second press fails to land on ``A0`` -- the toggle
            protocol drifted and the session must not run on a skewed
            viewport model.
    """
    _wait_until_in_game(page)
    enabled = _press_and_read(page, messages)
    if enabled:
        # The setting WAS off; the probe press turned it on -- undo.
        if _press_and_read(page, messages):
            raise RuntimeError("autoscroll stuck ON after corrective press")
        log.info("Autoscroll verified OFF (was off; press-verify round trip)")
        return
    log.info("Autoscroll verified OFF (was ON at session start -- corrected)")


__all__ = [
    "ensure_autoscroll_off",
]
