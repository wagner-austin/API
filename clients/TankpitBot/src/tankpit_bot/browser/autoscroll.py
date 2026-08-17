"""Wire-verified autoscroll-off enforcement at session start.

Autoscroll recenters the viewport when a walk reaches an edge tile
([[viewport-shift-protocol]]); the bot's whole scan/landing accounting
assumes the viewport is FIXED, so a session that starts with the
toggle ON silently skews everything downstream. The setting is
SERVER-PERSISTED per account and history shows it flipping by
accident (the 2026-07-24 key probe left it ON for a day; user ruling
2026-07-29: "we need to make sure autoscroll is always off otherwise
it throws the bot off. its a toggle... sometimes it resets to on").

The instrument is the ``A{enabled}`` settings command
([[client-commands]]): the client emits the plaintext two-byte text
``"A0"``/``"A1"`` stating the DESIRED state — absolute, not a blind
toggle — and the server acks by echoing it back plaintext. The
enforcement therefore sends ``A0`` directly over the captured
websocket and requires the ``A0`` echo: idempotent (re-sending
requests the same state), no toggle parity to track, and no
dependence on page focus or key handling.

It used to press the ``a`` KEY instead, which broke on the first
fresh account (2026-08-13, arterial): hotkey maps are PER-ACCOUNT
server state (the ``H`` command), and a fresh account's default binds
``a`` to a scope pan — the capture shows both presses emitting
``03 5a 06`` (scope shift, direction 6 = west), never the autoscroll
command. Fresh accounts may also START with autoscroll enabled
(user, 2026-08-13); the absolute ``A0`` handles either starting
state by construction.

The command only works IN-GAME (user ruling 2026-07-29: "you cant
enable or disable autoscroll til the bot is in the game btw") -- a
send on the entry screen acks nothing, which is exactly how the
first live firing failed at 23:08 ("game ready" is still pre-spawn).
The enforcement therefore waits for the wire to establish
``self_state`` (the tank's position broadcast proves the tank is in
the game) before the first send.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol, PageWaitProtocol
from tankpit_bot.browser.cdp_utils import send_websocket_bytes
from tankpit_bot.capture.frames import split_payload_frames
from tankpit_bot.protocol.decoders.text import try_decode_plaintext_ack
from tankpit_bot.protocol.framing import encode_frame
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.types.message import CapturedMessage

log = get_logger(__name__)


_TOGGLE_SETTLE_MS = 1500
"""Wait after the command send for the server ack to land in the capture."""

_SEND_ATTEMPTS = 3
"""Sends to spend before declaring the setting unverifiable.

One send suffices on a healthy socket; the command is absolute, so a
re-send after a slow ack requests the same state and cannot skew
anything.
"""

_AUTOSCROLL_OFF_COMMAND = b"A0"
"""The plaintext settings command requesting autoscroll OFF."""

_IN_GAME_POLL_MS = 500
"""Poll interval while waiting for the wire to prove the tank spawned."""

_IN_GAME_WAIT_BUDGET_MS = 30_000
"""Hard budget for the spawn wait; a tank with no position broadcast
after this long is a broken session, not a slow one."""


def _wait_until_in_game(page: PageWaitProtocol, ws: WorldService) -> None:
    """Block until the wire establishes ``self_state`` (tank spawned).

    Args:
        page: Live game page (its wait pumps the event loop so CDP
            handlers keep filling the world service while we wait).
        ws: The session's world service; the spawn check reads it.

    Raises:
        RuntimeError: When no position broadcast arrives within the
            budget -- the command would silently ack nothing pre-spawn,
            and an unverified setting must never be guessed at.
    """
    waited_ms = 0
    while waited_ms < _IN_GAME_WAIT_BUDGET_MS:
        if ws.get_world_state()["self_state"] is not None:
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
        start_index: Buffer length at the moment of the send.

    Returns:
        The LAST acked enabled flag in the window, or ``None`` when no
        ack has arrived. Last, not first: every ack states the
        RESULTING state, so the latest one is always the current truth.

    Raises:
        FramingError: If a live payload is corrupt. This used to
            re-derive the frame walk inline and drop a torn tail with a
            silent ``break`` ([[session-state-deglobalisation]]).
    """
    latest: bool | None = None
    for captured in messages[start_index:]:
        if captured["direction"] != "received":
            continue
        for body in split_payload_frames(captured["payload"]):
            ack = try_decode_plaintext_ack(body)
            if ack is not None and ack["msg_type"] == "autoscroll_ack":
                latest = ack["enabled"]
    return latest


def ensure_autoscroll_off(
    page: PageWaitProtocol,
    cdp: CDPSessionProtocol,
    messages: list[CapturedMessage],
    ws: WorldService,
) -> None:
    """Leave the session with autoscroll wire-verified OFF.

    Args:
        page: Live game page (waits pump the event loop).
        cdp: Active CDP session carrying the game websocket.
        messages: Capture buffer shared with the CDP service.
        ws: The session's world service; the spawn wait reads it.

    Raises:
        RuntimeError: When the tank never spawns, the send fails, no
            ack arrives after :data:`_SEND_ATTEMPTS` sends, or the
            server acks ``A1`` to an ``A0`` request -- the protocol
            drifted and the session must not run on a skewed viewport
            model.
    """
    _wait_until_in_game(page, ws)
    start_index = len(messages)
    for attempt in range(_SEND_ATTEMPTS):
        result = send_websocket_bytes(
            cdp, encode_frame(_AUTOSCROLL_OFF_COMMAND), label="autoscroll_off"
        )
        if not result.startswith("SENT_"):
            raise RuntimeError(f"autoscroll A0 send failed: {result}")
        page.wait_for_timeout(float(_TOGGLE_SETTLE_MS))
        enabled = _read_autoscroll_ack(messages, start_index)
        if enabled is False:
            log.info("Autoscroll verified OFF (A0 acked)")
            return
        if enabled is True:
            raise RuntimeError("server acked autoscroll ON after an A0 request; protocol drifted")
        log.info(
            "Autoscroll A0 send %d drew no ack yet - re-sending",
            attempt + 1,
        )
    raise RuntimeError(f"no autoscroll ack after {_SEND_ATTEMPTS} A0 sends; setting unverified")


__all__ = [
    "ensure_autoscroll_off",
]
