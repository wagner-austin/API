"""Step (c) completion: the live seam — the real bot plays the sim.

:class:`SimCDPSession` implements the same ``CDPSessionProtocol`` the
production tick loop talks to, answering each ``Runtime.evaluate``
from SIM WORLD TRUTH:

- the page-client snapshot query (``window.__tankpitActiveGame``) is
  answered with a truthfully-built :class:`PageClientSnapshotDict` —
  the link is genuinely open (``ws_ready_state=1``), the client is
  genuinely present (the sim IS the client), and fields with no sim
  counterpart carry the type's honest "not captured" forms
  (``None`` / empty dicts), which the tick loop already handles;
- the injected websocket send (``atob('<b64>')``) is decoded through
  the sim transport into typed commands and queued on the server;
- the sent-frame metadata and raw-message hooks report empty — true:
  no browser hook exists.

:func:`run_sim_session` drives the PRODUCTION ``_tick_once`` against
a :class:`SimServer`: handshake into the bot's real message buffer,
then tick after tick of decide -> command bytes -> sim -> wire batch.

Step (e) addition: the link keeps a ``wire_log`` of every frame that
crossed it, in both directions, as :class:`CapturedMessage` records —
:func:`build_capture_session` assembles them into the standard
``CaptureSession`` shape so ``make audit``'s validators can re-derive
the archive claims from sim-generated wire.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import JSONObject, require_str

from tankpit_bot.action_lab.page_client_snapshot import (
    PageClientSnapshotDict,
    encode_page_client_snapshot,
)
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.transport import decode_client_payload, encode_tick_payload
from tankpit_bot.types import CapturedMessage, CaptureSession
from tankpit_bot.wire.helpers import EncodeError

_WS_OPEN = 1
_SIM_WS_URL = "wss://sim.tankpit.local/ws"


class SimCDPSession:
    """``CDPSessionProtocol`` implementation backed by a :class:`SimServer`.

    Commands the bot sends are decoded and queued on the sim server;
    snapshot queries answer from the sim world. The session records
    every decoded command for assertions and diagnostics.
    """

    def __init__(self, server: SimServer, table: bytes) -> None:
        """Bind the session to a sim server and its XOR table.

        Args:
            server: The sim server this link speaks to.
            table: Session XOR table (the bot must share it).
        """
        self.server = server
        self.table = table
        self.sent_commands: list[str] = []
        self.wire_log: list[CapturedMessage] = []
        self.map_visible = False
        self._last_send_ms: int | None = None
        self._detached = False

    def _snapshot(self) -> PageClientSnapshotDict:
        """Build the truthful page-client snapshot from sim world state.

        Returns:
            The snapshot the tick loop's health gate and diagnostics
            consume. ``self_fields``/``world_fields``/``map_fields``/
            ``world_collections`` are empty — the "client object not
            yet captured" form — because no minified JS heap exists;
            the alignment samplers treat empty as not-captured.
        """
        now = get_current_time_ms()
        last_send = self._last_send_ms
        return PageClientSnapshotDict(
            timestamp_ms=now,
            client_present=True,
            map_visible=self.map_visible,
            client_state=None,
            client_busy=False,
            pending_actions=0,
            heartbeat_age_ms=0,
            last_page_client_send_age_ms=None,
            last_bot_send_age_ms=None if last_send is None else now - last_send,
            ws_ready_state=_WS_OPEN,
            current_send_label=None,
            sent_frame_meta_queue_length=0,
            self_fields={},
            world_fields={},
            map_fields={},
            world_collections={},
        )

    def _handle_send_expression(self, expression: str, start: int) -> JSONObject:
        """Decode one injected websocket send and queue its commands.

        Args:
            expression: The ``Runtime.evaluate`` expression carrying
                ``atob('<b64>')`` with the framed command bytes.
            start: Offset of the ``atob('`` marker in the expression.

        Returns:
            The CDP-shaped result the production sender expects.
        """
        end = expression.find("')", start)
        payload = expression[start + len("atob('") : end]
        commands = decode_client_payload(payload, self.table)
        for command in commands:
            self.sent_commands.append(command["kind"])
            if command["kind"] == "map_open":
                self.map_visible = True
            if command["kind"] == "teleport":
                self.map_visible = False
            self.server.queue_command(self.server.client_id, command)
        now = get_current_time_ms()
        self._last_send_ms = now
        self.wire_log.append(
            CapturedMessage(
                timestamp_ms=now,
                direction="sent",
                payload=payload,
                ws_url=_SIM_WS_URL,
            )
        )
        byte_count = len(payload) * 3 // 4
        return {"result": {"value": f"SENT_{byte_count}_BYTES via {_SIM_WS_URL}"}}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Answer one CDP command from sim truth.

        Args:
            method: CDP method name.
            params: Method parameters.

        Returns:
            The CDP-shaped response.

        Raises:
            EncodeError: For ``Runtime.evaluate`` expressions the sim
                does not model — loud, never best-effort.
        """
        if method != "Runtime.evaluate":
            return {"result": {"value": None}}
        if params is None:
            raise EncodeError("sim session: Runtime.evaluate without params")
        expression = require_str(params, "expression")
        if "__tankpitActiveGame" in expression:
            return {"result": {"value": encode_page_client_snapshot(self._snapshot())}}
        atob_start = expression.find("atob('")
        if atob_start != -1:
            return self._handle_send_expression(expression, atob_start)
        if "__sentFrameMetaQueue" in expression:
            return {"result": {"value": None}}
        if "__rawMsgs" in expression:
            return {"result": {"value": []}}
        if "document.body" in expression:
            return {"result": {"value": ""}}
        raise EncodeError(f"sim session: unmodeled evaluate expression: {expression[:80]!r}")

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Accept an event registration (the sim delivers via the buffer).

        Args:
            event: CDP event name.
            handler: The handler (unused — batches are delivered
                straight into the bot's message buffer).
        """
        del event, handler

    def detach(self) -> None:
        """Mark the session detached."""
        self._detached = True


def deliver_batch(buffer: list[str], messages: list[BinaryMessage], link: SimCDPSession) -> None:
    """Encode one sim batch to wire bytes and append it to the bot buffer.

    The encoded frame is also recorded on the link's ``wire_log`` so a
    finished seam session can be assembled into a standard capture via
    :func:`build_capture_session`.

    Args:
        buffer: The bot's received-message buffer
            (``bot._cdp_message_buffer`` — base64 payload strings, the
            exact shape ``world_sync.drain_messages`` consumes).
        messages: The sim's decoded batch.
        link: The seam link (provides the XOR table and the wire log).
    """
    if not messages:
        return
    payload = encode_tick_payload(messages, link.table)
    link.wire_log.append(
        CapturedMessage(
            timestamp_ms=get_current_time_ms(),
            direction="received",
            payload=payload,
            ws_url=_SIM_WS_URL,
        )
    )
    buffer.append(payload)


def build_capture_session(link: SimCDPSession, magic: str, session_id: str) -> CaptureSession:
    """Assemble the link's recorded traffic as a standard capture session.

    The result is byte-compatible with what the production sniffer
    writes to ``runs/*/<id>.capture_session.json`` — the ``make
    audit`` validators consume it unchanged, which is exactly the
    point: the same instruments that watch the real server judge the
    sim's wire.

    Args:
        link: The seam link whose ``wire_log`` holds the session.
        magic: The session's XOR magic key (the validators rebuild the
            table from it, so it must be the one the seam was booted
            with).
        session_id: Identifier for the assembled session.

    Returns:
        The capture session, messages in recorded order.

    Raises:
        EncodeError: If the link recorded no traffic — an empty
            capture means the seam never ran, which is a harness bug,
            not a session.
    """
    if not link.wire_log:
        raise EncodeError("sim session recorded no wire traffic — nothing to capture")
    return CaptureSession(
        session_id=session_id,
        start_timestamp_ms=link.wire_log[0]["timestamp_ms"],
        end_timestamp_ms=link.wire_log[-1]["timestamp_ms"],
        base_url="https://sim.tankpit.local/",
        messages=list(link.wire_log),
        magic=magic,
        game_log=[],
        tank_names={},
    )


__all__ = [
    "SimCDPSession",
    "build_capture_session",
    "deliver_batch",
]
