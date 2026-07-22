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
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import JSONObject, require_str

from tankpit_bot.action_lab.page_client_snapshot import (
    PageClientSnapshotDict,
    encode_page_client_snapshot,
)
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.protocol.helpers import EncodeError
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.transport import decode_client_payload, encode_tick_payload

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
        self._last_send_ms = get_current_time_ms()
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


def deliver_batch(buffer: list[str], messages: list[BinaryMessage], table: bytes) -> None:
    """Encode one sim batch to wire bytes and append it to the bot buffer.

    Args:
        buffer: The bot's received-message buffer
            (``bot._cdp_message_buffer`` — base64 payload strings, the
            exact shape ``world_sync.drain_messages`` consumes).
        messages: The sim's decoded batch.
        table: Session XOR table.
    """
    if not messages:
        return
    buffer.append(encode_tick_payload(messages, table))


__all__ = [
    "SimCDPSession",
    "deliver_batch",
]
