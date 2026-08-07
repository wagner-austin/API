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
from tankpit_bot.capture.xor import build_session_xor_table, require_static_key, xor_decode_body
from tankpit_bot.protocol.commands import CMD_STATISTICS, COMMAND_PREFIX, TYPE_QUERY
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.commands import decode_client_command
from tankpit_bot.sim.lobby import SimLobby, build_auth_frame
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.transport import (
    encode_plaintext_payload,
    encode_tick_payload,
    split_client_frames,
)
from tankpit_bot.types import CapturedMessage, CaptureSession
from tankpit_bot.wire.helpers import EncodeError

_WS_OPEN = 1
_SIM_WS_URL = "wss://sim.tankpit.local/ws"
_SIM_PAGE_URL = "https://sim.tankpit.local/"
_SIM_TPCLIENT_URL = "https://sim.tankpit.local/tpclient.js"
_RAW_MESSAGE_WINDOW = 500
"""The page hook keeps the most recent 500 received payloads."""

# The AUTH frame's account fields. They are the page client's identity,
# not the bot's, and nothing downstream reads them — only the trailing
# magic matters ([[session-state-deglobalisation]]).
_SIM_ACCOUNT_ID = "62913"
_SIM_SESSION_TOKEN = "00000000000000000000000000000000"
_SIM_AUTH_STAMP = "0"
_AUTOSCROLL_KEY = "a"
_STATISTICS_KEY = "c"
_KEY_DOWN = "keyDown"


class SimKeyboard:
    """The page client's keyboard, as far as the sim models one.

    Only the ``a`` autoscroll toggle is wired. Pressing it is what the
    PAGE does — the browser sends ``A1``/``A0`` (the flipped state, not
    a bare "toggle"; the archive's 75 sends are all one or the other and
    the server echoes them back verbatim). Every other key raises,
    because a silently-ignored press would let a probe believe it acted
    ([[session-state-deglobalisation]]).
    """

    def __init__(self, link: SimCDPSession) -> None:
        """Bind the keyboard to the link whose socket it types on.

        Args:
            link: The sim link standing in for the page client.
        """
        self._link = link

    def press(self, key: str, *, delay: float | None = None) -> None:
        """Press one key.

        Args:
            key: The key name.
            delay: Hold time — irrelevant with no real input stack.

        Raises:
            EncodeError: For any key but the autoscroll toggle.
        """
        del delay
        if key != _AUTOSCROLL_KEY:
            raise EncodeError(f"sim keyboard: unmodeled key press {key!r}")
        self._link.autoscroll_enabled = not self._link.autoscroll_enabled
        flag = "1" if self._link.autoscroll_enabled else "0"
        self._link.send_page_frame(f"{_AUTOSCROLL_KEY.upper()}{flag}".encode())


class SimCDPSession:
    """``CDPSessionProtocol`` implementation backed by a :class:`SimServer`.

    Commands the bot sends are decoded and queued on the sim server;
    snapshot queries answer from the sim world. The session records
    every decoded command for assertions and diagnostics.
    """

    def __init__(self, server: SimServer, magic: str, lobby: SimLobby | None = None) -> None:
        """Bind the session to a sim server and its session magic.

        The magic is the parameter rather than the table because the
        table is DERIVED from it and the AUTH frame carries it — a link
        holding both could hold two that disagree.

        Args:
            server: The sim server this link speaks to.
            magic: The session magic. Its table is built here, and the
                bot must end up with the same one.
            lobby: The pre-play protocol half. When given, the link
                stands in for the page client as well as the socket:
                it opens by sending the AUTH frame and answers the
                production ``join_room`` flow from
                :mod:`tankpit_bot.sim.lobby`. When omitted the link
                starts already in-room, which is what the seam tests
                want ([[session-state-deglobalisation]]).
        """
        self.server = server
        self.magic = magic
        self.table = build_session_xor_table(magic)
        self.lobby = lobby
        self.sent_commands: list[str] = []
        self.wire_log: list[CapturedMessage] = []
        self.raw_messages: list[str] = []
        """Received payloads, the ``window.__rawMsgs`` hook's contents.

        Received only — the browser hook pushes from the socket's
        ``message`` handler, so sent frames never appear there."""
        self.map_visible = False
        self.autoscroll_enabled = False
        """The server-persisted autoscroll setting this connection sees.

        Starts OFF, which is the state the production enforcer's
        press-verify round trip proves and restores."""
        self.url = _SIM_PAGE_URL
        """The page URL, one field of the room-entry metadata."""
        self._last_send_ms: int | None = None
        self._detached = False

    @property
    def keyboard(self) -> SimKeyboard:
        """The page client's keyboard, for the autoscroll toggle dance."""
        return SimKeyboard(self)

    def send_page_frame(self, body: bytes) -> None:
        """Send one frame the PAGE CLIENT writes, not the bot.

        The autoscroll toggle, the quit frame and the key-driven
        statistics command are the page's, not the bot's — our code
        presses a key and reads the answer off the wire. They take the
        same route as an injected send so there is one place that
        decides command-vs-lobby.

        Args:
            body: The frame body, including its lead byte.
        """
        self.route_client_payload(encode_plaintext_payload([body]))

    def wait_for_timeout(self, timeout: float) -> None:
        """Satisfy the join flow's poll delay without sleeping.

        The production loop pumps the browser's event loop between
        polls. The sim answers every query synchronously from world
        truth, so there is nothing to wait FOR — a real sleep here
        would only add ten seconds of dead time to every soak.

        Args:
            timeout: The requested delay, in milliseconds.
        """
        del timeout

    def _deliver_plaintext(self, frames: list[bytes]) -> None:
        """Record lobby replies as received wire and page-hook traffic.

        ONE payload per frame. The browser hook pushes once per
        websocket ``message`` event and ``decode_captured_body`` treats
        a second frame in the same payload as corruption, so a batched
        room list would arrive as an error rather than two rooms — and
        the archive shows the real server sending each row separately
        anyway.

        Args:
            frames: The lobby's plaintext reply frames.
        """
        for body in frames:
            payload = encode_plaintext_payload([body])
            self.raw_messages.append(payload)
            self.wire_log.append(
                CapturedMessage(
                    timestamp_ms=get_current_time_ms(),
                    direction="received",
                    payload=payload,
                    ws_url=_SIM_WS_URL,
                )
            )

    def open_lobby(self) -> None:
        """Send the page client's AUTH frame and take the room list.

        The AUTH frame is the page client's, not the bot's — our code
        only reads it, to lift the session magic. This link IS the page
        client, so it sends one, and the room list arrives as the
        server's answer rather than as pre-seeded state.

        Raises:
            EncodeError: If the link was built without a lobby.
        """
        if self.lobby is None:
            raise EncodeError("sim session: open_lobby on a link with no lobby")
        self.send_page_frame(
            build_auth_frame(_SIM_ACCOUNT_ID, _SIM_SESSION_TOKEN, _SIM_AUTH_STAMP, self.magic)
        )

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
        self.route_client_payload(payload)
        byte_count = len(payload) * 3 // 4
        return {"result": {"value": f"SENT_{byte_count}_BYTES via {_SIM_WS_URL}"}}

    def route_client_payload(self, payload: str) -> None:
        """Route one outbound payload, whoever wrote it.

        Everything the client puts on this socket arrives here: the
        bot's injected sends, the page client's lobby frames, and the
        page client's key-driven commands. One socket carries two
        protocols, told apart by the lead byte — ``!`` is an in-game
        command, anything else is lobby — and the split has to precede
        that question, so frames are split once and routed per frame
        ([[session-state-deglobalisation]]).

        Args:
            payload: Base64 wire payload of framed client bytes.
        """
        lobby_replies: list[bytes] = []
        for body in split_client_frames(payload):
            if body[0] != COMMAND_PREFIX:
                lobby_replies.extend(self._handle_lobby_frame(body))
                continue
            command = decode_client_command(xor_decode_body(body, self.table, offset=1))
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
        self._deliver_plaintext(lobby_replies)

    def _handle_key_event(self, params: JSONObject | None) -> JSONObject:
        """Turn a dispatched key event into the frame the page would send.

        The account-stats capture presses ``c`` through CDP and reads
        the answer; in a browser the page's own script turns that press
        into the ``CMD_STATISTICS`` command frame. The sim IS that
        script, so the press really does put the command on the wire —
        which is what gives the 0x56 answer something to answer
        ([[session-state-deglobalisation]]).

        Only the down edge sends: the capture dispatches ``keyDown``
        then ``keyUp``, and acting on both would double every press.

        Args:
            params: The CDP event parameters.

        Returns:
            The CDP-shaped response.

        Raises:
            EncodeError: For a key the sim does not model. A silently
                swallowed press would let a probe believe it acted.
        """
        if params is None:
            raise EncodeError("sim session: Input.dispatchKeyEvent without params")
        if require_str(params, "type") != _KEY_DOWN:
            return {"result": {"value": None}}
        key = require_str(params, "key")
        if key != _STATISTICS_KEY:
            raise EncodeError(f"sim session: unmodeled dispatched key {key!r}")
        self.send_page_frame(
            bytes([COMMAND_PREFIX])
            + xor_decode_body(bytes([TYPE_QUERY, CMD_STATISTICS]), self.table)
        )
        return {"result": {"value": None}}

    def _handle_lobby_frame(self, body: bytes) -> list[bytes]:
        """Route one non-command client frame to the lobby.

        Args:
            body: The plaintext frame body.

        Returns:
            The lobby's reply frames.

        Raises:
            EncodeError: If the link has no lobby. A plaintext frame on
                an in-room link is a harness bug, not a condition to
                absorb — the seam would silently drop the bot's join.
        """
        if self.lobby is None:
            raise EncodeError(
                f"sim session: plaintext frame 0x{body[0]:02X} on a link with no lobby"
            )
        return self.lobby.handle_frame(body)

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
        if method == "Input.dispatchKeyEvent":
            return self._handle_key_event(params)
        if method != "Runtime.evaluate":
            return {"result": {"value": None}}
        if params is None:
            raise EncodeError("sim session: Runtime.evaluate without params")
        expression = require_str(params, "expression")
        atob_start = expression.find("atob('")
        if atob_start != -1:
            return self._handle_send_expression(expression, atob_start)
        return self._answer_query(expression)

    def _answer_query(self, expression: str) -> JSONObject:
        """Answer one read-only page query from sim truth.

        Args:
            expression: The evaluate expression.

        Returns:
            The CDP-shaped response.

        Raises:
            EncodeError: For an expression the sim does not model —
                loud, never best-effort.
        """
        if "__tankpitActiveGame" in expression:
            return {"result": {"value": encode_page_client_snapshot(self._snapshot())}}
        if "__sentFrameMetaQueue" in expression:
            return {"result": {"value": None}}
        if "__rawMsgs" in expression:
            # The page hook keeps the most recent 500 RECEIVED
            # payloads; the join flow reads it to find its room list,
            # join confirm and enter response.
            return {"result": {"value": list(self.raw_messages[-_RAW_MESSAGE_WINDOW:])}}
        if "tankpit.magic" in expression:
            return {"result": {"value": self.magic}}
        if "tpclient" in expression:
            return self._answer_tpclient_query(expression)
        if "document.body" in expression:
            # No DOM. The C-panel's account-lifetime totals are exactly
            # the numbers the wire never carries, and fabricating a
            # panel would invent an account history the sim has not
            # played ([[session-state-deglobalisation]]).
            return {"result": {"value": ""}}
        raise EncodeError(f"sim session: unmodeled evaluate expression: {expression[:80]!r}")

    def _answer_tpclient_query(self, expression: str) -> JSONObject:
        """Answer the two tpclient queries the room-entry step makes.

        The join flow asks for the loaded script's URL and then fetches
        its source to lift the static key. The sim has no browser, but
        it HAS the key — the same ``xor_static_key.txt`` the production
        cipher reads — so it answers with a source line carrying it, in
        the shape ``load_tpclient_static_key`` matches.

        Args:
            expression: The evaluate expression.

        Returns:
            The CDP-shaped response.
        """
        if expression.lstrip().startswith("fetch("):
            return {"result": {"value": f'var Ub="{require_static_key()}";'}}
        return {"result": {"value": _SIM_TPCLIENT_URL}}

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
    "SimKeyboard",
    "build_capture_session",
    "deliver_batch",
]
