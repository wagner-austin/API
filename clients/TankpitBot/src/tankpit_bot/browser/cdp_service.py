"""CDP session service — WebSocket event handling and message capture.

Encapsulates CDP event wiring, WebSocket frame recording, magic key
extraction, and console listener setup. BrowserSession delegates to
this service for all CDP-level concerns.
"""

from __future__ import annotations

import base64
from collections.abc import Callable

from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.browser.cdp_utils import (
    CDPClock,
    _is_valid_base64,
    _pop_sent_frame_metadata,
    send_websocket_bytes,
)
from tankpit_bot.browser.inject_script import BROWSER_HOOK_SOURCE
from tankpit_bot.protocol.codec import extract_magic_from_auth_payload
from tankpit_bot.types import (
    CapturedMessage,
    MessageDirection,
    decode_cdp_websocket_created_event,
    decode_cdp_websocket_frame_event,
)

log = get_logger(__name__)


class CDPService:
    """Owns CDP event handling, WebSocket frame capture, and magic key extraction.

    Stores captured messages and the magic key. BrowserSession exposes
    these via property delegation.
    """

    def __init__(self) -> None:
        """Initialize empty CDP service state."""
        self.messages: list[CapturedMessage] = []
        self.ws_urls: dict[str, str] = {}
        self.magic: str | None = None
        self.cdp: CDPSessionProtocol | None = None
        self._on_message_captured: Callable[[CapturedMessage], None] | None = None
        self._on_magic_captured: Callable[[str], None] | None = None
        # The CDP clock's origin is this session's; anchoring it here
        # means a second session cannot read its frames through the
        # first session's offset ([[session-state-deglobalisation]]).
        self._clock = CDPClock()

    def set_callbacks(
        self,
        *,
        on_message_captured: Callable[[CapturedMessage], None],
        on_magic_captured: Callable[[str], None],
    ) -> None:
        """Wire callbacks after construction.

        Args:
            on_message_captured: Called for each captured WebSocket message.
            on_magic_captured: Called when magic key is first extracted.
        """
        self._on_message_captured = on_message_captured
        self._on_magic_captured = on_magic_captured

    def setup_cdp_handlers(self, cdp: CDPSessionProtocol) -> None:
        """Set up CDP event handlers for WebSocket capture.

        Args:
            cdp: CDP session to wire events on.
        """
        self.cdp = cdp
        cdp.send("Page.enable")
        cdp.send(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": BROWSER_HOOK_SOURCE},
        )
        cdp.send("Network.enable")
        cdp.on("Network.webSocketCreated", self._on_websocket_created)
        cdp.on("Network.webSocketFrameReceived", self._on_websocket_frame_received)
        cdp.on("Network.webSocketFrameSent", self._on_websocket_frame_sent)

    def setup_console_listener(self, cdp: CDPSessionProtocol) -> None:
        """Set up console message listener for WebSocket debug info.

        Args:
            cdp: CDP session.
        """
        cdp.send("Runtime.enable")

        def on_console(params: JSONObject) -> None:
            msg_type = params.get("type", "?")
            args = params.get("args", [])
            if isinstance(args, list):
                texts = []
                for a in args:
                    if isinstance(a, dict):
                        val = a.get("value", a.get("description", "?"))
                        texts.append(str(val) if val is not None else "?")
                text = " ".join(texts)
                # Errors and warnings are kept whatever they say. The
                # substring filter below is a NOISE control for the
                # game's chatty info logging, and it used to apply to
                # every level — so a page-side "TypeError" reached this
                # handler and was dropped for not containing "WS". The
                # bot drives a page it could not see the failures of.
                if msg_type in ("error", "warning"):
                    log.warning("[Console %s] %s", msg_type, text)
                elif "WS" in text or "Hook" in text or "WebSocket" in text:
                    log.info("[Console %s] %s", msg_type, text)

        def on_exception(params: JSONObject) -> None:
            """Report an uncaught page exception.

            Nothing subscribed to this before, so an exception in the
            game's own render loop was invisible: rAF does not
            reschedule after a throw, so one uncaught error freezes the
            canvas permanently while the WebSocket keeps updating game
            state. The bot plays on, the picture does not — which is
            exactly the symptom this listener exists to name.
            """
            details = params.get("exceptionDetails", {})
            if isinstance(details, dict):
                log.warning("[Page exception] %s", details.get("text", "?"))

        cdp.on("Runtime.consoleAPICalled", on_console)
        cdp.on("Runtime.exceptionThrown", on_exception)

    def send_bytes(self, cdp: CDPSessionProtocol, data: bytes, label: str) -> str:
        """Send raw bytes via the captured WebSocket.

        Args:
            cdp: CDP session.
            data: Raw bytes to send.
            label: Bot-side label for outbound provenance logging.

        Returns:
            Status string from the browser-side send helper.
        """
        return send_websocket_bytes(cdp, data, label)

    def log_websocket_urls(self) -> None:
        """Log all captured WebSocket URLs."""
        ws_urls = list(self.ws_urls.values())
        log.info("Captured WebSocket URLs: %s", ws_urls)

    def _on_websocket_created(self, params: JSONObject) -> None:
        """Handle Network.webSocketCreated CDP event.

        Args:
            params: CDP event parameters.
        """
        event = decode_cdp_websocket_created_event(params)
        self.ws_urls[event["requestId"]] = event["url"]

    def _on_websocket_frame_received(self, params: JSONObject) -> None:
        """Handle Network.webSocketFrameReceived CDP event.

        Args:
            params: CDP event parameters.
        """
        self._record_frame(params, "received")

    def _on_websocket_frame_sent(self, params: JSONObject) -> None:
        """Handle Network.webSocketFrameSent CDP event.

        Args:
            params: CDP event parameters.
        """
        self._record_frame(params, "sent")

    def _record_frame(self, params: JSONObject, direction: MessageDirection) -> None:
        """Record a WebSocket frame.

        Args:
            params: CDP event parameters.
            direction: Whether the frame was sent or received.
        """
        event = decode_cdp_websocket_frame_event(params)
        request_id = event["requestId"]
        ws_url = self.ws_urls.get(request_id, "unknown")
        payload = event["response"]["payloadData"]

        message = CapturedMessage(
            timestamp_ms=self._clock.to_unix_ms(event["timestamp"]),
            direction=direction,
            payload=payload,
            ws_url=ws_url,
        )
        if direction == "sent" and self.cdp is not None:
            metadata = _pop_sent_frame_metadata(self.cdp)
            if metadata is not None:
                message["sent_origin"] = metadata["origin"]
                if metadata["label"]:
                    message["sent_label"] = metadata["label"]
                if metadata["stack"]:
                    message["sent_stack"] = metadata["stack"]
        self.messages.append(message)
        self._extract_magic_and_notify(message)

    def _extract_magic_and_notify(self, message: CapturedMessage) -> None:
        """Extract magic key from AUTH messages and invoke callbacks.

        Args:
            message: The captured message.
        """
        if self._on_message_captured is not None:
            self._on_message_captured(message)

        if message["direction"] == "sent" and self.magic is None:
            payload = message["payload"]
            if not _is_valid_base64(payload):
                return
            data = base64.b64decode(payload)
            magic = extract_magic_from_auth_payload(data)
            if magic is not None:
                self.magic = magic
                log.info("Captured magic key: %s...", magic[:20])
                if self._on_magic_captured is not None:
                    self._on_magic_captured(magic)


__all__ = [
    "CDPService",
]
