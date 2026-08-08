"""Shared CDPService composition base for Bot and ProbeBase.

Owns CDPService + CommandService via composition, delegates message
storage, magic key, and WebSocket URL state to CDPService via
properties. Subclasses add domain-specific behavior (Bot adds state
machine and HFSM; ProbeBase adds action tracking).
"""

from __future__ import annotations

import uuid

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.bot.command_service import CommandService
from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.browser.cdp_utils import send_websocket_bytes
from tankpit_bot.capture.xor import build_session_xor_table
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.types import CapturedMessage

log = get_logger(__name__)


class SessionBase:
    """Shared composition base owning CDPService and CommandService.

    Provides property delegations, CDP setup, WebSocket send, and the
    common magic/message callbacks. Bot and ProbeBase inherit this to
    avoid duplicating the same 12 methods.
    """

    def __init__(
        self,
        target_url: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
        cdp_service: CDPService | None = None,
        command_service: CommandService | None = None,
    ) -> None:
        """Initialize session base with composed services.

        Args:
            target_url: URL to navigate to.
            headless: Whether to run browser in headless mode.
            prefer_account: Whether to prefer account login.
            cdp_service: Injected CDPService. Created internally if None.
            command_service: Injected CommandService. Created internally if None.
        """
        self._target_url = target_url
        self._headless = headless
        self._prefer_account = prefer_account
        self._session_id = str(uuid.uuid4())
        self._start_timestamp_ms = 0
        self._cdp_service = cdp_service if cdp_service is not None else CDPService()
        self._cdp_service.set_callbacks(
            on_message_captured=self._on_message_captured,
            on_magic_captured=self._on_magic_captured,
        )
        self._cdp: CDPSessionProtocol | None = None
        self._static_key: str | None = None
        #: This session's world state. Bound to the process singleton
        #: while step 8 is in flight -- the decoder still writes through
        #: ``get_world_service()``, so a session holding a DIFFERENT
        #: instance would read an empty world. The flip to
        #: ``WorldService()`` and the deletion of the singleton are the
        #: last two edits of the step, not the first
        #: ([[session-state-deglobalisation]]).
        self.world: WorldService = get_world_service()
        #: This session's XOR table, None until its magic is captured.
        #: Public because the decode path reads it through
        #: ``BufferedMessageSourceProtocol``.
        self.xor_table: bytes | None = None
        self._commands = (
            command_service
            if command_service is not None
            else CommandService(send_ws_bytes=self._send_websocket_bytes)
        )
        self._cdp_message_buffer: list[str] = []

    # -----------------------------------------------------------------
    # Properties delegating to CDPService
    # -----------------------------------------------------------------

    @property
    def _messages(self) -> list[CapturedMessage]:
        """Delegate message storage to CDPService."""
        return self._cdp_service.messages

    @_messages.setter
    def _messages(self, value: list[CapturedMessage]) -> None:
        self._cdp_service.messages = value

    @property
    def _ws_urls(self) -> dict[str, str]:
        """Delegate WebSocket URL storage to CDPService."""
        return self._cdp_service.ws_urls

    @_ws_urls.setter
    def _ws_urls(self, value: dict[str, str]) -> None:
        self._cdp_service.ws_urls = value

    @property
    def _magic(self) -> str | None:
        """Delegate magic key storage to CDPService."""
        return self._cdp_service.magic

    @_magic.setter
    def _magic(self, value: str | None) -> None:
        self._cdp_service.magic = value

    def captured_message_count(self) -> int:
        """Return how many WebSocket messages have been captured so far.

        Returns:
            Length of the session's captured-message list.
        """
        return len(self._cdp_service.messages)

    # -----------------------------------------------------------------
    # CDP setup (delegated to CDPService)
    # -----------------------------------------------------------------

    def _setup_cdp_handlers(self, cdp: CDPSessionProtocol) -> None:
        """Set up CDP event handlers for WebSocket capture.

        Args:
            cdp: CDP session.
        """
        self._cdp_service.setup_cdp_handlers(cdp)

    def _setup_console_listener(self, cdp: CDPSessionProtocol) -> None:
        """Set up console message listener.

        Args:
            cdp: CDP session.
        """
        self._cdp_service.setup_console_listener(cdp)

    # -----------------------------------------------------------------
    # Command dispatch
    # -----------------------------------------------------------------

    def _send_websocket_bytes(
        self,
        cdp: CDPSessionProtocol,
        data: bytes,
        label: str = "direct_send",
    ) -> str:
        """Send raw bytes via the captured WebSocket.

        Args:
            cdp: CDP session.
            data: Raw bytes to send.
            label: Bot-side label for outbound provenance logging.

        Returns:
            Status string from the browser-side send helper.
        """
        return send_websocket_bytes(cdp, data, label)

    def _send_bytes(self, data: bytes, cmd_name: str) -> bool:
        """XOR encode and send command bytes via WebSocket.

        Args:
            data: Framed command bytes (with 2-byte length header).
            cmd_name: Command name for logging.

        Returns:
            True if sent, False if CDP session not available.
        """
        self._commands.cdp = self._cdp
        return self._commands.send_bytes(data, cmd_name)

    # -----------------------------------------------------------------
    # Lifecycle hooks (called by CDPService)
    # -----------------------------------------------------------------

    def _on_message_captured(self, message: CapturedMessage) -> None:
        """Buffer received messages for sync.

        Args:
            message: The captured message.
        """
        if message["direction"] == "received":
            self._cdp_message_buffer.append(message["payload"])

    def _on_magic_captured(self, magic: str) -> None:
        """Build this session's XOR table.

        One table serves both directions: the command service encodes
        with it and the decode path decodes with it. It used to be built
        TWICE from the same inputs — once into a module global for
        decode, once onto the command service for encode — which is what
        made two sessions in one process impossible
        ([[session-state-deglobalisation]] step 1).

        It also used to arm twelve global capture trackers here. Eleven
        were never fed a message and the twelfth is sniffer-only, so the
        base session — which the BOT also runs — no longer arms any of
        them (step 9). Subclasses that own a tracker override this.

        Args:
            magic: The session magic string.

        Raises:
            XorStaticKeyUnavailableError: If the static key cannot be
                read. Decoding against a missing key yields garbage, so
                the session must not continue.
        """
        self.xor_table = build_session_xor_table(magic)
        self._commands.xor_table = self.xor_table
        log.info("Built session XOR table for command encoding and frame decoding")


__all__ = [
    "SessionBase",
]
