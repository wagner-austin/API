"""Browser session management for WebSocket capture.

Provides a base class that handles:
- Playwright browser launch and CDP setup
- WebSocket event handlers and message capture
- Magic key capture for XOR decoding
- Login flow integration
"""

from __future__ import annotations

import base64
import re
import time
import uuid
from pathlib import Path
from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    optional_str,
    require_dict,
    require_list,
    require_str,
)
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import (
    BrowserContextProtocol,
    BrowserProtocol,
    CDPSessionProtocol,
    PageProtocol,
)
from tankpit_bot.browser.dom_scraper import (
    GameLogEntry,
    GameLogScraper,
)
from tankpit_bot.browser.fuel_probe import FuelProber, FuelProbeResult
from tankpit_bot.browser.inject_script import BROWSER_HOOK_SOURCE
from tankpit_bot.browser.key_discovery import (
    extract_xor_first_bytes,
    find_best_static_byte,
    load_static_key,
    save_static_key,
)
from tankpit_bot.browser.types import (
    STATIC_KEY_PATH,
    GameNotJoinedError,
    PlaywrightNotInstalledError,
)
from tankpit_bot.combat import CombatEvent, CombatTracker
from tankpit_bot.inventory import InventoryChange, InventoryScraper
from tankpit_bot.protocol.codec import extract_magic_from_auth_payload
from tankpit_bot.types import (
    CapturedMessage,
    MessageDirection,
    decode_cdp_websocket_created_event,
    decode_cdp_websocket_frame_event,
)
from tankpit_bot.types.literals import SentFrameOrigin, require_sent_frame_origin

log = get_logger(__name__)

# Teardown bound: artifacts are saved before cleanup starts, so a
# teardown that outlives this is converted into a recorded forced exit
# instead of an eternal hang (runs 20260611-083908/092159 each sat 10+
# minutes inside sync Playwright teardown after saving).
_TEARDOWN_WATCHDOG_SECONDS = 30.0
# EX_TEMPFAIL-style distinct code so orchestration can tell a forced
# teardown exit apart from both clean exits and crashes.
_TEARDOWN_HANG_EXIT_CODE = 75


def _handle_teardown_hang() -> None:
    """Force the process to exit after a hung browser teardown."""
    log.error(
        "Teardown exceeded %.0fs; forcing process exit (artifacts were saved before cleanup)",
        _TEARDOWN_WATCHDOG_SECONDS,
    )
    _test_hooks.force_exit(_TEARDOWN_HANG_EXIT_CODE)


# Base64 validation pattern: A-Z, a-z, 0-9, +, /, and = for padding
_BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")
_BROWSER_HOOK_SOURCE = """
            (function() {
                window.__capturedWS = null;
                window.__allWS = [];
                window.__rawMsgs = [];
                window.__wsRecvCount = 0;
                window.__codexCurrentSendLabel = null;
                window.__sentFrameMetaQueue = [];
                window.__lastPageClientSendPerfMs = null;
                window.__lastBotInjectedSendPerfMs = null;
                window.__tankpitActiveGame = null;

                function maybeCaptureGameClient(candidate) {
                    if (!candidate || typeof candidate !== 'object') {
                        return;
                    }
                    const mapObject =
                        candidate.map && typeof candidate.map === 'object'
                            ? candidate.map
                            : null;
                    const worldObject =
                        candidate.h && typeof candidate.h === 'object'
                            ? candidate.h
                            : null;
                    const selfTank =
                        candidate.i && typeof candidate.i === 'object'
                            ? candidate.i
                            : null;
                    const transport =
                        candidate.va && typeof candidate.va === 'object'
                            ? candidate.va
                            : null;
                    const actionQueue =
                        worldObject &&
                        worldObject.j &&
                        typeof worldObject.j === 'object' &&
                        Array.isArray(worldObject.j.actions)
                            ? worldObject.j.actions
                            : null;
                    if (
                        mapObject !== null &&
                        worldObject !== null &&
                        selfTank !== null &&
                        transport !== null &&
                        actionQueue !== null &&
                        typeof candidate.s === 'number' &&
                        typeof candidate.Ha === 'boolean'
                    ) {
                        window.__tankpitActiveGame = candidate;
                    }
                }

                function installClientProbe(propertyName) {
                    const storageName = '__codexProbeValue_' + propertyName;
                    Object.defineProperty(Object.prototype, propertyName, {
                        configurable: true,
                        enumerable: false,
                        get: function() {
                            if (Object.prototype.hasOwnProperty.call(this, storageName)) {
                                return this[storageName];
                            }
                            return undefined;
                        },
                        set: function(value) {
                            Object.defineProperty(this, storageName, {
                                value: value,
                                writable: true,
                                configurable: true,
                                enumerable: false
                            });
                            Object.defineProperty(this, propertyName, {
                                value: value,
                                writable: true,
                                configurable: true,
                                enumerable: true
                            });
                            maybeCaptureGameClient(this);
                        }
                    });
                }

                installClientProbe('map');
                installClientProbe('h');
                installClientProbe('i');
                installClientProbe('va');
                installClientProbe('Ha');
                installClientProbe('s');

                // Hook EventTarget.prototype.addEventListener globally.
                // This catches ALL addEventListener calls, including those
                // made by the game on WebSocket instances.
                const origAEL = EventTarget.prototype.addEventListener;
                EventTarget.prototype.addEventListener = function(type, fn, opts) {
                    if (this instanceof WebSocket && type === 'message') {
                        if (window.__allWS.indexOf(this) === -1) {
                            window.__allWS.push(this);
                        }
                        const ws = this;
                        const origFn = fn;
                        fn = function(event) {
                            window.__wsRecvCount++;
                            if (ws.readyState === 1) window.__capturedWS = ws;
                            try {
                                if (event.data instanceof Blob) {
                                    const reader = new FileReader();
                                    reader.onload = function() {
                                        const bytes = new Uint8Array(reader.result);
                                        let b = '';
                                        for (let i = 0; i < bytes.length; i += 8192) {
                                            b += String.fromCharCode.apply(null,
                                                bytes.subarray(i, i + 8192));
                                        }
                                        window.__rawMsgs.push(btoa(b));
                                        if (window.__rawMsgs.length > 500) {
                                            window.__rawMsgs = window.__rawMsgs.slice(-200);
                                        }
                                    };
                                    reader.readAsArrayBuffer(event.data);
                                }
                            } catch(e) {}
                            return origFn.call(this, event);
                        };
                    }
                    return origAEL.call(this, type, fn, opts);
                };

                // Hook send for command injection
                const origSend = WebSocket.prototype.send;
                WebSocket.prototype.send = function(data) {
                    if (!window.__capturedWS || window.__capturedWS.readyState !== 1) {
                        if (this.readyState === 1) window.__capturedWS = this;
                    }
                    if (window.__allWS.indexOf(this) === -1) {
                        window.__allWS.push(this);
                    }
                    const currentLabel =
                        typeof window.__codexCurrentSendLabel === 'string'
                            ? window.__codexCurrentSendLabel
                            : null;
                    const perfNow = performance.now();
                    const err = new Error();
                    const stack = typeof err.stack === 'string' ? err.stack : '';
                    if (currentLabel) {
                        window.__lastBotInjectedSendPerfMs = perfNow;
                    } else {
                        window.__lastPageClientSendPerfMs = perfNow;
                    }
                    window.__sentFrameMetaQueue.push({
                        origin: currentLabel ? 'bot_injected' : 'page_client',
                        label: currentLabel || '',
                        stack: stack
                    });
                    if (window.__sentFrameMetaQueue.length > 500) {
                        window.__sentFrameMetaQueue = window.__sentFrameMetaQueue.slice(-200);
                    }
                    return origSend.call(this, data);
                };
            })();
            """


def _is_valid_base64(payload: str) -> bool:
    """Check if payload is valid base64.

    Args:
        payload: String to validate.

    Returns:
        True if valid base64, False otherwise.
    """
    if not payload:
        return False
    if not _BASE64_PATTERN.match(payload):
        return False
    return len(payload) % 4 == 0


def _extract_runtime_value(result: JSONObject) -> JSONValue:
    """Return the `Runtime.evaluate` value field.

    Args:
        result: Raw CDP result object.

    Returns:
        The evaluated JavaScript value.

    Raises:
        ValueError: If the CDP result is missing the value field.
    """
    result_obj = require_dict(result, "result")
    if "value" not in result_obj:
        raise ValueError(f"Runtime.evaluate result missing value: {result_obj}")
    return result_obj["value"]


def get_captured_raw_messages(cdp: CDPSessionProtocol) -> list[str]:
    """Return the captured raw WebSocket message buffer from the page hook.

    Args:
        cdp: Active CDP session.

    Returns:
        Captured raw message payloads as base64 strings.

    Raises:
        ValueError: If the page hook returned malformed data.
    """
    result = cdp.send(
        "Runtime.evaluate",
        {
            "expression": """
            (() => Array.isArray(window.__rawMsgs) ? window.__rawMsgs.slice(-500) : [])()
            """,
            "returnByValue": True,
        },
    )
    raw_value = _extract_runtime_value(result)
    payloads_raw = require_list({"items": raw_value}, "items")
    payloads: list[str] = []
    for payload in payloads_raw:
        payloads.append(require_str({"payload": payload}, "payload"))
    return payloads


def send_websocket_bytes(cdp: CDPSessionProtocol, data: bytes, label: str = "direct_send") -> str:
    """Send raw bytes via the captured WebSocket.

    Uses the WebSocket instance captured by the prototype hook installed in
    `_setup_cdp_handlers`, with fallbacks to the game globals used by older
    client builds.

    Args:
        cdp: Active CDP session.
        data: Raw framed bytes to send.
        label: Bot-side label for outbound provenance logging.

    Returns:
        Status string returned by the browser-side send helper.
    """
    b64 = base64.b64encode(data).decode()
    send_js = """
    (() => {
        let ws = window.__capturedWS;
        if (!ws && typeof tankpit !== 'undefined' && tankpit.ws) {
            ws = tankpit.ws;
        }
        if (!ws && typeof window.ws !== 'undefined') {
            ws = window.ws;
        }
        if (!ws) {
            const status = window.__capturedWS ? 'exists' : 'null';
            return 'NO_WEBSOCKET_FOUND (__capturedWS=' + status + ')';
        }
        if (ws.readyState !== 1) {
            return 'WEBSOCKET_NOT_OPEN: ' + ws.readyState;
        }
        const binary = atob('%B64%');
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        window.__codexCurrentSendLabel = %LABEL%;
        try {
            ws.send(bytes.buffer);
        } finally {
            window.__codexCurrentSendLabel = null;
        }
        return 'SENT_' + bytes.length + '_BYTES via ' + ws.url;
    })()
    """
    send_js = send_js.replace("%B64%", b64)
    send_js = send_js.replace("%LABEL%", repr(label))
    result = cdp.send("Runtime.evaluate", {"expression": send_js, "returnByValue": True})
    runtime_value = _extract_runtime_value(result)
    return require_str({"value": runtime_value}, "value")


class SentFrameMetadata(TypedDict):
    """Metadata captured at outbound WebSocket send time.

    Attributes:
        origin: Whether the send came from bot injection or the page client.
        label: Bot-side send label when known.
        stack: JavaScript stack recorded at send time.
    """

    origin: SentFrameOrigin
    label: str
    stack: str


def _pop_sent_frame_metadata(cdp: CDPSessionProtocol) -> SentFrameMetadata | None:
    """Pop the next outbound frame metadata record from the browser hook queue.

    Args:
        cdp: Active CDP session.

    Returns:
        The next queued outbound metadata record, or None when unavailable.

    Raises:
        ValueError: If the hook returned malformed metadata.
    """
    result = cdp.send(
        "Runtime.evaluate",
        {
            "expression": """
            (() => {
                if (!Array.isArray(window.__sentFrameMetaQueue)) {
                    return null;
                }
                if (window.__sentFrameMetaQueue.length === 0) {
                    return null;
                }
                return window.__sentFrameMetaQueue.shift();
            })()
            """,
            "returnByValue": True,
        },
    )
    raw_value = _extract_runtime_value(result)
    if raw_value is None or raw_value == "":
        return None
    metadata_obj = require_dict({"metadata": raw_value}, "metadata")
    return SentFrameMetadata(
        origin=require_sent_frame_origin(metadata_obj, "origin"),
        label=optional_str(metadata_obj, "label") or "",
        stack=optional_str(metadata_obj, "stack") or "",
    )


def get_current_time_ms() -> int:
    """Get current time in milliseconds.

    Returns:
        Current Unix timestamp in milliseconds.
    """
    return int(time.time() * 1000)


# Global offset for CDP timestamp conversion (set on first message)
_cdp_time_offset_ms: int | None = None


def cdp_timestamp_to_ms(timestamp: float) -> int:
    """Convert CDP monotonic timestamp to Unix milliseconds.

    CDP timestamps are monotonic seconds since browser start.
    We calculate an offset on first call to convert to Unix time.

    Args:
        timestamp: CDP monotonic timestamp in seconds.

    Returns:
        Unix timestamp in milliseconds.
    """
    global _cdp_time_offset_ms
    cdp_ms = int(timestamp * 1000)
    if _cdp_time_offset_ms is None:
        # First call - calculate offset from current time
        _cdp_time_offset_ms = get_current_time_ms() - cdp_ms
    return cdp_ms + _cdp_time_offset_ms


def reset_cdp_time_offset() -> None:
    """Reset CDP time offset for new browser session."""
    global _cdp_time_offset_ms
    _cdp_time_offset_ms = None


class BrowserSession:
    """Base class for browser-based WebSocket capture.

    Handles common functionality:
    - Browser/CDP setup
    - WebSocket event handlers
    - Message capture and storage
    - Magic key capture

    Subclasses implement specific behavior (passive sniffing vs active probing).
    """

    def __init__(
        self,
        target_url: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
    ) -> None:
        """Initialize browser session.

        Args:
            target_url: URL to navigate to.
            headless: Whether to run browser in headless mode.
            prefer_account: Skip guest login and use account credentials.
        """
        self._target_url = target_url
        self._headless = headless
        self._prefer_account = prefer_account
        self._session_id = str(uuid.uuid4())
        self._start_timestamp_ms = 0
        self._messages: list[CapturedMessage] = []
        self._ws_urls: dict[str, str] = {}  # requestId -> url mapping
        self._cdp: CDPSessionProtocol | None = None
        self._page: PageProtocol | None = None
        self._magic: str | None = None
        self._static_key: str | None = None
        self._game_log_scraper: GameLogScraper | None = None
        self._inventory_scraper: InventoryScraper | None = None
        self._combat_tracker: CombatTracker | None = None
        self._fuel_prober: FuelProber | None = None
        self._last_fuel_result: FuelProbeResult | None = None

    def captured_message_count(self) -> int:
        """Return how many WebSocket messages have been captured so far.

        Returns:
            Length of the session's captured-message list.
        """
        return len(self._messages)

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._session_id

    @property
    def messages(self) -> list[CapturedMessage]:
        """Get captured messages."""
        return self._messages

    @property
    def magic(self) -> str | None:
        """Get captured magic key for XOR decoding."""
        return self._magic

    @property
    def static_key(self) -> str | None:
        """Get captured static XOR key from game JS."""
        return self._static_key

    def _init_game_log_scraper(self, cdp: CDPSessionProtocol) -> None:
        """Initialize the game log scraper.

        Args:
            cdp: CDP session for DOM access.
        """
        self._game_log_scraper = GameLogScraper(cdp)
        log.info("Game log scraper initialized")

    def _poll_game_log(self) -> list[GameLogEntry]:
        """Poll for new game log entries, log them, and process combat.

        Returns:
            List of new entries found since last poll.
        """
        if self._game_log_scraper is None:
            return []
        new_entries = self._game_log_scraper.get_new_entries()
        for entry in new_entries:
            self._process_game_log_entry(entry)
        return new_entries

    def _process_game_log_entry(self, entry: GameLogEntry) -> None:
        """Process a single game log entry.

        Args:
            entry: The game log entry to process.
        """
        prefix = f"[GAME:{entry['category'].upper()}]"
        log.info("%s %s", prefix, entry["text"])
        # Process combat events
        if entry["category"] != "combat" or self._combat_tracker is None:
            return
        event = self._combat_tracker.process_log_line(entry["text"])
        if event is None:
            return
        self._combat_tracker.log_event(event)

    def _init_combat_tracker(self) -> None:
        """Initialize the combat tracker."""
        self._combat_tracker = CombatTracker()
        log.info("Combat tracker initialized")

    def _get_combat_events(self) -> list[CombatEvent]:
        """Get all recorded combat events.

        Returns:
            List of CombatEvents, or empty list if tracker not initialized.
        """
        if self._combat_tracker is None:
            return []
        return self._combat_tracker.get_events()

    def _init_inventory_scraper(self, cdp: CDPSessionProtocol) -> None:
        """Initialize the inventory scraper.

        Args:
            cdp: CDP session for DOM access.
        """
        self._inventory_scraper = InventoryScraper(cdp)
        log.info("Inventory scraper initialized")

    def _poll_inventory(self) -> list[InventoryChange]:
        """Poll for inventory changes and log them.

        Returns:
            List of changes found since last poll.
        """
        if self._inventory_scraper is None:
            return []
        return self._inventory_scraper.log_changes()

    def _init_fuel_prober(self, cdp: CDPSessionProtocol) -> None:
        """Initialize the fuel prober.

        Args:
            cdp: CDP session for JavaScript execution.
        """
        self._fuel_prober = FuelProber(cdp)
        log.info("Fuel prober initialized")

    def _poll_fuel(self) -> FuelProbeResult | None:
        """Poll for fuel values and log findings.

        Returns:
            FuelProbeResult if prober initialized, None otherwise.
        """
        if self._fuel_prober is None:
            return None

        result = self._fuel_prober.probe()

        # Log any interesting findings
        if result["js_variables"]:
            for var in result["js_variables"]:
                log.info("[FUEL:JS] %s = %s", var["path"], var["value"])

        if result["dom_bars"]:
            for bar in result["dom_bars"]:
                if bar["width"]:
                    log.info(
                        "[FUEL:DOM] %s width=%s class=%s",
                        bar["tag"],
                        bar["width"],
                        bar["class_name"][:20] if bar["class_name"] else "",
                    )

        self._last_fuel_result = result
        return result

    def _on_websocket_created(self, params: JSONObject) -> None:
        """Handle Network.webSocketCreated CDP event.

        Args:
            params: CDP event parameters.
        """
        event = decode_cdp_websocket_created_event(params)
        self._ws_urls[event["requestId"]] = event["url"]

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
        ws_url = self._ws_urls.get(request_id, "unknown")
        payload = event["response"]["payloadData"]

        message = CapturedMessage(
            timestamp_ms=cdp_timestamp_to_ms(event["timestamp"]),
            direction=direction,
            payload=payload,
            ws_url=ws_url,
        )
        if direction == "sent" and self._cdp is not None:
            metadata = _pop_sent_frame_metadata(self._cdp)
            if metadata is not None:
                message["sent_origin"] = metadata["origin"]
                if metadata["label"]:
                    message["sent_label"] = metadata["label"]
                if metadata["stack"]:
                    message["sent_stack"] = metadata["stack"]
        self._messages.append(message)
        self._on_message_captured(message)

    def _on_message_captured(self, message: CapturedMessage) -> None:
        """Extract magic key from AUTH messages and notify subclasses.

        Subclasses should call super()._on_message_captured(message) first,
        then perform their own message processing.

        Args:
            message: The captured message.
        """
        if message["direction"] == "sent" and self._magic is None:
            payload = message["payload"]
            if not _is_valid_base64(payload):
                return
            data = base64.b64decode(payload)
            magic = extract_magic_from_auth_payload(data)
            if magic is not None:
                self._magic = magic
                log.info("Captured magic key: %s...", magic[:20])
                self._on_magic_captured(magic)

    def _on_magic_captured(self, magic: str) -> None:
        """Called when magic key is first captured from AUTH message.

        Override in subclasses to perform setup that requires the magic key
        (e.g., initializing XOR tables, trackers).

        Args:
            magic: The session magic string.
        """

    def _setup_cdp_handlers(self, cdp: CDPSessionProtocol) -> None:
        """Set up CDP event handlers for WebSocket capture.

        Also installs a WebSocket prototype hook to capture the game's
        WebSocket instance for later command injection.

        Args:
            cdp: CDP session.
        """
        # Enable Page domain first - required for script injection to work
        cdp.send("Page.enable")

        # Install hooks BEFORE any page scripts load.
        # Hooks EventTarget.prototype.addEventListener to intercept ALL
        # WebSocket message handlers, capturing raw binary data as base64.
        # Also hooks WebSocket.prototype.send for command injection.
        cdp.send(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": BROWSER_HOOK_SOURCE},
        )

        # Enable Network domain for WebSocket frame capture
        cdp.send("Network.enable")
        cdp.on("Network.webSocketCreated", self._on_websocket_created)
        cdp.on("Network.webSocketFrameReceived", self._on_websocket_frame_received)
        cdp.on("Network.webSocketFrameSent", self._on_websocket_frame_sent)

    def _setup_console_listener(self, cdp: CDPSessionProtocol) -> None:
        """Set up console message listener for WebSocket debug info.

        Logs console messages containing 'WS', 'Hook', or 'WebSocket'.

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
                if "WS" in text or "Hook" in text or "WebSocket" in text:
                    log.info("[Console %s] %s", msg_type, text)

        cdp.on("Runtime.consoleAPICalled", on_console)

    def _log_websocket_urls(self) -> None:
        """Log all captured WebSocket URLs."""
        ws_urls = list(self._ws_urls.values())
        log.info("Captured WebSocket URLs: %s", ws_urls)

    def _debug_js_websocket(self, cdp: CDPSessionProtocol) -> None:
        """Check for WebSocket instances in JavaScript and log findings.

        Args:
            cdp: CDP session.
        """
        debug_js = """
        (() => {
            let found = [];
            for (let key in window) {
                try {
                    if (window[key] instanceof WebSocket) {
                        found.push('window.' + key + ' (state=' + window[key].readyState + ')');
                    }
                } catch(e) {}
            }
            if (typeof tankpit !== 'undefined') {
                for (let key in tankpit) {
                    try {
                        if (tankpit[key] instanceof WebSocket) {
                            let s = tankpit[key].readyState;
                            found.push('tankpit.' + key + ' (state=' + s + ')');
                        }
                    } catch(e) {}
                }
            }
            if (window.__capturedWS) {
                found.push('__capturedWS (state=' + window.__capturedWS.readyState + ')');
            }
            return found.length > 0 ? found.join(', ') : 'NO WebSocket found';
        })()
        """
        debug_result = cdp.send("Runtime.evaluate", {"expression": debug_js, "returnByValue": True})
        debug_obj = debug_result.get("result")
        if isinstance(debug_obj, dict):
            debug_val = debug_obj.get("value", "?")
            log.info("JS WebSocket check: %s", debug_val)

    def _log_script_urls(self, page: PageProtocol) -> None:
        """Log all loaded script URLs for protocol analysis.

        Args:
            page: Playwright page.
        """
        script_urls = page.evaluate(
            "Array.from(document.querySelectorAll('script[src]')).map(s => s.src)"
        )
        if script_urls and isinstance(script_urls, list):
            log.info("Loaded scripts (%d):", len(script_urls))
            for url in script_urls:
                if isinstance(url, str):
                    log.info("  - %s", url)

    def _gather_intel(self, page: PageProtocol, cdp: CDPSessionProtocol) -> None:
        """Gather and log all available intel after login.

        Logs WebSocket URLs, JS WebSocket instances, and script URLs.

        Args:
            page: Playwright page.
            cdp: CDP session.
        """
        self._log_websocket_urls()
        self._debug_js_websocket(cdp)
        self._log_script_urls(page)
        self._capture_static_key(page)

    def _capture_static_key(self, page: PageProtocol) -> None:
        """Extract static XOR key from tpclient JS source.

        Args:
            page: Playwright page.
        """
        import re

        # Wait for tpclient script to be loaded
        js_check_loaded = (
            "Array.from(document.querySelectorAll('script[src]'))"
            ".some(s => s.src.includes('tpclient'))"
        )
        page.wait_for_function(js_check_loaded, timeout=10000)

        # Get the URL
        js_get_url = (
            "Array.from(document.querySelectorAll('script[src]'))"
            ".find(s => s.src.includes('tpclient'))?.src"
        )
        tpclient_url = page.evaluate(js_get_url)
        if not isinstance(tpclient_url, str):
            log.warning("Could not find tpclient script URL")
            return

        # Fetch the JS content
        js_content = page.evaluate(f"fetch('{tpclient_url}').then(r => r.text())")
        if not isinstance(js_content, str):
            log.warning("Could not fetch tpclient JS content")
            return

        # Save full JS file for protocol analysis
        js_path = Path("tpclient.js")
        _test_hooks.write_text(js_path, js_content)
        log.info("Saved tpclient JS to %s (%d bytes)", js_path, len(js_content))

        # Extract 1000-char static key: any 1000-char quoted string
        match = re.search(r'"([^"]{1000})"', js_content)
        if not match:
            log.warning("Could not find static key in tpclient JS")
            return

        static_key: str = match.group(1)
        self._static_key = static_key
        save_static_key(static_key)
        log.info("Captured static key: %s...", static_key[:20])

    def _derive_static_key_from_messages(self) -> None:
        """Derive static key by analyzing captured messages.

        Uses brute-force to find which static[0] value makes captured messages
        decode to known protocol signatures. This is robust against game updates.

        Raises:
            FileNotFoundError: If static key file does not exist.
            ValueError: If static key file has invalid content.
        """
        if not self._magic or not self._messages:
            return

        raw_first_bytes = extract_xor_first_bytes(self._messages)
        if not raw_first_bytes:
            log.warning("No binary messages to derive static key from")
            return

        magic_0 = ord(self._magic[0])
        finder = _test_hooks.find_best_static_byte or find_best_static_byte
        best_static_0, best_coverage = finder(raw_first_bytes, magic_0)

        if best_coverage == 0:
            log.warning("Could not derive static key - no known signatures matched")
            return

        pct = 100 * best_coverage / len(raw_first_bytes)
        log.info(
            "Derived static[0]=0x%02x (%r) with %.1f%% coverage (%d/%d)",
            best_static_0,
            chr(best_static_0),
            pct,
            best_coverage,
            len(raw_first_bytes),
        )

        current_key = load_static_key()
        new_key = chr(best_static_0) + current_key[1:]

        if new_key != current_key:
            self._static_key = new_key
            save_static_key(new_key)
            log.info("Updated static key file: %s", STATIC_KEY_PATH)

    def _send_websocket_bytes(
        self,
        cdp: CDPSessionProtocol,
        data: bytes,
        label: str = "direct_send",
    ) -> str:
        """Send raw bytes via the captured WebSocket.

        Uses the WebSocket instance captured by the prototype hook installed
        in _setup_cdp_handlers, with fallbacks to tankpit.ws and window.ws.

        Args:
            cdp: CDP session.
            data: Raw bytes to send.
            label: Bot-side label for outbound provenance logging.

        Returns:
            Status string: 'SENT_N_BYTES via URL' on success, error message otherwise.
        """
        return send_websocket_bytes(cdp, data, label)

    def _wait_for_game_ready(self, page: PageProtocol) -> None:
        """Wait for game to fully load (message flow stabilizes).

        Args:
            page: Playwright page.

        Raises:
            GameNotJoinedError: If no messages captured.
        """
        log.info("Waiting for game to initialize...")
        page.wait_for_timeout(2000.0)  # Initial wait

        # Wait until no new messages for 1 second
        last_count = len(self._messages)
        stable_checks = 0
        while stable_checks < 3:  # Need 3 consecutive stable checks
            page.wait_for_timeout(500.0)
            current_count = len(self._messages)
            if current_count == last_count:
                stable_checks += 1
            else:
                stable_checks = 0
                last_count = current_count

        if len(self._messages) == 0:
            raise GameNotJoinedError("No WebSocket messages captured - game may not have loaded")

        log.info("Game ready, captured %d initial messages", len(self._messages))

    def _launch_browser(
        self,
    ) -> tuple[BrowserProtocol, BrowserContextProtocol, PageProtocol, CDPSessionProtocol]:
        """Launch browser and set up CDP session.

        Returns:
            Tuple of (browser, context, page, cdp).

        Raises:
            PlaywrightNotInstalledError: If Playwright not installed.
        """
        if _test_hooks.sync_playwright is None:
            raise PlaywrightNotInstalledError(
                "Playwright is not installed. Call _test_hooks._install_real_playwright() first."
            )

        # This returns a context manager, we need to handle it properly
        # The caller should use this within a with block or manage cleanup
        playwright = _test_hooks.sync_playwright()
        pw = playwright.__enter__()
        browser = pw.chromium.launch(headless=self._headless)
        context = browser.new_context()
        page = context.new_page()
        cdp = context.new_cdp_session(page)

        # Reset CDP time offset for new session
        reset_cdp_time_offset()
        self._setup_cdp_handlers(cdp)

        return browser, context, page, cdp

    def _navigate_and_login(
        self,
        page: PageProtocol,
        cdp: CDPSessionProtocol,
        *,
        tank_name_prefix: str = "T",
        auto_join_room: bool = False,
    ) -> None:
        """Navigate to target URL and handle login.

        Args:
            page: Playwright page.
            cdp: CDP session.
            tank_name_prefix: Prefix for tank name.
            auto_join_room: Whether to automatically join a room.
        """
        from tankpit_bot.browser.login import handle_login_flow

        page.goto(self._target_url, wait_until="domcontentloaded")
        log.info("Navigated to %s", page.url)

        success = handle_login_flow(
            page,
            cdp,
            tank_name_prefix=tank_name_prefix,
            prefer_account=self._prefer_account,
            auto_join_room=auto_join_room,
        )
        if not success:
            raise GameNotJoinedError("login or room join did not complete successfully")

    def _cleanup(
        self,
        cdp: CDPSessionProtocol,
        page: PageProtocol,
        context: BrowserContextProtocol,
        browser: BrowserProtocol,
    ) -> None:
        """Close the browser, bounded by a teardown watchdog.

        ``browser.close()`` tears down every context, page, and
        attached CDP session in one protocol call. The previous
        four-step sequence (detach, page close, context close, browser
        close) gave sync Playwright four separate chances to deadlock,
        and runs 20260611-083908 and 20260611-092159 each sat 10+
        minutes hung after saving their captures. The watchdog converts
        any remaining hang -- including ``playwright.stop()`` after
        this method returns -- into a recorded forced exit: every
        artifact is saved before cleanup starts, so a bounded exit
        with a logged cause strictly beats an unrecorded eternal hang.

        Args:
            cdp: CDP session (closed implicitly by the browser close;
                kept for the shared cleanup protocol signature).
            page: Playwright page (closed implicitly; see ``cdp``).
            context: Browser context (closed implicitly; see ``cdp``).
            browser: Browser instance.
        """
        del cdp, page, context
        _test_hooks.start_watchdog(_TEARDOWN_WATCHDOG_SECONDS, _handle_teardown_hang)
        log.info("Teardown: closing browser")
        try:
            browser.close()
        except (OSError, RuntimeError) as exc:
            log.debug("Browser close failed (already closed): %s", exc)
        log.info("Teardown: browser closed")


__all__ = [
    "BrowserSession",
    "cdp_timestamp_to_ms",
    "get_current_time_ms",
    "reset_cdp_time_offset",
]
