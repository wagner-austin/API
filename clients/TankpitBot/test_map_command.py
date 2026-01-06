"""Test sending map command directly via WebSocket."""

import base64
from pathlib import Path

from platform_core.logging import get_logger, setup_rich_logging

from tankpit_bot import _test_hooks
from tankpit_bot.browser import BrowserSession, get_current_time_ms

log = get_logger(__name__)


def load_static_key() -> str:
    """Load the static XOR key."""
    path = Path(__file__).parent / "xor_static_key.txt"
    return path.read_text().strip()


def build_xor_table(static_key: str, magic: str) -> bytes:
    """Build XOR table from static key and magic."""
    table = bytearray(len(static_key))
    for i in range(len(static_key)):
        table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
    return bytes(table)


def encode_command(cmd_type: int, cmd_id: int, xor_table: bytes) -> bytes:
    """Encode a command with XOR and length header."""
    # Raw command: ! + type + id
    raw = bytes([0x21, cmd_type, cmd_id])

    # XOR encode (skip the ! prefix)
    encoded = bytearray(len(raw))
    encoded[0] = raw[0]  # '!' stays as-is
    for i in range(1, len(raw)):
        encoded[i] = raw[i] ^ xor_table[i - 1]

    # Add 2-byte length header (little-endian)
    length = len(encoded)
    header = bytes([length & 0xFF, (length >> 8) & 0xFF])

    return header + bytes(encoded)


def encode_text_command(text: str) -> bytes:
    """Encode a plain text command with length header."""
    raw = text.encode("utf-8")
    length = len(raw)
    header = bytes([length & 0xFF, (length >> 8) & 0xFF])
    return header + raw


class MapCommandTester(BrowserSession):
    """Test sending map command directly."""

    def __init__(self, target_url: str) -> None:
        super().__init__(target_url, headless=False, prefer_account=True)
        self._static_key = load_static_key()

    def _install_websocket_hook(self, cdp: object) -> None:
        """Install a hook to capture WebSocket instances."""
        hook_js = """
        (() => {
            if (window.__wsInstances) return 'already installed';
            window.__wsInstances = [];
            const OWS = window.WebSocket;
            window.WebSocket = function(url, protocols) {
                const ws = protocols ? new OWS(url, protocols) : new OWS(url);
                window.__wsInstances.push(ws);
                console.log('[WS Hook] Captured WebSocket to:', url);
                return ws;
            };
            window.WebSocket.prototype = OWS.prototype;
            return 'installed';
        })()
        """
        result = cdp.send("Runtime.evaluate", {"expression": hook_js, "returnByValue": True})
        log.info("WebSocket hook: %s", result.get("result", {}).get("value", "?"))

    def _send_websocket_command(self, cdp: object, encoded_bytes: bytes) -> str:
        """Send command via WebSocket using JavaScript."""
        # Convert bytes to base64 for JavaScript
        b64 = base64.b64encode(encoded_bytes).decode()

        send_js = f"""
        (() => {{
            // Use captured WebSocket from prototype hook
            let ws = window.__capturedWS;

            // Fallback: try common locations
            if (!ws && typeof tankpit !== 'undefined' && tankpit.ws) {{
                ws = tankpit.ws;
            }}
            if (!ws && typeof window.ws !== 'undefined') {{
                ws = window.ws;
            }}

            if (!ws) {{
                let status = window.__capturedWS ? 'exists' : 'null';
                return 'NO_WEBSOCKET_FOUND (__capturedWS=' + status + ')';
            }}

            if (ws.readyState !== 1) {{
                return 'WEBSOCKET_NOT_OPEN: ' + ws.readyState;
            }}

            // Decode base64 to binary
            const b64 = '{b64}';
            const binary = atob(b64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {{
                bytes[i] = binary.charCodeAt(i);
            }}

            // Send as binary
            ws.send(bytes.buffer);
            return 'SENT_' + bytes.length + '_BYTES via ' + ws.url;
        }})()
        """
        result = cdp.send("Runtime.evaluate", {"expression": send_js, "returnByValue": True})
        result_obj = result.get("result")
        if isinstance(result_obj, dict):
            val = result_obj.get("value", "?")
            return str(val) if val is not None else "?"
        return "?"

    def run_test(self) -> None:
        """Run the map command test."""
        if _test_hooks.sync_playwright is None:
            raise RuntimeError("Playwright not installed")

        self._start_timestamp_ms = get_current_time_ms()
        self._messages = []
        self._ws_urls = {}

        with _test_hooks.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=self._headless)
            context = browser.new_context()
            page = context.new_page()
            cdp = context.new_cdp_session(page)

            # Enable Page domain for script injection to work
            cdp.send("Page.enable")
            cdp.send("Runtime.enable")

            # Listen for ALL console messages to debug
            def on_console(params: dict[str, object]) -> None:
                msg_type = params.get("type", "?")
                args = params.get("args", [])
                text = " ".join(str(a.get("value", a.get("description", "?"))) for a in args)
                if "WS" in text or "Hook" in text:
                    log.info("[Console %s] %s", msg_type, text)

            cdp.on("Runtime.consoleAPICalled", on_console)

            # Hook WebSocket.prototype.send BEFORE page loads to capture instances
            result = cdp.send(
                "Page.addScriptToEvaluateOnNewDocument",
                {
                    "source": """
                (function() {
                    window.__capturedWS = null;
                    const origSend = WebSocket.prototype.send;
                    WebSocket.prototype.send = function(data) {
                        if (!window.__capturedWS && this.readyState === 1) {
                            window.__capturedWS = this;
                            console.log('[WS Hook] Captured WebSocket via send:', this.url);
                        }
                        return origSend.call(this, data);
                    };
                    console.log('[WS Hook] hooked on', window.location.href);
                })();
                """
                },
            )
            log.info("WebSocket prototype hook installed, result: %s", result)

            self._setup_cdp_handlers(cdp)
            self._navigate_and_login(page, cdp, tank_name_prefix="M", auto_join_room=True)

            # Debug: Check what WebSocket URLs we've captured (keys are requestIds)
            log.info("Captured WS URLs: %s", list(self._ws_urls.values()))
            log.info("Captured %d messages so far", len(self._messages))

            # Debug: Check if WebSocket exists in JavaScript
            debug_js = """
            (() => {
                let found = [];
                // Check window properties
                for (let key in window) {
                    try {
                        if (window[key] instanceof WebSocket) {
                            found.push('window.' + key + ' (state=' + window[key].readyState + ')');
                        }
                    } catch(e) {}
                }
                // Check if there's a tankpit object
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
                return found.length > 0 ? found.join(', ') : 'NO WebSocket found';
            })()
            """
            result = cdp.send("Runtime.evaluate", {"expression": debug_js, "returnByValue": True})
            log.info("JS WebSocket check: %s", result.get("result", {}).get("value", "?"))

            self._wait_for_game_ready(page)

            log.info("Magic key: %s", self._magic)

            if not self._magic:
                log.error("No magic key captured!")
                return

            # Build XOR table
            xor_table = build_xor_table(self._static_key, self._magic)
            log.info("XOR table first 10 bytes: %s", xor_table[:10].hex())

            # Encode map command: type=2, id=108
            map_cmd = encode_command(2, 108, xor_table)
            log.info("Encoded map command: %s", map_cmd.hex())
            log.info("As base64: %s", base64.b64encode(map_cmd).decode())

            # === STEP 1: Open Map ===
            log.info("=== STEP 1: Opening map ===")
            msg_before = len(self._messages)
            result = self._send_websocket_command(cdp, map_cmd)
            log.info("Map OPEN: %s", result)
            page.wait_for_timeout(2000.0)

            new_msgs = self._messages[msg_before:]
            log.info("  -> %d new messages", len(new_msgs))
            for i, msg in enumerate(new_msgs):
                payload_len = len(msg["payload"]) if msg["payload"] else 0
                preview = msg["payload"][:20] if msg["payload"] else ""
                log.info("    Msg %d: len=%d, payload[:20]=%r", i, payload_len, preview)

            # === STEP 2: Close Map (JavaScript keypress - worked earlier) ===
            log.info("=== STEP 2: Closing map ===")
            close_js = """
            (() => {
                const targets = [document, window, document.body,
                                 document.querySelector('canvas')];
                for (let target of targets) {
                    if (!target) continue;
                    const event = new KeyboardEvent('keydown', {
                        key: 'f', code: 'KeyF', keyCode: 70, which: 70,
                        bubbles: true, cancelable: true
                    });
                    target.dispatchEvent(event);
                }
                return 'ok';
            })()
            """
            cdp.send("Runtime.evaluate", {"expression": close_js, "returnByValue": True})
            log.info("Map CLOSE: dispatched 'f' to all targets")
            page.wait_for_timeout(1500.0)

            # === STEP 3: Open Radar ===
            log.info("=== STEP 3: Opening radar ===")
            radar_cmd = encode_command(2, 102, xor_table)  # type=2, id=102 ('d' key)
            msg_before = len(self._messages)
            result = self._send_websocket_command(cdp, radar_cmd)
            log.info("Radar OPEN: %s", result)
            page.wait_for_timeout(5000.0)
            log.info("  -> %d new messages", len(self._messages) - msg_before)

            # === STEP 4: Send 'q' command ===
            log.info("=== STEP 4: Sending 'q' command ('-') ===")
            quit_cmd = encode_text_command("-")  # '-' is the quit command
            msg_before = len(self._messages)
            result = self._send_websocket_command(cdp, quit_cmd)
            log.info("Quit command: %s", result)
            page.wait_for_timeout(5000.0)
            log.info("  -> %d new messages", len(self._messages) - msg_before)

            # === STEP 5: Quit ===
            log.info("=== STEP 5: Quitting ===")

            self._cleanup(cdp, page, context, browser)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    setup_rich_logging(level="INFO")

    _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    tester = MapCommandTester("https://tankpit.com/play")
    tester.run_test()


if __name__ == "__main__":
    main()
