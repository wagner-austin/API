"""Browser session management for WebSocket capture.

Provides a base class that handles:
- Playwright browser launch and CDP setup
- WebSocket event handlers and message capture
- Magic key capture for XOR decoding
- Login flow integration
"""

from __future__ import annotations

import re

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import (
    CDPSessionProtocol,
    PageProtocol,
)
from tankpit_bot.browser.cdp_utils import (
    get_current_time_ms,
    reset_cdp_time_offset,
)
from tankpit_bot.browser.dom_scraper import (
    GameLogEntry,
    GameLogScraper,
)
from tankpit_bot.browser.fuel_probe import FuelProber, FuelProbeResult
from tankpit_bot.browser.session_base import SessionBase
from tankpit_bot.combat import CombatEvent
from tankpit_bot.combat_tracker import CombatTracker
from tankpit_bot.inventory import InventoryChange, InventoryScraper
from tankpit_bot.types import (
    CapturedMessage,
)

log = get_logger(__name__)

# Teardown bound: artifacts are saved before cleanup starts, so a
# teardown that outlives this is converted into a recorded forced exit
# instead of an eternal hang (runs 20260611-083908/092159 each sat 10+
# minutes inside sync Playwright teardown after saving).
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


class BrowserSession(SessionBase):
    """Base class for browser-based WebSocket capture.

    Inherits CDPService composition from SessionBase. Adds sniffer-specific
    scrapers (game log, combat, inventory, fuel), browser lifecycle methods,
    and intel gathering.
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
        super().__init__(target_url, headless=headless, prefer_account=prefer_account)
        self._page: PageProtocol | None = None
        self._game_log_scraper: GameLogScraper | None = None
        self._inventory_scraper: InventoryScraper | None = None
        self._combat_tracker: CombatTracker | None = None
        self._fuel_prober: FuelProber | None = None
        self._last_fuel_result: FuelProbeResult | None = None

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


__all__ = [
    "BrowserSession",
    "get_current_time_ms",
    "reset_cdp_time_offset",
]
