"""Fuel/HP value probing via DOM and JavaScript inspection.

Provides utilities to find fuel values in the browser by:
1. Searching DOM elements for progress bars
2. Searching JavaScript globals for game state
3. Inspecting canvas elements

This complements WebSocket decoding by finding values directly in the UI.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol

log = get_logger(__name__)


# =============================================================================
# TypedDicts
# =============================================================================


class DOMBarElement(TypedDict):
    """A potential fuel/HP bar found in the DOM."""

    tag: str
    id: str
    class_name: str
    width: str
    computed_width: str
    parent_class: str


class JSVariable(TypedDict):
    """A JavaScript variable that might contain fuel."""

    name: str
    value: float | int | str | None
    path: str


class FuelProbeResult(TypedDict):
    """Result of probing for fuel values."""

    dom_bars: list[DOMBarElement]
    js_variables: list[JSVariable]
    numeric_globals: list[tuple[str, float]]


# =============================================================================
# JavaScript for probing
# =============================================================================

# Find potential progress bar elements
FIND_BARS_JS = """
(() => {
    const bars = [];

    // Find all divs with percentage or pixel widths
    document.querySelectorAll('div, span').forEach(el => {
        const width = el.style.width;
        if (width && (width.includes('%') || width.includes('px'))) {
            const computed = getComputedStyle(el);
            const bg = computed.backgroundColor;

            // Only include if has a background color (likely a bar)
            if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
                bars.push({
                    tag: el.tagName,
                    id: el.id || '',
                    class_name: el.className || '',
                    width: width,
                    computed_width: computed.width,
                    parent_class: el.parentElement?.className || ''
                });
            }
        }
    });

    return bars;
})()
"""

# Find game-related JavaScript variables
FIND_GAME_VARS_JS = """
(() => {
    const vars = [];

    // Common patterns for game state objects
    const patterns = [
        'game', 'Game', 'GAME', 'gameState', 'GameState',
        'player', 'Player', 'PLAYER', 'playerState',
        'tank', 'Tank', 'TANK', 'tankState',
        'state', 'State', 'STATE',
        'fuel', 'Fuel', 'FUEL',
        'hp', 'HP', 'health', 'Health', 'HEALTH'
    ];

    for (const pattern of patterns) {
        try {
            if (window[pattern] !== undefined) {
                const obj = window[pattern];
                const type = typeof obj;

                if (type === 'number') {
                    vars.push({
                        name: pattern,
                        value: obj,
                        path: pattern
                    });
                } else if (type === 'object' && obj !== null) {
                    // Search for fuel/hp properties in the object
                    const keywords = ['fuel', 'hp', 'health', 'energy', 'power'];
                    for (const key of Object.keys(obj)) {
                        if (keywords.some(k => key.toLowerCase().includes(k))) {
                            const val = obj[key];
                            if (typeof val === 'number') {
                                vars.push({
                                    name: key,
                                    value: val,
                                    path: `${pattern}.${key}`
                                });
                            }
                        }
                    }
                }
            }
        } catch (e) {}
    }

    return vars;
})()
"""

# Get all numeric window properties in reasonable fuel range
FIND_NUMERIC_GLOBALS_JS = """
(() => {
    const nums = [];
    for (const key in window) {
        try {
            const val = window[key];
            if (typeof val === 'number' && !isNaN(val) && isFinite(val)) {
                // Only include values in possible fuel range (0-2000)
                if (val >= 0 && val <= 2000) {
                    nums.push([key, val]);
                }
            }
        } catch (e) {}
    }
    return nums.sort((a, b) => b[1] - a[1]);
})()
"""

# Try to read specific common game variable paths
READ_COMMON_PATHS_JS = """
(() => {
    const paths = [
        'game.player.fuel',
        'game.player.hp',
        'game.fuel',
        'player.fuel',
        'player.hp',
        'gameState.fuel',
        'gameState.player.fuel',
        'Tank.fuel',
        'tank.fuel',
        'state.fuel',
        'window.fuel',
        'window.hp'
    ];

    const results = [];

    for (const path of paths) {
        try {
            const parts = path.split('.');
            let obj = window;
            for (const part of parts) {
                if (part === 'window') continue;
                obj = obj[part];
                if (obj === undefined) break;
            }
            if (typeof obj === 'number') {
                results.push([path, obj]);
            }
        } catch (e) {}
    }

    return results;
})()
"""


# =============================================================================
# Probe Functions
# =============================================================================


def probe_dom_bars(cdp: CDPSessionProtocol) -> list[DOMBarElement]:
    """Find potential progress bar elements in the DOM.

    Args:
        cdp: CDP session for executing JavaScript.

    Returns:
        List of DOM elements that look like progress bars.
    """
    result: JSONObject = cdp.send(
        "Runtime.evaluate",
        {"expression": FIND_BARS_JS, "returnByValue": True},
    )
    result_obj = result.get("result")
    if isinstance(result_obj, dict):
        value = result_obj.get("value")
        if isinstance(value, list):
            bars: list[DOMBarElement] = []
            for item in value:
                if isinstance(item, dict):
                    bars.append(
                        DOMBarElement(
                            tag=str(item.get("tag", "")),
                            id=str(item.get("id", "")),
                            class_name=str(item.get("class_name", "")),
                            width=str(item.get("width", "")),
                            computed_width=str(item.get("computed_width", "")),
                            parent_class=str(item.get("parent_class", "")),
                        )
                    )
            return bars
    return []


def probe_game_variables(cdp: CDPSessionProtocol) -> list[JSVariable]:
    """Find game-related JavaScript variables.

    Args:
        cdp: CDP session for executing JavaScript.

    Returns:
        List of variables that might contain fuel values.
    """
    result: JSONObject = cdp.send(
        "Runtime.evaluate",
        {"expression": FIND_GAME_VARS_JS, "returnByValue": True},
    )
    result_obj = result.get("result")
    if isinstance(result_obj, dict):
        value = result_obj.get("value")
        if isinstance(value, list):
            variables: list[JSVariable] = []
            for item in value:
                if isinstance(item, dict):
                    val = item.get("value")
                    variables.append(
                        JSVariable(
                            name=str(item.get("name", "")),
                            value=val if isinstance(val, (int, float, str)) else None,
                            path=str(item.get("path", "")),
                        )
                    )
            return variables
    return []


def probe_numeric_globals(cdp: CDPSessionProtocol) -> list[tuple[str, float]]:
    """Get all numeric window properties in fuel range.

    Args:
        cdp: CDP session for executing JavaScript.

    Returns:
        List of (name, value) tuples for numeric globals.
    """
    result: JSONObject = cdp.send(
        "Runtime.evaluate",
        {"expression": FIND_NUMERIC_GLOBALS_JS, "returnByValue": True},
    )
    result_obj = result.get("result")
    if isinstance(result_obj, dict):
        value = result_obj.get("value")
        if isinstance(value, list):
            nums: list[tuple[str, float]] = []
            for item in value:
                if isinstance(item, list) and len(item) == 2:
                    name = str(item[0])
                    val = item[1]
                    if isinstance(val, (int, float)):
                        nums.append((name, float(val)))
            return nums
    return []


def probe_common_paths(cdp: CDPSessionProtocol) -> list[tuple[str, float]]:
    """Try to read common game variable paths.

    Args:
        cdp: CDP session for executing JavaScript.

    Returns:
        List of (path, value) tuples for found variables.
    """
    result: JSONObject = cdp.send(
        "Runtime.evaluate",
        {"expression": READ_COMMON_PATHS_JS, "returnByValue": True},
    )
    result_obj = result.get("result")
    if isinstance(result_obj, dict):
        value = result_obj.get("value")
        if isinstance(value, list):
            paths: list[tuple[str, float]] = []
            for item in value:
                if isinstance(item, list) and len(item) == 2:
                    path = str(item[0])
                    val = item[1]
                    if isinstance(val, (int, float)):
                        paths.append((path, float(val)))
            return paths
    return []


def probe_all(cdp: CDPSessionProtocol) -> FuelProbeResult:
    """Run all fuel probes and return combined results.

    Args:
        cdp: CDP session for executing JavaScript.

    Returns:
        Combined probe results.
    """
    return FuelProbeResult(
        dom_bars=probe_dom_bars(cdp),
        js_variables=probe_game_variables(cdp),
        numeric_globals=probe_numeric_globals(cdp),
    )


# =============================================================================
# FuelProber Class
# =============================================================================


class FuelProber:
    """Tracks fuel values over time by probing the browser.

    Use this to find and monitor fuel values when WebSocket
    decoding is uncertain.
    """

    def __init__(self, cdp: CDPSessionProtocol) -> None:
        """Initialize the prober.

        Args:
            cdp: CDP session for JavaScript execution.
        """
        self._cdp = cdp
        self._last_result: FuelProbeResult | None = None

    def probe(self) -> FuelProbeResult:
        """Probe for fuel values.

        Returns:
            Current probe results.
        """
        result = probe_all(self._cdp)
        self._last_result = result
        return result

    def log_results(self) -> None:
        """Log current probe results."""
        result = self.probe()

        if result["dom_bars"]:
            log.info("Found %d potential progress bars:", len(result["dom_bars"]))
            for bar in result["dom_bars"]:
                log.info(
                    "  %s width=%s computed=%s class=%s",
                    bar["tag"],
                    bar["width"],
                    bar["computed_width"],
                    bar["class_name"][:30] if bar["class_name"] else "",
                )

        if result["js_variables"]:
            log.info("Found %d game variables:", len(result["js_variables"]))
            for var in result["js_variables"]:
                log.info("  %s = %s", var["path"], var["value"])

        if result["numeric_globals"]:
            log.info("Found %d numeric globals (0-2000):", len(result["numeric_globals"]))
            for name, value in result["numeric_globals"][:10]:
                log.info("  %s = %s", name, value)


__all__ = [
    "DOMBarElement",
    "FuelProbeResult",
    "FuelProber",
    "JSVariable",
    "probe_all",
    "probe_common_paths",
    "probe_dom_bars",
    "probe_game_variables",
    "probe_numeric_globals",
]
