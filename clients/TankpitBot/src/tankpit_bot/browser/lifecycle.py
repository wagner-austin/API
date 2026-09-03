"""Browser lifecycle functions — standalone navigation, login, and cleanup.

These functions operate on page/CDP session handles directly, without
requiring a BrowserSession instance. Used by probe_runtime to bootstrap
probes via composition instead of inheritance.
"""

from __future__ import annotations

import sys
import threading
import traceback

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import (
    BrowserProtocol,
    CDPSessionProtocol,
    PageProtocol,
)
from tankpit_bot.browser.types import GameNotJoinedError
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.types import CapturedMessage

log = get_logger(__name__)

_TEARDOWN_REMEDY_SECONDS = 15.0
"""Grace before the ladder's second rung kills the browser engine.

The close is normally sub-second, but this host's filesystem
minifilter has been measured inspecting browser teardown for tens of
seconds (``tests/browser/test_lifecycle.py`` fixture note), so a
short fuse would kill closes that were merely slow. Fifteen seconds
is past every observed CLEAN close and well before the terminal
watchdog, leaving the driver a full window to notice the engine's
death and resolve the pending ``browser.close()``."""

_TEARDOWN_WATCHDOG_SECONDS = 60.0
"""Terminal deadline on the whole teardown. Was 30 s, which sat
INSIDE the measured tens-of-seconds slow-teardown band and forced
exit 75 on sessions that had completed cleanly (12 of 266 archived
runs; operator flag 2026-09-03: ``make run`` exited before its
scorecard). Sixty seconds clears that band and still bounds a true
wedge; the exit code is the session's outcome, not a blanket 75."""

_TEARDOWN_HANG_EXIT_CODE = 75


def _thread_stacks_snapshot() -> str:
    """Render every live thread's current stack for the hang autopsy.

    The watchdog fires on its own timer thread while the thread that
    called ``browser.close()`` is the one that is stuck, so the
    snapshot must cross threads: ``sys._current_frames`` hands over
    each live thread's innermost frame, and the formatted stacks name
    the exact call the teardown is wedged in. Both 240 s fleet runs on
    2026-08-14 ended in a 30 s close hang with identical logs and no
    surviving evidence — this snapshot is what turns the next
    occurrence into a diagnosis instead of another mystery exit 75.

    Returns:
        One section per live thread — its name, ident, and formatted
        stack — joined into a single loggable string.
    """
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    sections: list[str] = []
    for ident, frame in sys._current_frames().items():
        name = names.get(ident, "unnamed")
        stack = "".join(traceback.format_stack(frame))
        sections.append(f"--- thread {name!r} (ident={ident}) ---\n{stack}")
    return "\n".join(sections)


def _session_outcome_exit_code() -> int:
    """The exit code a forced teardown exit must carry.

    The teardown inherits the SESSION's fate, not its own: a hung
    ``browser.close()`` after a completed session (quit sent,
    artifacts saved) is an environment cleanup problem, loudly
    logged, and must not turn the run's exit code into a failure —
    that is exactly what made ``make run`` abort before its
    scorecard (operator flag 2026-09-03). ``cleanup_browser`` runs in
    its callers' ``finally`` blocks, so an exception in flight here
    IS the session crashing, and the forced exit keeps a failure
    code for it.

    Returns:
        ``0`` when no exception is propagating (session completed or
        was caught upstream), :data:`_TEARDOWN_HANG_EXIT_CODE`
        otherwise.
    """
    if sys.exc_info()[0] is None:
        return 0
    return _TEARDOWN_HANG_EXIT_CODE


def _remedy_close_hang(closed: threading.Event) -> None:
    """Second rung: kill the browser engine under a stalled close.

    Fires :data:`_TEARDOWN_REMEDY_SECONDS` after the close began. The
    engine processes are killed directly while the Playwright node
    driver is SPARED — the driver is the peer that must observe the
    engine's death to resolve the pending ``browser.close()`` on the
    main thread, which is what un-wedges the teardown without a
    forced exit.

    Args:
        closed: Set by ``cleanup_browser`` the moment the close
            returns; a set event means there is nothing to remedy.
    """
    if closed.is_set():
        return
    log.error(
        "Teardown: browser close still pending after %.0fs; killing the browser "
        "engine (driver spared so it can resolve the close)",
        _TEARDOWN_REMEDY_SECONDS,
    )
    killed = _test_hooks.kill_browser_processes()
    log.error("Teardown: killed browser engine pids %s", killed)


def _handle_teardown_hang(closed: threading.Event, hang_exit_code: int) -> None:
    """Terminal rung: force the process to exit after a hung teardown.

    Args:
        closed: Set when ``browser.close()`` returned; distinguishes
            a close that never came back from a process that closed
            fine but then wedged in Playwright/interpreter shutdown
            (both shapes exist in the run corpus — bot-20260729-010551
            is the post-close shape).
        hang_exit_code: The session-outcome code computed when the
            teardown began (see :func:`_session_outcome_exit_code`).
    """
    if closed.is_set():
        # The close completed. On the CLI this thread only exists
        # because process exit is wedged; in the long-running service
        # the process is HEALTHY and must not be shot — before this
        # guard, the uncancelled watchdog force-exited a serving
        # process 30 s after every clean session teardown
        # (bot-20260729-010551: "browser closed" at :35, forced exit
        # at :29:04). The CLI's post-session deadline lives in
        # ``bot/entry.py``, armed only after ``bot.run`` returns.
        return
    log.error(
        "Teardown exceeded %.0fs; forcing process exit %d (artifacts were saved before cleanup)",
        _TEARDOWN_WATCHDOG_SECONDS,
        hang_exit_code,
    )
    log.error("Thread stacks at the moment of the hang:\n%s", _thread_stacks_snapshot())
    _test_hooks.force_exit(hang_exit_code)


_SESSION_EXIT_DEADLINE_SECONDS = 30.0
"""CLI tail bound: seconds a finished ``tankpit-bot`` process gets
between ``bot.run`` returning and actual process exit. The tail is
Playwright's context-manager stop plus interpreter shutdown — the
post-close wedge shape (run bot-20260729-010551 wedged there and was
only saved by an unrelated timer). Thirty seconds is far past any
clean shutdown; the forced exit carries 0 because the session is
already complete and its artifacts saved."""


def _handle_session_exit_wedge() -> None:
    """Force a finished CLI process out of a wedged shutdown tail."""
    log.error(
        "Process still alive %.0fs after the session ended; forcing exit 0 "
        "(post-session wedge in Playwright/interpreter shutdown; artifacts already saved)",
        _SESSION_EXIT_DEADLINE_SECONDS,
    )
    log.error("Thread stacks at the moment of the hang:\n%s", _thread_stacks_snapshot())
    _test_hooks.force_exit(0)


def arm_session_exit_deadline() -> None:
    """Bound the tail between a completed session and process exit.

    CLI-only (``bot/entry.py``, armed AFTER ``bot.run`` returns): the
    daemon timer dies unnoticed with a normal exit and fires only when
    shutdown wedges. The long-running service must NEVER arm this —
    its process is healthy after a session ends and outliving the
    session is its job.
    """
    _test_hooks.start_watchdog(_SESSION_EXIT_DEADLINE_SECONDS, _handle_session_exit_wedge)


def navigate_and_login(
    page: PageProtocol,
    cdp: CDPSessionProtocol,
    ws: WorldService,
    *,
    target_url: str,
    prefer_account: bool,
    tank_name_prefix: str = "TP",
    auto_join_room: bool = True,
) -> None:
    """Navigate to the game URL and handle login.

    Args:
        page: Playwright page.
        cdp: CDP session.
        ws: The session's world service; the joined room lands here.
        target_url: URL to navigate to.
        prefer_account: Whether to prefer account login.
        tank_name_prefix: Prefix for tank name.
        auto_join_room: Whether to automatically join a room.

    Raises:
        GameNotJoinedError: If login or room join fails.
    """
    from tankpit_bot.browser.login import handle_login_flow

    page.goto(target_url, wait_until="domcontentloaded")
    log.info("Navigated to %s", page.url)

    success = handle_login_flow(
        page,
        cdp,
        ws,
        tank_name_prefix=tank_name_prefix,
        prefer_account=prefer_account,
        auto_join_room=auto_join_room,
    )
    if not success:
        raise GameNotJoinedError("login or room join did not complete successfully")


def wait_for_game_ready(
    page: PageProtocol,
    messages: list[CapturedMessage],
) -> None:
    """Wait for game to fully load (message flow stabilizes).

    Args:
        page: Playwright page.
        messages: Captured message list to monitor for stability.

    Raises:
        GameNotJoinedError: If no messages captured after stabilization.
    """
    log.info("Waiting for game to initialize...")
    page.wait_for_timeout(2000.0)

    last_count = len(messages)
    stable_checks = 0
    while stable_checks < 3:
        page.wait_for_timeout(500.0)
        current_count = len(messages)
        if current_count == last_count:
            stable_checks += 1
        else:
            stable_checks = 0
            last_count = current_count

    if len(messages) == 0:
        raise GameNotJoinedError("No WebSocket messages captured - game may not have loaded")

    log.info("Game ready, captured %d initial messages", len(messages))


def cleanup_browser(browser: BrowserProtocol) -> None:
    """Close the browser under the teardown escalation ladder.

    Rung 1 is the normal ``browser.close()``. Rung 2
    (:func:`_remedy_close_hang`, at :data:`_TEARDOWN_REMEDY_SECONDS`)
    kills the browser engine directly so the spared driver can
    resolve the close. Rung 3 (:func:`_handle_teardown_hang`, at
    :data:`_TEARDOWN_WATCHDOG_SECONDS`) force-exits with the
    session's outcome code. A close that returns disarms both timers
    via the shared event — they still fire, and do nothing.

    Args:
        browser: Browser instance to close.
    """
    closed = threading.Event()
    hang_exit_code = _session_outcome_exit_code()

    def remedy() -> None:
        _remedy_close_hang(closed)

    def terminal() -> None:
        _handle_teardown_hang(closed, hang_exit_code)

    from playwright._impl._errors import Error as PlaywrightError

    _test_hooks.start_watchdog(_TEARDOWN_REMEDY_SECONDS, remedy)
    _test_hooks.start_watchdog(_TEARDOWN_WATCHDOG_SECONDS, terminal)
    log.info("Teardown: closing browser")
    try:
        browser.close()
    except (OSError, RuntimeError) as exc:
        log.debug("Browser close failed (already closed): %s", exc)
    except PlaywrightError as exc:
        # The close resolving by exception is rung 2's designed
        # outcome: the engine was killed under it and the driver
        # reports the disconnect instead of a clean close. Either
        # way the browser is gone, which is all teardown needs.
        log.error("Teardown: browser close resolved by disconnect: %s", exc)
    closed.set()
    log.info("Teardown: browser closed")


def gather_intel(
    page: PageProtocol,
    cdp: CDPSessionProtocol,
) -> str | None:
    """Gather and log all available intel after login.

    Args:
        page: Playwright page.
        cdp: CDP session.

    Returns:
        Captured static XOR key, or None if not found.
    """
    _debug_js_websocket(cdp)
    _log_script_urls(page)
    return _capture_static_key(page)


def _debug_js_websocket(cdp: CDPSessionProtocol) -> None:
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
    result_obj = debug_result.get("result", {})
    debug_val = result_obj.get("value", "?") if isinstance(result_obj, dict) else "?"
    log.info("JS WebSocket check: %s", debug_val)


def _log_script_urls(page: PageProtocol) -> None:
    """Log all loaded script URLs for protocol analysis.

    Args:
        page: Playwright page.
    """
    script_urls = page.evaluate(
        "Array.from(document.querySelectorAll('script[src]')).map(s => s.src)"
    )
    if isinstance(script_urls, list) and script_urls:
        log.info("Loaded scripts (%d):", len(script_urls))
        for url in script_urls:
            log.info("  - %s", url)


def _capture_static_key(page: PageProtocol) -> str | None:
    """Extract static XOR key from tpclient JS source.

    Args:
        page: Playwright page.

    Returns:
        Static key string, or None if not found.
    """
    import re

    from tankpit_bot.browser.key_discovery import save_static_key

    js_check_loaded = (
        "Array.from(document.querySelectorAll('script[src]')).some(s => s.src.includes('tpclient'))"
    )
    page.wait_for_function(js_check_loaded, timeout=10000)

    js_get_url = (
        "Array.from(document.querySelectorAll('script[src]'))"
        ".find(s => s.src.includes('tpclient'))?.src"
    )
    tpclient_url = page.evaluate(js_get_url)
    if not isinstance(tpclient_url, str):
        log.warning("Could not find tpclient script URL")
        return None

    js_content = page.evaluate(f"fetch('{tpclient_url}').then(r => r.text())")
    if not isinstance(js_content, str) or not js_content:
        # A failed fetch used to become the empty string and be written
        # anyway, which replaced the tracked reference copy with nothing.
        # The checked-in tpclient.js IS the artifact later sessions read,
        # so a fetch that returned no source has nothing to save.
        log.warning("Fetched no tpclient source from %s", tpclient_url)
        return None

    from pathlib import Path

    js_path = Path("tpclient.js")
    _test_hooks.write_text(js_path, js_content)
    log.info("Saved tpclient JS to %s (%d bytes)", js_path, len(js_content))

    match = re.search(r'"([^"]{1000})"', js_content)
    if not match:
        log.warning("Could not find static key in tpclient JS")
        return None

    static_key: str = match.group(1)
    save_static_key(static_key)
    log.info("Captured static key: %s...", static_key[:20])
    return static_key


__all__ = [
    "arm_session_exit_deadline",
    "cleanup_browser",
    "gather_intel",
    "navigate_and_login",
    "wait_for_game_ready",
]
