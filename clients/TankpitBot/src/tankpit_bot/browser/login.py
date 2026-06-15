"""Shared login logic for Tankpit browser automation.

Provides unified guest and account login functionality used by both
the sniffer and probe modules.
"""

from __future__ import annotations

import base64
import re
import uuid
from typing import TypedDict

from platform_core.json_utils import JSONObject, require_dict, require_str
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import CDPSessionProtocol, PageProtocol
from tankpit_bot.browser.accounts import resolve_account
from tankpit_bot.browser.session import get_captured_raw_messages, send_websocket_bytes
from tankpit_bot.parser import RoomInfo, is_room_info_text
from tankpit_bot.parser_messages import parse_room_info
from tankpit_bot.protocol.codec import ProtocolCodec
from tankpit_bot.protocol.framing import decode_frame
from tankpit_bot.protocol.lobby import (
    ROOM_ENTRY_DEFAULT_X,
    ROOM_ENTRY_DEFAULT_Y,
    RoomEnterRequestDict,
    RoomSelectRequestDict,
    build_room_enter_metadata,
    serialize_room_enter_request,
    serialize_room_select_request,
)

log = get_logger(__name__)
_ROOM_DISCOVERY_TIMEOUT_MS = 10000
_JOIN_CONFIRM_TIMEOUT_MS = 10000
_ROOM_ENTER_TIMEOUT_MS = 10000
_JOIN_POLL_INTERVAL_MS = 100.0
_STATIC_KEY_PATTERN = re.compile(r'"([^"]{1000})"')


class GuestLoginResult(TypedDict):
    """Result of guest login attempt.

    Attributes:
        success: Whether the login was successful.
        rate_limited: Whether guest creation was rate-limited.
        error_message: Error message if any.
    """

    success: bool
    rate_limited: bool
    error_message: str


class AccountLoginResult(TypedDict):
    """Result of account login attempt.

    Attributes:
        success: Whether the login was successful.
        error_message: Error message if any.
    """

    success: bool
    error_message: str


def _fill_tank_name(cdp: CDPSessionProtocol, tank_name: str) -> str:
    """Fill the tank name input field.

    Args:
        cdp: CDP session for JavaScript evaluation.
        tank_name: Name to fill in the input.

    Returns:
        Result message from JavaScript evaluation.
    """
    fill_js = f"""
    (() => {{
        const input = document.querySelector('input[name="tank_name"]');
        if (input) {{
            input.value = '{tank_name}';
            return 'filled';
        }}
        return 'input not found';
    }})()
    """
    result = cdp.send("Runtime.evaluate", {"expression": fill_js, "returnByValue": True})
    result_obj = result.get("result")
    if isinstance(result_obj, dict):
        val = result_obj.get("value", "?")
        return str(val) if val is not None else "?"
    return "?"


def _click_play_now(cdp: CDPSessionProtocol) -> str:
    """Click the Play Now button.

    Args:
        cdp: CDP session for JavaScript evaluation.

    Returns:
        Result message from JavaScript evaluation.
    """
    submit_js = """
    (() => {
        const btn = document.querySelector('input[value="Play Now"]');
        if (btn) {
            btn.click();
            return 'clicked Play Now';
        }
        const form = document.querySelector('form[action="/guest/create-tank"]');
        if (form) {
            form.submit();
            return 'submitted form';
        }
        return 'nothing found';
    })()
    """
    result = cdp.send("Runtime.evaluate", {"expression": submit_js, "returnByValue": True})
    result_obj = result.get("result")
    if isinstance(result_obj, dict):
        val = result_obj.get("value", "?")
        return str(val) if val is not None else "?"
    return "?"


def _check_page_errors(cdp: CDPSessionProtocol) -> str:
    """Check for error messages on the page.

    Args:
        cdp: CDP session for JavaScript evaluation.

    Returns:
        Concatenated error messages or empty string.
    """
    error_js = """
    (() => {
        const errors = document.querySelectorAll('.error, .alert, [class*=error]');
        return Array.from(errors).map(e => e.textContent.trim()).join(' | ');
    })()
    """
    result = cdp.send("Runtime.evaluate", {"expression": error_js, "returnByValue": True})
    result_obj = result.get("result")
    raw_val = result_obj.get("value", "") if isinstance(result_obj, dict) else ""
    return str(raw_val) if raw_val else ""


def handle_guest_login(
    page: PageProtocol,
    cdp: CDPSessionProtocol,
    *,
    tank_name_prefix: str = "B",
) -> GuestLoginResult:
    """Attempt guest login on the before-playing page.

    Args:
        page: Playwright page.
        cdp: CDP session for JavaScript evaluation.
        tank_name_prefix: Prefix for generated tank name.

    Returns:
        GuestLoginResult with success status and error info.
    """
    if "before-playing" not in page.url:
        return GuestLoginResult(success=True, rate_limited=False, error_message="")

    log.info("Attempting guest login...")
    page.wait_for_timeout(2000.0)

    # Generate tank name and fill input
    tank_name = f"{tank_name_prefix}{uuid.uuid4().hex[:8]}"
    fill_result = _fill_tank_name(cdp, tank_name)
    log.info("Fill result: %s", fill_result)

    # Click submit
    submit_result = _click_play_now(cdp)
    log.info("Submit result: %s", submit_result)

    # Wait for navigation
    page.wait_for_timeout(3000.0)
    log.info("After submit, URL: %s", page.url)

    # Check for errors
    error_msg = _check_page_errors(cdp)
    if error_msg:
        log.info("Page errors: %s", error_msg)

    # Check if rate-limited
    if "too many tanks" in error_msg.lower():
        return GuestLoginResult(success=False, rate_limited=True, error_message=error_msg)

    # Check if we're still on before-playing (indicates failure)
    if "before-playing" in page.url:
        return GuestLoginResult(success=False, rate_limited=False, error_message=error_msg)

    return GuestLoginResult(success=True, rate_limited=False, error_message="")


def handle_account_login(
    page: PageProtocol,
    cdp: CDPSessionProtocol,
    username: str,
    password: str,
) -> AccountLoginResult:
    """Attempt account login using credentials.

    Args:
        page: Playwright page.
        cdp: CDP session for JavaScript evaluation.
        username: Account username.
        password: Account password.

    Returns:
        AccountLoginResult with success status and error info.
    """
    log.info("Logging in as %s...", username)

    # Step 1: Open login overlay
    open_js = """
    (() => {
        const loginLink = document.querySelector('a[href="#login"]');
        if (loginLink) {
            loginLink.click();
            return 'opened login';
        }
        return 'login link not found';
    })()
    """
    cdp.send("Runtime.evaluate", {"expression": open_js, "returnByValue": True})

    # Step 2: Wait for login form to be visible
    wait_js = """
    (() => {
        const userInput = document.querySelector('#login-username');
        const passInput = document.querySelector(
            'form[action="/guest/sign-in"] input[name="password"]'
        );
        if (!userInput || !passInput) return 'waiting';
        const userVisible = userInput.offsetParent !== null;
        const passVisible = passInput.offsetParent !== null;
        return userVisible && passVisible ? 'ready' : 'waiting';
    })()
    """
    for _ in range(10):
        result = cdp.send("Runtime.evaluate", {"expression": wait_js, "returnByValue": True})
        result_obj = result.get("result")
        if isinstance(result_obj, dict) and result_obj.get("value") == "ready":
            break
        page.wait_for_timeout(100.0)
    else:
        log.warning("Login form not ready after waiting")

    # Step 3: Fill credentials with proper event dispatch
    fill_login_js = f"""
    (() => {{
        const userInput = document.querySelector('#login-username');
        const passInput = document.querySelector(
            'form[action="/guest/sign-in"] input[name="password"]'
        );
        if (!userInput || !passInput) return 'inputs not found';

        // Set values and dispatch events to trigger form validation
        userInput.focus();
        userInput.value = '{username}';
        userInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
        userInput.dispatchEvent(new Event('change', {{ bubbles: true }}));

        passInput.focus();
        passInput.value = '{password}';
        passInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
        passInput.dispatchEvent(new Event('change', {{ bubbles: true }}));

        return 'filled: user=' + userInput.value.length + ' pass=' + passInput.value.length;
    }})()
    """
    fill_result = cdp.send("Runtime.evaluate", {"expression": fill_login_js, "returnByValue": True})
    fill_obj = fill_result.get("result")
    fill_val = fill_obj.get("value", "?") if isinstance(fill_obj, dict) else "?"
    log.info("Fill: %s", fill_val)

    # Step 4: Submit login
    submit_login_js = """
    (() => {
        const submit = document.querySelector(
            'form[action="/guest/sign-in"] input[type="submit"]'
        );
        if (submit) {
            submit.click();
            return 'clicked login';
        }
        return 'submit not found';
    })()
    """
    result = cdp.send("Runtime.evaluate", {"expression": submit_login_js, "returnByValue": True})
    result_obj = result.get("result")
    login_val = result_obj.get("value", "?") if isinstance(result_obj, dict) else "?"
    log.info("Login: %s", login_val)

    # Poll for navigation or error state (login redirects to /play on success)
    for _ in range(30):  # 30 x 100ms = 3 seconds max
        page.wait_for_timeout(100.0)
        current_url = page.url

        # Success: navigated away from login page
        if "/play" in current_url:
            log.info("Login successful, navigated to: %s", current_url)
            return AccountLoginResult(success=True, error_message="")

        # Still on before-playing - check if there's an error or still processing
        if "before-playing" in current_url:
            # Check for error messages
            login_err_js = """
            (() => {
                const errors = document.querySelectorAll(
                    '.error, .alert, [class*=error], #login .message'
                );
                const texts = Array.from(errors).map(e => e.textContent.trim());
                return texts.filter(t => t.length > 0).join(' | ');
            })()
            """
            err_result = cdp.send(
                "Runtime.evaluate", {"expression": login_err_js, "returnByValue": True}
            )
            err_obj = err_result.get("result")
            err_raw = err_obj.get("value", "") if isinstance(err_obj, dict) else ""
            error_msg = str(err_raw) if err_raw else ""

            if error_msg:
                log.warning("Login errors: %s", error_msg)
                return AccountLoginResult(success=False, error_message=error_msg)
            # No error yet, keep polling

    # Timeout waiting for login completion
    log.warning("Login timeout: still on %s after 3 seconds", page.url)
    return AccountLoginResult(success=False, error_message="Login timeout")


def ensure_on_play_page(page: PageProtocol) -> None:
    """Navigate to the play page if not already there.

    Args:
        page: Playwright page.
    """
    if "/play" not in page.url:
        log.info("Navigating to game...")
        page.goto("https://tankpit.com/play", wait_until="domcontentloaded")
        page.wait_for_timeout(2000.0)
        log.info("Game URL: %s", page.url)


def _decode_captured_body(payload: str) -> bytes:
    """Decode one captured raw WebSocket payload to its body bytes.

    Args:
        payload: Base64-encoded framed message payload.

    Returns:
        Decoded frame body bytes.
    """
    framed = base64.b64decode(payload)
    body, remaining = decode_frame(framed)
    if remaining:
        raise ValueError(f"unexpected trailing bytes in framed message: {remaining.hex()}")
    return body


def _evaluate_string(
    cdp: CDPSessionProtocol,
    expression: str,
    *,
    await_promise: bool = False,
) -> str:
    """Evaluate JavaScript and return the string result.

    Args:
        cdp: Active CDP session.
        expression: JavaScript expression to evaluate.
        await_promise: Whether Runtime.evaluate should await a returned Promise.

    Returns:
        String value returned by the expression.
    """
    params: JSONObject = {
        "expression": expression,
        "returnByValue": True,
    }
    if await_promise:
        params["awaitPromise"] = True
    result = cdp.send("Runtime.evaluate", params)
    result_obj = require_dict(result, "result")
    return require_str(result_obj, "value")


def _get_magic_key(cdp: CDPSessionProtocol) -> str:
    """Return the current session magic key from the page runtime.

    Args:
        cdp: Active CDP session.

    Returns:
        The current ``tankpit.magic`` value, or an empty string when absent.
    """
    return _evaluate_string(
        cdp,
        """
        (() => {
            if (typeof tankpit !== 'undefined' && typeof tankpit.magic === 'string') {
                return tankpit.magic;
            }
            return '';
        })()
        """,
    )


def _get_tpclient_url(cdp: CDPSessionProtocol) -> str:
    """Return the loaded tpclient script URL.

    Args:
        cdp: Active CDP session.

    Returns:
        Loaded tpclient script URL, or an empty string when not found.
    """
    return _evaluate_string(
        cdp,
        """
        (() => {
            const script = Array.from(document.querySelectorAll('script[src]')).find(
                (item) => item.src.includes('tpclient')
            );
            return script ? script.src : '';
        })()
        """,
    )


def _load_tpclient_static_key(cdp: CDPSessionProtocol, tpclient_url: str) -> str:
    """Fetch the loaded tpclient source and extract the current static key.

    Args:
        cdp: Active CDP session.
        tpclient_url: Loaded tpclient script URL.

    Returns:
        Current 1000-character static key string.

    Raises:
        ValueError: If the loaded script does not contain the expected key.
    """
    js_content = _evaluate_string(
        cdp,
        f"fetch({tpclient_url!r}).then((response) => response.text())",
        await_promise=True,
    )
    match = _STATIC_KEY_PATTERN.search(js_content)
    if match is None:
        raise ValueError("tpclient static key was not found in loaded script")
    return match.group(1)


def _collect_room_entries(cdp: CDPSessionProtocol) -> list[RoomInfo]:
    """Return room metadata decoded from captured ROOM_LIST messages.

    Args:
        cdp: Active CDP session.

    Returns:
        Ordered room entries decoded from ROOM_LIST traffic.
    """
    entries: list[RoomInfo] = []
    for payload in get_captured_raw_messages(cdp):
        body = _decode_captured_body(payload)
        if not body or body[0] != ord("+"):
            continue
        text = body.decode("utf-8")
        if not is_room_info_text(text[1:]):
            continue
        entries.append(parse_room_info(text[1:]))
    return entries


def _register_room_entries(entries: list[RoomInfo]) -> None:
    """Register discovered room images for later terrain-map loading.

    Args:
        entries: Room entries decoded from ROOM_LIST messages.
    """
    from tankpit_bot.sniffer.world_state import register_room_image

    for entry in entries:
        register_room_image(entry["room_id"], entry["image"])


def _resolve_room_entry(
    entries: list[RoomInfo],
    room_name: str,
) -> RoomInfo | None:
    """Resolve the desired room entry from decoded room metadata.

    Args:
        entries: Room entries decoded from ROOM_LIST messages.
        room_name: Desired room name from configuration.

    Returns:
        Matching room entry, or `None` if no room matches.
    """
    normalized_target = room_name.strip().lower()
    prefix_match: RoomInfo | None = None
    for entry in entries:
        normalized_candidate = entry["name"].strip().lower()
        if normalized_candidate == normalized_target:
            return entry
        if normalized_candidate.startswith(
            normalized_target + " "
        ) or normalized_candidate.startswith(normalized_target + "("):
            prefix_match = entry
    return prefix_match


def _resolve_room_id(
    entries: list[RoomInfo],
    room_name: str,
) -> str | None:
    """Resolve the desired room ID from decoded room metadata.

    Args:
        entries: Room entries decoded from ROOM_LIST messages.
        room_name: Desired room name from configuration.

    Returns:
        Matching room ID, or `None` if no room matches.
    """
    entry = _resolve_room_entry(entries, room_name)
    if entry is None:
        return None
    return entry["room_id"]


def _wait_for_room_entry(
    page: PageProtocol,
    cdp: CDPSessionProtocol,
    room_name: str,
) -> RoomInfo | None:
    """Wait for the desired room entry to appear in captured ROOM_LIST traffic.

    Args:
        page: Playwright page used for polling delays.
        cdp: Active CDP session.
        room_name: Desired room name from configuration.

    Returns:
        Matching room entry, or `None` if the room never appears.
    """
    waited_ms = 0
    while waited_ms < _ROOM_DISCOVERY_TIMEOUT_MS:
        entries = _collect_room_entries(cdp)
        _register_room_entries(entries)
        room_entry = _resolve_room_entry(entries, room_name)
        if room_entry is not None:
            return room_entry
        page.wait_for_timeout(_JOIN_POLL_INTERVAL_MS)
        waited_ms += int(_JOIN_POLL_INTERVAL_MS)
    return None


def _wait_for_room_id(
    page: PageProtocol,
    cdp: CDPSessionProtocol,
    room_name: str,
) -> str | None:
    """Wait for the desired room to appear in captured ROOM_LIST traffic.

    Args:
        page: Playwright page used for polling delays.
        cdp: Active CDP session.
        room_name: Desired room name from configuration.

    Returns:
        Matching room ID, or `None` if the room never appears.
    """
    room_entry = _wait_for_room_entry(page, cdp, room_name)
    if room_entry is None:
        return None
    return room_entry["room_id"]


def _has_join_confirm(cdp: CDPSessionProtocol, room_id: str, *, start_index: int = 0) -> bool:
    """Return whether a JOIN_CONFIRM for the selected room was captured.

    Args:
        cdp: Active CDP session.
        room_id: Expected joined room ID.
        start_index: Raw-message index where matching should begin.

    Returns:
        True when a matching JOIN_CONFIRM has been captured.
    """
    expected_prefix = f"={room_id}|"
    payloads = get_captured_raw_messages(cdp)
    for payload in payloads[start_index:]:
        body = _decode_captured_body(payload)
        if not body or body[0] != ord("="):
            continue
        if body.decode("utf-8").startswith(expected_prefix):
            return True
    return False


def _wait_for_join_confirm(
    page: PageProtocol,
    cdp: CDPSessionProtocol,
    room_id: str,
    *,
    start_index: int = 0,
) -> bool:
    """Wait for a JOIN_CONFIRM message for the selected room.

    Args:
        page: Playwright page used for polling delays.
        cdp: Active CDP session.
        room_id: Selected room ID.
        start_index: Raw-message index where matching should begin.

    Returns:
        True when the matching JOIN_CONFIRM arrives, False on timeout.
    """
    waited_ms = 0
    while waited_ms < _JOIN_CONFIRM_TIMEOUT_MS:
        if _has_join_confirm(cdp, room_id, start_index=start_index):
            return True
        page.wait_for_timeout(_JOIN_POLL_INTERVAL_MS)
        waited_ms += int(_JOIN_POLL_INTERVAL_MS)
    return False


def _has_enter_response(cdp: CDPSessionProtocol, room_id: str, *, start_index: int = 0) -> bool:
    """Return whether an enter response for the selected room was captured.

    Args:
        cdp: Active CDP session.
        room_id: Expected entered room ID.
        start_index: Raw-message index where matching should begin.

    Returns:
        True when a matching ``$room_id|...`` response has been captured.
    """
    expected_prefix = f"${room_id}|"
    payloads = get_captured_raw_messages(cdp)
    for payload in payloads[start_index:]:
        body = _decode_captured_body(payload)
        if not body or body[0] != ord("$"):
            continue
        if body.decode("utf-8").startswith(expected_prefix):
            return True
    return False


def _wait_for_enter_response(
    page: PageProtocol,
    cdp: CDPSessionProtocol,
    room_id: str,
    *,
    start_index: int = 0,
) -> bool:
    """Wait for the room-enter response for the selected room.

    Args:
        page: Playwright page used for polling delays.
        cdp: Active CDP session.
        room_id: Selected room ID.
        start_index: Raw-message index where matching should begin.

    Returns:
        True when the matching enter response arrives, False on timeout.
    """
    waited_ms = 0
    while waited_ms < _ROOM_ENTER_TIMEOUT_MS:
        if _has_enter_response(cdp, room_id, start_index=start_index):
            return True
        page.wait_for_timeout(_JOIN_POLL_INTERVAL_MS)
        waited_ms += int(_JOIN_POLL_INTERVAL_MS)
    return False


def join_room(
    page: PageProtocol,
    cdp: CDPSessionProtocol,
) -> bool:
    """Join the configured room through the lobby websocket protocol.

    Args:
        page: Playwright page.
        cdp: Active CDP session.

    Returns:
        True if the room was confirmed and the enter response arrived.
    """
    log.info("Joining game...")
    room_name = _test_hooks.get_env("TANKPIT_ROOM") or "Practice"
    room_entry = _wait_for_room_entry(page, cdp, room_name)
    if room_entry is None:
        log.info("Room select failed: room list never exposed %s", room_name)
        return False
    room_id = room_entry["room_id"]
    join_confirm_start = len(get_captured_raw_messages(cdp))
    select_request: RoomSelectRequestDict = {"room_id": room_id}
    select_result = send_websocket_bytes(cdp, serialize_room_select_request(select_request))
    log.info("Room select: room=%s name=%s -> %s", room_id, room_name, select_result)
    if not select_result.startswith("SENT_"):
        return False

    if not _wait_for_join_confirm(page, cdp, room_id, start_index=join_confirm_start):
        log.info("Join confirm timeout: room=%s name=%s", room_id, room_name)
        return False
    from tankpit_bot.sniffer.world_state import set_selected_room

    set_selected_room(room_id)
    magic = _get_magic_key(cdp)
    if len(magic) == 0:
        log.info("Enter game failed: tankpit.magic was unavailable")
        return False
    tpclient_url = _get_tpclient_url(cdp)
    if len(tpclient_url) == 0:
        log.info("Enter game failed: tpclient script URL was unavailable")
        return False
    static_key = _load_tpclient_static_key(cdp, tpclient_url)
    metadata = build_room_enter_metadata(page.url, tpclient_url)
    codec = ProtocolCodec(static_key, magic)
    enter_request: RoomEnterRequestDict = {
        "room_id": room_id,
        "troop": room_entry["default_troop"],
        "preview_x": ROOM_ENTRY_DEFAULT_X,
        "preview_y": ROOM_ENTRY_DEFAULT_Y,
        "metadata": metadata,
    }
    enter_response_start = len(get_captured_raw_messages(cdp))
    enter_result = send_websocket_bytes(
        cdp,
        serialize_room_enter_request(enter_request, codec),
    )
    log.info(
        "Enter game: room=%s troop=%d -> %s",
        room_id,
        room_entry["default_troop"],
        enter_result,
    )
    if not enter_result.startswith("SENT_"):
        return False
    if not _wait_for_enter_response(page, cdp, room_id, start_index=enter_response_start):
        log.info("Enter response timeout: room=%s name=%s", room_id, room_name)
        return False
    return True


def _try_account_login_with_env(
    page: PageProtocol,
    cdp: CDPSessionProtocol,
) -> bool | None:
    """Attempt account login using resolved credentials.

    Resolves credentials via the account registry:
    1. TANKPIT_USERNAME + TANKPIT_PASSWORD env vars (explicit override)
    2. TANKPIT_ACCOUNT selector + accounts.json
    3. First account in accounts.json (default)

    Args:
        page: Playwright page.
        cdp: CDP session for JavaScript evaluation.

    Returns:
        True if login succeeded, False if login failed, None if no account configured.

    Raises:
        AccountNotFoundError: If TANKPIT_ACCOUNT selector is invalid.
        InvalidJsonError: If accounts.json is malformed.
        JSONTypeError: If account data is structurally invalid.
    """
    account = resolve_account()
    if account is None:
        return None

    account_result = handle_account_login(page, cdp, account["username"], account["password"])
    return account_result["success"]


def _do_login(
    page: PageProtocol,
    cdp: CDPSessionProtocol,
    *,
    tank_name_prefix: str,
    allow_account_fallback: bool,
    prefer_account: bool,
) -> bool:
    """Perform the login flow without room joining.

    Args:
        page: Playwright page.
        cdp: CDP session for JavaScript evaluation.
        tank_name_prefix: Prefix for generated tank name.
        allow_account_fallback: Whether to try account login when rate-limited.
        prefer_account: Skip guest login and use account credentials directly.

    Returns:
        True if login succeeded, False otherwise.
    """
    if "before-playing" not in page.url:
        return True

    # If prefer_account, skip guest login and go straight to account
    if prefer_account:
        result = _try_account_login_with_env(page, cdp)
        if result is None:
            log.warning("prefer_account=True but TANKPIT_USERNAME/TANKPIT_PASSWORD not set.")
            return False
        if result:
            ensure_on_play_page(page)
        return result

    # Try guest login
    guest_result = handle_guest_login(page, cdp, tank_name_prefix=tank_name_prefix)

    if guest_result["success"]:
        ensure_on_play_page(page)
        return True

    # If rate-limited, try account login
    if guest_result["rate_limited"] and allow_account_fallback:
        result = _try_account_login_with_env(page, cdp)
        if result is None:
            log.warning("Rate limited. Set TANKPIT_USERNAME and TANKPIT_PASSWORD in .env to login.")
            return False
        if result:
            ensure_on_play_page(page)
        return result

    # Guest login failed for other reasons - still go to play page
    ensure_on_play_page(page)
    return True


def handle_login_flow(
    page: PageProtocol,
    cdp: CDPSessionProtocol,
    *,
    tank_name_prefix: str = "B",
    allow_account_fallback: bool = True,
    prefer_account: bool = False,
    auto_join_room: bool = False,
) -> bool:
    """Handle the complete login flow with optional account fallback.

    Attempts guest login first, falls back to account login if rate-limited
    and credentials are available. If prefer_account is True, skips guest
    login entirely and uses account credentials directly.

    Args:
        page: Playwright page.
        cdp: CDP session for JavaScript evaluation.
        tank_name_prefix: Prefix for generated tank name.
        allow_account_fallback: Whether to try account login when rate-limited.
        prefer_account: Skip guest login and use account credentials directly.
        auto_join_room: Whether to automatically join a room after login.

    Returns:
        True if login succeeded, False otherwise.
    """
    success = _do_login(
        page,
        cdp,
        tank_name_prefix=tank_name_prefix,
        allow_account_fallback=allow_account_fallback,
        prefer_account=prefer_account,
    )

    if success and auto_join_room:
        return join_room(page, cdp)

    return success


__all__ = [
    "AccountLoginResult",
    "GuestLoginResult",
    "ensure_on_play_page",
    "handle_account_login",
    "handle_guest_login",
    "handle_login_flow",
    "join_room",
]
