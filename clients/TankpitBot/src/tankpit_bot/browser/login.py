"""Authentication logic for Tankpit browser automation.

Provides guest login, account login, and the unified login flow.
Room discovery and join protocol live in ``room_join.py``.
"""

from __future__ import annotations

import re
import uuid
from typing import TypedDict

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol, PageProtocol
from tankpit_bot.browser.accounts import resolve_account
from tankpit_bot.browser.room_join import join_room

log = get_logger(__name__)
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
]
