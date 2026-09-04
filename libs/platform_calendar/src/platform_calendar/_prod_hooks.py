"""Default (production) implementations behind the calendar hooks.

The hooks container and protocols live in
:mod:`platform_calendar.testing`; the shared fakes in
:mod:`platform_calendar.fakes`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from platform_core.errors import AppError, CalendarErrorCode
from platform_core.json_utils import JSONObject
from rich.console import Console

from platform_calendar.types import (
    OAuthCredentials,
    OAuthTokens,
)


def _prod_http_get(url: str, headers: dict[str, str]) -> str:
    """Production HTTP GET using urllib."""
    import urllib.request
    from http.client import HTTPResponse

    req = urllib.request.Request(url)
    for key, value in headers.items():
        req.add_header(key, value)
    response = urllib.request.urlopen(req, timeout=30)
    assert isinstance(response, HTTPResponse)
    try:
        body = response.read()
        return body.decode("utf-8")
    finally:
        response.close()


def _prod_http_post(url: str, headers: dict[str, str], body: str) -> str:
    """Production HTTP POST using urllib."""
    import urllib.request
    from http.client import HTTPResponse

    req = urllib.request.Request(url, data=body.encode("utf-8"), method="POST")
    for key, value in headers.items():
        req.add_header(key, value)
    response = urllib.request.urlopen(req, timeout=30)
    assert isinstance(response, HTTPResponse)
    try:
        response_body = response.read()
        return response_body.decode("utf-8")
    finally:
        response.close()


def _prod_http_patch(url: str, headers: dict[str, str], body: str) -> str:
    """Production HTTP PATCH using urllib.

    Args:
        url: URL to send PATCH request to.
        headers: HTTP headers to include.
        body: Request body as string.

    Returns:
        Response body as string.
    """
    import urllib.request
    from http.client import HTTPResponse

    req = urllib.request.Request(url, data=body.encode("utf-8"), method="PATCH")
    for key, value in headers.items():
        req.add_header(key, value)
    response = urllib.request.urlopen(req, timeout=30)
    assert isinstance(response, HTTPResponse)
    try:
        response_body = response.read()
        return response_body.decode("utf-8")
    finally:
        response.close()


def _prod_http_delete(url: str, headers: dict[str, str]) -> None:
    """Production HTTP DELETE using urllib."""
    import urllib.request
    from http.client import HTTPResponse

    req = urllib.request.Request(url, method="DELETE")
    for key, value in headers.items():
        req.add_header(key, value)
    response = urllib.request.urlopen(req, timeout=30)
    assert isinstance(response, HTTPResponse)
    response.close()


def _prod_load_tokens(path: str | None = None) -> OAuthTokens | None:
    """Production token loader.

    Loads OAuth tokens from environment variables or file.

    Environment variables (checked first):
        GOOGLE_CALENDAR_ACCESS_TOKEN: OAuth access token
        GOOGLE_CALENDAR_REFRESH_TOKEN: OAuth refresh token
        GOOGLE_CALENDAR_TOKEN_EXPIRES_AT: Token expiry (Unix timestamp as string)

    If any token env var is set, all must be set.
    If no env vars are set, reads from file path.

    Args:
        path: Optional file path. Defaults to ~/.google/calendar_tokens.json

    Returns:
        OAuthTokens if found, None if no tokens configured.

    Raises:
        AppError[CalendarErrorCode]: If tokens are partially configured in environment.
    """

    from platform_core.config import config_test_hooks
    from platform_core.json_utils import (
        InvalidJsonError,
        JSONTypeError,
        load_json_str,
        narrow_json_to_dict,
    )

    from platform_calendar.types import OAuthTokens, decode_oauth_tokens

    # Check environment variables first using centralized hook
    env_access_token = config_test_hooks.get_env("GOOGLE_CALENDAR_ACCESS_TOKEN")
    env_refresh_token = config_test_hooks.get_env("GOOGLE_CALENDAR_REFRESH_TOKEN")
    env_expires_at = config_test_hooks.get_env("GOOGLE_CALENDAR_TOKEN_EXPIRES_AT")

    # If any token env var is set, validate all are present
    if env_access_token is not None or env_refresh_token is not None or env_expires_at is not None:
        missing: list[str] = []
        if env_access_token is None:
            missing.append("GOOGLE_CALENDAR_ACCESS_TOKEN")
        if env_refresh_token is None:
            missing.append("GOOGLE_CALENDAR_REFRESH_TOKEN")
        if env_expires_at is None:
            missing.append("GOOGLE_CALENDAR_TOKEN_EXPIRES_AT")
        if missing:
            msg = f"Partial tokens in environment. Missing: {', '.join(missing)}"
            raise AppError(CalendarErrorCode.AUTH_FAILED, msg, http_status=401)
        # All env vars present - narrow types for mypy
        assert env_access_token is not None
        assert env_refresh_token is not None
        assert env_expires_at is not None
        return OAuthTokens(
            access_token=env_access_token,
            refresh_token=env_refresh_token,
            expires_at=int(env_expires_at),
            token_type="Bearer",
        )

    # No env vars set - read from file
    tokens_path = Path(path) if path else Path.home() / ".google" / "calendar_tokens.json"
    if not tokens_path.exists():
        return None
    content = tokens_path.read_text(encoding="utf-8")
    try:
        raw_value = load_json_str(content)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError):
        return None
    return decode_oauth_tokens(data)


def _prod_save_tokens(tokens: OAuthTokens, path: str | None = None) -> None:
    """Production token saver - writes to ~/.google/calendar_tokens.json."""

    from platform_core.json_utils import dump_json_str

    from platform_calendar.types import encode_oauth_tokens

    tokens_path = Path(path) if path else Path.home() / ".google" / "calendar_tokens.json"
    tokens_path.parent.mkdir(parents=True, exist_ok=True)
    content = dump_json_str(encode_oauth_tokens(tokens), indent=2)
    tokens_path.write_text(content, encoding="utf-8")


def _prod_load_credentials(path: str | None = None) -> OAuthCredentials:
    """Production credentials loader.

    Loads OAuth credentials from environment variables or file.

    Environment variables (checked first):
        GOOGLE_CALENDAR_CLIENT_ID: OAuth client ID
        GOOGLE_CALENDAR_CLIENT_SECRET: OAuth client secret
        GOOGLE_CALENDAR_REDIRECT_URI: Redirect URI (defaults to "http://localhost")

    If any credential env var is set, all required ones must be set.
    If no env vars are set, reads from file path.

    Args:
        path: Optional file path. Defaults to ~/.google/calendar_credentials.json

    Returns:
        OAuthCredentials with client_id, client_secret, redirect_uri.

    Raises:
        AppError[CalendarErrorCode]: If credentials not found or partially configured.
    """

    from platform_core.config import config_test_hooks
    from platform_core.json_utils import (
        InvalidJsonError,
        JSONTypeError,
        load_json_str,
        narrow_json_to_dict,
    )

    from platform_calendar.types import OAuthCredentials, decode_google_credentials_file

    # Check environment variables first using centralized hook
    env_client_id = config_test_hooks.get_env("GOOGLE_CALENDAR_CLIENT_ID")
    env_client_secret = config_test_hooks.get_env("GOOGLE_CALENDAR_CLIENT_SECRET")
    env_redirect_uri = config_test_hooks.get_env("GOOGLE_CALENDAR_REDIRECT_URI")

    # If any credential env var is set, validate all required ones are present
    if env_client_id is not None or env_client_secret is not None:
        missing: list[str] = []
        if env_client_id is None:
            missing.append("GOOGLE_CALENDAR_CLIENT_ID")
        if env_client_secret is None:
            missing.append("GOOGLE_CALENDAR_CLIENT_SECRET")
        if missing:
            msg = f"Partial credentials in environment. Missing: {', '.join(missing)}"
            raise AppError(CalendarErrorCode.CREDENTIALS_NOT_FOUND, msg, http_status=401)
        # All required env vars present - narrow types for mypy
        assert env_client_id is not None
        assert env_client_secret is not None
        return OAuthCredentials(
            client_id=env_client_id,
            client_secret=env_client_secret,
            redirect_uri=env_redirect_uri if env_redirect_uri is not None else "http://localhost",
        )

    # No env vars set - read from file
    creds_path = Path(path) if path else Path.home() / ".google" / "calendar_credentials.json"
    if not creds_path.exists():
        msg = f"Credentials file not found at {creds_path}"
        raise AppError(CalendarErrorCode.CREDENTIALS_NOT_FOUND, msg, http_status=401)
    content = creds_path.read_text(encoding="utf-8")
    try:
        raw_value = load_json_str(content)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError) as e:
        msg = f"Credentials file is not valid JSON: {e}"
        raise AppError(CalendarErrorCode.CREDENTIALS_NOT_FOUND, msg, http_status=401) from e
    google_creds = decode_google_credentials_file(data)
    installed = google_creds["installed"]
    redirect_uri = installed["redirect_uris"][0] if installed["redirect_uris"] else ""
    return OAuthCredentials(
        client_id=installed["client_id"],
        client_secret=installed["client_secret"],
        redirect_uri=redirect_uri,
    )


def _prod_open_browser(
    url: str,
    _opener: Callable[[str], bool] | None = None,
) -> None:
    """Production browser opener."""
    import webbrowser

    opener = _opener if _opener is not None else webbrowser.open
    opener(url)


def _prod_current_time() -> int:
    """Production current time in seconds since epoch."""
    import time

    return int(time.time())


def _prod_read_file(path: str) -> str:
    """Production file reader."""

    return Path(path).read_text(encoding="utf-8")


def _prod_write_file(path: str, content: str) -> None:
    """Production file writer."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _prod_file_exists(path: str) -> bool:
    """Production file exists check."""

    return Path(path).exists()


def _prod_console_output(message: str) -> None:
    """Production console output using print."""
    import sys

    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def _prod_console_input(
    prompt: str,
    _input_func: Callable[[str], str] | None = None,
) -> str:
    """Production console input using input."""
    input_func = _input_func if _input_func is not None else input
    return input_func(prompt)


# =============================================================================
# CLI Production Implementations
# =============================================================================

# Module-level cache for CLI environment and console
_cli_env_loaded: bool = False
_cli_env_cache: dict[str, str] = {}
_cli_default_console: Console | None = None

#: Where :func:`_prod_cli_get_env` looks for its ``.env``.
#:
#: A module global with the real location as its value, reset by
#: ``testing.reset_hooks`` alongside the two cache globals beside it, because
#: the path used to be computed inline from ``__file__``. That made the
#: parse loop reachable ONLY on a machine that happened to have a ``.env``
#: in the package root -- which is gitignored, so it exists on a developer's
#: box and nowhere else. The five lines that read and split the file were
#: covered locally and uncovered in CI, and the package's 100% gate was
#: being met by an untracked file rather than by a test.
_cli_env_path: str = ""


def _default_cli_env_path() -> str:
    """Name the real ``.env``, in the package root.

    Returns:
        The path, as a string. A function rather than an inline expression
        so ``testing.reset_hooks`` can restore the real location after a
        test has pointed :data:`_cli_env_path` somewhere else.
    """
    return str(Path(__file__).parent.parent.parent / ".env")


_cli_env_path = _default_cli_env_path()


def _prod_cli_api_get(access_token: str, url: str) -> JSONObject:
    """Production CLI API GET request.

    Args:
        access_token: OAuth access token.
        url: Full API URL.

    Returns:
        Parsed JSON response.
    """
    import http.client
    import urllib.request

    from platform_core.json_utils import load_json_str, narrow_json_to_dict

    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    resp: http.client.HTTPResponse = urllib.request.urlopen(req)
    body = resp.read().decode("utf-8")
    raw = load_json_str(body)
    return narrow_json_to_dict(raw)


def _prod_cli_api_post(access_token: str, url: str, request_body: JSONObject) -> JSONObject:
    """Production CLI API POST request.

    Args:
        access_token: OAuth access token.
        url: Full API URL.
        request_body: JSON request body.

    Returns:
        Parsed JSON response.
    """
    import http.client
    import urllib.request

    from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

    data = dump_json_str(request_body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    resp: http.client.HTTPResponse = urllib.request.urlopen(req)
    body = resp.read().decode("utf-8")
    raw = load_json_str(body)
    return narrow_json_to_dict(raw)


def _prod_cli_api_delete(access_token: str, url: str) -> None:
    """Production CLI API DELETE request.

    Args:
        access_token: OAuth access token.
        url: Full API URL.
    """
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        method="DELETE",
    )
    urllib.request.urlopen(req)


def _prod_cli_get_env(key: str) -> str | None:
    """Production CLI environment variable getter.

    Loads from .env file in the platform_calendar package directory.

    Args:
        key: Environment variable name.

    Returns:
        Value if found, None otherwise.
    """
    global _cli_env_loaded, _cli_env_cache

    if not _cli_env_loaded:
        env_path = Path(_cli_env_path)
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    _cli_env_cache[k] = v
        _cli_env_loaded = True

    return _cli_env_cache.get(key)


def _prod_cli_set_env(key: str, value: str) -> None:
    """Production CLI environment variable setter.

    Updates the in-memory cache with the new value.

    Args:
        key: Environment variable name.
        value: Value to set.
    """
    global _cli_env_cache
    _cli_env_cache[key] = value


def _prod_cli_get_now() -> datetime:
    """Production CLI current datetime.

    Returns:
        Current datetime.
    """
    return datetime.now()


def _prod_cli_prompt_ask(
    message: str,
    _prompt_func: Callable[[str], str] | None = None,
) -> str:
    """Production CLI prompt using Rich.

    Args:
        message: Prompt message.
        _prompt_func: Optional override for testing.

    Returns:
        User input.
    """
    from rich.prompt import Prompt

    prompt_func = _prompt_func if _prompt_func is not None else Prompt.ask
    return prompt_func(message)


def _prod_cli_confirm_ask(
    message: str,
    _confirm_func: Callable[[str], bool] | None = None,
) -> bool:
    """Production CLI confirm using Rich.

    Args:
        message: Prompt message.
        _confirm_func: Optional override for testing.

    Returns:
        True if confirmed.
    """
    from rich.prompt import Confirm

    confirm_func = _confirm_func if _confirm_func is not None else Confirm.ask
    return confirm_func(message)


def _prod_cli_get_console() -> Console:
    """Production CLI console getter.

    Returns:
        Rich Console instance (cached).
    """
    global _cli_default_console

    if _cli_default_console is None:
        _cli_default_console = Console()
    return _cli_default_console
