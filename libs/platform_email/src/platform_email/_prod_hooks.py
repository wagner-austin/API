"""Default (production) implementations behind the email hooks."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from platform_core.errors import AppError, EmailErrorCode

from platform_email.types import (
    OAuthCredentials,
    OAuthTokens,
    OutlookOAuthConfig,
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
    body = response.read()
    response.close()
    return body.decode("utf-8")


def _prod_http_post(url: str, headers: dict[str, str], body: str) -> str:
    """Production HTTP POST using urllib."""
    import urllib.request
    from http.client import HTTPResponse

    req = urllib.request.Request(url, data=body.encode("utf-8"), method="POST")
    for key, value in headers.items():
        req.add_header(key, value)
    response = urllib.request.urlopen(req, timeout=30)
    assert isinstance(response, HTTPResponse)
    response_body = response.read()
    response.close()
    return response_body.decode("utf-8")


def _prod_http_patch(url: str, headers: dict[str, str], body: str) -> str:
    """Production HTTP PATCH using urllib."""
    import urllib.request
    from http.client import HTTPResponse

    req = urllib.request.Request(url, data=body.encode("utf-8"), method="PATCH")
    for key, value in headers.items():
        req.add_header(key, value)
    response = urllib.request.urlopen(req, timeout=30)
    assert isinstance(response, HTTPResponse)
    response_body = response.read()
    response.close()
    return response_body.decode("utf-8")


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


def _prod_load_outlook_tokens() -> OAuthTokens | None:
    """Production Outlook token loader."""
    from platform_core.json_utils import (
        InvalidJsonError,
        JSONTypeError,
        load_json_str,
        narrow_json_to_dict,
    )

    from platform_email import testing as _testing
    from platform_email.types import decode_oauth_tokens

    tokens_path = Path(_testing.hooks.outlook_tokens_path())
    if not tokens_path.exists():
        return None
    content = tokens_path.read_text(encoding="utf-8")
    try:
        raw_value = load_json_str(content)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError):
        return None
    return decode_oauth_tokens(data)


def _prod_save_outlook_tokens(tokens: OAuthTokens) -> None:
    """Production Outlook token saver."""
    from platform_core.json_utils import dump_json_str

    from platform_email import testing as _testing
    from platform_email.types import encode_oauth_tokens

    tokens_path = Path(_testing.hooks.outlook_tokens_path())
    tokens_path.parent.mkdir(parents=True, exist_ok=True)
    content = dump_json_str(encode_oauth_tokens(tokens), indent=2)
    tokens_path.write_text(content, encoding="utf-8")


def _prod_load_outlook_config() -> OutlookOAuthConfig:
    """Production Outlook config loader."""
    from platform_core.json_utils import (
        InvalidJsonError,
        JSONTypeError,
        load_json_str,
        narrow_json_to_dict,
    )

    from platform_email import testing as _testing
    from platform_email.types import decode_outlook_oauth_config

    creds_path = Path(_testing.hooks.outlook_credentials_path())
    if not creds_path.exists():
        msg = f"Outlook credentials file not found at {creds_path}"
        raise AppError(EmailErrorCode.CREDENTIALS_NOT_FOUND, msg, http_status=401)
    content = creds_path.read_text(encoding="utf-8")
    try:
        raw_value = load_json_str(content)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError) as e:
        msg = f"Outlook credentials file is not valid JSON: {e}"
        raise AppError(EmailErrorCode.CREDENTIALS_NOT_FOUND, msg, http_status=401) from e
    return decode_outlook_oauth_config(data)


def _prod_load_gmail_tokens() -> OAuthTokens | None:
    """Production Gmail token loader."""
    from platform_core.json_utils import (
        InvalidJsonError,
        JSONTypeError,
        load_json_str,
        narrow_json_to_dict,
    )

    from platform_email import testing as _testing
    from platform_email.types import decode_oauth_tokens

    tokens_path = Path(_testing.hooks.gmail_tokens_path())
    if not tokens_path.exists():
        return None
    content = tokens_path.read_text(encoding="utf-8")
    try:
        raw_value = load_json_str(content)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError):
        return None
    return decode_oauth_tokens(data)


def _prod_save_gmail_tokens(tokens: OAuthTokens) -> None:
    """Production Gmail token saver."""
    from platform_core.json_utils import dump_json_str

    from platform_email import testing as _testing
    from platform_email.types import encode_oauth_tokens

    tokens_path = Path(_testing.hooks.gmail_tokens_path())
    tokens_path.parent.mkdir(parents=True, exist_ok=True)
    content = dump_json_str(encode_oauth_tokens(tokens), indent=2)
    tokens_path.write_text(content, encoding="utf-8")


def _prod_load_gmail_credentials() -> OAuthCredentials:
    """Production Gmail credentials loader."""
    from platform_core.json_utils import (
        InvalidJsonError,
        JSONTypeError,
        load_json_str,
        narrow_json_to_dict,
        require_list,
        require_str,
    )

    from platform_email import testing as _testing

    creds_path = Path(_testing.hooks.gmail_credentials_path())
    if not creds_path.exists():
        msg = f"Gmail credentials file not found at {creds_path}"
        raise AppError(EmailErrorCode.CREDENTIALS_NOT_FOUND, msg, http_status=401)
    content = creds_path.read_text(encoding="utf-8")
    try:
        raw_value = load_json_str(content)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError) as e:
        msg = f"Gmail credentials file is not valid JSON: {e}"
        raise AppError(EmailErrorCode.CREDENTIALS_NOT_FOUND, msg, http_status=401) from e
    # Google credentials file has "installed" wrapper
    installed_raw = data.get("installed")
    if not isinstance(installed_raw, dict):
        msg = "Gmail credentials file missing 'installed' section"
        raise AppError(EmailErrorCode.CREDENTIALS_NOT_FOUND, msg, http_status=401)
    installed = installed_raw
    redirect_uris_raw = require_list(installed, "redirect_uris")
    redirect_uri = redirect_uris_raw[0] if redirect_uris_raw else "http://localhost"
    if not isinstance(redirect_uri, str):
        redirect_uri = "http://localhost"
    return OAuthCredentials(
        client_id=require_str(installed, "client_id"),
        client_secret=require_str(installed, "client_secret"),
        redirect_uri=redirect_uri,
    )


def _prod_open_browser(url: str) -> None:
    """Production browser opener."""
    import webbrowser

    webbrowser.open(url)


def _prod_current_time() -> int:
    """Production current time in seconds since epoch."""
    import time

    return int(time.time())


def _prod_read_file(path: str) -> str:
    """Production file reader."""

    return Path(path).read_text(encoding="utf-8")


def _prod_read_file_bytes(path: str) -> bytes:
    """Production binary file reader.

    Args:
        path: Path to the file.

    Returns:
        Raw bytes of the file.
    """

    return Path(path).read_bytes()


def _prod_write_file(path: str, content: str) -> None:
    """Production file writer."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _prod_file_exists(path: str) -> bool:
    """Production file exists check."""

    return Path(path).exists()


def _prod_console_output(message: str) -> None:
    """Production console output."""
    import sys

    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def _prod_console_input(prompt: str) -> str:
    """Production console input."""
    return input(prompt)


def _prod_outlook_tokens_path() -> str:
    """Production Outlook tokens path."""
    from platform_email.config import DEFAULT_OUTLOOK_TOKENS_PATH

    return str(DEFAULT_OUTLOOK_TOKENS_PATH)


def _prod_outlook_credentials_path() -> str:
    """Production Outlook credentials path."""
    from platform_email.config import DEFAULT_OUTLOOK_CREDENTIALS_PATH

    return str(DEFAULT_OUTLOOK_CREDENTIALS_PATH)


def _prod_gmail_tokens_path() -> str:
    """Production Gmail tokens path."""
    from platform_email.config import DEFAULT_GMAIL_TOKENS_PATH

    return str(DEFAULT_GMAIL_TOKENS_PATH)


def _prod_gmail_credentials_path() -> str:
    """Production Gmail credentials path."""
    from platform_email.config import DEFAULT_GMAIL_CREDENTIALS_PATH

    return str(DEFAULT_GMAIL_CREDENTIALS_PATH)


# =============================================================================
# CLI Production Implementations
# =============================================================================

# Module-level cache for CLI environment
_cli_env_loaded: bool = False
_cli_env_cache: dict[str, str] = {}


def _prod_cli_get_env(key: str) -> str | None:
    """Production CLI environment variable getter.

    Loads from .env file in the platform_email package directory.

    Args:
        key: Environment variable name.

    Returns:
        Value if found, None otherwise.
    """
    import os

    global _cli_env_loaded, _cli_env_cache

    if not _cli_env_loaded:
        # Load from .env file relative to this module
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
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


__all__ = []
