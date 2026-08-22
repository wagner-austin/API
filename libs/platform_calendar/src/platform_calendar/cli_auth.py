"""Account and OAuth token machinery for the calendar CLI.

The commands live in :mod:`platform_calendar.cli`.
"""

from __future__ import annotations

import urllib.parse
from datetime import datetime
from typing import TypedDict

from platform_core.json_utils import JSONObject, narrow_json_to_dict

from platform_calendar.testing import hooks

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


def _get_now() -> datetime:
    """Get current datetime.

    Returns:
        Current datetime.
    """
    return hooks.cli_get_now()


class TokenRefreshResponse(TypedDict):
    """Response from Google OAuth token refresh endpoint."""

    access_token: str
    expires_in: int
    token_type: str


def require_str(data: JSONObject, key: str) -> str:
    """Require a string value from a JSON object.

    Args:
        data: JSON object to extract from.
        key: Key to look up.

    Returns:
        String value.

    Raises:
        KeyError: If key not found.
        TypeError: If value is not a string.
    """
    value = data[key]
    if not isinstance(value, str):
        msg = f"Expected str for {key}, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def require_int(data: JSONObject, key: str) -> int:
    """Require an int value from a JSON object.

    Args:
        data: JSON object to extract from.
        key: Key to look up.

    Returns:
        Int value.

    Raises:
        KeyError: If key not found.
        TypeError: If value is not an int.
    """
    value = data[key]
    if not isinstance(value, int):
        msg = f"Expected int for {key}, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def decode_token_refresh_response(data: JSONObject) -> TokenRefreshResponse:
    """Decode token refresh response from JSON.

    Args:
        data: JSON object from token endpoint.

    Returns:
        Typed TokenRefreshResponse.

    Raises:
        KeyError: If required field missing.
        TypeError: If field has wrong type.
    """
    return TokenRefreshResponse(
        access_token=require_str(data, "access_token"),
        expires_in=require_int(data, "expires_in"),
        token_type=require_str(data, "token_type"),
    )


# =============================================================================
# Config
# =============================================================================


class Account:
    """Account configuration."""

    def __init__(
        self,
        name: str,
        email: str,
        token_env: str,
        refresh_token_env: str,
        expires_at_env: str,
        default_calendar: str = "primary",
    ) -> None:
        """Initialize account.

        Args:
            name: Display name for the account.
            email: Email address associated with the account.
            token_env: Environment variable name for access token.
            refresh_token_env: Environment variable name for refresh token.
            expires_at_env: Environment variable name for token expiration.
            default_calendar: Default calendar ID.
        """
        self.name = name
        self.email = email
        self.token_env = token_env
        self.refresh_token_env = refresh_token_env
        self.expires_at_env = expires_at_env
        self.default_calendar = default_calendar


ACCOUNTS = [
    Account(
        name="Personal",
        email="austin.o.wagner@gmail.com",
        token_env="GOOGLE_CALENDAR_ACCESS_TOKEN",
        refresh_token_env="GOOGLE_CALENDAR_REFRESH_TOKEN",
        expires_at_env="GOOGLE_CALENDAR_TOKEN_EXPIRES_AT",
    ),
    Account(
        name="Interns",
        email="interns@liuforirvine.com",
        token_env="GOOGLE_CALENDAR_INTERNS_ACCESS_TOKEN",
        refresh_token_env="GOOGLE_CALENDAR_INTERNS_REFRESH_TOKEN",
        expires_at_env="GOOGLE_CALENDAR_INTERNS_EXPIRES_AT",
    ),
]


def _get_env(key: str) -> str | None:
    """Get environment variable.

    Args:
        key: Environment variable name.

    Returns:
        Value if found, None otherwise.
    """
    return hooks.cli_get_env(key)


def _set_env(key: str, value: str) -> None:
    """Set environment variable in cache.

    Args:
        key: Environment variable name.
        value: Value to set.
    """
    hooks.cli_set_env(key, value)


def _is_token_expired(expires_at_str: str) -> bool:
    """Check if token is expired or will expire within 60 seconds.

    Args:
        expires_at_str: Unix timestamp as string.

    Returns:
        True if token is expired or expiring soon.
    """
    expires_at = int(expires_at_str)
    current_time = int(_get_now().timestamp())
    buffer_seconds = 60
    return current_time >= (expires_at - buffer_seconds)


def _refresh_token(
    client_id: str,
    client_secret: str,
    refresh_token: str,
) -> TokenRefreshResponse:
    """Refresh an access token using the refresh token.

    Args:
        client_id: OAuth client ID.
        client_secret: OAuth client secret.
        refresh_token: Refresh token.

    Returns:
        TokenRefreshResponse with new access token.

    Raises:
        urllib.error.HTTPError: If refresh fails.
        KeyError: If response missing required fields.
        TypeError: If response fields have wrong types.
    """
    from platform_core.json_utils import load_json_str

    body_params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    body = urllib.parse.urlencode(body_params)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = hooks.http_post(GOOGLE_TOKEN_URL, headers, body)
    raw_value = load_json_str(response)
    data = narrow_json_to_dict(raw_value)
    return decode_token_refresh_response(data)


def _get_valid_token_for_account(account: Account) -> str | None:
    """Get a valid access token for an account, refreshing if expired.

    Args:
        account: Account to get token for.

    Returns:
        Valid access token, or None if no token configured.
    """
    access_token = _get_env(account.token_env)
    if not access_token:
        return None

    refresh_token = _get_env(account.refresh_token_env)
    expires_at = _get_env(account.expires_at_env)

    # If we have expiration info and token is expired, refresh
    if expires_at and refresh_token and _is_token_expired(expires_at):
        client_id = _get_env("GOOGLE_CALENDAR_CLIENT_ID")
        client_secret = _get_env("GOOGLE_CALENDAR_CLIENT_SECRET")

        if client_id and client_secret:
            response = _refresh_token(client_id, client_secret, refresh_token)
            access_token = response["access_token"]
            new_expires_at = int(_get_now().timestamp()) + response["expires_in"]

            # Update cache with new token
            _set_env(account.token_env, access_token)
            _set_env(account.expires_at_env, str(new_expires_at))

    return access_token


def _get_token(account_name: str) -> str | None:
    """Get valid token for an account by name.

    Args:
        account_name: Account name to look up.

    Returns:
        Valid access token if found, None otherwise.
    """
    for account in ACCOUNTS:
        if account.name.lower() == account_name.lower():
            return _get_valid_token_for_account(account)
    return None


# =============================================================================
# Hookable Functions
# =============================================================================
