"""Account and OAuth token machinery for the email CLI."""

from __future__ import annotations

import base64
import hashlib
import secrets
import urllib.parse
from datetime import datetime
from typing import TypedDict

from platform_core.json_utils import JSONObject, load_json_str, narrow_json_to_dict
from platform_core.oauth import expires_at_is_past

from platform_email.config.outlook import (
    OUTLOOK_EMAIL_SCOPES,
    outlook_auth_url,
    outlook_token_url,
)
from platform_email.testing import hooks

STYLE_HEADER = "bold cyan"
STYLE_FROM = "bold yellow"
STYLE_SUBJECT = "white"
STYLE_DATE = "green"
STYLE_UNREAD = "bold white"
STYLE_READ = "dim white"
STYLE_ERROR = "bold red"
STYLE_SUCCESS = "bold green"
STYLE_DIM = "dim"
STYLE_FOLDER = "cyan"

MICROSOFT_AUTH_URL: str = outlook_auth_url("common")
MICROSOFT_TOKEN_URL: str = outlook_token_url("common")


class TokenRefreshResponse(TypedDict):
    """Response from Microsoft OAuth token refresh endpoint."""

    access_token: str
    refresh_token: str
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


def decode_token_response(data: JSONObject) -> TokenRefreshResponse:
    """Decode token response from JSON.

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
        refresh_token=require_str(data, "refresh_token"),
        expires_in=require_int(data, "expires_in"),
        token_type=require_str(data, "token_type"),
    )


# =============================================================================
# Account Config
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
        client_id_env: str,
        client_secret_env: str,
    ) -> None:
        """Initialize account.

        Args:
            name: Display name for the account.
            email: Email address associated with the account.
            token_env: Environment variable name for access token.
            refresh_token_env: Environment variable name for refresh token.
            expires_at_env: Environment variable name for token expiration.
            client_id_env: Environment variable name for client ID.
            client_secret_env: Environment variable name for client secret.
        """
        self.name = name
        self.email = email
        self.token_env = token_env
        self.refresh_token_env = refresh_token_env
        self.expires_at_env = expires_at_env
        self.client_id_env = client_id_env
        self.client_secret_env = client_secret_env


# Default account - user can customize
ACCOUNTS = [
    Account(
        name="Outlook",
        email="",  # Will be filled after auth
        token_env="OUTLOOK_ACCESS_TOKEN",
        refresh_token_env="OUTLOOK_REFRESH_TOKEN",
        expires_at_env="OUTLOOK_TOKEN_EXPIRES_AT",
        client_id_env="OUTLOOK_CLIENT_ID",
        client_secret_env="OUTLOOK_CLIENT_SECRET",
    ),
]


# =============================================================================
# Environment Helpers
# =============================================================================


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


def _get_now() -> datetime:
    """Get current datetime.

    Returns:
        Current datetime.
    """
    return hooks.cli_get_now()


def _is_token_expired(expires_at_str: str) -> bool:
    """Check if token is expired or will expire within 60 seconds.

    Args:
        expires_at_str: Unix timestamp as string.

    Returns:
        True if token is expired or expiring soon.
    """
    return expires_at_is_past(int(expires_at_str), int(_get_now().timestamp()))


# =============================================================================
# PKCE Helpers
# =============================================================================


def _generate_code_verifier() -> str:
    """Generate a PKCE code verifier.

    Returns:
        Random code verifier string.
    """
    return secrets.token_urlsafe(64)[:128]


def _generate_code_challenge(verifier: str) -> str:
    """Generate a PKCE code challenge from verifier.

    Args:
        verifier: Code verifier string.

    Returns:
        Base64url-encoded SHA256 hash.
    """
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


# =============================================================================
# Token Operations
# =============================================================================


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
        OSError: If refresh fails.
        KeyError: If response missing required fields.
        TypeError: If response fields have wrong types.
    """
    body_params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
        "scope": " ".join(OUTLOOK_EMAIL_SCOPES),
    }
    body = urllib.parse.urlencode(body_params)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = hooks.http_post(MICROSOFT_TOKEN_URL, headers, body)
    raw_value = load_json_str(response)
    data = narrow_json_to_dict(raw_value)
    return decode_token_response(data)


def _exchange_code_for_tokens(
    client_id: str,
    client_secret: str,
    code: str,
    code_verifier: str,
    redirect_uri: str,
) -> TokenRefreshResponse:
    """Exchange authorization code for tokens.

    Args:
        client_id: OAuth client ID.
        client_secret: OAuth client secret.
        code: Authorization code from user.
        code_verifier: PKCE code verifier.
        redirect_uri: Redirect URI used in auth request.

    Returns:
        TokenRefreshResponse with tokens.

    Raises:
        OSError: If exchange fails.
        KeyError: If response missing required fields.
        TypeError: If response fields have wrong types.
    """
    body_params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "code_verifier": code_verifier,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
        "scope": " ".join(OUTLOOK_EMAIL_SCOPES),
    }
    body = urllib.parse.urlencode(body_params)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = hooks.http_post(MICROSOFT_TOKEN_URL, headers, body)
    raw_value = load_json_str(response)
    data = narrow_json_to_dict(raw_value)
    return decode_token_response(data)


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
        client_id = _get_env(account.client_id_env)
        client_secret = _get_env(account.client_secret_env)

        if client_id and client_secret:
            response = _refresh_token(client_id, client_secret, refresh_token)
            access_token = response["access_token"]
            new_expires_at = int(_get_now().timestamp()) + response["expires_in"]

            # Update cache with new token
            _set_env(account.token_env, access_token)
            _set_env(account.refresh_token_env, response["refresh_token"])
            _set_env(account.expires_at_env, str(new_expires_at))

    return access_token


def _get_token() -> str | None:
    """Get valid token for the default account.

    Returns:
        Valid access token if found, None otherwise.
    """
    return _get_valid_token_for_account(ACCOUNTS[0])


# =============================================================================
# Console Helpers
# =============================================================================


def _print(message: str) -> None:
    """Print message to console.

    Args:
        message: Message to print.
    """
    hooks.console_output(message)


def _input(prompt: str) -> str:
    """Get input from user.

    Args:
        prompt: Prompt to display.

    Returns:
        User input string.
    """
    return hooks.console_input(prompt)


# =============================================================================
# API Helpers
# =============================================================================

GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"
