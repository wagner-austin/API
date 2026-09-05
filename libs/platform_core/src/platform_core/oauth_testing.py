"""Public test utilities for OAuth functionality.

Provides fake implementations and factory functions for testing OAuth flows.
These utilities are exported for use by consuming libraries and services.

Pattern: Libs export testing.py with public test utilities.
Services use _test_hooks.py for internal dependency injection.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import dump_json_str
from platform_core.oauth_types import OAuthCredentials, OAuthTokenResponse, OAuthTokens

# =============================================================================
# HTTP Hook Fakes
# =============================================================================


def make_fake_http_post(response: str) -> Callable[[str, dict[str, str], str], str]:
    """Create an HTTP POST hook that returns a fixed response.

    Args:
        response: Response body to return for all requests.

    Returns:
        A hook answering with the fixed response.
    """

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        _ = (url, headers, body)
        return response

    return _hook


def make_raising_http_post(exc: BaseException) -> Callable[[str, dict[str, str], str], str]:
    """Create an HTTP POST hook that raises an exception.

    Args:
        exc: Exception to raise on each request.

    Returns:
        A hook that raises the exception.
    """

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        _ = (url, headers, body)
        raise exc

    return _hook


def make_sequenced_http_post(
    responses: list[str | BaseException],
) -> Callable[[str, dict[str, str], str], str]:
    """Create an HTTP POST hook that returns different responses in sequence.

    Useful for testing retry logic or multi-step flows where different
    calls should produce different results.

    Args:
        responses: List of responses or exceptions to return/raise in order.

    Returns:
        A hook cycling through the responses.
    """
    call_count = [0]

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        _ = (url, headers, body)
        index = call_count[0]
        call_count[0] += 1

        if index >= len(responses):
            msg = f"No more responses configured (call {index + 1})"
            raise RuntimeError(msg)

        response = responses[index]
        if isinstance(response, BaseException):
            raise response
        return response

    return _hook


# =============================================================================
# Time Hook Fakes
# =============================================================================


def make_fake_current_time(timestamp: int) -> Callable[[], int]:
    """Create a current time hook that returns a fixed timestamp.

    Args:
        timestamp: Unix timestamp to return.

    Returns:
        A hook answering with the fixed timestamp.
    """

    def _hook() -> int:
        return timestamp

    return _hook


def make_advancing_current_time(start: int, increment: int = 1) -> Callable[[], int]:
    """Create a current time hook that advances on each call.

    Useful for testing token expiry with multiple time-sensitive operations.

    Args:
        start: Initial timestamp.
        increment: Amount to add on each subsequent call.

    Returns:
        A hook advancing time on each call.
    """
    current = [start]

    def _hook() -> int:
        result = current[0]
        current[0] += increment
        return result

    return _hook


# =============================================================================
# Token Response Helpers
# =============================================================================


def make_token_response_json(
    *,
    access_token: str = "test_access_token",
    refresh_token: str | None = "test_refresh_token",
    expires_in: int = 3600,
    token_type: str = "Bearer",
) -> str:
    """Create a JSON token response string for testing.

    Args:
        access_token: Access token value.
        refresh_token: Refresh token value (None to omit).
        expires_in: Token lifetime in seconds.
        token_type: Token type string.

    Returns:
        JSON-encoded token response string.
    """
    response: dict[str, str | int | None] = {
        "access_token": access_token,
        "expires_in": expires_in,
        "token_type": token_type,
    }
    if refresh_token is not None:
        response["refresh_token"] = refresh_token
    return dump_json_str(response)


def make_error_response_json(
    *,
    error: str = "invalid_grant",
    error_description: str | None = None,
) -> str:
    """Create a JSON error response string for testing.

    Args:
        error: OAuth error code.
        error_description: Human-readable error description.

    Returns:
        JSON-encoded error response string.
    """
    response: dict[str, str] = {"error": error}
    if error_description is not None:
        response["error_description"] = error_description
    return dump_json_str(response)


# =============================================================================
# Credential and Token Factories
# =============================================================================


def make_test_credentials(
    *,
    client_id: str = "test_client_id",
    client_secret: str = "test_client_secret",
    redirect_uri: str = "http://localhost:8080/callback",
) -> OAuthCredentials:
    """Create test OAuth credentials.

    Args:
        client_id: OAuth client ID.
        client_secret: OAuth client secret.
        redirect_uri: Redirect URI.

    Returns:
        OAuthCredentials for testing.
    """
    return OAuthCredentials(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
    )


def make_test_tokens(
    *,
    access_token: str = "test_access_token",
    refresh_token: str = "test_refresh_token",
    expires_at: int = 1735200000,
    expired: bool = False,
    current_time: int | None = None,
) -> OAuthTokens:
    """Create test OAuth tokens.

    Args:
        access_token: Access token value.
        refresh_token: Refresh token value.
        expires_at: Token expiry timestamp (ignored if expired=True).
        expired: If True, set expires_at to be in the past relative to current_time.
        current_time: Reference time for calculating expired tokens.

    Returns:
        OAuthTokens for testing.
    """
    if expired:
        base_time = current_time if current_time is not None else 1735200000
        actual_expires_at = base_time - 100
    else:
        actual_expires_at = expires_at

    return OAuthTokens(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=actual_expires_at,
        token_type="Bearer",
    )


def make_test_token_response(
    *,
    access_token: str = "test_access_token",
    refresh_token: str | None = "test_refresh_token",
    expires_in: int = 3600,
    token_type: str = "Bearer",
) -> OAuthTokenResponse:
    """Create test OAuth token response.

    Args:
        access_token: Access token value.
        refresh_token: Refresh token value (None for refresh responses).
        expires_in: Token lifetime in seconds.
        token_type: Token type string.

    Returns:
        OAuthTokenResponse for testing.
    """
    return OAuthTokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
        token_type=token_type,
    )


__all__ = [
    "make_advancing_current_time",
    "make_error_response_json",
    "make_fake_current_time",
    "make_fake_http_post",
    "make_raising_http_post",
    "make_sequenced_http_post",
    "make_test_credentials",
    "make_test_token_response",
    "make_test_tokens",
    "make_token_response_json",
]
