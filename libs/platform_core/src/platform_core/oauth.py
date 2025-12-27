"""OAuth 2.0 utilities for authorization flows.

Provides reusable OAuth 2.0 functionality including:
- PKCE (Proof Key for Code Exchange) generation
- Token expiry checking
- Authorization URL building
- Token exchange and refresh

All functions are designed to be provider-agnostic. Provider-specific
configuration (URLs, scopes, error handling) is passed as parameters.

Hook dependencies (http_post, current_time) are passed explicitly to functions
rather than using module-level state, enabling clean testing without global state.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import urllib.parse
from collections.abc import Callable

from platform_core.errors import AppError, OAuthErrorCode
from platform_core.json_utils import (
    InvalidJsonError,
    JSONTypeError,
    load_json_str,
    narrow_json_to_dict,
)
from platform_core.oauth_types import (
    OAuthCredentials,
    OAuthTokenResponse,
    OAuthTokens,
    decode_oauth_token_response,
)

# =============================================================================
# Hook Type Definitions
# =============================================================================

HttpPostHook = Callable[[str, dict[str, str], str], str]
"""HTTP POST function: (url, headers, body) -> response_body."""

CurrentTimeHook = Callable[[], int]
"""Returns current Unix timestamp in seconds."""


# =============================================================================
# PKCE (Proof Key for Code Exchange)
# =============================================================================


def generate_code_verifier(*, length: int = 64) -> str:
    """Generate a random code verifier for PKCE.

    Creates a cryptographically random URL-safe string suitable for use
    as a PKCE code_verifier. The verifier should be stored during the
    authorization request and used when exchanging the code for tokens.

    Args:
        length: Number of random bytes to generate (default 64, produces ~86 char string).

    Returns:
        URL-safe base64-encoded random string without padding.
    """
    return secrets.token_urlsafe(length)


def generate_code_challenge(verifier: str) -> str:
    """Generate code challenge from verifier using S256 method.

    Creates a SHA-256 hash of the verifier and base64url encodes it
    without padding, as specified in RFC 7636.

    Args:
        verifier: The code verifier string.

    Returns:
        Base64url-encoded SHA-256 hash of the verifier (no padding).
    """
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    # Base64url encode without padding
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


# =============================================================================
# Token Expiry
# =============================================================================


def is_token_expired(
    tokens: OAuthTokens,
    current_time: int,
    *,
    buffer_seconds: int = 60,
) -> bool:
    """Check if access token is expired or will expire soon.

    Args:
        tokens: OAuth tokens to check.
        current_time: Current Unix timestamp in seconds.
        buffer_seconds: Consider expired if within this many seconds of expiry.

    Returns:
        True if token is expired or will expire within buffer.
    """
    return tokens["expires_at"] <= current_time + buffer_seconds


# =============================================================================
# Authorization URL
# =============================================================================


def build_authorization_url(
    auth_endpoint: str,
    client_id: str,
    redirect_uri: str,
    *,
    code_challenge: str,
    state: str,
    scopes: tuple[str, ...],
    access_type: str = "offline",
    prompt: str = "consent",
) -> str:
    """Build an OAuth 2.0 authorization URL with PKCE.

    Constructs a URL for the authorization endpoint with all required
    parameters for a PKCE authorization code flow.

    Args:
        auth_endpoint: Base URL of the authorization endpoint.
        client_id: OAuth client ID.
        redirect_uri: Redirect URI for the callback.
        code_challenge: PKCE code challenge (from generate_code_challenge).
        state: Random state string for CSRF protection.
        scopes: Tuple of OAuth scopes to request.
        access_type: Access type, typically "offline" for refresh tokens.
        prompt: Prompt behavior, typically "consent" for refresh tokens.

    Returns:
        Complete authorization URL to open in browser.
    """
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "access_type": access_type,
        "prompt": prompt,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return auth_endpoint + "?" + urllib.parse.urlencode(params)


def generate_state() -> str:
    """Generate a random state string for CSRF protection.

    Returns:
        URL-safe random string for OAuth state parameter.
    """
    return secrets.token_urlsafe(32)


# =============================================================================
# Token Exchange
# =============================================================================


def exchange_authorization_code(
    token_endpoint: str,
    credentials: OAuthCredentials,
    code: str,
    code_verifier: str,
    *,
    http_post: HttpPostHook,
    current_time: int,
) -> OAuthTokens:
    """Exchange authorization code for access and refresh tokens.

    Performs the token exchange step of the OAuth 2.0 authorization code flow.

    Args:
        token_endpoint: URL of the token endpoint.
        credentials: OAuth client credentials.
        code: Authorization code from the callback.
        code_verifier: PKCE code verifier used during authorization.
        http_post: HTTP POST function for making the request.
        current_time: Current Unix timestamp for calculating expiry.

    Returns:
        OAuthTokens with access_token, refresh_token, expires_at.

    Raises:
        AppError[OAuthErrorCode]: If token exchange fails.
    """
    body_params = {
        "client_id": credentials["client_id"],
        "client_secret": credentials["client_secret"],
        "code": code,
        "code_verifier": code_verifier,
        "grant_type": "authorization_code",
        "redirect_uri": credentials["redirect_uri"],
    }
    body = urllib.parse.urlencode(body_params)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = _post_to_token_endpoint(
        token_endpoint,
        headers,
        body,
        http_post=http_post,
        error_code=OAuthErrorCode.TOKEN_EXCHANGE_FAILED,
        error_prefix="Failed to exchange authorization code",
    )

    token_response = _decode_token_response(
        response,
        error_code=OAuthErrorCode.TOKEN_EXCHANGE_FAILED,
    )

    refresh_token = token_response["refresh_token"]
    if refresh_token is None:
        msg = "No refresh token in response"
        raise AppError(OAuthErrorCode.MISSING_REFRESH_TOKEN, msg, http_status=401)

    expires_at = current_time + token_response["expires_in"]

    return OAuthTokens(
        access_token=token_response["access_token"],
        refresh_token=refresh_token,
        expires_at=expires_at,
        token_type="Bearer",
    )


def refresh_access_token(
    token_endpoint: str,
    credentials: OAuthCredentials,
    refresh_token: str,
    *,
    http_post: HttpPostHook,
    current_time: int,
) -> OAuthTokens:
    """Refresh an expired access token.

    Uses the refresh token to obtain a new access token without
    requiring user interaction.

    Args:
        token_endpoint: URL of the token endpoint.
        credentials: OAuth client credentials.
        refresh_token: The refresh token.
        http_post: HTTP POST function for making the request.
        current_time: Current Unix timestamp for calculating expiry.

    Returns:
        OAuthTokens with new access_token and same refresh_token.

    Raises:
        AppError[OAuthErrorCode]: If refresh fails.
    """
    body_params = {
        "client_id": credentials["client_id"],
        "client_secret": credentials["client_secret"],
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    body = urllib.parse.urlencode(body_params)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = _post_to_token_endpoint(
        token_endpoint,
        headers,
        body,
        http_post=http_post,
        error_code=OAuthErrorCode.TOKEN_REFRESH_FAILED,
        error_prefix="Failed to refresh token",
    )

    token_response = _decode_token_response(
        response,
        error_code=OAuthErrorCode.TOKEN_REFRESH_FAILED,
    )

    expires_at = current_time + token_response["expires_in"]

    return OAuthTokens(
        access_token=token_response["access_token"],
        refresh_token=refresh_token,  # Keep original refresh token
        expires_at=expires_at,
        token_type="Bearer",
    )


# =============================================================================
# Internal Helpers
# =============================================================================


def _post_to_token_endpoint(
    url: str,
    headers: dict[str, str],
    body: str,
    *,
    http_post: HttpPostHook,
    error_code: OAuthErrorCode,
    error_prefix: str,
) -> str:
    """Make HTTP POST to token endpoint with error handling.

    Args:
        url: Token endpoint URL.
        headers: HTTP headers.
        body: Request body.
        http_post: HTTP POST function.
        error_code: Error code to use on failure.
        error_prefix: Prefix for error messages.

    Returns:
        Response body as string.

    Raises:
        AppError[OAuthErrorCode]: On network errors.
    """
    try:
        return http_post(url, headers, body)
    except ConnectionError as e:
        msg = f"{error_prefix}: {e}"
        raise AppError(error_code, msg, http_status=401) from e
    except OSError as e:
        msg = f"{error_prefix}: {e}"
        raise AppError(error_code, msg, http_status=401) from e


def _decode_token_response(
    response: str,
    *,
    error_code: OAuthErrorCode,
) -> OAuthTokenResponse:
    """Decode and validate token endpoint response.

    Args:
        response: Raw response body.
        error_code: Error code to use on failure.

    Returns:
        Decoded OAuthTokenResponse.

    Raises:
        AppError[OAuthErrorCode]: On invalid JSON or error response.
    """
    try:
        raw_value = load_json_str(response)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError) as e:
        msg = f"Invalid JSON response from token endpoint: {e}"
        raise AppError(error_code, msg, http_status=401) from e

    # Check for OAuth error response
    if "error" in data:
        error_val = data.get("error")
        error_desc_val = data.get("error_description")
        error_str = str(error_val) if error_val else "Unknown error"
        error_desc = str(error_desc_val) if error_desc_val else error_str
        msg = f"Token endpoint error: {error_desc}"
        raise AppError(error_code, msg, http_status=401)

    return decode_oauth_token_response(data)


__all__ = [
    "CurrentTimeHook",
    "HttpPostHook",
    "build_authorization_url",
    "exchange_authorization_code",
    "generate_code_challenge",
    "generate_code_verifier",
    "generate_state",
    "is_token_expired",
    "refresh_access_token",
]
