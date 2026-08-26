"""OAuth 2.0 authentication flow for Google Calendar API.

Uses centralized OAuth utilities from platform_core.oauth for:
- PKCE generation (generate_code_verifier, generate_code_challenge)
- Authorization URL building (build_authorization_url)
- Token expiry checking (is_token_expired)

Google-specific functionality (error codes, hooks, API endpoints) remains here.
"""

from __future__ import annotations

import urllib.parse

from platform_core.errors import AppError, CalendarErrorCode
from platform_core.json_utils import (
    InvalidJsonError,
    JSONTypeError,
    load_json_str,
    narrow_json_to_dict,
)
from platform_core.logging import get_logger
from platform_core.oauth import (
    generate_code_challenge,
    generate_code_verifier,
    generate_state,
)
from platform_core.oauth import (
    is_token_expired as _core_is_token_expired,
)

from platform_calendar.config import (
    CALENDAR_SCOPES,
    GOOGLE_AUTH_URL,
    GOOGLE_TOKEN_URL,
)
from platform_calendar.testing import hooks
from platform_calendar.types import (
    OAuthCredentials,
    OAuthTokens,
    decode_oauth_token_response,
)

# =============================================================================
# Authorization URL
# =============================================================================


def build_auth_url(
    credentials: OAuthCredentials,
    *,
    code_challenge: str,
    state: str,
) -> str:
    """Build the Google OAuth authorization URL.

    Args:
        credentials: OAuth client credentials.
        code_challenge: PKCE code challenge.
        state: Random state for CSRF protection.

    Returns:
        Authorization URL to open in browser.
    """
    params = {
        "client_id": credentials["client_id"],
        "redirect_uri": credentials["redirect_uri"],
        "response_type": "code",
        "scope": " ".join(CALENDAR_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return GOOGLE_AUTH_URL + "?" + urllib.parse.urlencode(params)


# =============================================================================
# Token Exchange
# =============================================================================


def exchange_code_for_tokens(
    credentials: OAuthCredentials,
    *,
    code: str,
    code_verifier: str,
) -> OAuthTokens:
    """Exchange authorization code for access and refresh tokens.

    Args:
        credentials: OAuth client credentials.
        code: Authorization code from callback.
        code_verifier: PKCE code verifier.

    Returns:
        OAuthTokens with access_token, refresh_token, expires_at.

    Raises:
        AppError[CalendarErrorCode]: If token exchange fails.
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

    try:
        response = hooks.http_post(GOOGLE_TOKEN_URL, headers, body)
    except ConnectionError as e:
        msg = f"Failed to exchange authorization code: {e}"
        raise AppError(CalendarErrorCode.AUTH_FAILED, msg, http_status=401) from e
    except OSError as e:
        msg = f"Failed to exchange authorization code: {e}"
        raise AppError(CalendarErrorCode.AUTH_FAILED, msg, http_status=401) from e

    try:
        raw_value = load_json_str(response)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError) as e:
        msg = f"Invalid JSON response from token endpoint: {e}"
        raise AppError(CalendarErrorCode.AUTH_FAILED, msg, http_status=401) from e

    if "error" in data:
        error_val = data.get("error")
        error_desc_val = data.get("error_description")
        error_str = str(error_val) if error_val else "Unknown error"
        error_desc = str(error_desc_val) if error_desc_val else error_str
        msg = f"Token exchange failed: {error_desc}"
        raise AppError(CalendarErrorCode.AUTH_FAILED, msg, http_status=401)

    token_response = decode_oauth_token_response(data)
    current_time = hooks.current_time()
    expires_at = current_time + token_response["expires_in"]

    refresh_token = token_response["refresh_token"]
    if refresh_token is None:
        msg = "No refresh token in response"
        raise AppError(CalendarErrorCode.AUTH_FAILED, msg, http_status=401)

    return OAuthTokens(
        access_token=token_response["access_token"],
        refresh_token=refresh_token,
        expires_at=expires_at,
        token_type="Bearer",
    )


# =============================================================================
# Token Refresh
# =============================================================================


def refresh_access_token(
    credentials: OAuthCredentials,
    refresh_token: str,
) -> OAuthTokens:
    """Refresh an expired access token.

    Args:
        credentials: OAuth client credentials.
        refresh_token: The refresh token.

    Returns:
        OAuthTokens with new access_token and same refresh_token.

    Raises:
        AppError[CalendarErrorCode]: If refresh fails.
    """
    body_params = {
        "client_id": credentials["client_id"],
        "client_secret": credentials["client_secret"],
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    body = urllib.parse.urlencode(body_params)

    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    try:
        response = hooks.http_post(GOOGLE_TOKEN_URL, headers, body)
    except ConnectionError as e:
        msg = f"Failed to refresh token: {e}"
        raise AppError(CalendarErrorCode.TOKEN_EXPIRED, msg, http_status=401) from e
    except OSError as e:
        msg = f"Failed to refresh token: {e}"
        raise AppError(CalendarErrorCode.TOKEN_EXPIRED, msg, http_status=401) from e

    try:
        raw_value = load_json_str(response)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError) as e:
        msg = f"Invalid JSON response from token endpoint: {e}"
        raise AppError(CalendarErrorCode.TOKEN_EXPIRED, msg, http_status=401) from e

    if "error" in data:
        error_val = data.get("error")
        error_desc_val = data.get("error_description")
        error_str = str(error_val) if error_val else "Unknown error"
        error_desc = str(error_desc_val) if error_desc_val else error_str
        msg = f"Token refresh failed: {error_desc}"
        raise AppError(CalendarErrorCode.TOKEN_EXPIRED, msg, http_status=401)

    token_response = decode_oauth_token_response(data)
    current_time = hooks.current_time()
    expires_at = current_time + token_response["expires_in"]

    return OAuthTokens(
        access_token=token_response["access_token"],
        refresh_token=refresh_token,  # Keep original refresh token
        expires_at=expires_at,
        token_type="Bearer",
    )


# =============================================================================
# Token Management
# =============================================================================


def is_token_expired(tokens: OAuthTokens, *, buffer_seconds: int = 60) -> bool:
    """Check if access token is expired or will expire soon.

    Uses centralized is_token_expired from platform_core.oauth with
    current time from hooks.

    Args:
        tokens: OAuth tokens to check.
        buffer_seconds: Consider expired if within this many seconds of expiry.

    Returns:
        True if token is expired or will expire within buffer.
    """
    return _core_is_token_expired(
        tokens,
        hooks.current_time(),
        buffer_seconds=buffer_seconds,
    )


def get_valid_tokens(credentials: OAuthCredentials, tokens: OAuthTokens) -> OAuthTokens:
    """Get valid tokens, refreshing if necessary.

    Args:
        credentials: OAuth client credentials.
        tokens: Current tokens (may be expired).

    Returns:
        Valid OAuthTokens (either original or refreshed).

    Raises:
        AppError[CalendarErrorCode]: If refresh fails.
    """
    if not is_token_expired(tokens):
        return tokens

    new_tokens = refresh_access_token(credentials, tokens["refresh_token"])
    hooks.save_tokens(new_tokens)
    return new_tokens


# =============================================================================
# Authorization Flow
# =============================================================================


def authorize(credentials: OAuthCredentials) -> OAuthTokens:
    """Run the full OAuth authorization flow.

    Opens browser for user consent and waits for callback.

    Args:
        credentials: OAuth client credentials.

    Returns:
        OAuthTokens after successful authorization.

    Raises:
        AppError[CalendarErrorCode]: If authorization fails.
    """
    # Generate PKCE values using centralized utilities
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)
    state = generate_state()

    # Build and open auth URL
    auth_url = build_auth_url(credentials, code_challenge=code_challenge, state=state)
    hooks.open_browser(auth_url)

    # For desktop flow with urn:ietf:wg:oauth:2.0:oob, user copies code manually
    # In a real implementation, we'd either:
    # 1. Start a local HTTP server to catch the callback
    # 2. Use the OOB flow where user copies the code
    # For simplicity, we'll prompt for the code via hook
    hooks.console_output("\nAfter authorizing, copy the authorization code from the browser.")
    hooks.console_output("If using localhost redirect, the code is in the URL after 'code='")
    code = hooks.console_input("Enter authorization code: ").strip()

    if not code:
        msg = "No authorization code provided"
        raise AppError(CalendarErrorCode.AUTH_FAILED, msg, http_status=401)

    tokens = exchange_code_for_tokens(
        credentials,
        code=code,
        code_verifier=code_verifier,
    )

    hooks.save_tokens(tokens)
    return tokens


def load_or_authorize() -> OAuthTokens:
    """Load cached tokens or run authorization flow.

    Returns:
        Valid OAuthTokens (from cache, refreshed, or new authorization).

    Raises:
        AppError[CalendarErrorCode]: If credentials not found or authorization fails.
    """
    credentials = hooks.load_credentials()

    # Try to load cached tokens
    cached_tokens = hooks.load_tokens()
    if cached_tokens is not None:
        # Check if we need to refresh
        if is_token_expired(cached_tokens):
            try:
                new_tokens = refresh_access_token(credentials, cached_tokens["refresh_token"])
                hooks.save_tokens(new_tokens)
                return new_tokens
            except AppError as refresh_failed:
                # Falling through to a full authorization is the recovery, but a
                # refresh that keeps failing means the stored refresh token is
                # dead, and silently re-prompting hides that from whoever has to
                # explain why this account reauthorizes every run.
                get_logger(__name__).warning(
                    "Calendar token refresh failed, falling back to full authorization: %s",
                    refresh_failed,
                )
        else:
            return cached_tokens

    # No cached tokens or refresh failed, run auth flow
    return authorize(credentials)
