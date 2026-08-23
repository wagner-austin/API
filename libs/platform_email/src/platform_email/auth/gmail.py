"""Google/Gmail OAuth 2.0 authentication flow.

Provides OAuth authentication for Gmail API.
"""

from __future__ import annotations

import urllib.parse

from platform_core.errors import AppError, EmailErrorCode
from platform_core.json_utils import (
    InvalidJsonError,
    JSONTypeError,
    load_json_str,
    narrow_json_to_dict,
    optional_str,
    require_int,
    require_str,
)
from platform_core.logging import get_logger

from platform_email.auth.common import (
    generate_code_challenge,
    generate_code_verifier,
    generate_state,
    is_token_expired,
)
from platform_email.config import GMAIL_AUTH_URL, GMAIL_EMAIL_SCOPES, GMAIL_TOKEN_URL
from platform_email.testing import hooks
from platform_email.types import OAuthCredentials, OAuthTokens


def build_gmail_auth_url(
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
        "scope": " ".join(GMAIL_EMAIL_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return GMAIL_AUTH_URL + "?" + urllib.parse.urlencode(params)


def exchange_gmail_code_for_tokens(
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
        AppError[EmailErrorCode]: If token exchange fails.
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
        response = hooks.http_post(GMAIL_TOKEN_URL, headers, body)
    except ConnectionError as e:
        msg = f"Failed to exchange authorization code: {e}"
        raise AppError(EmailErrorCode.AUTH_FAILED, msg, http_status=401) from e
    except OSError as e:
        msg = f"Failed to exchange authorization code: {e}"
        raise AppError(EmailErrorCode.AUTH_FAILED, msg, http_status=401) from e

    try:
        raw_value = load_json_str(response)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError) as e:
        msg = f"Invalid JSON response from token endpoint: {e}"
        raise AppError(EmailErrorCode.AUTH_FAILED, msg, http_status=401) from e

    if "error" in data:
        error_val = data.get("error")
        error_desc_val = data.get("error_description")
        error_str = str(error_val) if error_val else "Unknown error"
        error_desc = str(error_desc_val) if error_desc_val else error_str
        msg = f"Token exchange failed: {error_desc}"
        raise AppError(EmailErrorCode.AUTH_FAILED, msg, http_status=401)

    access_token = require_str(data, "access_token")
    refresh_token = optional_str(data, "refresh_token")
    expires_in = require_int(data, "expires_in")
    current_time = hooks.current_time()
    expires_at = current_time + expires_in

    if refresh_token is None:
        msg = "No refresh token in response"
        raise AppError(EmailErrorCode.AUTH_FAILED, msg, http_status=401)

    return OAuthTokens(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        token_type="Bearer",
    )


def refresh_gmail_access_token(
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
        AppError[EmailErrorCode]: If refresh fails.
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
        response = hooks.http_post(GMAIL_TOKEN_URL, headers, body)
    except ConnectionError as e:
        msg = f"Failed to refresh token: {e}"
        raise AppError(EmailErrorCode.TOKEN_EXPIRED, msg, http_status=401) from e
    except OSError as e:
        msg = f"Failed to refresh token: {e}"
        raise AppError(EmailErrorCode.TOKEN_EXPIRED, msg, http_status=401) from e

    try:
        raw_value = load_json_str(response)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError) as e:
        msg = f"Invalid JSON response from token endpoint: {e}"
        raise AppError(EmailErrorCode.TOKEN_EXPIRED, msg, http_status=401) from e

    if "error" in data:
        error_val = data.get("error")
        error_desc_val = data.get("error_description")
        error_str = str(error_val) if error_val else "Unknown error"
        error_desc = str(error_desc_val) if error_desc_val else error_str
        msg = f"Token refresh failed: {error_desc}"
        raise AppError(EmailErrorCode.TOKEN_EXPIRED, msg, http_status=401)

    access_token = require_str(data, "access_token")
    expires_in = require_int(data, "expires_in")
    current_time = hooks.current_time()
    expires_at = current_time + expires_in

    return OAuthTokens(
        access_token=access_token,
        refresh_token=refresh_token,  # Keep original refresh token
        expires_at=expires_at,
        token_type="Bearer",
    )


def get_valid_gmail_tokens(credentials: OAuthCredentials, tokens: OAuthTokens) -> OAuthTokens:
    """Get valid tokens, refreshing if necessary.

    Args:
        credentials: OAuth client credentials.
        tokens: Current tokens (may be expired).

    Returns:
        Valid OAuthTokens (either original or refreshed).

    Raises:
        AppError[EmailErrorCode]: If refresh fails.
    """
    if not is_token_expired(tokens):
        return tokens

    new_tokens = refresh_gmail_access_token(credentials, tokens["refresh_token"])
    hooks.save_gmail_tokens(new_tokens)
    return new_tokens


def authorize_gmail(credentials: OAuthCredentials) -> OAuthTokens:
    """Run the full Gmail OAuth authorization flow.

    Opens browser for user consent and waits for callback.

    Args:
        credentials: OAuth client credentials.

    Returns:
        OAuthTokens after successful authorization.

    Raises:
        AppError[EmailErrorCode]: If authorization fails.
    """
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)
    state = generate_state()

    auth_url = build_gmail_auth_url(credentials, code_challenge=code_challenge, state=state)
    hooks.open_browser(auth_url)

    hooks.console_output("\nAfter authorizing, copy the authorization code from the browser.")
    hooks.console_output("If using localhost redirect, the code is in the URL after 'code='")
    code = hooks.console_input("Enter authorization code: ").strip()

    if not code:
        msg = "No authorization code provided"
        raise AppError(EmailErrorCode.AUTH_FAILED, msg, http_status=401)

    tokens = exchange_gmail_code_for_tokens(
        credentials,
        code=code,
        code_verifier=code_verifier,
    )

    hooks.save_gmail_tokens(tokens)
    return tokens


def gmail_load_or_authorize() -> OAuthTokens:
    """Load cached tokens or run Gmail authorization flow.

    Returns:
        Valid OAuthTokens (from cache, refreshed, or new authorization).

    Raises:
        AppError[EmailErrorCode]: If credentials not found or authorization fails.
    """
    credentials = hooks.load_gmail_credentials()

    cached_tokens = hooks.load_gmail_tokens()
    if cached_tokens is not None:
        if is_token_expired(cached_tokens):
            try:
                new_tokens = refresh_gmail_access_token(credentials, cached_tokens["refresh_token"])
                hooks.save_gmail_tokens(new_tokens)
                return new_tokens
            except AppError as refresh_failed:
                # Falling through to a full authorization is the recovery, but a
                # refresh that keeps failing means the stored refresh token is
                # dead, and silently re-prompting hides that from whoever has to
                # explain why this account reauthorizes every run.
                get_logger(__name__).warning(
                    "Gmail token refresh failed, falling back to full authorization: %s",
                    refresh_failed,
                )
        else:
            return cached_tokens

    return authorize_gmail(credentials)


__all__ = [
    "authorize_gmail",
    "build_gmail_auth_url",
    "exchange_gmail_code_for_tokens",
    "get_valid_gmail_tokens",
    "gmail_load_or_authorize",
    "refresh_gmail_access_token",
]
