"""Microsoft/Outlook OAuth 2.0 authentication flow.

Provides OAuth authentication for Microsoft Graph Mail API.
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

from platform_email.auth.common import (
    generate_code_challenge,
    generate_code_verifier,
    generate_state,
    is_token_expired,
)
from platform_email.config import OUTLOOK_EMAIL_SCOPES, outlook_auth_url, outlook_token_url
from platform_email.testing import hooks
from platform_email.types import OAuthTokens, OutlookOAuthConfig


def build_outlook_auth_url(
    config: OutlookOAuthConfig,
    *,
    code_challenge: str,
    state: str,
) -> str:
    """Build the Microsoft OAuth authorization URL.

    Args:
        config: Outlook OAuth configuration.
        code_challenge: PKCE code challenge.
        state: Random state for CSRF protection.

    Returns:
        Authorization URL to open in browser.
    """
    params = {
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "response_type": "code",
        "scope": " ".join(OUTLOOK_EMAIL_SCOPES),
        "response_mode": "query",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    base_url = outlook_auth_url(config["tenant_id"])
    return base_url + "?" + urllib.parse.urlencode(params)


def exchange_outlook_code_for_tokens(
    config: OutlookOAuthConfig,
    *,
    code: str,
    code_verifier: str,
) -> OAuthTokens:
    """Exchange authorization code for access and refresh tokens.

    Args:
        config: Outlook OAuth configuration.
        code: Authorization code from callback.
        code_verifier: PKCE code verifier.

    Returns:
        OAuthTokens with access_token, refresh_token, expires_at.

    Raises:
        AppError[EmailErrorCode]: If token exchange fails.
    """
    body_params = {
        "client_id": config["client_id"],
        "client_secret": config["client_secret"],
        "code": code,
        "code_verifier": code_verifier,
        "grant_type": "authorization_code",
        "redirect_uri": config["redirect_uri"],
        "scope": " ".join(OUTLOOK_EMAIL_SCOPES),
    }
    body = urllib.parse.urlencode(body_params)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    token_endpoint = outlook_token_url(config["tenant_id"])

    try:
        response = hooks.http_post(token_endpoint, headers, body)
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


def refresh_outlook_access_token(
    config: OutlookOAuthConfig,
    refresh_token: str,
) -> OAuthTokens:
    """Refresh an expired access token.

    Args:
        config: Outlook OAuth configuration.
        refresh_token: The refresh token.

    Returns:
        OAuthTokens with new access_token and same refresh_token.

    Raises:
        AppError[EmailErrorCode]: If refresh fails.
    """
    body_params = {
        "client_id": config["client_id"],
        "client_secret": config["client_secret"],
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
        "scope": " ".join(OUTLOOK_EMAIL_SCOPES),
    }
    body = urllib.parse.urlencode(body_params)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    token_endpoint = outlook_token_url(config["tenant_id"])

    try:
        response = hooks.http_post(token_endpoint, headers, body)
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
    new_refresh_token = optional_str(data, "refresh_token")
    expires_in = require_int(data, "expires_in")
    current_time = hooks.current_time()
    expires_at = current_time + expires_in

    # Use new refresh token if provided, otherwise keep original
    final_refresh_token = new_refresh_token if new_refresh_token is not None else refresh_token

    return OAuthTokens(
        access_token=access_token,
        refresh_token=final_refresh_token,
        expires_at=expires_at,
        token_type="Bearer",
    )


def get_valid_outlook_tokens(config: OutlookOAuthConfig, tokens: OAuthTokens) -> OAuthTokens:
    """Get valid tokens, refreshing if necessary.

    Args:
        config: Outlook OAuth configuration.
        tokens: Current tokens (may be expired).

    Returns:
        Valid OAuthTokens (either original or refreshed).

    Raises:
        AppError[EmailErrorCode]: If refresh fails.
    """
    if not is_token_expired(tokens):
        return tokens

    new_tokens = refresh_outlook_access_token(config, tokens["refresh_token"])
    hooks.save_outlook_tokens(new_tokens)
    return new_tokens


def authorize_outlook(config: OutlookOAuthConfig) -> OAuthTokens:
    """Run the full Outlook OAuth authorization flow.

    Opens browser for user consent and waits for callback.

    Args:
        config: Outlook OAuth configuration.

    Returns:
        OAuthTokens after successful authorization.

    Raises:
        AppError[EmailErrorCode]: If authorization fails.
    """
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)
    state = generate_state()

    auth_url = build_outlook_auth_url(config, code_challenge=code_challenge, state=state)
    hooks.open_browser(auth_url)

    hooks.console_output("\nAfter authorizing, copy the authorization code from the browser.")
    hooks.console_output("If using localhost redirect, the code is in the URL after 'code='")
    code = hooks.console_input("Enter authorization code: ").strip()

    if not code:
        msg = "No authorization code provided"
        raise AppError(EmailErrorCode.AUTH_FAILED, msg, http_status=401)

    tokens = exchange_outlook_code_for_tokens(
        config,
        code=code,
        code_verifier=code_verifier,
    )

    hooks.save_outlook_tokens(tokens)
    return tokens


def outlook_load_or_authorize() -> OAuthTokens:
    """Load cached tokens or run Outlook authorization flow.

    Returns:
        Valid OAuthTokens (from cache, refreshed, or new authorization).

    Raises:
        AppError[EmailErrorCode]: If credentials not found or authorization fails.
    """
    config = hooks.load_outlook_config()

    cached_tokens = hooks.load_outlook_tokens()
    if cached_tokens is not None:
        if is_token_expired(cached_tokens):
            try:
                new_tokens = refresh_outlook_access_token(config, cached_tokens["refresh_token"])
                hooks.save_outlook_tokens(new_tokens)
                return new_tokens
            except AppError:
                pass
        else:
            return cached_tokens

    return authorize_outlook(config)


__all__ = [
    "authorize_outlook",
    "build_outlook_auth_url",
    "exchange_outlook_code_for_tokens",
    "get_valid_outlook_tokens",
    "outlook_load_or_authorize",
    "refresh_outlook_access_token",
]
