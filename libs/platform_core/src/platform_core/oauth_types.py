"""OAuth 2.0 TypedDict definitions with encode/decode functions.

Provides reusable OAuth types for any OAuth 2.0 integration.
Google-specific types (GoogleCredentialsFile, etc.) remain in platform_calendar.

All TypedDicts use immutable semantics with proper validation on decode.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    optional_str,
    require_int,
    require_str,
)

# =============================================================================
# Literal Types
# =============================================================================

TokenType = Literal["Bearer"]


# =============================================================================
# Validation Helpers
# =============================================================================


def _require_token_type(obj: JSONObject, key: str) -> TokenType:
    """Extract and validate TokenType from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Key to extract.

    Returns:
        Validated TokenType literal.

    Raises:
        JSONTypeError: If value is not a valid TokenType.
    """
    value = require_str(obj, key)
    if value == "Bearer":
        return "Bearer"
    raise JSONTypeError(f"Field '{key}' must be Bearer, got '{value}'")


# =============================================================================
# OAuth Credentials
# =============================================================================


class OAuthCredentials(TypedDict):
    """OAuth 2.0 client credentials.

    Attributes:
        client_id: OAuth client ID from provider.
        client_secret: OAuth client secret from provider.
        redirect_uri: Redirect URI for authorization callback.
    """

    client_id: str
    client_secret: str
    redirect_uri: str


def encode_oauth_credentials(credentials: OAuthCredentials) -> JSONObject:
    """Encode OAuthCredentials to JSON-serializable dict.

    Args:
        credentials: OAuthCredentials to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "client_id": credentials["client_id"],
        "client_secret": credentials["client_secret"],
        "redirect_uri": credentials["redirect_uri"],
    }
    return result


def decode_oauth_credentials(data: JSONObject) -> OAuthCredentials:
    """Decode OAuthCredentials from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated OAuthCredentials.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return OAuthCredentials(
        client_id=require_str(data, "client_id"),
        client_secret=require_str(data, "client_secret"),
        redirect_uri=require_str(data, "redirect_uri"),
    )


# =============================================================================
# OAuth Tokens
# =============================================================================


class OAuthTokens(TypedDict):
    """OAuth 2.0 access and refresh tokens.

    Attributes:
        access_token: Bearer token for API requests.
        refresh_token: Token used to obtain new access tokens.
        expires_at: Unix timestamp when access_token expires.
        token_type: Token type, always "Bearer".
    """

    access_token: str
    refresh_token: str
    expires_at: int
    token_type: TokenType


def encode_oauth_tokens(tokens: OAuthTokens) -> JSONObject:
    """Encode OAuthTokens to JSON-serializable dict.

    Args:
        tokens: OAuthTokens to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "expires_at": tokens["expires_at"],
        "token_type": tokens["token_type"],
    }
    return result


def decode_oauth_tokens(data: JSONObject) -> OAuthTokens:
    """Decode OAuthTokens from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated OAuthTokens.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return OAuthTokens(
        access_token=require_str(data, "access_token"),
        refresh_token=require_str(data, "refresh_token"),
        expires_at=require_int(data, "expires_at"),
        token_type=_require_token_type(data, "token_type"),
    )


# =============================================================================
# OAuth Token Response (from token endpoint)
# =============================================================================


class OAuthTokenResponse(TypedDict):
    """Response from OAuth token endpoint.

    This represents the raw response from an OAuth provider's token endpoint.
    The refresh_token is optional since it's only returned on initial auth,
    not on token refresh.

    Attributes:
        access_token: The access token issued by the authorization server.
        refresh_token: The refresh token (only present on initial authorization).
        expires_in: Lifetime in seconds of the access token.
        token_type: Type of token issued (typically "Bearer").
    """

    access_token: str
    refresh_token: str | None
    expires_in: int
    token_type: str


def encode_oauth_token_response(response: OAuthTokenResponse) -> JSONObject:
    """Encode OAuthTokenResponse to JSON-serializable dict.

    Args:
        response: OAuthTokenResponse to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "access_token": response["access_token"],
        "refresh_token": response["refresh_token"],
        "expires_in": response["expires_in"],
        "token_type": response["token_type"],
    }
    return result


def decode_oauth_token_response(data: JSONObject) -> OAuthTokenResponse:
    """Decode OAuthTokenResponse from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated OAuthTokenResponse.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return OAuthTokenResponse(
        access_token=require_str(data, "access_token"),
        refresh_token=optional_str(data, "refresh_token"),
        expires_in=require_int(data, "expires_in"),
        token_type=require_str(data, "token_type"),
    )


__all__ = [
    "OAuthCredentials",
    "OAuthTokenResponse",
    "OAuthTokens",
    "TokenType",
    "decode_oauth_credentials",
    "decode_oauth_token_response",
    "decode_oauth_tokens",
    "encode_oauth_credentials",
    "encode_oauth_token_response",
    "encode_oauth_tokens",
]
