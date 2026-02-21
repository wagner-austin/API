"""OAuth configuration TypedDict definitions.

Provides OutlookOAuthConfig and GmailOAuthConfig types with encode/decode functions.

OAuth token types (OAuthCredentials, OAuthTokens, OAuthTokenResponse) are re-exported
from platform_core.oauth_types for consistency.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    require_str,
)

# Re-export OAuth types from platform_core
from platform_core.oauth_types import OAuthCredentials as OAuthCredentials
from platform_core.oauth_types import OAuthTokenResponse as OAuthTokenResponse
from platform_core.oauth_types import OAuthTokens as OAuthTokens
from platform_core.oauth_types import TokenType as TokenType
from platform_core.oauth_types import decode_oauth_credentials as decode_oauth_credentials
from platform_core.oauth_types import decode_oauth_token_response as decode_oauth_token_response
from platform_core.oauth_types import decode_oauth_tokens as decode_oauth_tokens
from platform_core.oauth_types import encode_oauth_credentials as encode_oauth_credentials
from platform_core.oauth_types import encode_oauth_token_response as encode_oauth_token_response
from platform_core.oauth_types import encode_oauth_tokens as encode_oauth_tokens

# =============================================================================
# OutlookOAuthConfig
# =============================================================================


class OutlookOAuthConfig(TypedDict):
    """Microsoft/Outlook OAuth configuration.

    Attributes:
        client_id: Azure application (client) ID.
        client_secret: Azure client secret.
        redirect_uri: OAuth redirect URI.
        tenant_id: Azure tenant ID (use "common" for multi-tenant).
    """

    client_id: str
    client_secret: str
    redirect_uri: str
    tenant_id: str


def encode_outlook_oauth_config(c: OutlookOAuthConfig) -> JSONObject:
    """Encode OutlookOAuthConfig to JSON-serializable dict.

    Args:
        c: OutlookOAuthConfig to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "client_id": c["client_id"],
        "client_secret": c["client_secret"],
        "redirect_uri": c["redirect_uri"],
        "tenant_id": c["tenant_id"],
    }
    return result


def decode_outlook_oauth_config(data: JSONObject) -> OutlookOAuthConfig:
    """Decode OutlookOAuthConfig from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated OutlookOAuthConfig.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return OutlookOAuthConfig(
        client_id=require_str(data, "client_id"),
        client_secret=require_str(data, "client_secret"),
        redirect_uri=require_str(data, "redirect_uri"),
        tenant_id=require_str(data, "tenant_id"),
    )


# =============================================================================
# GmailOAuthConfig
# =============================================================================


class GmailOAuthConfig(TypedDict):
    """Google/Gmail OAuth configuration.

    Attributes:
        client_id: Google OAuth client ID.
        client_secret: Google OAuth client secret.
        redirect_uri: OAuth redirect URI.
    """

    client_id: str
    client_secret: str
    redirect_uri: str


def encode_gmail_oauth_config(c: GmailOAuthConfig) -> JSONObject:
    """Encode GmailOAuthConfig to JSON-serializable dict.

    Args:
        c: GmailOAuthConfig to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "client_id": c["client_id"],
        "client_secret": c["client_secret"],
        "redirect_uri": c["redirect_uri"],
    }
    return result


def decode_gmail_oauth_config(data: JSONObject) -> GmailOAuthConfig:
    """Decode GmailOAuthConfig from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated GmailOAuthConfig.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return GmailOAuthConfig(
        client_id=require_str(data, "client_id"),
        client_secret=require_str(data, "client_secret"),
        redirect_uri=require_str(data, "redirect_uri"),
    )


__all__ = [
    "GmailOAuthConfig",
    "OAuthCredentials",
    "OAuthTokenResponse",
    "OAuthTokens",
    "OutlookOAuthConfig",
    "TokenType",
    "decode_gmail_oauth_config",
    "decode_oauth_credentials",
    "decode_oauth_token_response",
    "decode_oauth_tokens",
    "decode_outlook_oauth_config",
    "encode_gmail_oauth_config",
    "encode_oauth_credentials",
    "encode_oauth_token_response",
    "encode_oauth_tokens",
    "encode_outlook_oauth_config",
]
