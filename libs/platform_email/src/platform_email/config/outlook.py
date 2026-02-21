"""Microsoft Outlook/Graph API configuration.

Provides OAuth URLs, scopes, and API endpoints for Microsoft Graph Mail API.
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# Default Paths
# =============================================================================

DEFAULT_OUTLOOK_CREDENTIALS_PATH: Path = Path.home() / ".microsoft" / "email_credentials.json"
DEFAULT_OUTLOOK_TOKENS_PATH: Path = Path.home() / ".microsoft" / "email_tokens.json"

# =============================================================================
# OAuth URLs (tenant-based)
# =============================================================================


def outlook_auth_url(tenant_id: str) -> str:
    """Get Microsoft OAuth authorization URL for a tenant.

    Args:
        tenant_id: Azure tenant ID. Use "common" for multi-tenant apps.

    Returns:
        Authorization endpoint URL.
    """
    return f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize"


def outlook_token_url(tenant_id: str) -> str:
    """Get Microsoft OAuth token URL for a tenant.

    Args:
        tenant_id: Azure tenant ID. Use "common" for multi-tenant apps.

    Returns:
        Token endpoint URL.
    """
    return f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"


# =============================================================================
# API Base URL
# =============================================================================

OUTLOOK_API_BASE: str = "https://graph.microsoft.com/v1.0"

# =============================================================================
# OAuth Scopes
# =============================================================================

OUTLOOK_EMAIL_SCOPES: tuple[str, str, str, str] = (
    "https://graph.microsoft.com/Mail.Read",
    "https://graph.microsoft.com/Mail.Send",
    "https://graph.microsoft.com/Mail.ReadWrite",
    "offline_access",
)

# =============================================================================
# Path Helpers
# =============================================================================


def get_outlook_credentials_path() -> Path:
    """Get the default Outlook credentials file path.

    Returns:
        Path to ~/.microsoft/email_credentials.json
    """
    return DEFAULT_OUTLOOK_CREDENTIALS_PATH


def get_outlook_tokens_path() -> Path:
    """Get the default Outlook tokens file path.

    Returns:
        Path to ~/.microsoft/email_tokens.json
    """
    return DEFAULT_OUTLOOK_TOKENS_PATH


__all__ = [
    "DEFAULT_OUTLOOK_CREDENTIALS_PATH",
    "DEFAULT_OUTLOOK_TOKENS_PATH",
    "OUTLOOK_API_BASE",
    "OUTLOOK_EMAIL_SCOPES",
    "get_outlook_credentials_path",
    "get_outlook_tokens_path",
    "outlook_auth_url",
    "outlook_token_url",
]
