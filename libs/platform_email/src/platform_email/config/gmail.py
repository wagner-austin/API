"""Google Gmail API configuration.

Provides OAuth URLs, scopes, and API endpoints for Gmail API.
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# Default Paths
# =============================================================================

DEFAULT_GMAIL_CREDENTIALS_PATH: Path = Path.home() / ".google" / "email_credentials.json"
DEFAULT_GMAIL_TOKENS_PATH: Path = Path.home() / ".google" / "email_tokens.json"

# =============================================================================
# OAuth URLs
# =============================================================================

GMAIL_AUTH_URL: str = "https://accounts.google.com/o/oauth2/v2/auth"
GMAIL_TOKEN_URL: str = "https://oauth2.googleapis.com/token"

# =============================================================================
# API Base URL
# =============================================================================

GMAIL_API_BASE: str = "https://gmail.googleapis.com/gmail/v1"

# =============================================================================
# OAuth Scopes
# =============================================================================

GMAIL_EMAIL_SCOPES: tuple[str, str, str] = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
)

# =============================================================================
# Path Helpers
# =============================================================================


def get_gmail_credentials_path() -> Path:
    """Get the default Gmail credentials file path.

    Returns:
        Path to ~/.google/email_credentials.json
    """
    return DEFAULT_GMAIL_CREDENTIALS_PATH


def get_gmail_tokens_path() -> Path:
    """Get the default Gmail tokens file path.

    Returns:
        Path to ~/.google/email_tokens.json
    """
    return DEFAULT_GMAIL_TOKENS_PATH


__all__ = [
    "DEFAULT_GMAIL_CREDENTIALS_PATH",
    "DEFAULT_GMAIL_TOKENS_PATH",
    "GMAIL_API_BASE",
    "GMAIL_AUTH_URL",
    "GMAIL_EMAIL_SCOPES",
    "GMAIL_TOKEN_URL",
    "get_gmail_credentials_path",
    "get_gmail_tokens_path",
]
