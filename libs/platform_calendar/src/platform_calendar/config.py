"""Configuration and path management for platform_calendar."""

from __future__ import annotations

from pathlib import Path

from platform_core.oauth_types import OAuthCredentials

from platform_calendar.testing import hooks

# =============================================================================
# Default Paths
# =============================================================================

DEFAULT_CREDENTIALS_PATH: Path = Path.home() / ".google" / "calendar_credentials.json"
DEFAULT_TOKENS_PATH: Path = Path.home() / ".google" / "calendar_tokens.json"
DEFAULT_COMPETITIONS_PATH: Path = Path.home() / ".competitions" / "tracked.json"

# =============================================================================
# OAuth Scopes
# =============================================================================

CALENDAR_SCOPES: tuple[str, str] = (
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
)

# =============================================================================
# Google API Endpoints
# =============================================================================

GOOGLE_AUTH_URL: str = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL: str = "https://oauth2.googleapis.com/token"
GOOGLE_CALENDAR_API_BASE: str = "https://www.googleapis.com/calendar/v3"


# =============================================================================
# Configuration Functions
# =============================================================================


def load_credentials(path: Path | None = None) -> OAuthCredentials:
    """Load OAuth credentials from file.

    Args:
        path: Path to credentials JSON file. Defaults to ~/.google/calendar_credentials.json

    Returns:
        OAuthCredentials containing client_id, client_secret, redirect_uri

    Raises:
        CredentialsNotFoundError: If credentials file not found or invalid
    """
    return hooks.load_credentials()


def get_credentials_path() -> Path:
    """Get the default credentials file path."""
    return DEFAULT_CREDENTIALS_PATH


def get_tokens_path() -> Path:
    """Get the default tokens file path."""
    return DEFAULT_TOKENS_PATH


def get_competitions_path() -> Path:
    """Get the default competitions file path."""
    return DEFAULT_COMPETITIONS_PATH
