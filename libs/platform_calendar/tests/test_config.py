"""Tests for platform_calendar.config module."""

from __future__ import annotations

from platform_calendar.config import (
    CALENDAR_SCOPES,
    GOOGLE_AUTH_URL,
    GOOGLE_CALENDAR_API_BASE,
    GOOGLE_TOKEN_URL,
    get_competitions_path,
    get_credentials_path,
    get_tokens_path,
    load_credentials,
)
from platform_calendar.fakes import (
    make_fake_credentials,
)
from platform_calendar.testing import (
    hooks,
)
from platform_calendar.types import OAuthCredentials


class TestConstants:
    def test_calendar_scopes(self) -> None:
        assert len(CALENDAR_SCOPES) == 2
        assert "calendar.events" in CALENDAR_SCOPES[0]
        assert "calendar.readonly" in CALENDAR_SCOPES[1]

    def test_google_auth_url(self) -> None:
        assert "accounts.google.com" in GOOGLE_AUTH_URL

    def test_google_token_url(self) -> None:
        assert "oauth2.googleapis.com" in GOOGLE_TOKEN_URL

    def test_google_calendar_api_base(self) -> None:
        assert "googleapis.com/calendar/v3" in GOOGLE_CALENDAR_API_BASE


class TestGetPaths:
    def test_credentials_path(self) -> None:
        path = get_credentials_path()
        # Check Path-like attributes instead of isinstance
        assert path.name == "calendar_credentials.json"
        assert ".google" in str(path)

    def test_tokens_path(self) -> None:
        path = get_tokens_path()
        # Check Path-like attributes instead of isinstance
        assert path.name == "calendar_tokens.json"
        assert ".google" in str(path)

    def test_competitions_path(self) -> None:
        path = get_competitions_path()
        # Check Path-like attributes instead of isinstance
        assert path.name == "tracked.json"
        assert ".competitions" in str(path)


class TestLoadCredentials:
    def test_uses_hook(self) -> None:
        creds = OAuthCredentials(
            client_id="test_id",
            client_secret="test_secret",
            redirect_uri="http://localhost",
        )
        hooks.load_credentials = make_fake_credentials(creds)

        result = load_credentials()
        assert result["client_id"] == "test_id"
        assert result["client_secret"] == "test_secret"
