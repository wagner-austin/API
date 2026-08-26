"""OAuth, competition, and Google response types."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_calendar.types import (
    CompetitionsFile,
    OAuthCredentials,
    OAuthTokens,
    TrackedCompetition,
    decode_competitions_file,
    decode_google_credentials_file,
    decode_oauth_credentials,
    decode_oauth_tokens,
    decode_tracked_competition,
    encode_competitions_file,
    encode_oauth_credentials,
    encode_oauth_tokens,
    encode_tracked_competition,
)


class TestOAuthCredentials:
    def test_encode_oauth_credentials(self) -> None:
        creds = OAuthCredentials(
            client_id="id123",
            client_secret="secret456",
            redirect_uri="http://localhost",
        )
        encoded = encode_oauth_credentials(creds)
        assert encoded["client_id"] == "id123"
        assert encoded["client_secret"] == "secret456"

    def test_decode_oauth_credentials(self) -> None:
        data: JSONObject = {
            "client_id": "id123",
            "client_secret": "secret456",
            "redirect_uri": "http://localhost",
        }
        creds = decode_oauth_credentials(data)
        assert creds["client_id"] == "id123"

    def test_roundtrip_oauth_credentials(self) -> None:
        original = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )
        decoded = decode_oauth_credentials(encode_oauth_credentials(original))
        assert decoded == original


class TestOAuthTokens:
    def test_encode_oauth_tokens(self) -> None:
        tokens = OAuthTokens(
            access_token="access123",
            refresh_token="refresh456",
            expires_at=1735200000,
            token_type="Bearer",
        )
        encoded = encode_oauth_tokens(tokens)
        assert encoded["access_token"] == "access123"
        assert encoded["token_type"] == "Bearer"

    def test_decode_oauth_tokens(self) -> None:
        data: JSONObject = {
            "access_token": "access123",
            "refresh_token": "refresh456",
            "expires_at": 1735200000,
            "token_type": "Bearer",
        }
        tokens = decode_oauth_tokens(data)
        assert tokens["access_token"] == "access123"
        assert tokens["token_type"] == "Bearer"

    def test_decode_oauth_tokens_invalid_type(self) -> None:
        data: JSONObject = {
            "access_token": "access123",
            "refresh_token": "refresh456",
            "expires_at": 1735200000,
            "token_type": "Basic",
        }
        with pytest.raises(JSONTypeError, match="must be Bearer"):
            decode_oauth_tokens(data)

    def test_roundtrip_oauth_tokens(self) -> None:
        original = OAuthTokens(
            access_token="access",
            refresh_token="refresh",
            expires_at=1735200000,
            token_type="Bearer",
        )
        decoded = decode_oauth_tokens(encode_oauth_tokens(original))
        assert decoded == original


class TestTrackedCompetition:
    def test_encode_tracked_competition(self) -> None:
        comp = TrackedCompetition(
            id="devpost-test",
            source="devpost",
            name="Test Competition",
            deadline="2025-12-26T22:00:00Z",
            url="https://devpost.com/test",
            project_path="libs/test",
            calendar_event_id="event123",
            reminders=(1440, 60),
        )
        encoded = encode_tracked_competition(comp)
        assert encoded["id"] == "devpost-test"
        assert encoded["source"] == "devpost"
        assert encoded["reminders"] == [1440, 60]

    def test_decode_tracked_competition(self) -> None:
        data: JSONObject = {
            "id": "kaggle-test",
            "source": "kaggle",
            "name": "Test",
            "deadline": "2025-12-26T22:00:00Z",
            "url": "https://kaggle.com/test",
            "project_path": None,
            "calendar_event_id": None,
            "reminders": [1440],
        }
        comp = decode_tracked_competition(data)
        assert comp["source"] == "kaggle"
        assert comp["project_path"] is None

    def test_decode_tracked_competition_all_sources(self) -> None:
        for source in ("kaggle", "devpost", "manual"):
            data: JSONObject = {
                "id": "test",
                "source": source,
                "name": "Test",
                "deadline": "2025-12-26T22:00:00Z",
                "url": "https://example.com",
                "project_path": None,
                "calendar_event_id": None,
                "reminders": [],
            }
            comp = decode_tracked_competition(data)
            assert comp["source"] == source

    def test_decode_tracked_competition_invalid_source(self) -> None:
        data: JSONObject = {
            "id": "test",
            "source": "github",
            "name": "Test",
            "deadline": "2025-12-26T22:00:00Z",
            "url": "https://example.com",
            "project_path": None,
            "calendar_event_id": None,
            "reminders": [],
        }
        with pytest.raises(JSONTypeError, match="kaggle/devpost/manual"):
            decode_tracked_competition(data)

    def test_decode_tracked_competition_invalid_reminders(self) -> None:
        data: JSONObject = {
            "id": "test",
            "source": "manual",
            "name": "Test",
            "deadline": "2025-12-26T22:00:00Z",
            "url": "https://example.com",
            "project_path": None,
            "calendar_event_id": None,
            "reminders": ["not_an_int"],
        }
        with pytest.raises(JSONTypeError, match="must be an int"):
            decode_tracked_competition(data)

    def test_roundtrip_tracked_competition(self) -> None:
        original = TrackedCompetition(
            id="test",
            source="manual",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
            project_path=None,
            calendar_event_id=None,
            reminders=(1440,),
        )
        decoded = decode_tracked_competition(encode_tracked_competition(original))
        assert decoded == original


class TestCompetitionsFile:
    def test_encode_competitions_file(self) -> None:
        comp = TrackedCompetition(
            id="test",
            source="manual",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
            project_path=None,
            calendar_event_id=None,
            reminders=(1440,),
        )
        file = CompetitionsFile(competitions=(comp,))
        encoded = encode_competitions_file(file)
        decoded = decode_competitions_file(encoded)
        assert len(decoded["competitions"]) == 1

    def test_decode_competitions_file(self) -> None:
        data: JSONObject = {
            "competitions": [
                {
                    "id": "test",
                    "source": "manual",
                    "name": "Test",
                    "deadline": "2025-12-26T22:00:00Z",
                    "url": "https://example.com",
                    "project_path": None,
                    "calendar_event_id": None,
                    "reminders": [],
                }
            ]
        }
        file = decode_competitions_file(data)
        assert len(file["competitions"]) == 1

    def test_decode_competitions_file_empty(self) -> None:
        data: JSONObject = {"competitions": []}
        file = decode_competitions_file(data)
        assert len(file["competitions"]) == 0

    def test_decode_competitions_file_invalid_competition(self) -> None:
        data: JSONObject = {"competitions": ["not_a_dict"]}
        with pytest.raises(JSONTypeError, match="must be an object"):
            decode_competitions_file(data)

    def test_roundtrip_competitions_file(self) -> None:
        comp = TrackedCompetition(
            id="test",
            source="devpost",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
            project_path="libs/test",
            calendar_event_id="event123",
            reminders=(1440, 60),
        )
        original = CompetitionsFile(competitions=(comp,))
        decoded = decode_competitions_file(encode_competitions_file(original))
        assert decoded == original


class TestGoogleCredentialsFile:
    def test_decode_google_credentials_file(self) -> None:
        data: JSONObject = {
            "installed": {
                "client_id": "123.apps.googleusercontent.com",
                "client_secret": "secret123",
                "redirect_uris": ["http://localhost"],
            }
        }
        creds = decode_google_credentials_file(data)
        assert creds["installed"]["client_id"] == "123.apps.googleusercontent.com"
        assert len(creds["installed"]["redirect_uris"]) == 1

    def test_decode_google_credentials_file_invalid_installed(self) -> None:
        data: JSONObject = {"installed": "not_a_dict"}
        with pytest.raises(JSONTypeError, match="must be an object"):
            decode_google_credentials_file(data)

    def test_decode_google_credentials_file_invalid_redirect_uri(self) -> None:
        data: JSONObject = {
            "installed": {
                "client_id": "123",
                "client_secret": "secret",
                "redirect_uris": [123],
            }
        }
        with pytest.raises(JSONTypeError, match="must be a string"):
            decode_google_credentials_file(data)
