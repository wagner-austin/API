"""Tests for platform_calendar.types module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_calendar.types import (
    DEFAULT_REMINDERS,
    CalendarEvent,
    CalendarListItem,
    CompetitionsFile,
    EventDateTime,
    EventReminders,
    OAuthCredentials,
    OAuthTokens,
    ReminderOverride,
    TrackedCompetition,
    decode_calendar_event,
    decode_calendar_list_item,
    decode_competitions_file,
    decode_event_datetime,
    decode_event_reminders,
    decode_google_credentials_file,
    decode_google_token_response,
    decode_oauth_credentials,
    decode_oauth_tokens,
    decode_reminder_override,
    decode_tracked_competition,
    encode_calendar_event,
    encode_calendar_list_item,
    encode_competitions_file,
    encode_event_datetime,
    encode_event_reminders,
    encode_oauth_credentials,
    encode_oauth_tokens,
    encode_reminder_override,
    encode_tracked_competition,
)


class TestDefaultReminders:
    def test_default_reminders_values(self) -> None:
        assert DEFAULT_REMINDERS == (1440, 60)


class TestEventDateTime:
    def test_encode_event_datetime(self) -> None:
        dt = EventDateTime(
            dateTime="2025-12-26T14:00:00-08:00",
            timeZone="America/Los_Angeles",
        )
        encoded = encode_event_datetime(dt)
        assert encoded == {
            "dateTime": "2025-12-26T14:00:00-08:00",
            "timeZone": "America/Los_Angeles",
        }

    def test_decode_event_datetime(self) -> None:
        data: JSONObject = {
            "dateTime": "2025-12-26T14:00:00-08:00",
            "timeZone": "America/Los_Angeles",
        }
        dt = decode_event_datetime(data)
        assert dt["dateTime"] == "2025-12-26T14:00:00-08:00"
        assert dt["timeZone"] == "America/Los_Angeles"

    def test_decode_event_datetime_missing_field(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_event_datetime({"dateTime": "2025-12-26T14:00:00-08:00"})

    def test_roundtrip_event_datetime(self) -> None:
        original = EventDateTime(
            dateTime="2025-12-26T14:00:00Z",
            timeZone="UTC",
        )
        decoded = decode_event_datetime(encode_event_datetime(original))
        assert decoded == original


class TestReminderOverride:
    def test_encode_reminder_override(self) -> None:
        r = ReminderOverride(method="popup", minutes=60)
        encoded = encode_reminder_override(r)
        assert encoded == {"method": "popup", "minutes": 60}

    def test_decode_reminder_override_popup(self) -> None:
        data: JSONObject = {"method": "popup", "minutes": 60}
        r = decode_reminder_override(data)
        assert r["method"] == "popup"
        assert r["minutes"] == 60

    def test_decode_reminder_override_email(self) -> None:
        data: JSONObject = {"method": "email", "minutes": 1440}
        r = decode_reminder_override(data)
        assert r["method"] == "email"
        assert r["minutes"] == 1440

    def test_decode_reminder_override_invalid_method(self) -> None:
        data: JSONObject = {"method": "sms", "minutes": 60}
        with pytest.raises(JSONTypeError, match="must be email/popup"):
            decode_reminder_override(data)


class TestEventReminders:
    def test_encode_event_reminders(self) -> None:
        r = EventReminders(
            useDefault=False,
            overrides=(
                ReminderOverride(method="popup", minutes=60),
                ReminderOverride(method="email", minutes=1440),
            ),
        )
        encoded = encode_event_reminders(r)
        assert encoded["useDefault"] is False
        decoded = decode_event_reminders(encoded)
        assert len(decoded["overrides"]) == 2

    def test_decode_event_reminders(self) -> None:
        data: JSONObject = {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": 60},
            ],
        }
        r = decode_event_reminders(data)
        assert r["useDefault"] is False
        assert len(r["overrides"]) == 1
        assert r["overrides"][0]["method"] == "popup"

    def test_decode_event_reminders_invalid_override(self) -> None:
        data: JSONObject = {
            "useDefault": False,
            "overrides": ["not_a_dict"],
        }
        with pytest.raises(JSONTypeError, match="must be an object"):
            decode_event_reminders(data)

    def test_roundtrip_event_reminders(self) -> None:
        original = EventReminders(
            useDefault=True,
            overrides=(),
        )
        decoded = decode_event_reminders(encode_event_reminders(original))
        assert decoded == original


class TestCalendarEvent:
    def test_encode_calendar_event(self) -> None:
        event = CalendarEvent(
            id="event123",
            summary="Test Event",
            description="Test description",
            start=EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC"),
            end=EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC"),
            status="confirmed",
            reminders=EventReminders(useDefault=True, overrides=()),
            location="123 Main St",
            recurrence=("RRULE:FREQ=WEEKLY;COUNT=10",),
        )
        encoded = encode_calendar_event(event)
        assert encoded["id"] == "event123"
        assert encoded["summary"] == "Test Event"
        assert encoded["status"] == "confirmed"
        assert encoded["location"] == "123 Main St"
        assert encoded["recurrence"] == ["RRULE:FREQ=WEEKLY;COUNT=10"]

    def test_decode_calendar_event(self) -> None:
        data: JSONObject = {
            "id": "event123",
            "summary": "Test Event",
            "description": "Test description",
            "start": {"dateTime": "2025-12-26T14:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2025-12-26T15:00:00Z", "timeZone": "UTC"},
            "status": "confirmed",
            "reminders": {"useDefault": True, "overrides": []},
            "location": "Office",
            "recurrence": [],
        }
        event = decode_calendar_event(data)
        assert event["id"] == "event123"
        assert event["status"] == "confirmed"
        assert event["location"] == "Office"
        assert event["recurrence"] == ()

    def test_decode_calendar_event_with_recurrence(self) -> None:
        data: JSONObject = {
            "id": "event123",
            "summary": "Weekly Meeting",
            "description": "",
            "start": {"dateTime": "2025-12-26T14:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2025-12-26T15:00:00Z", "timeZone": "UTC"},
            "status": "confirmed",
            "reminders": {"useDefault": True, "overrides": []},
            "location": "",
            "recurrence": ["RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR"],
        }
        event = decode_calendar_event(data)
        assert event["recurrence"] == ("RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR",)

    def test_decode_calendar_event_defaults_location(self) -> None:
        """Test that missing location defaults to empty string."""
        data: JSONObject = {
            "id": "event123",
            "summary": "Test",
            "description": "",
            "start": {"dateTime": "2025-12-26T14:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2025-12-26T15:00:00Z", "timeZone": "UTC"},
            "status": "confirmed",
            "reminders": {"useDefault": True, "overrides": []},
        }
        event = decode_calendar_event(data)
        assert event["location"] == ""
        assert event["recurrence"] == ()

    def test_decode_calendar_event_tentative(self) -> None:
        data: JSONObject = {
            "id": "event123",
            "summary": "Test",
            "description": "",
            "start": {"dateTime": "2025-12-26T14:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2025-12-26T15:00:00Z", "timeZone": "UTC"},
            "status": "tentative",
            "reminders": {"useDefault": True, "overrides": []},
        }
        event = decode_calendar_event(data)
        assert event["status"] == "tentative"

    def test_decode_calendar_event_cancelled(self) -> None:
        data: JSONObject = {
            "id": "event123",
            "summary": "Test",
            "description": "",
            "start": {"dateTime": "2025-12-26T14:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2025-12-26T15:00:00Z", "timeZone": "UTC"},
            "status": "cancelled",
            "reminders": {"useDefault": True, "overrides": []},
        }
        event = decode_calendar_event(data)
        assert event["status"] == "cancelled"

    def test_decode_calendar_event_invalid_status(self) -> None:
        data: JSONObject = {
            "id": "event123",
            "summary": "Test",
            "description": "",
            "start": {"dateTime": "2025-12-26T14:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2025-12-26T15:00:00Z", "timeZone": "UTC"},
            "status": "invalid",
            "reminders": {"useDefault": True, "overrides": []},
        }
        with pytest.raises(JSONTypeError, match="must be confirmed/tentative/cancelled"):
            decode_calendar_event(data)

    def test_decode_calendar_event_invalid_start(self) -> None:
        data: JSONObject = {
            "id": "event123",
            "summary": "Test",
            "description": "",
            "start": "not_a_dict",
            "end": {"dateTime": "2025-12-26T15:00:00Z", "timeZone": "UTC"},
            "status": "confirmed",
            "reminders": {"useDefault": True, "overrides": []},
        }
        with pytest.raises(JSONTypeError, match="must be an object"):
            decode_calendar_event(data)

    def test_decode_calendar_event_invalid_recurrence(self) -> None:
        data: JSONObject = {
            "id": "event123",
            "summary": "Test",
            "description": "",
            "start": {"dateTime": "2025-12-26T14:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2025-12-26T15:00:00Z", "timeZone": "UTC"},
            "status": "confirmed",
            "reminders": {"useDefault": True, "overrides": []},
            "recurrence": "not_a_list",
        }
        with pytest.raises(JSONTypeError, match="must be a list"):
            decode_calendar_event(data)

    def test_decode_calendar_event_invalid_recurrence_item(self) -> None:
        data: JSONObject = {
            "id": "event123",
            "summary": "Test",
            "description": "",
            "start": {"dateTime": "2025-12-26T14:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2025-12-26T15:00:00Z", "timeZone": "UTC"},
            "status": "confirmed",
            "reminders": {"useDefault": True, "overrides": []},
            "recurrence": [123],
        }
        with pytest.raises(JSONTypeError, match="must be a string"):
            decode_calendar_event(data)

    def test_roundtrip_calendar_event(self) -> None:
        original = CalendarEvent(
            id="event123",
            summary="Test Event",
            description="Test description",
            start=EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC"),
            end=EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC"),
            status="confirmed",
            reminders=EventReminders(useDefault=True, overrides=()),
            location="Meeting Room A",
            recurrence=("RRULE:FREQ=DAILY;COUNT=5",),
        )
        decoded = decode_calendar_event(encode_calendar_event(original))
        assert decoded == original


class TestCalendarListItem:
    def test_encode_calendar_list_item(self) -> None:
        item = CalendarListItem(
            id="primary",
            summary="My Calendar",
            description="Personal calendar",
            primary=True,
            accessRole="owner",
            timeZone="America/Los_Angeles",
        )
        encoded = encode_calendar_list_item(item)
        assert encoded["id"] == "primary"
        assert encoded["primary"] is True
        assert encoded["accessRole"] == "owner"

    def test_decode_calendar_list_item(self) -> None:
        data: JSONObject = {
            "id": "primary",
            "summary": "My Calendar",
            "description": "",
            "primary": True,
            "accessRole": "owner",
            "timeZone": "America/Los_Angeles",
        }
        item = decode_calendar_list_item(data)
        assert item["id"] == "primary"
        assert item["accessRole"] == "owner"

    def test_decode_calendar_list_item_all_roles(self) -> None:
        for role in ("freeBusyReader", "reader", "writer", "owner"):
            data: JSONObject = {
                "id": "cal",
                "summary": "Cal",
                "description": "",
                "primary": False,
                "accessRole": role,
                "timeZone": "UTC",
            }
            item = decode_calendar_list_item(data)
            assert item["accessRole"] == role

    def test_decode_calendar_list_item_invalid_role(self) -> None:
        data: JSONObject = {
            "id": "cal",
            "summary": "Cal",
            "description": "",
            "primary": False,
            "accessRole": "admin",
            "timeZone": "UTC",
        }
        with pytest.raises(JSONTypeError, match="freeBusyReader/reader/writer/owner"):
            decode_calendar_list_item(data)


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


class TestGoogleTokenResponse:
    def test_decode_google_token_response(self) -> None:
        data: JSONObject = {
            "access_token": "access123",
            "refresh_token": "refresh456",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        resp = decode_google_token_response(data)
        assert resp["access_token"] == "access123"
        assert resp["refresh_token"] == "refresh456"
        assert resp["expires_in"] == 3600

    def test_decode_google_token_response_no_refresh(self) -> None:
        data: JSONObject = {
            "access_token": "access123",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        resp = decode_google_token_response(data)
        assert resp["refresh_token"] is None
