"""Calendar client: event creation and updates."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, CalendarErrorCode
from platform_core.json_utils import dump_json_str
from platform_core.oauth_types import OAuthTokens

from platform_calendar.client import google_calendar_client
from platform_calendar.fakes import (
    make_fake_http_post,
    make_raising_http_patch,
    make_raising_http_post,
)
from platform_calendar.testing import (
    hooks,
)
from platform_calendar.types import EventDateTime


def _test_tokens() -> OAuthTokens:
    """Create test OAuth tokens."""
    return OAuthTokens(
        access_token="test_access_token",
        refresh_token="test_refresh_token",
        expires_at=9999999999,
        token_type="Bearer",
    )


class TestCreateEvent:
    def test_create_event_success(self) -> None:
        response = dump_json_str(
            {
                "id": "new_event",
                "summary": "Test Event",
                "description": "Test description",
                "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
                "status": "confirmed",
                "reminders": {"useDefault": False, "overrides": []},
            }
        )
        hooks.http_post = make_fake_http_post(response)

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        start: EventDateTime = {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"}
        end: EventDateTime = {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"}

        event = client.create_event(
            calendar_id="primary",
            summary="Test Event",
            description="Test description",
            start=start,
            end=end,
            reminders=(60, 1440),
        )

        assert event["id"] == "new_event"
        assert event["summary"] == "Test Event"

    def test_create_event_adds_missing_fields(self) -> None:
        # Response without description, status, reminders
        response = dump_json_str(
            {
                "id": "new_event",
                "summary": "Test Event",
                "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
            }
        )
        hooks.http_post = make_fake_http_post(response)

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        start: EventDateTime = {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"}
        end: EventDateTime = {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"}

        event = client.create_event(
            calendar_id="primary",
            summary="Test Event",
            description="Test description",
            start=start,
            end=end,
            reminders=(60,),
        )

        assert event["id"] == "new_event"

    def test_create_event_connection_error(self) -> None:
        hooks.http_post = make_raising_http_post(ConnectionError("Network error"))

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        start: EventDateTime = {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"}
        end: EventDateTime = {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"}

        with pytest.raises(AppError) as exc_info:
            client.create_event(
                calendar_id="primary",
                summary="Test",
                description="Test",
                start=start,
                end=end,
                reminders=(60,),
            )
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.CALENDAR_API_ERROR

    def test_create_event_os_error(self) -> None:
        hooks.http_post = make_raising_http_post(OSError("OS error"))

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        start: EventDateTime = {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"}
        end: EventDateTime = {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"}

        with pytest.raises(AppError) as exc_info:
            client.create_event(
                calendar_id="primary",
                summary="Test",
                description="Test",
                start=start,
                end=end,
                reminders=(60,),
            )
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.CALENDAR_API_ERROR

    def test_create_event_with_location(self) -> None:
        response = dump_json_str(
            {
                "id": "new_event",
                "summary": "Meeting",
                "description": "",
                "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
                "status": "confirmed",
                "reminders": {"useDefault": False, "overrides": []},
                "location": "Conference Room B",
                "recurrence": [],
            }
        )
        hooks.http_post = make_fake_http_post(response)

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        start: EventDateTime = {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"}
        end: EventDateTime = {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"}

        event = client.create_event(
            calendar_id="primary",
            summary="Meeting",
            description="",
            start=start,
            end=end,
            reminders=(),
            location="Conference Room B",
        )

        assert event["location"] == "Conference Room B"

    def test_create_event_with_recurrence(self) -> None:
        response = dump_json_str(
            {
                "id": "new_event",
                "summary": "Weekly Standup",
                "description": "",
                "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
                "status": "confirmed",
                "reminders": {"useDefault": False, "overrides": []},
                "location": "",
                "recurrence": ["RRULE:FREQ=WEEKLY;BYDAY=MO"],
            }
        )
        hooks.http_post = make_fake_http_post(response)

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        start: EventDateTime = {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"}
        end: EventDateTime = {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"}

        event = client.create_event(
            calendar_id="primary",
            summary="Weekly Standup",
            description="",
            start=start,
            end=end,
            reminders=(),
            recurrence=("RRULE:FREQ=WEEKLY;BYDAY=MO",),
        )

        assert event["recurrence"] == ("RRULE:FREQ=WEEKLY;BYDAY=MO",)


class TestUpdateEvent:
    def test_update_event_summary(self) -> None:
        def mock_patch(url: str, headers: dict[str, str], body: str) -> str:
            return dump_json_str(
                {
                    "id": "event1",
                    "summary": "New Title",
                    "description": "Description",
                    "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                    "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
                    "status": "confirmed",
                    "reminders": {"useDefault": True, "overrides": []},
                    "location": "",
                    "recurrence": [],
                }
            )

        hooks.http_patch = mock_patch

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        event = client.update_event(
            calendar_id="primary",
            event_id="event1",
            summary="New Title",
        )

        assert event["summary"] == "New Title"

    def test_update_event_description(self) -> None:
        def mock_patch(url: str, headers: dict[str, str], body: str) -> str:
            return dump_json_str(
                {
                    "id": "event1",
                    "summary": "Title",
                    "description": "New Desc",
                    "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                    "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
                    "status": "confirmed",
                    "reminders": {"useDefault": True, "overrides": []},
                    "location": "",
                    "recurrence": [],
                }
            )

        hooks.http_patch = mock_patch

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        event = client.update_event(
            calendar_id="primary",
            event_id="event1",
            description="New Desc",
        )

        assert event["description"] == "New Desc"

    def test_update_event_adds_missing_fields(self) -> None:
        def mock_patch(url: str, headers: dict[str, str], body: str) -> str:
            # Response without optional fields
            return dump_json_str(
                {
                    "id": "event1",
                    "summary": "Title",
                    "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                    "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
                }
            )

        hooks.http_patch = mock_patch

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        event = client.update_event(
            calendar_id="primary",
            event_id="event1",
        )

        assert event["description"] == ""
        assert event["status"] == "confirmed"
        assert event["location"] == ""
        assert event["recurrence"] == ()

    def test_update_event_start_and_end(self) -> None:
        def mock_patch(url: str, headers: dict[str, str], body: str) -> str:
            return dump_json_str(
                {
                    "id": "event1",
                    "summary": "Meeting",
                    "description": "",
                    "start": {"dateTime": "2025-12-27T10:00:00Z", "timeZone": "UTC"},
                    "end": {"dateTime": "2025-12-27T11:00:00Z", "timeZone": "UTC"},
                    "status": "confirmed",
                    "reminders": {"useDefault": True, "overrides": []},
                    "location": "",
                    "recurrence": [],
                }
            )

        hooks.http_patch = mock_patch

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        event = client.update_event(
            calendar_id="primary",
            event_id="event1",
            start=EventDateTime(dateTime="2025-12-27T10:00:00Z", timeZone="UTC"),
            end=EventDateTime(dateTime="2025-12-27T11:00:00Z", timeZone="UTC"),
        )

        assert event["start"]["dateTime"] == "2025-12-27T10:00:00Z"
        assert event["end"]["dateTime"] == "2025-12-27T11:00:00Z"

    def test_update_event_reminders(self) -> None:
        def mock_patch(url: str, headers: dict[str, str], body: str) -> str:
            return dump_json_str(
                {
                    "id": "event1",
                    "summary": "Meeting",
                    "description": "",
                    "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                    "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
                    "status": "confirmed",
                    "reminders": {
                        "useDefault": False,
                        "overrides": [{"method": "popup", "minutes": 30}],
                    },
                    "location": "",
                    "recurrence": [],
                }
            )

        hooks.http_patch = mock_patch

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        event = client.update_event(
            calendar_id="primary",
            event_id="event1",
            reminders=(30,),
        )

        assert event["reminders"]["useDefault"] is False
        assert len(event["reminders"]["overrides"]) == 1
        assert event["reminders"]["overrides"][0]["minutes"] == 30

    def test_update_event_location(self) -> None:
        def mock_patch(url: str, headers: dict[str, str], body: str) -> str:
            return dump_json_str(
                {
                    "id": "event1",
                    "summary": "Meeting",
                    "description": "",
                    "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                    "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
                    "status": "confirmed",
                    "reminders": {"useDefault": True, "overrides": []},
                    "location": "Conference Room A",
                    "recurrence": [],
                }
            )

        hooks.http_patch = mock_patch

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        event = client.update_event(
            calendar_id="primary",
            event_id="event1",
            location="Conference Room A",
        )

        assert event["location"] == "Conference Room A"

    def test_update_event_recurrence(self) -> None:
        def mock_patch(url: str, headers: dict[str, str], body: str) -> str:
            return dump_json_str(
                {
                    "id": "event1",
                    "summary": "Weekly Meeting",
                    "description": "",
                    "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                    "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
                    "status": "confirmed",
                    "reminders": {"useDefault": True, "overrides": []},
                    "location": "",
                    "recurrence": ["RRULE:FREQ=WEEKLY;COUNT=10"],
                }
            )

        hooks.http_patch = mock_patch

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        event = client.update_event(
            calendar_id="primary",
            event_id="event1",
            recurrence=("RRULE:FREQ=WEEKLY;COUNT=10",),
        )

        assert event["recurrence"] == ("RRULE:FREQ=WEEKLY;COUNT=10",)

    def test_update_event_connection_error(self) -> None:
        def mock_patch(url: str, headers: dict[str, str], body: str) -> str:
            raise ConnectionError("Network failed")

        hooks.http_patch = mock_patch

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)

        with pytest.raises(AppError) as exc_info:
            client.update_event(calendar_id="primary", event_id="event1", summary="Test")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.CALENDAR_API_ERROR
        assert "Request failed" in error.message

    def test_update_event_os_error(self) -> None:
        def mock_patch(url: str, headers: dict[str, str], body: str) -> str:
            raise OSError("IO failed")

        hooks.http_patch = mock_patch

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)

        with pytest.raises(AppError) as exc_info:
            client.update_event(calendar_id="primary", event_id="event1", summary="Test")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.CALENDAR_API_ERROR
        assert "Request failed" in error.message

    def test_update_event_invalid_json_response(self) -> None:
        def mock_patch(url: str, headers: dict[str, str], body: str) -> str:
            return "not valid json"

        hooks.http_patch = mock_patch

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)

        with pytest.raises(AppError) as exc_info:
            client.update_event(calendar_id="primary", event_id="event1", summary="Test")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.CALENDAR_API_ERROR
        assert "Invalid response" in error.message

    def test_update_event_http_error(self) -> None:
        """Test update_event with HTTP error response (e.g., 404)."""

        class FakeHTTPError(OSError):
            code = 404

            def read(self) -> bytes:
                return b'{"error": {"message": "Event not found"}}'

        hooks.http_patch = make_raising_http_patch(FakeHTTPError("Not found"))

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)

        with pytest.raises(AppError) as exc_info:
            client.update_event(calendar_id="primary", event_id="event1", summary="Test")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.EVENT_NOT_FOUND
