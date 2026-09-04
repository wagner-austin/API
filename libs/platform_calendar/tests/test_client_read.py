"""Calendar client: factory, listing, event reads."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, CalendarErrorCode
from platform_core.json_utils import JSONTypeError, dump_json_str
from platform_core.oauth_types import OAuthTokens

from platform_calendar.client import google_calendar_client
from platform_calendar.fakes import (
    make_fake_http_get,
    make_raising_http_get,
)
from platform_calendar.testing import (
    hooks,
)


def _test_tokens() -> OAuthTokens:
    """Create test OAuth tokens."""
    return OAuthTokens(
        access_token="test_access_token",
        refresh_token="test_refresh_token",
        expires_at=9999999999,
        token_type="Bearer",
    )


class TestGoogleCalendarClientFactory:
    def test_creates_client(self) -> None:
        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        # Verify it implements the protocol by checking callable methods exist
        assert callable(client.list_calendars)
        assert callable(client.get_events)
        assert callable(client.create_event)


class TestListCalendars:
    def test_list_single_page(self) -> None:
        response = dump_json_str(
            {
                "items": [
                    {
                        "id": "primary",
                        "summary": "My Calendar",
                        "description": "Main calendar",
                        "primary": True,
                        "timeZone": "America/Los_Angeles",
                        "accessRole": "owner",
                    }
                ]
            }
        )
        hooks.http_get = make_fake_http_get(response)

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        calendars = client.list_calendars()

        assert len(calendars) == 1
        assert calendars[0]["id"] == "primary"
        assert calendars[0]["summary"] == "My Calendar"
        assert calendars[0]["primary"] is True

    def test_list_with_pagination(self) -> None:
        # Track which call we're on
        call_count = [0]

        def paginated_get(url: str, headers: dict[str, str]) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return dump_json_str(
                    {
                        "items": [
                            {
                                "id": "cal1",
                                "summary": "Cal 1",
                                "timeZone": "UTC",
                                "accessRole": "owner",
                            }
                        ],
                        "nextPageToken": "page2",
                    }
                )
            return dump_json_str(
                {
                    "items": [
                        {
                            "id": "cal2",
                            "summary": "Cal 2",
                            "timeZone": "UTC",
                            "accessRole": "owner",
                        }
                    ],
                }
            )

        hooks.http_get = paginated_get

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        calendars = client.list_calendars()

        assert len(calendars) == 2
        assert calendars[0]["id"] == "cal1"
        assert calendars[1]["id"] == "cal2"

    def test_list_items_not_list(self) -> None:
        response = dump_json_str({"items": "not a list"})
        hooks.http_get = make_fake_http_get(response)

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        calendars = client.list_calendars()

        assert len(calendars) == 0

    def test_list_skips_non_dict_items(self) -> None:
        response = dump_json_str(
            {
                "items": [
                    {"id": "valid", "summary": "Valid", "timeZone": "UTC", "accessRole": "owner"},
                    "invalid_item",
                    123,
                ]
            }
        )
        hooks.http_get = make_fake_http_get(response)

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        calendars = client.list_calendars()

        assert len(calendars) == 1
        assert calendars[0]["id"] == "valid"

    def test_list_handles_connection_error(self) -> None:
        hooks.http_get = make_raising_http_get(ConnectionError("Network error"))

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)

        with pytest.raises(AppError) as exc_info:
            client.list_calendars()
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.CALENDAR_API_ERROR
        assert "Request failed" in error.message

    def test_list_handles_os_error(self) -> None:
        hooks.http_get = make_raising_http_get(OSError("OS error"))

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)

        with pytest.raises(AppError) as exc_info:
            client.list_calendars()
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.CALENDAR_API_ERROR
        assert "Request failed" in error.message


class TestGetEvents:
    def test_get_events_single_page(self) -> None:
        response = dump_json_str(
            {
                "items": [
                    {
                        "id": "event1",
                        "summary": "Meeting",
                        "description": "Team meeting",
                        "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                        "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
                        "status": "confirmed",
                        "reminders": {"useDefault": True, "overrides": []},
                    }
                ]
            }
        )
        hooks.http_get = make_fake_http_get(response)

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        events = client.get_events(
            calendar_id="primary",
            time_min="2025-12-01T00:00:00Z",
            time_max="2025-12-31T23:59:59Z",
        )

        assert len(events) == 1
        assert events[0]["id"] == "event1"
        assert events[0]["summary"] == "Meeting"

    def test_get_events_with_pagination(self) -> None:
        call_count = [0]

        def paginated_get(url: str, headers: dict[str, str]) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return dump_json_str(
                    {
                        "items": [
                            {
                                "id": "event1",
                                "summary": "Event 1",
                                "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                                "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
                                "reminders": {"useDefault": True, "overrides": []},
                            }
                        ],
                        "nextPageToken": "page2",
                    }
                )
            return dump_json_str(
                {
                    "items": [
                        {
                            "id": "event2",
                            "summary": "Event 2",
                            "start": {"dateTime": "2025-12-27T10:00:00Z", "timeZone": "UTC"},
                            "end": {"dateTime": "2025-12-27T11:00:00Z", "timeZone": "UTC"},
                            "reminders": {"useDefault": True, "overrides": []},
                        }
                    ],
                }
            )

        hooks.http_get = paginated_get

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        events = client.get_events(
            calendar_id="primary",
            time_min="2025-12-01T00:00:00Z",
            time_max="2025-12-31T23:59:59Z",
        )

        assert len(events) == 2

    def test_get_events_items_not_list(self) -> None:
        response = dump_json_str({"items": "not a list"})
        hooks.http_get = make_fake_http_get(response)

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        events = client.get_events(
            calendar_id="primary",
            time_min="2025-12-01T00:00:00Z",
            time_max="2025-12-31T23:59:59Z",
        )

        assert len(events) == 0

    def test_get_events_adds_missing_fields(self) -> None:
        # Event without description, status, or reminders
        response = dump_json_str(
            {
                "items": [
                    {
                        "id": "event1",
                        "summary": "Event",
                        "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                        "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
                    }
                ]
            }
        )
        hooks.http_get = make_fake_http_get(response)

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        events = client.get_events(
            calendar_id="primary",
            time_min="2025-12-01T00:00:00Z",
            time_max="2025-12-31T23:59:59Z",
        )

        assert len(events) == 1
        assert events[0]["description"] == ""
        assert events[0]["status"] == "confirmed"

    def test_get_events_handles_reminders_dict(self) -> None:
        # Event with partial reminders
        response = dump_json_str(
            {
                "items": [
                    {
                        "id": "event1",
                        "summary": "Event",
                        "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                        "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
                        "reminders": {},  # Empty dict
                    }
                ]
            }
        )
        hooks.http_get = make_fake_http_get(response)

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        events = client.get_events(
            calendar_id="primary",
            time_min="2025-12-01T00:00:00Z",
            time_max="2025-12-31T23:59:59Z",
        )

        assert len(events) == 1

    def test_get_events_skips_non_dict_items(self) -> None:
        # Response with some non-dict items that should be skipped
        response = dump_json_str(
            {
                "items": [
                    {
                        "id": "valid_event",
                        "summary": "Valid Event",
                        "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                        "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
                        "reminders": {"useDefault": True, "overrides": []},
                    },
                    "invalid_string_item",
                    123,
                    None,
                ]
            }
        )
        hooks.http_get = make_fake_http_get(response)

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        events = client.get_events(
            calendar_id="primary",
            time_min="2025-12-01T00:00:00Z",
            time_max="2025-12-31T23:59:59Z",
        )

        # Only the valid dict item should be processed
        assert len(events) == 1
        assert events[0]["id"] == "valid_event"

    def test_get_events_non_dict_reminders_raises(self) -> None:
        """Test that non-dict reminders causes decode error (covers 239->242 branch)."""
        response = dump_json_str(
            {
                "items": [
                    {
                        "id": "event1",
                        "summary": "Event",
                        "description": "",
                        "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                        "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
                        "status": "confirmed",
                        "reminders": "not a dict",  # Invalid - should be a dict
                    }
                ]
            }
        )
        hooks.http_get = make_fake_http_get(response)

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)

        with pytest.raises(JSONTypeError, match="reminders must be an object"):
            client.get_events(
                calendar_id="primary",
                time_min="2025-12-01T00:00:00Z",
                time_max="2025-12-31T23:59:59Z",
            )


class TestGetEvent:
    def test_get_event_success(self) -> None:
        response = dump_json_str(
            {
                "id": "event123",
                "summary": "Test Event",
                "description": "Test description",
                "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
                "status": "confirmed",
                "reminders": {"useDefault": True, "overrides": []},
                "location": "Meeting Room A",
                "recurrence": ["RRULE:FREQ=WEEKLY;COUNT=10"],
            }
        )
        hooks.http_get = make_fake_http_get(response)

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        event = client.get_event(calendar_id="primary", event_id="event123")

        assert event["id"] == "event123"
        assert event["summary"] == "Test Event"
        assert event["location"] == "Meeting Room A"
        assert event["recurrence"] == ("RRULE:FREQ=WEEKLY;COUNT=10",)

    def test_get_event_normalizes_missing_fields(self) -> None:
        response = dump_json_str(
            {
                "id": "event123",
                "summary": "Test Event",
                "start": {"dateTime": "2025-12-26T10:00:00Z", "timeZone": "UTC"},
                "end": {"dateTime": "2025-12-26T11:00:00Z", "timeZone": "UTC"},
            }
        )
        hooks.http_get = make_fake_http_get(response)

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        event = client.get_event(calendar_id="primary", event_id="event123")

        assert event["description"] == ""
        assert event["status"] == "confirmed"
        assert event["location"] == ""
        assert event["recurrence"] == ()

    def test_get_event_connection_error(self) -> None:
        hooks.http_get = make_raising_http_get(ConnectionError("Network failed"))

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)

        with pytest.raises(AppError) as exc_info:
            client.get_event(calendar_id="primary", event_id="event123")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.CALENDAR_API_ERROR
