"""Calendar client: deletes and error handling."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, CalendarErrorCode

from platform_calendar.client import _GoogleCalendarClient, google_calendar_client
from platform_calendar.fakes import (
    make_fake_http_delete,
    make_fake_http_get,
    make_fake_http_post,
    make_raising_http_delete,
    make_raising_http_get,
    make_raising_http_post,
)
from platform_calendar.testing import (
    hooks,
)
from platform_calendar.types import EventDateTime, OAuthTokens


def _test_tokens() -> OAuthTokens:
    """Create test OAuth tokens."""
    return OAuthTokens(
        access_token="test_access_token",
        refresh_token="test_refresh_token",
        expires_at=9999999999,
        token_type="Bearer",
    )


class TestDeleteEvent:
    def test_delete_event_success(self) -> None:
        """Test successful event deletion."""
        hooks.http_delete = make_fake_http_delete()

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)
        # Should not raise
        client.delete_event(calendar_id="primary", event_id="event1")

    def test_delete_event_connection_error(self) -> None:
        """Test delete event with connection error."""
        hooks.http_delete = make_raising_http_delete(ConnectionError("Network failed"))

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)

        with pytest.raises(AppError) as exc_info:
            client.delete_event(calendar_id="primary", event_id="event1")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.CALENDAR_API_ERROR
        assert "Delete request failed" in error.message

    def test_delete_event_os_error(self) -> None:
        """Test delete event with generic OS error."""
        hooks.http_delete = make_raising_http_delete(OSError("IO failed"))

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)

        with pytest.raises(AppError) as exc_info:
            client.delete_event(calendar_id="primary", event_id="event1")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.CALENDAR_API_ERROR
        assert "Delete request failed" in error.message

    def test_delete_event_http_error_404(self) -> None:
        """Test delete event with HTTP 404 error."""

        class FakeHTTPError(OSError):
            code = 404

            def read(self) -> bytes:
                return b'{"error": {"message": "Event not found"}}'

        hooks.http_delete = make_raising_http_delete(FakeHTTPError("Not found"))

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)

        with pytest.raises(AppError) as exc_info:
            client.delete_event(calendar_id="primary", event_id="event1")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.EVENT_NOT_FOUND


class TestAPIErrors:
    def test_get_invalid_json_response(self) -> None:
        hooks.http_get = make_fake_http_get("not json")

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)

        with pytest.raises(AppError) as exc_info:
            client.list_calendars()
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.CALENDAR_API_ERROR
        assert "Invalid response" in error.message

    def test_post_invalid_json_response(self) -> None:
        hooks.http_post = make_fake_http_post("not json")

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
        assert "Invalid response" in error.message

    def test_get_not_object_response(self) -> None:
        hooks.http_get = make_fake_http_get("[]")

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)

        with pytest.raises(AppError) as exc_info:
            client.list_calendars()
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.CALENDAR_API_ERROR
        assert "Invalid response" in error.message


class TestHTTPErrorHandling:
    def test_calendar_not_found_error(self) -> None:
        class FakeHTTPError(OSError):
            code = 404

            def read(self) -> bytes:
                return b"Not found"

        hooks.http_get = make_raising_http_get(FakeHTTPError("Not found"))

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)

        with pytest.raises(AppError) as exc_info:
            client.list_calendars()  # Contains "calendar" in endpoint
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.CALENDAR_NOT_FOUND

    def test_event_not_found_error(self) -> None:
        """Test 404 on event endpoint returns EVENT_NOT_FOUND."""

        class FakeHTTPError(OSError):
            code = 404

            def read(self) -> bytes:
                return b"Not found"

        hooks.http_get = make_raising_http_get(FakeHTTPError("Not found"))

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)

        with pytest.raises(AppError) as exc_info:
            client.get_events(
                calendar_id="primary",
                time_min="2025-01-01T00:00:00Z",
                time_max="2025-12-31T23:59:59Z",
            )
        error: AppError[CalendarErrorCode] = exc_info.value
        # The endpoint /calendars/{id}/events contains "events" -> EVENT_NOT_FOUND
        assert error.code == CalendarErrorCode.EVENT_NOT_FOUND

    def test_404_fallback_without_calendar_or_events(self) -> None:
        """Test 404 fallback when context has neither 'calendar' nor 'events'."""
        client = _GoogleCalendarClient(access_token="test_token")

        # Call _handle_error directly with a context that has neither keyword
        with pytest.raises(AppError) as exc_info:
            client._handle_error(404, "Not found", "/some/other/path")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.EVENT_NOT_FOUND
        assert "Not found" in error.message

    def test_other_api_error(self) -> None:
        class FakeHTTPError(OSError):
            code = 500

            def read(self) -> bytes:
                return b"Server error"

        hooks.http_get = make_raising_http_get(FakeHTTPError("Server error"))

        tokens = _test_tokens()
        client = google_calendar_client(tokens=tokens)

        with pytest.raises(AppError) as exc_info:
            client.list_calendars()
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.CALENDAR_API_ERROR

    def test_post_http_error(self) -> None:
        class FakeHTTPError(OSError):
            code = 403

            def read(self) -> bytes:
                return b"Forbidden"

        hooks.http_post = make_raising_http_post(FakeHTTPError("Forbidden"))

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


class TestHandleError:
    def test_event_not_found_404_without_calendar_in_context(self) -> None:
        # Test EVENT_NOT_FOUND by calling _handle_error directly
        # with a context that doesn't contain "calendar"
        client = _GoogleCalendarClient(access_token="test")
        with pytest.raises(AppError) as exc_info:
            client._handle_error(404, "Not found", "/events/123")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.EVENT_NOT_FOUND
        assert "Event not found" in error.message
