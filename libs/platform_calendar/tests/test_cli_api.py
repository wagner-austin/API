"""CLI Graph API access and event formatting."""

from __future__ import annotations

from platform_core.json_utils import JSONObject

from platform_calendar.cli_api import (
    EventInfo,
    _collect_events,
    _create_event,
    _delete_event,
    _fetch_calendars,
    _fetch_events,
    _format_time,
    _parse_time,
    _sanitize,
)
from platform_calendar.testing import hooks

# =============================================================================
# Test Typed Argument Structures
# =============================================================================


class TestApiGet:
    """Tests for API GET function."""

    def test_api_get_hooked(self) -> None:
        """Test API GET with hook."""
        hooks.cli_api_get = lambda token, url: {"id": "123", "summary": "Test"}
        from platform_calendar.cli_api import (
            _api_get,
        )

        result = _api_get("token123", "https://example.com/api")
        assert result["id"] == "123"
        assert result["summary"] == "Test"


class TestApiPost:
    """Tests for API POST function."""

    def test_api_post_hooked(self) -> None:
        """Test API POST with hook."""
        hooks.cli_api_post = lambda token, url, body: {"id": "created", "body": body}
        from platform_calendar.cli_api import (
            _api_post,
        )

        result = _api_post("token123", "https://example.com/api", {"title": "Test"})
        assert result["id"] == "created"


class TestApiDelete:
    """Tests for API DELETE function."""

    def test_api_delete_hooked(self) -> None:
        """Test API DELETE with hook."""
        deleted_urls: list[str] = []

        def capture_delete(token: str, url: str) -> None:
            deleted_urls.append(url)

        hooks.cli_api_delete = capture_delete
        from platform_calendar.cli_api import (
            _api_delete,
        )

        _api_delete("token123", "https://example.com/api/123")
        assert "https://example.com/api/123" in deleted_urls


# =============================================================================
# Test Fetch Functions
# =============================================================================


class TestFetchCalendars:
    """Tests for _fetch_calendars."""

    def test_fetch_calendars(self) -> None:
        """Test fetching calendars."""
        hooks.cli_api_get = lambda token, url: {
            "items": [
                {"id": "cal1", "summary": "Calendar 1"},
                {"id": "cal2", "summary": "Calendar 2", "primary": True},
            ]
        }
        result = _fetch_calendars("token123")
        assert len(result) == 2
        assert result[0]["id"] == "cal1"
        assert result[1]["primary"] is True

    def test_fetch_calendars_no_items(self) -> None:
        """Test fetching calendars with no items."""
        hooks.cli_api_get = lambda token, url: {}
        result = _fetch_calendars("token123")
        assert result == []

    def test_fetch_calendars_items_not_list(self) -> None:
        """Test fetching calendars when items is not a list."""
        hooks.cli_api_get = lambda token, url: {"items": "not a list"}
        result = _fetch_calendars("token123")
        assert result == []


class TestFetchEvents:
    """Tests for _fetch_events."""

    def test_fetch_events(self) -> None:
        """Test fetching events."""
        hooks.cli_api_get = lambda token, url: {
            "items": [
                {
                    "id": "evt1",
                    "summary": "Event 1",
                    "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                }
            ]
        }
        result = _fetch_events("token123", "cal123", "2026-02-20")
        assert len(result) == 1
        assert result[0]["id"] == "evt1"

    def test_fetch_events_empty(self) -> None:
        """Test fetching events with no results."""
        hooks.cli_api_get = lambda token, url: {"items": []}
        result = _fetch_events("token123", "cal123", "2026-02-20")
        assert result == []

    def test_fetch_events_items_not_list(self) -> None:
        """Test fetching events when items is not a list."""
        hooks.cli_api_get = lambda token, url: {"items": "not a list"}
        result = _fetch_events("token123", "cal123", "2026-02-20")
        assert result == []


class TestCreateEvent:
    """Tests for _create_event."""

    def test_create_event(self) -> None:
        """Test creating event."""
        created: list[JSONObject] = []

        def capture_post(token: str, url: str, body: JSONObject) -> JSONObject:
            created.append(body)
            return {"id": "new_evt"}

        hooks.cli_api_post = capture_post
        result = _create_event(
            "token123",
            "cal123",
            "Test Meeting",
            "2026-02-20T14:00:00",
            "2026-02-20T15:00:00",
            location="Office",
        )
        assert result["id"] == "new_evt"
        assert len(created) == 1
        assert created[0]["summary"] == "Test Meeting"
        assert created[0]["location"] == "Office"

    def test_create_event_no_location(self) -> None:
        """Test creating event without location."""
        created: list[JSONObject] = []

        def capture_post(token: str, url: str, body: JSONObject) -> JSONObject:
            created.append(body)
            return {"id": "new_evt"}

        hooks.cli_api_post = capture_post
        _create_event(
            "token123",
            "cal123",
            "Test",
            "2026-02-20T14:00:00",
            "2026-02-20T15:00:00",
        )
        assert "location" not in created[0]


class TestDeleteEvent:
    """Tests for _delete_event."""

    def test_delete_event(self) -> None:
        """Test deleting event."""
        deleted_urls: list[str] = []
        hooks.cli_api_delete = lambda token, url: deleted_urls.append(url)
        _delete_event("token123", "cal123", "evt123")
        assert len(deleted_urls) == 1
        assert "evt123" in deleted_urls[0]


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestFormatTime:
    """Tests for _format_time."""

    def test_format_time_with_datetime(self) -> None:
        """Test formatting event with dateTime."""
        event: JSONObject = {"start": {"dateTime": "2026-02-20T14:30:00-08:00"}}
        time_str, sort_key = _format_time(event)
        assert time_str == "14:30"
        assert sort_key == "2026-02-20T14:30:00-08:00"

    def test_format_time_all_day(self) -> None:
        """Test formatting all-day event."""
        event: JSONObject = {"start": {"date": "2026-02-20"}}
        time_str, sort_key = _format_time(event)
        assert time_str == "all-day"
        assert sort_key == "00:00"

    def test_format_time_no_start(self) -> None:
        """Test formatting event with no start."""
        event: JSONObject = {}
        time_str, sort_key = _format_time(event)
        assert time_str == "all-day"
        assert sort_key == "00:00"

    def test_format_time_start_not_dict(self) -> None:
        """Test formatting event with start not a dict."""
        event: JSONObject = {"start": "not a dict"}
        time_str, sort_key = _format_time(event)
        assert time_str == "all-day"
        assert sort_key == "00:00"


class TestSanitize:
    """Tests for _sanitize."""

    def test_sanitize_ascii(self) -> None:
        """Test sanitizing ASCII text."""
        result = _sanitize("Hello World")
        assert result == "Hello World"

    def test_sanitize_removes_non_ascii(self) -> None:
        """Test sanitizing removes non-ASCII."""
        result = _sanitize("Hello\u2019World")
        assert result == "HelloWorld"


class TestParseTime:
    """Tests for _parse_time."""

    def test_parse_time_pm(self) -> None:
        """Test parsing PM time."""
        assert _parse_time("2pm") == "14:00"
        assert _parse_time("3PM") == "15:00"

    def test_parse_time_am(self) -> None:
        """Test parsing AM time."""
        assert _parse_time("9am") == "09:00"
        assert _parse_time("10AM") == "10:00"

    def test_parse_time_noon(self) -> None:
        """Test parsing noon."""
        assert _parse_time("12pm") == "12:00"

    def test_parse_time_midnight(self) -> None:
        """Test parsing midnight."""
        assert _parse_time("12am") == "00:00"

    def test_parse_time_24h(self) -> None:
        """Test parsing 24-hour format."""
        assert _parse_time("14:30") == "14:30"
        assert _parse_time("09:00") == "09:00"


# =============================================================================
# Test Event Info
# =============================================================================


class TestEventInfo:
    """Tests for EventInfo class."""

    def test_event_info_initialization(self) -> None:
        """Test EventInfo initialization."""
        info = EventInfo(
            sort_key="2026-02-20T14:00:00",
            time_str="14:00",
            summary="Meeting",
            calendar="Work",
            account="Personal",
            event_id="evt123",
            cal_id="cal123",
        )
        assert info.sort_key == "2026-02-20T14:00:00"
        assert info.time_str == "14:00"
        assert info.summary == "Meeting"
        assert info.calendar == "Work"
        assert info.account == "Personal"
        assert info.event_id == "evt123"
        assert info.cal_id == "cal123"


# =============================================================================
# Test Collect Events
# =============================================================================


class TestCollectEvents:
    """Tests for _collect_events."""

    def test_collect_events_from_multiple_accounts(self) -> None:
        """Test collecting events from multiple accounts."""

        def fake_get_env(key: str) -> str | None:
            if key == "GOOGLE_CALENDAR_ACCESS_TOKEN":
                return "token_personal"
            if key == "GOOGLE_CALENDAR_INTERNS_ACCESS_TOKEN":
                return "token_interns"
            return None

        call_count = [0]

        def fake_api_get(token: str, url: str) -> JSONObject:
            call_count[0] += 1
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": f"evt{call_count[0]}",
                        "summary": f"Event {call_count[0]}",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        hooks.cli_get_env = fake_get_env
        hooks.cli_api_get = fake_api_get

        result = _collect_events("2026-02-20")
        # Verify we got events from at least one account
        assert result != []
        assert result[0].summary.startswith("Event")

    def test_collect_events_no_tokens(self) -> None:
        """Test collecting events when no tokens available."""
        hooks.cli_get_env = lambda key: None
        result = _collect_events("2026-02-20")
        assert result == []

    def test_collect_events_skips_holidays(self) -> None:
        """Test collecting events skips holiday calendars."""

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {
                    "items": [
                        {"id": "main", "summary": "Main"},
                        {"id": "holiday", "summary": "US Holidays"},
                    ]
                }
            if "holiday" in url:
                return {"items": [{"id": "h1", "summary": "Holiday Event"}]}
            return {"items": []}

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get

        result = _collect_events("2026-02-20")
        # Holiday events should be filtered out
        assert all("Holiday" not in ev.summary for ev in result)


# =============================================================================
# Test Commands
# =============================================================================
