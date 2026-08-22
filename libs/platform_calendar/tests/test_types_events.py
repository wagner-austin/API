"""Event, reminder, and calendar list types."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_calendar.types import (
    DEFAULT_REMINDERS,
    CalendarEvent,
    CalendarListItem,
    EventDateTime,
    EventReminders,
    ReminderOverride,
    decode_calendar_event,
    decode_calendar_list_item,
    decode_event_datetime,
    decode_event_reminders,
    decode_reminder_override,
    encode_calendar_event,
    encode_calendar_list_item,
    encode_event_datetime,
    encode_event_reminders,
    encode_reminder_override,
    is_all_day_event,
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

    def test_decode_event_datetime_missing_timezone_defaults_utc(self) -> None:
        dt = decode_event_datetime({"dateTime": "2025-12-26T14:00:00-08:00"})
        assert dt["dateTime"] == "2025-12-26T14:00:00-08:00"
        assert dt["timeZone"] == "UTC"

    def test_decode_event_datetime_missing_both_fields(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_event_datetime({})

    def test_decode_event_datetime_all_day(self) -> None:
        dt = decode_event_datetime({"date": "2025-12-26"})
        assert dt["date"] == "2025-12-26"
        assert "dateTime" not in dt

    def test_is_all_day_event_true(self) -> None:
        dt = EventDateTime(date="2025-12-26")
        assert is_all_day_event(dt) is True

    def test_is_all_day_event_false(self) -> None:
        dt = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        assert is_all_day_event(dt) is False

    def test_encode_event_datetime_all_day(self) -> None:
        dt = EventDateTime(date="2025-12-26")
        encoded = encode_event_datetime(dt)
        assert encoded == {"date": "2025-12-26"}

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
