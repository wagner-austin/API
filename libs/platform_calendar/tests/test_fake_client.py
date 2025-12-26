"""Tests for FakeCalendarClient."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, CalendarErrorCode

from platform_calendar.testing import (
    FakeCalendarClient,
    make_fake_event,
)
from platform_calendar.types import EventDateTime


class TestFakeCalendarClient:
    def test_implements_protocol(self) -> None:
        client = FakeCalendarClient()
        # Verify protocol methods exist by checking they are callable
        assert callable(client.list_calendars)
        assert callable(client.get_events)
        assert callable(client.create_event)
        assert callable(client.update_event)
        assert callable(client.delete_event)

    def test_add_calendar(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")
        calendars = client.list_calendars()
        assert len(calendars) == 1
        assert calendars[0]["id"] == "primary"
        assert calendars[0]["summary"] == "My Calendar"

    def test_add_calendar_with_options(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(
            calendar_id="work",
            summary="Work",
            description="Work calendar",
            primary=False,
            time_zone="America/New_York",
        )
        calendars = client.list_calendars()
        assert calendars[0]["description"] == "Work calendar"
        assert calendars[0]["primary"] is False
        assert calendars[0]["timeZone"] == "America/New_York"

    def test_create_event(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        event = client.create_event(
            calendar_id="primary",
            summary="Test Event",
            description="Test description",
            start=start,
            end=end,
            reminders=(60, 1440),
        )

        assert event["id"].startswith("fake_event_")
        assert event["summary"] == "Test Event"
        assert event["status"] == "confirmed"
        assert len(event["reminders"]["overrides"]) == 2

    def test_get_created_events(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        client.create_event(
            calendar_id="primary",
            summary="Event 1",
            description="",
            start=start,
            end=end,
            reminders=(),
        )
        client.create_event(
            calendar_id="primary",
            summary="Event 2",
            description="",
            start=start,
            end=end,
            reminders=(),
        )

        created = client.get_created_events()
        assert len(created) == 2
        assert created[0]["summary"] == "Event 1"
        assert created[1]["summary"] == "Event 2"

    def test_get_events(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        client.create_event(
            calendar_id="primary",
            summary="Test",
            description="",
            start=start,
            end=end,
            reminders=(),
        )

        events = client.get_events(
            calendar_id="primary",
            time_min="2025-12-26T00:00:00Z",
            time_max="2025-12-27T00:00:00Z",
        )
        assert len(events) == 1

    def test_get_event_success(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        created = client.create_event(
            calendar_id="primary",
            summary="Test Event",
            description="Test description",
            start=start,
            end=end,
            reminders=(60,),
            location="Room A",
            recurrence=("RRULE:FREQ=DAILY",),
        )

        event = client.get_event(calendar_id="primary", event_id=created["id"])

        assert event["id"] == created["id"]
        assert event["summary"] == "Test Event"
        assert event["location"] == "Room A"
        assert event["recurrence"] == ("RRULE:FREQ=DAILY",)

    def test_get_event_not_found(self) -> None:
        client = FakeCalendarClient()
        with pytest.raises(AppError) as exc_info:
            client.get_event(calendar_id="primary", event_id="nonexistent")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.EVENT_NOT_FOUND

    def test_get_events_empty_calendar(self) -> None:
        client = FakeCalendarClient()
        events = client.get_events(
            calendar_id="nonexistent",
            time_min="2025-12-26T00:00:00Z",
            time_max="2025-12-27T00:00:00Z",
        )
        assert len(events) == 0

    def test_update_event(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        created = client.create_event(
            calendar_id="primary",
            summary="Original",
            description="Original desc",
            start=start,
            end=end,
            reminders=(),
        )

        updated = client.update_event(
            calendar_id="primary",
            event_id=created["id"],
            summary="Updated",
            description="Updated desc",
        )

        assert updated["summary"] == "Updated"
        assert updated["description"] == "Updated desc"

    def test_update_event_partial(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        created = client.create_event(
            calendar_id="primary",
            summary="Original",
            description="Original desc",
            start=start,
            end=end,
            reminders=(),
        )

        updated = client.update_event(
            calendar_id="primary",
            event_id=created["id"],
            summary="Updated",
        )

        assert updated["summary"] == "Updated"
        assert updated["description"] == "Original desc"

    def test_update_event_not_found(self) -> None:
        client = FakeCalendarClient()
        with pytest.raises(AppError) as exc_info:
            client.update_event(
                calendar_id="primary",
                event_id="nonexistent",
                summary="Test",
            )
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.EVENT_NOT_FOUND

    def test_get_updated_events(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        created = client.create_event(
            calendar_id="primary",
            summary="Original",
            description="",
            start=start,
            end=end,
            reminders=(),
        )

        client.update_event(
            calendar_id="primary",
            event_id=created["id"],
            summary="Updated",
        )

        updated = client.get_updated_events()
        assert len(updated) == 1
        assert updated[0]["summary"] == "Updated"

    def test_delete_event(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        created = client.create_event(
            calendar_id="primary",
            summary="To Delete",
            description="",
            start=start,
            end=end,
            reminders=(),
        )

        client.delete_event(calendar_id="primary", event_id=created["id"])

        deleted = client.get_deleted_events()
        assert len(deleted) == 1
        assert deleted[0] == ("primary", created["id"])

        events = client.get_events(
            calendar_id="primary",
            time_min="2025-12-26T00:00:00Z",
            time_max="2025-12-27T00:00:00Z",
        )
        assert len(events) == 0

    def test_add_event(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        event = make_fake_event(event_id="existing")
        client.add_event(calendar_id="primary", event=event)

        events = client.get_events(
            calendar_id="primary",
            time_min="2025-01-01T00:00:00Z",
            time_max="2025-12-31T00:00:00Z",
        )
        assert len(events) == 1
        assert events[0]["id"] == "existing"


class TestFakeCalendarClientBranches:
    def test_add_event_to_new_calendar(self) -> None:
        """Test add_event when calendar_id is not yet in _events dict."""
        client = FakeCalendarClient()
        # Don't add the calendar first - just add an event directly
        event = make_fake_event(event_id="test123")
        client.add_event(calendar_id="new_calendar", event=event)

        events = client.get_events(
            calendar_id="new_calendar",
            time_min="2025-01-01T00:00:00Z",
            time_max="2025-12-31T00:00:00Z",
        )
        assert len(events) == 1
        assert events[0]["id"] == "test123"

    def test_create_event_on_new_calendar(self) -> None:
        """Test create_event when calendar_id is not yet in _events dict."""
        client = FakeCalendarClient()
        # Don't add the calendar first - create an event directly
        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        event = client.create_event(
            calendar_id="new_calendar",
            summary="Test Event",
            description="Test",
            start=start,
            end=end,
            reminders=(),
        )

        assert event["id"].startswith("fake_event_")
        events = client.get_events(
            calendar_id="new_calendar",
            time_min="2025-01-01T00:00:00Z",
            time_max="2025-12-31T00:00:00Z",
        )
        assert len(events) == 1

    def test_update_event_iterates_through_multiple(self) -> None:
        """Test update_event when target is not the first event (covers loop branch)."""
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        # Create multiple events
        client.create_event(
            calendar_id="primary",
            summary="Event 1",
            description="First",
            start=start,
            end=end,
            reminders=(),
        )
        second_event = client.create_event(
            calendar_id="primary",
            summary="Event 2",
            description="Second",
            start=start,
            end=end,
            reminders=(),
        )
        client.create_event(
            calendar_id="primary",
            summary="Event 3",
            description="Third",
            start=start,
            end=end,
            reminders=(),
        )

        # Update the second event (requires iterating past the first)
        updated = client.update_event(
            calendar_id="primary",
            event_id=second_event["id"],
            summary="Updated Event 2",
        )

        assert updated["id"] == second_event["id"]
        assert updated["summary"] == "Updated Event 2"

    def test_get_event_iterates_through_multiple(self) -> None:
        """Test get_event when target is not the first event (covers loop branch)."""
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        # Create multiple events
        client.create_event(
            calendar_id="primary",
            summary="Event 1",
            description="First",
            start=start,
            end=end,
            reminders=(),
        )
        second_event = client.create_event(
            calendar_id="primary",
            summary="Event 2",
            description="Second",
            start=start,
            end=end,
            reminders=(),
        )
        client.create_event(
            calendar_id="primary",
            summary="Event 3",
            description="Third",
            start=start,
            end=end,
            reminders=(),
        )

        # Get the second event (requires iterating past the first)
        event = client.get_event(calendar_id="primary", event_id=second_event["id"])

        assert event["id"] == second_event["id"]
        assert event["summary"] == "Event 2"

    def test_update_event_with_reminders(self) -> None:
        """Test update_event with reminders parameter (covers reminders branch)."""
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        created = client.create_event(
            calendar_id="primary",
            summary="Original",
            description="Test",
            start=start,
            end=end,
            reminders=(30,),
        )

        # Update with new reminders
        updated = client.update_event(
            calendar_id="primary",
            event_id=created["id"],
            reminders=(60, 1440),
        )

        assert updated["id"] == created["id"]
        assert len(updated["reminders"]["overrides"]) == 2
        assert updated["reminders"]["overrides"][0]["minutes"] == 60
        assert updated["reminders"]["overrides"][1]["minutes"] == 1440
        assert updated["reminders"]["useDefault"] is False
