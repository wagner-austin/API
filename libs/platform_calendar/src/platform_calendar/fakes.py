"""Shared test doubles and hook factories for platform_calendar.

The hooks container and protocols live in
:mod:`platform_calendar.testing`.
"""

from __future__ import annotations

from platform_core.errors import AppError, CalendarErrorCode
from platform_core.oauth_types import (
    OAuthCredentials,
    OAuthTokens,
)

from platform_calendar.testing import (
    CalendarClientProtocol,
    ConsoleInputHook,
    ConsoleOutputHook,
    CurrentTimeHook,
    FileExistsHook,
    HttpDeleteHook,
    HttpGetHook,
    HttpPatchHook,
    HttpPostHook,
    LoadCredentialsHook,
    LoadTokensHook,
    ReadFileHook,
    WriteFileHook,
)
from platform_calendar.types import (
    CalendarEvent,
    CalendarListItem,
    EventDateTime,
    EventReminders,
    ReminderOverride,
)


class FakeCalendarClient(CalendarClientProtocol):
    """In-memory fake calendar client for testing."""

    def __init__(self) -> None:
        """Initialize the fake client with empty state."""
        self._calendars: list[CalendarListItem] = []
        self._events: dict[str, list[CalendarEvent]] = {}
        self._next_event_id: int = 1
        self._created_events: list[CalendarEvent] = []
        self._updated_events: list[CalendarEvent] = []
        self._deleted_events: list[tuple[str, str]] = []

    # -------------------------------------------------------------------------
    # Test Helpers
    # -------------------------------------------------------------------------

    def add_calendar(
        self,
        *,
        calendar_id: str,
        summary: str,
        description: str = "",
        primary: bool = False,
        time_zone: str = "UTC",
    ) -> None:
        """Add a fake calendar for testing.

        Args:
            calendar_id: Unique calendar ID.
            summary: Calendar name.
            description: Calendar description.
            primary: Whether this is the primary calendar.
            time_zone: Calendar timezone.
        """
        item = CalendarListItem(
            id=calendar_id,
            summary=summary,
            description=description,
            primary=primary,
            accessRole="owner",
            timeZone=time_zone,
        )
        self._calendars.append(item)
        self._events[calendar_id] = []

    def add_event(self, *, calendar_id: str, event: CalendarEvent) -> None:
        """Add a fake event for testing.

        Args:
            calendar_id: Calendar to add event to.
            event: Event to add.
        """
        if calendar_id not in self._events:
            self._events[calendar_id] = []
        self._events[calendar_id].append(event)

    def get_created_events(self) -> list[CalendarEvent]:
        """Get all events created via create_event().

        Returns:
            List of created events.
        """
        return list(self._created_events)

    def get_updated_events(self) -> list[CalendarEvent]:
        """Get all events updated via update_event().

        Returns:
            List of updated events.
        """
        return list(self._updated_events)

    def get_deleted_events(self) -> list[tuple[str, str]]:
        """Get all (calendar_id, event_id) pairs deleted via delete_event().

        Returns:
            List of (calendar_id, event_id) tuples.
        """
        return list(self._deleted_events)

    # -------------------------------------------------------------------------
    # Protocol Implementation
    # -------------------------------------------------------------------------

    def list_calendars(self) -> tuple[CalendarListItem, ...]:
        """List all calendars.

        Returns:
            Tuple of all calendars.
        """
        return tuple(self._calendars)

    def get_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
    ) -> CalendarEvent:
        """Get a single event by ID.

        Args:
            calendar_id: Calendar containing the event.
            event_id: ID of the event to retrieve.

        Returns:
            The CalendarEvent.

        Raises:
            AppError[CalendarErrorCode]: If event not found.
        """
        events = self._events.get(calendar_id, [])
        for event in events:
            if event["id"] == event_id:
                return event
        msg = f"Event '{event_id}' not found in calendar '{calendar_id}'"
        raise AppError(CalendarErrorCode.EVENT_NOT_FOUND, msg, http_status=404)

    def get_events(
        self,
        *,
        calendar_id: str,
        time_min: str,
        time_max: str,
    ) -> tuple[CalendarEvent, ...]:
        """Get events in a time range.

        Args:
            calendar_id: Calendar to query.
            time_min: Start of time range.
            time_max: End of time range.

        Returns:
            Tuple of events (simple filtering, no datetime comparison).
        """
        events = self._events.get(calendar_id, [])
        return tuple(events)

    def create_event(
        self,
        *,
        calendar_id: str,
        summary: str,
        description: str,
        start: EventDateTime,
        end: EventDateTime,
        reminders: tuple[int, ...],
        location: str = "",
        recurrence: tuple[str, ...] = (),
    ) -> CalendarEvent:
        """Create a new calendar event.

        Args:
            calendar_id: Calendar to create event in.
            summary: Event title.
            description: Event description.
            start: Event start time.
            end: Event end time.
            reminders: Reminder times in minutes.
            location: Event location string.
            recurrence: RRULE strings for recurring events.

        Returns:
            The created CalendarEvent.
        """
        event_id = f"fake_event_{self._next_event_id}"
        self._next_event_id += 1

        overrides: list[ReminderOverride] = []
        for minutes in reminders:
            overrides.append(ReminderOverride(method="popup", minutes=minutes))

        event = CalendarEvent(
            id=event_id,
            summary=summary,
            description=description,
            start=start,
            end=end,
            status="confirmed",
            reminders=EventReminders(
                useDefault=False,
                overrides=tuple(overrides),
            ),
            location=location,
            recurrence=recurrence,
        )

        if calendar_id not in self._events:
            self._events[calendar_id] = []
        self._events[calendar_id].append(event)
        self._created_events.append(event)

        return event

    def update_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
        summary: str | None = None,
        description: str | None = None,
        start: EventDateTime | None = None,
        end: EventDateTime | None = None,
        reminders: tuple[int, ...] | None = None,
        location: str | None = None,
        recurrence: tuple[str, ...] | None = None,
    ) -> CalendarEvent:
        """Update an existing calendar event.

        Args:
            calendar_id: Calendar containing the event.
            event_id: ID of the event to update.
            summary: New event title (None to keep existing).
            description: New description (None to keep existing).
            start: New start time (None to keep existing).
            end: New end time (None to keep existing).
            reminders: New reminder times (None to keep existing).
            location: New location (None to keep existing).
            recurrence: New recurrence rules (None to keep existing).

        Returns:
            The updated CalendarEvent.

        Raises:
            AppError[CalendarErrorCode]: If event not found.
        """
        events = self._events.get(calendar_id, [])
        for i, event in enumerate(events):
            if event["id"] == event_id:
                # Build new reminders if provided
                new_reminders: EventReminders
                if reminders is not None:
                    overrides: list[ReminderOverride] = []
                    for minutes in reminders:
                        overrides.append(ReminderOverride(method="popup", minutes=minutes))
                    new_reminders = EventReminders(
                        useDefault=False,
                        overrides=tuple(overrides),
                    )
                else:
                    new_reminders = event["reminders"]

                updated = CalendarEvent(
                    id=event["id"],
                    summary=summary if summary is not None else event["summary"],
                    description=description if description is not None else event["description"],
                    start=start if start is not None else event["start"],
                    end=end if end is not None else event["end"],
                    status=event["status"],
                    reminders=new_reminders,
                    location=location if location is not None else event["location"],
                    recurrence=recurrence if recurrence is not None else event["recurrence"],
                )
                self._events[calendar_id][i] = updated
                self._updated_events.append(updated)
                return updated

        # Event not found - raise error
        msg = f"Event '{event_id}' not found in calendar '{calendar_id}'"
        raise AppError(CalendarErrorCode.EVENT_NOT_FOUND, msg, http_status=404)

    def delete_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
    ) -> None:
        """Delete a calendar event.

        Args:
            calendar_id: Calendar containing the event.
            event_id: ID of the event to delete.
        """
        self._deleted_events.append((calendar_id, event_id))
        events = self._events.get(calendar_id, [])
        self._events[calendar_id] = [e for e in events if e["id"] != event_id]


# =============================================================================
# Factory Helpers for Tests
# =============================================================================


def make_fake_http_get(response: str) -> HttpGetHook:
    """Create a hook that returns a fixed response."""

    def _hook(url: str, headers: dict[str, str]) -> str:
        return response

    return _hook


def make_fake_http_post(response: str) -> HttpPostHook:
    """Create a hook that returns a fixed response."""

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        return response

    return _hook


def make_raising_http_get(exc: BaseException) -> HttpGetHook:
    """Create a hook that raises an exception."""

    def _hook(url: str, headers: dict[str, str]) -> str:
        raise exc

    return _hook


def make_raising_http_post(exc: BaseException) -> HttpPostHook:
    """Create a hook that raises an exception."""

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        raise exc

    return _hook


def make_fake_http_delete() -> HttpDeleteHook:
    """Create a hook that does nothing (successful delete)."""

    def _hook(url: str, headers: dict[str, str]) -> None:
        pass

    return _hook


def make_raising_http_delete(exc: BaseException) -> HttpDeleteHook:
    """Create a hook that raises an exception."""

    def _hook(url: str, headers: dict[str, str]) -> None:
        raise exc

    return _hook


def make_fake_http_patch(response: str) -> HttpPatchHook:
    """Create a hook that returns a fixed response.

    Args:
        response: Response body to return.

    Returns:
        HttpPatchHook that returns the fixed response.
    """

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        return response

    return _hook


def make_raising_http_patch(exc: BaseException) -> HttpPatchHook:
    """Create a hook that raises an exception.

    Args:
        exc: Exception to raise.

    Returns:
        HttpPatchHook that raises the exception.
    """

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        raise exc

    return _hook


def make_fake_tokens(tokens: OAuthTokens) -> LoadTokensHook:
    """Create a hook that returns fixed tokens."""

    def _hook() -> OAuthTokens | None:
        return tokens

    return _hook


def make_fake_no_tokens() -> LoadTokensHook:
    """Create a hook that returns None (no cached tokens)."""

    def _hook() -> OAuthTokens | None:
        return None

    return _hook


def make_fake_credentials(creds: OAuthCredentials) -> LoadCredentialsHook:
    """Create a hook that returns fixed credentials."""

    def _hook() -> OAuthCredentials:
        return creds

    return _hook


def make_fake_current_time(timestamp: int) -> CurrentTimeHook:
    """Create a hook that returns a fixed timestamp."""

    def _hook() -> int:
        return timestamp

    return _hook


def make_fake_file_system(
    files: dict[str, str],
) -> tuple[ReadFileHook, WriteFileHook, FileExistsHook]:
    """Create hooks that use an in-memory file system."""
    storage = dict(files)

    def _read(path: str) -> str:
        if path not in storage:
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)
        return storage[path]

    def _write(path: str, content: str) -> None:
        storage[path] = content

    def _exists(path: str) -> bool:
        return path in storage

    return _read, _write, _exists


def make_fake_console(inputs: list[str]) -> tuple[ConsoleOutputHook, ConsoleInputHook]:
    """Create hooks for fake console I/O.

    Args:
        inputs: List of strings to return from console_input in order.

    Returns:
        Tuple of (output_hook, input_hook).
    """
    outputs: list[str] = []
    input_index = [0]  # Use list for closure mutation

    def _output(message: str) -> None:
        outputs.append(message)

    def _input(prompt: str) -> str:
        if input_index[0] >= len(inputs):
            return ""
        result = inputs[input_index[0]]
        input_index[0] += 1
        return result

    return _output, _input


def make_fake_event(
    *,
    event_id: str = "test_event_1",
    summary: str = "Test Event",
    description: str = "Test description",
    start_datetime: str = "2025-12-26T14:00:00-08:00",
    end_datetime: str = "2025-12-26T15:00:00-08:00",
    time_zone: str = "America/Los_Angeles",
    status: str = "confirmed",
    location: str = "",
    recurrence: tuple[str, ...] = (),
) -> CalendarEvent:
    """Create a fake CalendarEvent for testing.

    Args:
        event_id: Event ID.
        summary: Event title.
        description: Event description.
        start_datetime: Start time in RFC 3339 format.
        end_datetime: End time in RFC 3339 format.
        time_zone: Timezone for start/end.
        status: Event status (confirmed, tentative, cancelled).
        location: Event location string.
        recurrence: RRULE strings for recurring events.

    Returns:
        CalendarEvent with the specified values.
    """
    event_status: str = status
    return CalendarEvent(
        id=event_id,
        summary=summary,
        description=description,
        start=EventDateTime(dateTime=start_datetime, timeZone=time_zone),
        end=EventDateTime(dateTime=end_datetime, timeZone=time_zone),
        status=(
            "confirmed"
            if event_status == "confirmed"
            else ("tentative" if event_status == "tentative" else "cancelled")
        ),
        reminders=EventReminders(useDefault=True, overrides=()),
        location=location,
        recurrence=recurrence,
    )


def make_fake_calendar(
    *,
    calendar_id: str = "primary",
    summary: str = "Primary Calendar",
    description: str = "",
    primary: bool = True,
    time_zone: str = "America/Los_Angeles",
) -> CalendarListItem:
    """Create a fake CalendarListItem for testing."""
    return CalendarListItem(
        id=calendar_id,
        summary=summary,
        description=description,
        primary=primary,
        accessRole="owner",
        timeZone=time_zone,
    )


__all__ = [
    "FakeCalendarClient",
    "make_fake_calendar",
    "make_fake_console",
    "make_fake_credentials",
    "make_fake_current_time",
    "make_fake_event",
    "make_fake_file_system",
    "make_fake_http_delete",
    "make_fake_http_get",
    "make_fake_http_patch",
    "make_fake_http_post",
    "make_fake_no_tokens",
    "make_fake_tokens",
    "make_raising_http_delete",
    "make_raising_http_get",
    "make_raising_http_patch",
    "make_raising_http_post",
]
