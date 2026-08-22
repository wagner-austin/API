"""CLI command handlers."""

from __future__ import annotations

import io
from datetime import datetime

from platform_core.json_utils import JSONObject
from rich.console import Console

from platform_calendar.cli import (
    CreateArgs,
    DeleteArgs,
    ListArgs,
    _get_delete_choice,
    _show_events_for_delete,
    cmd_calendars,
    cmd_create,
    cmd_delete,
    cmd_list,
    cmd_tomorrow,
    cmd_week,
)
from platform_calendar.cli_api import (
    EventInfo,
)
from platform_calendar.testing import hooks

# =============================================================================
# Test Typed Argument Structures
# =============================================================================


class TestCmdList:
    """Tests for cmd_list command."""

    def test_cmd_list_with_events(self) -> None:
        """Test listing events."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt1",
                        "summary": "Test Event",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get

        cmd_list(ListArgs(date="2026-02-20"))
        result = output.getvalue()
        assert "Test Event" in result

    def test_cmd_list_no_events(self) -> None:
        """Test listing when no events."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        hooks.cli_get_env = lambda key: None

        cmd_list(ListArgs(date="2026-02-20"))
        result = output.getvalue()
        assert "No events found" in result


class TestCmdCalendars:
    """Tests for cmd_calendars command."""

    def test_cmd_calendars_with_calendars(self) -> None:
        """Test listing calendars."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            return {
                "items": [
                    {"id": "primary", "summary": "Main", "primary": True},
                    {"id": "work", "summary": "Work"},
                ]
            }

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get

        cmd_calendars()
        result = output.getvalue()
        assert "Main" in result
        # Primary marker is present (rich formatting adds escape codes)
        assert "primary" in result

    def test_cmd_calendars_no_token(self) -> None:
        """Test listing calendars when no token."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        hooks.cli_get_env = lambda key: None

        cmd_calendars()
        result = output.getvalue()
        assert "No token" in result


class TestCmdCreate:
    """Tests for cmd_create command."""

    def test_cmd_create_success(self) -> None:
        """Test creating event successfully."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_post = lambda token, url, body: {"id": "created"}

        cmd_create(
            CreateArgs(
                title="New Meeting",
                time="14:00",
                date="2026-02-20",
                duration=60,
                location="Office",
                account="Personal",
            )
        )
        result = output.getvalue()
        assert "Created" in result
        assert "New Meeting" in result
        assert "Location" in result

    def test_cmd_create_no_token(self) -> None:
        """Test creating event when no token."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        hooks.cli_get_env = lambda key: None

        cmd_create(
            CreateArgs(
                title="Test",
                time="14:00",
                date="2026-02-20",
                duration=60,
                location="",
                account="Personal",
            )
        )
        result = output.getvalue()
        assert "Unknown account" in result

    def test_cmd_create_no_location(self) -> None:
        """Test creating event without location doesn't print location line."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_post = lambda token, url, body: {"id": "created"}

        cmd_create(
            CreateArgs(
                title="TestEvent",
                time="14:00",
                date="2026-02-20",
                duration=60,
                location="",
                account="Personal",
            )
        )
        result = output.getvalue()
        assert "Created" in result
        # No "Location:" label printed when location is empty
        assert "Location:" not in result


class TestCmdDelete:
    """Tests for cmd_delete command."""

    def test_cmd_delete_no_events(self) -> None:
        """Test deleting when no events."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        hooks.cli_get_env = lambda key: None

        cmd_delete(DeleteArgs(date="2026-02-20"))
        result = output.getvalue()
        assert "No events to delete" in result

    def test_cmd_delete_cancelled(self) -> None:
        """Test deleting when user cancels."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt1",
                        "summary": "Event",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get
        hooks.cli_prompt_ask = lambda msg: "q"

        cmd_delete(DeleteArgs(date="2026-02-20"))
        result = output.getvalue()
        assert "Events for" in result

    def test_cmd_delete_invalid_input(self) -> None:
        """Test deleting with invalid input."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt1",
                        "summary": "Event",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get
        hooks.cli_prompt_ask = lambda msg: "invalid"

        cmd_delete(DeleteArgs(date="2026-02-20"))
        result = output.getvalue()
        assert "Invalid input" in result

    def test_cmd_delete_success(self) -> None:
        """Test deleting event successfully."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt1",
                        "summary": "Test Event",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        deleted: list[str] = []

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get
        hooks.cli_api_delete = lambda token, url: deleted.append(url)
        hooks.cli_prompt_ask = lambda msg: "1"
        hooks.cli_confirm_ask = lambda msg: True

        cmd_delete(DeleteArgs(date="2026-02-20"))
        result = output.getvalue()
        assert "Deleted" in result
        assert len(deleted) == 1

    def test_cmd_delete_not_confirmed(self) -> None:
        """Test deleting when not confirmed."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt1",
                        "summary": "Event",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get
        hooks.cli_prompt_ask = lambda msg: "1"
        hooks.cli_confirm_ask = lambda msg: False

        cmd_delete(DeleteArgs(date="2026-02-20"))
        result = output.getvalue()
        assert "Deleted" not in result


class TestCmdTomorrow:
    """Tests for cmd_tomorrow command."""

    def test_cmd_tomorrow(self) -> None:
        """Test showing tomorrow's events."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        hooks.cli_get_env = lambda key: None
        hooks.cli_get_now = lambda: datetime(2026, 2, 20, 10, 0, 0)

        cmd_tomorrow()
        result = output.getvalue()
        assert "February 21" in result


class TestCmdWeek:
    """Tests for cmd_week command."""

    def test_cmd_week(self) -> None:
        """Test showing week's events."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        hooks.cli_get_env = lambda key: None
        hooks.cli_get_now = lambda: datetime(2026, 2, 20, 10, 0, 0)

        cmd_week()
        result = output.getvalue()
        assert "February 20" in result
        assert "February 21" in result
        assert "February 26" in result


# =============================================================================
# Test Helper Display Functions
# =============================================================================


class TestShowEventsForDelete:
    """Tests for _show_events_for_delete."""

    def test_show_events_for_delete(self) -> None:
        """Test displaying events for deletion."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        events = [
            EventInfo(
                sort_key="2026-02-20T14:00:00",
                time_str="14:00",
                summary="Meeting",
                calendar="Work",
                account="Personal",
                event_id="evt1",
                cal_id="cal1",
            )
        ]

        _show_events_for_delete(events, "2026-02-20")
        result = output.getvalue()
        assert "Meeting" in result
        assert "14:00" in result


class TestGetDeleteChoice:
    """Tests for _get_delete_choice."""

    def test_get_delete_choice_valid(self) -> None:
        """Test getting valid delete choice."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_prompt_ask = lambda msg: "1"

        events = [
            EventInfo(
                sort_key="2026-02-20T14:00:00",
                time_str="14:00",
                summary="Meeting",
                calendar="Work",
                account="Personal",
                event_id="evt1",
                cal_id="cal1",
            )
        ]

        result = _get_delete_choice(events)
        assert result == 0

    def test_get_delete_choice_quit(self) -> None:
        """Test quitting delete choice."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_prompt_ask = lambda msg: "q"

        events = [
            EventInfo(
                sort_key="2026-02-20T14:00:00",
                time_str="14:00",
                summary="Meeting",
                calendar="Work",
                account="Personal",
                event_id="evt1",
                cal_id="cal1",
            )
        ]

        result = _get_delete_choice(events)
        assert result is None

    def test_get_delete_choice_out_of_range(self) -> None:
        """Test delete choice out of range."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_prompt_ask = lambda msg: "99"

        events = [
            EventInfo(
                sort_key="2026-02-20T14:00:00",
                time_str="14:00",
                summary="Meeting",
                calendar="Work",
                account="Personal",
                event_id="evt1",
                cal_id="cal1",
            )
        ]

        result = _get_delete_choice(events)
        assert result is None
        assert "Invalid number" in output.getvalue()

    def test_get_delete_choice_zero(self) -> None:
        """Test delete choice of zero."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_prompt_ask = lambda msg: "0"

        events = [
            EventInfo(
                sort_key="2026-02-20T14:00:00",
                time_str="14:00",
                summary="Meeting",
                calendar="Work",
                account="Personal",
                event_id="evt1",
                cal_id="cal1",
            )
        ]

        result = _get_delete_choice(events)
        assert result is None


# =============================================================================
# Test Main Function
# =============================================================================
