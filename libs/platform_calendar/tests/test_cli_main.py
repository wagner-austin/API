"""CLI parser, dispatch, main, and hook wiring."""

from __future__ import annotations

import io
import runpy
import sys
from datetime import datetime

from platform_core.json_utils import JSONObject
from rich.console import Console

from platform_calendar.cli import (
    DeleteArgs,
    ListArgs,
    cmd_calendars,
    cmd_delete,
    cmd_list,
    main,
)
from platform_calendar.cli_api import (
    _collect_events,
)
from platform_calendar.testing import hooks, reset_hooks

# =============================================================================
# Test Typed Argument Structures
# =============================================================================


class TestMain:
    """Tests for main entry point."""

    def test_main_list(self) -> None:
        """Test main with list command."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_get_env = lambda key: None

        import sys

        old_argv = sys.argv
        sys.argv = ["calendar", "list", "2026-02-20"]
        main()
        sys.argv = old_argv

        result = output.getvalue()
        assert "February 20" in result

    def test_main_default_list(self) -> None:
        """Test main with no command defaults to list."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_get_env = lambda key: None
        hooks.cli_get_now = lambda: datetime(2026, 2, 20, 10, 0, 0)

        import sys

        old_argv = sys.argv
        sys.argv = ["calendar"]
        main()
        sys.argv = old_argv

        result = output.getvalue()
        assert "2026" in result

    def test_main_calendars(self) -> None:
        """Test main with calendars command."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_get_env = lambda key: None

        import sys

        old_argv = sys.argv
        sys.argv = ["calendar", "calendars"]
        main()
        sys.argv = old_argv

        result = output.getvalue()
        assert "No token" in result

    def test_main_tomorrow(self) -> None:
        """Test main with tomorrow command."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_get_env = lambda key: None
        hooks.cli_get_now = lambda: datetime(2026, 2, 20, 10, 0, 0)

        import sys

        old_argv = sys.argv
        sys.argv = ["calendar", "tomorrow"]
        main()
        sys.argv = old_argv

        result = output.getvalue()
        assert "February 21" in result

    def test_main_week(self) -> None:
        """Test main with week command."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_get_env = lambda key: None
        hooks.cli_get_now = lambda: datetime(2026, 2, 20, 10, 0, 0)

        import sys

        old_argv = sys.argv
        sys.argv = ["calendar", "week"]
        main()
        sys.argv = old_argv

        result = output.getvalue()
        assert "February 20" in result

    def test_main_create(self) -> None:
        """Test main with create command."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_post = lambda t, u, b: {"id": "new"}

        import sys

        old_argv = sys.argv
        sys.argv = ["calendar", "create", "Meeting", "14:00", "-d", "2026-02-20"]
        main()
        sys.argv = old_argv

        result = output.getvalue()
        assert "Created" in result

    def test_main_delete(self) -> None:
        """Test main with delete command - no events."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_get_env = lambda key: None

        import sys

        old_argv = sys.argv
        sys.argv = ["calendar", "delete", "2026-02-20"]
        main()
        sys.argv = old_argv

        result = output.getvalue()
        assert "No events to delete" in result


# =============================================================================
# Test Hooks Module
# =============================================================================


class TestHooksModule:
    """Tests for hooks module."""

    def test_reset_hooks(self) -> None:
        """Test resetting all hooks restores production implementations."""
        # Set fake hooks
        hooks.cli_api_get = lambda t, u: {}
        hooks.cli_api_post = lambda t, u, b: {}
        hooks.cli_api_delete = lambda t, u: None
        hooks.cli_get_env = lambda k: "test_value"
        hooks.cli_get_now = lambda: datetime(2000, 1, 1)
        hooks.cli_prompt_ask = lambda m: "fake"
        hooks.cli_confirm_ask = lambda m: False
        fake_console = Console(file=io.StringIO())
        hooks.cli_get_console = lambda: fake_console

        # Verify fakes are set
        assert hooks.cli_get_env("any") == "test_value"
        assert hooks.cli_get_now() == datetime(2000, 1, 1)

        # Reset hooks
        reset_hooks()

        # Verify hooks are callable (production implementations restored)
        # Production cli_get_env returns None for unknown keys
        assert hooks.cli_get_env("NONEXISTENT_KEY_12345") is None
        # Production cli_get_now returns current time (not our fixed fake)
        now = hooks.cli_get_now()
        assert now.year >= 2026


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_collect_events_calendar_id_not_string(self) -> None:
        """Test collecting events when calendar ID is not a string."""

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": 123, "summary": "Bad Calendar"}]}
            return {"items": []}

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get

        result = _collect_events("2026-02-20")
        assert result == []

    def test_collect_events_calendar_name_not_string(self) -> None:
        """Test collecting events when calendar name is not a string."""

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "cal1", "summary": 123}]}
            return {"items": []}

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get

        result = _collect_events("2026-02-20")
        assert result == []

    def test_cmd_delete_recurring_event(self) -> None:
        """Test deleting recurring event instance."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt1_20260220T140000Z",  # Recurring event instance
                        "summary": "Recurring Event",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        deleted_urls: list[str] = []

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get
        hooks.cli_api_delete = lambda token, url: deleted_urls.append(url)
        hooks.cli_prompt_ask = lambda msg: "1"
        hooks.cli_confirm_ask = lambda msg: True

        cmd_delete(DeleteArgs(date="2026-02-20"))

        # Should use base event ID (evt1) not the instance ID
        assert len(deleted_urls) == 1
        assert "evt1" in deleted_urls[0]
        assert "_" not in deleted_urls[0].split("/")[-1]

    def test_cmd_delete_no_token_for_account(self) -> None:
        """Test deleting when account has no token after confirmation."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        # Track calls to determine when to return None for token
        # The flow: fetch events for all accounts, then _get_token for delete
        confirmed = [False]

        def fake_get_env(key: str) -> str | None:
            # Return token during fetch, but None after user confirms delete
            if "ACCESS_TOKEN" in key:
                if confirmed[0]:
                    return None
                return "token"
            return None

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

        def fake_confirm(msg: str) -> bool:
            # After confirmation, stop returning tokens
            confirmed[0] = True
            return True

        hooks.cli_get_env = fake_get_env
        hooks.cli_api_get = fake_api_get
        hooks.cli_api_delete = lambda t, u: None
        hooks.cli_prompt_ask = lambda msg: "1"
        hooks.cli_confirm_ask = fake_confirm

        cmd_delete(DeleteArgs(date="2026-02-20"))

        result = output.getvalue()
        assert "No token for" in result

    def test_cmd_list_all_day_event(self) -> None:
        """Test listing all-day events."""
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
                        "summary": "All Day Event",
                        "start": {"date": "2026-02-20"},
                    }
                ]
            }

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get

        cmd_list(ListArgs(date="2026-02-20"))
        result = output.getvalue()
        assert "All Day Event" in result
        assert "all-day" in result

    def test_collect_events_sorts_all_day_first(self) -> None:
        """Test that all-day events are sorted first."""

        # Only use Personal account to avoid duplicates
        def fake_get_env(key: str) -> str | None:
            if key == "GOOGLE_CALENDAR_ACCESS_TOKEN":
                return "token"
            return None

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt2",
                        "summary": "Timed Event",
                        "start": {"dateTime": "2026-02-20T09:00:00-08:00"},
                    },
                    {
                        "id": "evt1",
                        "summary": "All Day",
                        "start": {"date": "2026-02-20"},
                    },
                ]
            }

        hooks.cli_get_env = fake_get_env
        hooks.cli_api_get = fake_api_get

        result = _collect_events("2026-02-20")
        # All-day events should come first (sort key "00:00" is compared)
        # Find the all-day event
        all_day_indices = [i for i, e in enumerate(result) if e.time_str == "all-day"]
        timed_indices = [i for i, e in enumerate(result) if e.time_str != "all-day"]
        # All all-day events should come before timed events
        if all_day_indices and timed_indices:
            assert max(all_day_indices) < min(timed_indices)

    def test_cmd_calendars_no_summary(self) -> None:
        """Test listing calendars with no summary falls back to id."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            return {"items": [{"id": "cal123"}]}

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get

        cmd_calendars()
        result = output.getvalue()
        assert "cal123" in result

    def test_collect_events_no_summary(self) -> None:
        """Test collecting events with no summary."""

        # Only use Personal account to get exactly one event
        def fake_get_env(key: str) -> str | None:
            if key == "GOOGLE_CALENDAR_ACCESS_TOKEN":
                return "token"
            return None

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt1",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        hooks.cli_get_env = fake_get_env
        hooks.cli_api_get = fake_api_get

        result = _collect_events("2026-02-20")
        # Find the event without title
        no_title_events = [e for e in result if e.summary == "(no title)"]
        assert no_title_events != []
        assert no_title_events[0].summary == "(no title)"


class TestRunningTheModuleDirectly:
    """`python -m platform_calendar.cli` is how this CLI is invoked.

    The package declares no console script, so the __main__ block is the
    ONLY production path into main(). Without it this module's seven
    tests were the only callers -- a CLI with no way to run it.
    """

    def test_the_module_runs_as_main(self) -> None:
        """Reaches main() through the __main__ block, in-process."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_get_env = lambda key: None
        hooks.cli_get_now = lambda: datetime(2026, 2, 20, 10, 0, 0)

        original = list(sys.argv)
        sys.argv[:] = ["calendar", "list", "2026-02-20"]
        sys.modules.pop("platform_calendar.cli", None)
        try:
            runpy.run_module("platform_calendar.cli", run_name="__main__")
        finally:
            sys.argv[:] = original

        assert "February 20" in output.getvalue()
