#!/usr/bin/env python
"""Calendar CLI - Manage events across all accounts and calendars."""

from __future__ import annotations

import argparse
import urllib.parse
from datetime import datetime, timedelta
from typing import TypedDict

from platform_core.json_utils import JSONObject, narrow_json_to_dict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from platform_calendar.testing import hooks

# =============================================================================
# Styles
# =============================================================================

STYLE_HEADER = "bold cyan"
STYLE_ACCOUNT = "bold yellow"
STYLE_CALENDAR = "dim white"
STYLE_TIME = "green"
STYLE_ALLDAY = "blue"
STYLE_EVENT = "white"
STYLE_ERROR = "bold red"
STYLE_SUCCESS = "bold green"
STYLE_DIM = "dim"


# =============================================================================
# Typed Argument Structures
# =============================================================================


class ListArgs(TypedDict):
    """Arguments for list command."""

    date: str


class CreateArgs(TypedDict):
    """Arguments for create command."""

    title: str
    time: str
    date: str
    duration: int
    location: str
    account: str


class DeleteArgs(TypedDict):
    """Arguments for delete command."""

    date: str


def _extract_str(ns: argparse.Namespace, key: str, default: str) -> str:
    """Extract string attribute from namespace.

    Args:
        ns: Namespace to extract from.
        key: Attribute name.
        default: Default value if not found or not a string.

    Returns:
        String value.
    """
    val: str | None = getattr(ns, key, default)
    return val if isinstance(val, str) else default


def _extract_str_or_none(ns: argparse.Namespace, key: str) -> str | None:
    """Extract optional string attribute from namespace.

    Args:
        ns: Namespace to extract from.
        key: Attribute name.

    Returns:
        String value or None.
    """
    val: str | None = getattr(ns, key, None)
    return val if isinstance(val, str) else None


def _extract_int(ns: argparse.Namespace, key: str, default: int) -> int:
    """Extract int attribute from namespace.

    Args:
        ns: Namespace to extract from.
        key: Attribute name.
        default: Default value if not found or not an int.

    Returns:
        Int value.
    """
    val: int | None = getattr(ns, key, default)
    return val if isinstance(val, int) else default


def decode_list_args(args: argparse.Namespace) -> ListArgs:
    """Decode list arguments from argparse.Namespace.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Typed ListArgs structure.
    """
    raw_date = _extract_str_or_none(args, "date")
    date = raw_date if raw_date is not None else _get_now().strftime("%Y-%m-%d")
    return ListArgs(date=date)


def decode_create_args(args: argparse.Namespace) -> CreateArgs:
    """Decode create arguments from argparse.Namespace.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Typed CreateArgs structure.
    """
    title = _extract_str(args, "title", "")
    time_val = _extract_str(args, "time", "")
    raw_date = _extract_str_or_none(args, "date")
    date = raw_date if raw_date is not None else _get_now().strftime("%Y-%m-%d")
    duration = _extract_int(args, "duration", 60)
    raw_location = _extract_str_or_none(args, "location")
    location = raw_location if raw_location is not None else ""
    account = _extract_str(args, "account", "Personal")

    return CreateArgs(
        title=title,
        time=time_val,
        date=date,
        duration=duration,
        location=location,
        account=account,
    )


def decode_delete_args(args: argparse.Namespace) -> DeleteArgs:
    """Decode delete arguments from argparse.Namespace.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Typed DeleteArgs structure.
    """
    raw_date = _extract_str_or_none(args, "date")
    date = raw_date if raw_date is not None else _get_now().strftime("%Y-%m-%d")
    return DeleteArgs(date=date)


# =============================================================================
# Token Refresh Types
# =============================================================================

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


class TokenRefreshResponse(TypedDict):
    """Response from Google OAuth token refresh endpoint."""

    access_token: str
    expires_in: int
    token_type: str


def require_str(data: JSONObject, key: str) -> str:
    """Require a string value from a JSON object.

    Args:
        data: JSON object to extract from.
        key: Key to look up.

    Returns:
        String value.

    Raises:
        KeyError: If key not found.
        TypeError: If value is not a string.
    """
    value = data[key]
    if not isinstance(value, str):
        msg = f"Expected str for {key}, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def require_int(data: JSONObject, key: str) -> int:
    """Require an int value from a JSON object.

    Args:
        data: JSON object to extract from.
        key: Key to look up.

    Returns:
        Int value.

    Raises:
        KeyError: If key not found.
        TypeError: If value is not an int.
    """
    value = data[key]
    if not isinstance(value, int):
        msg = f"Expected int for {key}, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def decode_token_refresh_response(data: JSONObject) -> TokenRefreshResponse:
    """Decode token refresh response from JSON.

    Args:
        data: JSON object from token endpoint.

    Returns:
        Typed TokenRefreshResponse.

    Raises:
        KeyError: If required field missing.
        TypeError: If field has wrong type.
    """
    return TokenRefreshResponse(
        access_token=require_str(data, "access_token"),
        expires_in=require_int(data, "expires_in"),
        token_type=require_str(data, "token_type"),
    )


# =============================================================================
# Config
# =============================================================================


class Account:
    """Account configuration."""

    def __init__(
        self,
        name: str,
        email: str,
        token_env: str,
        refresh_token_env: str,
        expires_at_env: str,
        default_calendar: str = "primary",
    ) -> None:
        """Initialize account.

        Args:
            name: Display name for the account.
            email: Email address associated with the account.
            token_env: Environment variable name for access token.
            refresh_token_env: Environment variable name for refresh token.
            expires_at_env: Environment variable name for token expiration.
            default_calendar: Default calendar ID.
        """
        self.name = name
        self.email = email
        self.token_env = token_env
        self.refresh_token_env = refresh_token_env
        self.expires_at_env = expires_at_env
        self.default_calendar = default_calendar


ACCOUNTS = [
    Account(
        name="Personal",
        email="austin.o.wagner@gmail.com",
        token_env="GOOGLE_CALENDAR_ACCESS_TOKEN",
        refresh_token_env="GOOGLE_CALENDAR_REFRESH_TOKEN",
        expires_at_env="GOOGLE_CALENDAR_TOKEN_EXPIRES_AT",
    ),
    Account(
        name="Interns",
        email="interns@liuforirvine.com",
        token_env="GOOGLE_CALENDAR_INTERNS_ACCESS_TOKEN",
        refresh_token_env="GOOGLE_CALENDAR_INTERNS_REFRESH_TOKEN",
        expires_at_env="GOOGLE_CALENDAR_INTERNS_EXPIRES_AT",
    ),
]


def _get_env(key: str) -> str | None:
    """Get environment variable.

    Args:
        key: Environment variable name.

    Returns:
        Value if found, None otherwise.
    """
    return hooks.cli_get_env(key)


def _set_env(key: str, value: str) -> None:
    """Set environment variable in cache.

    Args:
        key: Environment variable name.
        value: Value to set.
    """
    hooks.cli_set_env(key, value)


def _is_token_expired(expires_at_str: str) -> bool:
    """Check if token is expired or will expire within 60 seconds.

    Args:
        expires_at_str: Unix timestamp as string.

    Returns:
        True if token is expired or expiring soon.
    """
    expires_at = int(expires_at_str)
    current_time = int(_get_now().timestamp())
    buffer_seconds = 60
    return current_time >= (expires_at - buffer_seconds)


def _refresh_token(
    client_id: str,
    client_secret: str,
    refresh_token: str,
) -> TokenRefreshResponse:
    """Refresh an access token using the refresh token.

    Args:
        client_id: OAuth client ID.
        client_secret: OAuth client secret.
        refresh_token: Refresh token.

    Returns:
        TokenRefreshResponse with new access token.

    Raises:
        urllib.error.HTTPError: If refresh fails.
        KeyError: If response missing required fields.
        TypeError: If response fields have wrong types.
    """
    from platform_core.json_utils import load_json_str

    body_params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    body = urllib.parse.urlencode(body_params)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = hooks.http_post(GOOGLE_TOKEN_URL, headers, body)
    raw_value = load_json_str(response)
    data = narrow_json_to_dict(raw_value)
    return decode_token_refresh_response(data)


def _get_valid_token_for_account(account: Account) -> str | None:
    """Get a valid access token for an account, refreshing if expired.

    Args:
        account: Account to get token for.

    Returns:
        Valid access token, or None if no token configured.
    """
    access_token = _get_env(account.token_env)
    if not access_token:
        return None

    refresh_token = _get_env(account.refresh_token_env)
    expires_at = _get_env(account.expires_at_env)

    # If we have expiration info and token is expired, refresh
    if expires_at and refresh_token and _is_token_expired(expires_at):
        client_id = _get_env("GOOGLE_CALENDAR_CLIENT_ID")
        client_secret = _get_env("GOOGLE_CALENDAR_CLIENT_SECRET")

        if client_id and client_secret:
            response = _refresh_token(client_id, client_secret, refresh_token)
            access_token = response["access_token"]
            new_expires_at = int(_get_now().timestamp()) + response["expires_in"]

            # Update cache with new token
            _set_env(account.token_env, access_token)
            _set_env(account.expires_at_env, str(new_expires_at))

    return access_token


def _get_token(account_name: str) -> str | None:
    """Get valid token for an account by name.

    Args:
        account_name: Account name to look up.

    Returns:
        Valid access token if found, None otherwise.
    """
    for account in ACCOUNTS:
        if account.name.lower() == account_name.lower():
            return _get_valid_token_for_account(account)
    return None


# =============================================================================
# Hookable Functions
# =============================================================================


def _get_console() -> Console:
    """Get console instance.

    Returns:
        Console instance for output.
    """
    return hooks.cli_get_console()


def set_console(console: Console) -> None:
    """Set console instance for testing.

    Args:
        console: Console instance to use.
    """
    hooks.cli_get_console = lambda: console


def _get_now() -> datetime:
    """Get current datetime.

    Returns:
        Current datetime.
    """
    return hooks.cli_get_now()


def _prompt_ask(message: str) -> str:
    """Ask user for input.

    Args:
        message: Prompt message.

    Returns:
        User input string.
    """
    return hooks.cli_prompt_ask(message)


def _confirm_ask(message: str) -> bool:
    """Ask user for confirmation.

    Args:
        message: Prompt message.

    Returns:
        True if confirmed, False otherwise.
    """
    return hooks.cli_confirm_ask(message)


# =============================================================================
# API Functions
# =============================================================================


def _api_get(access_token: str, url: str) -> JSONObject:
    """Make GET request to Google Calendar API.

    Args:
        access_token: OAuth access token for authentication.
        url: Full URL for the API endpoint.

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        urllib.error.URLError: If the request fails.
    """
    return hooks.cli_api_get(access_token, url)


def _api_post(access_token: str, url: str, request_body: JSONObject) -> JSONObject:
    """Make POST request to Google Calendar API.

    Args:
        access_token: OAuth access token for authentication.
        url: Full URL for the API endpoint.
        request_body: Request body as JSON-compatible dict.

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        urllib.error.URLError: If the request fails.
    """
    return hooks.cli_api_post(access_token, url, request_body)


def _api_delete(access_token: str, url: str) -> None:
    """Make DELETE request to Google Calendar API.

    Args:
        access_token: OAuth access token for authentication.
        url: Full URL for the API endpoint.

    Raises:
        urllib.error.URLError: If the request fails.
    """
    hooks.cli_api_delete(access_token, url)


def _fetch_calendars(access_token: str) -> list[JSONObject]:
    """Fetch all calendars for an account.

    Args:
        access_token: OAuth access token for authentication.

    Returns:
        List of calendar objects.

    Raises:
        urllib.error.URLError: If the request fails.
    """
    url = "https://www.googleapis.com/calendar/v3/users/me/calendarList"
    data = _api_get(access_token, url)
    items = data.get("items", [])
    if not isinstance(items, list):
        return []
    return [narrow_json_to_dict(item) for item in items if isinstance(item, dict)]


def _fetch_events(access_token: str, calendar_id: str, date: str) -> list[JSONObject]:
    """Fetch events for a calendar on a specific date.

    Args:
        access_token: OAuth access token for authentication.
        calendar_id: ID of the calendar to fetch events from.
        date: Date in YYYY-MM-DD format.

    Returns:
        List of event objects.

    Raises:
        urllib.error.URLError: If the request fails.
    """
    encoded_id = urllib.parse.quote(calendar_id, safe="")
    url = (
        f"https://www.googleapis.com/calendar/v3/calendars/{encoded_id}/events"
        f"?timeMin={date}T00:00:00-08:00"
        f"&timeMax={date}T23:59:59-08:00"
        f"&singleEvents=true"
        f"&orderBy=startTime"
    )
    data = _api_get(access_token, url)
    items = data.get("items", [])
    if not isinstance(items, list):
        return []
    return [narrow_json_to_dict(item) for item in items if isinstance(item, dict)]


def _create_event(
    access_token: str,
    calendar_id: str,
    summary: str,
    start_dt: str,
    end_dt: str,
    location: str = "",
) -> JSONObject:
    """Create an event via API.

    Args:
        access_token: OAuth access token for authentication.
        calendar_id: ID of the calendar to create event in.
        summary: Event title.
        start_dt: Start datetime in ISO format.
        end_dt: End datetime in ISO format.
        location: Event location.

    Returns:
        Created event object.

    Raises:
        urllib.error.URLError: If the request fails.
    """
    encoded_id = urllib.parse.quote(calendar_id, safe="")
    url = f"https://www.googleapis.com/calendar/v3/calendars/{encoded_id}/events"

    body: JSONObject = {
        "summary": summary,
        "start": {"dateTime": start_dt, "timeZone": "America/Los_Angeles"},
        "end": {"dateTime": end_dt, "timeZone": "America/Los_Angeles"},
    }
    if location:
        body["location"] = location

    return _api_post(access_token, url, body)


def _delete_event(access_token: str, calendar_id: str, event_id: str) -> None:
    """Delete an event via API.

    Args:
        access_token: OAuth access token for authentication.
        calendar_id: ID of the calendar containing the event.
        event_id: ID of the event to delete.

    Raises:
        urllib.error.URLError: If the request fails.
    """
    encoded_cal = urllib.parse.quote(calendar_id, safe="")
    encoded_event = urllib.parse.quote(event_id, safe="")
    url = f"https://www.googleapis.com/calendar/v3/calendars/{encoded_cal}/events/{encoded_event}"
    _api_delete(access_token, url)


# =============================================================================
# Helpers
# =============================================================================


def _format_time(event: JSONObject) -> tuple[str, str]:
    """Extract time string and sort key from event.

    Args:
        event: Event object from API.

    Returns:
        Tuple of (display time, sort key).
    """
    start = event.get("start", {})
    if not isinstance(start, dict):
        return "all-day", "00:00"
    date_time = start.get("dateTime")
    if isinstance(date_time, str):
        time_part = date_time.split("T")[1][:5]
        return time_part, date_time
    return "all-day", "00:00"


def _sanitize(text: str) -> str:
    """Remove non-ASCII characters.

    Args:
        text: Input text.

    Returns:
        ASCII-safe text.
    """
    return text.encode("ascii", "ignore").decode()


def _parse_time(time_str: str) -> str:
    """Parse time string (e.g., '2pm', '14:00') to HH:MM format.

    Args:
        time_str: Time string to parse.

    Returns:
        Time in HH:MM format.
    """
    t = time_str.lower().replace(" ", "")
    if "pm" in t:
        hour = int(t.replace("pm", ""))
        if hour != 12:
            hour += 12
        return f"{hour:02d}:00"
    if "am" in t:
        hour = int(t.replace("am", ""))
        if hour == 12:
            hour = 0
        return f"{hour:02d}:00"
    return time_str


# =============================================================================
# Event Collection
# =============================================================================


class EventInfo:
    """Container for event information."""

    def __init__(
        self,
        sort_key: str,
        time_str: str,
        summary: str,
        calendar: str,
        account: str,
        event_id: str,
        cal_id: str,
    ) -> None:
        """Initialize event info.

        Args:
            sort_key: Key for sorting events.
            time_str: Display time string.
            summary: Event title.
            calendar: Calendar name.
            account: Account name.
            event_id: Event ID.
            cal_id: Calendar ID.
        """
        self.sort_key = sort_key
        self.time_str = time_str
        self.summary = summary
        self.calendar = calendar
        self.account = account
        self.event_id = event_id
        self.cal_id = cal_id


def _collect_events(date: str) -> list[EventInfo]:
    """Collect all events across all accounts for a date.

    Args:
        date: Date in YYYY-MM-DD format.

    Returns:
        List of events sorted by time.
    """
    all_events: list[EventInfo] = []

    for account in ACCOUNTS:
        token = _get_valid_token_for_account(account)
        if not token:
            continue

        calendars = _fetch_calendars(token)
        for cal in calendars:
            cal_id = cal.get("id")
            if not isinstance(cal_id, str):
                continue
            cal_name = cal.get("summary", cal_id)
            if not isinstance(cal_name, str):
                cal_name = cal_id

            if "holiday" in cal_name.lower():
                continue

            events = _fetch_events(token, cal_id, date)
            for event in events:
                time_str, sort_key = _format_time(event)
                raw_summary = event.get("summary", "(no title)")
                summary = _sanitize(str(raw_summary))
                cal_display = _sanitize(cal_name)
                raw_event_id = event.get("id", "")
                event_id = str(raw_event_id)
                all_events.append(
                    EventInfo(
                        sort_key=sort_key,
                        time_str=time_str,
                        summary=summary,
                        calendar=cal_display,
                        account=account.name,
                        event_id=event_id,
                        cal_id=cal_id,
                    )
                )

    all_events.sort(key=lambda x: (x.sort_key != "00:00", x.sort_key))
    return all_events


# =============================================================================
# Commands
# =============================================================================


def cmd_list(cmd_args: ListArgs) -> None:
    """List all events across all accounts for a date.

    Args:
        cmd_args: Typed command arguments.
    """
    console = _get_console()
    date = cmd_args["date"]
    all_events = _collect_events(date)

    date_display = datetime.strptime(date, "%Y-%m-%d").strftime("%A, %B %d, %Y")
    console.print(Panel(f"[{STYLE_HEADER}]{date_display}[/]", expand=False))
    console.print()

    if not all_events:
        console.print("[dim]No events found.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style=STYLE_DIM, width=3)
    table.add_column("Time", style=STYLE_TIME, width=8)
    table.add_column("Event", style=STYLE_EVENT)
    table.add_column("Calendar", style=STYLE_CALENDAR)
    table.add_column("Account", style=STYLE_ACCOUNT)

    for i, ev in enumerate(all_events, 1):
        time_style = STYLE_ALLDAY if ev.time_str == "all-day" else STYLE_TIME
        table.add_row(
            str(i),
            f"[{time_style}]{ev.time_str}[/]",
            ev.summary,
            ev.calendar,
            ev.account,
        )

    console.print(table)
    console.print(f"\n[dim]{len(all_events)} events total[/dim]")


def cmd_calendars() -> None:
    """List all calendars across all accounts."""
    console = _get_console()
    for account in ACCOUNTS:
        token = _get_valid_token_for_account(account)
        if not token:
            console.print(f"[{STYLE_ERROR}]No token for {account.name}[/]")
            continue

        console.print(f"\n[{STYLE_ACCOUNT}]{account.name}[/] ({account.email})")

        calendars = _fetch_calendars(token)
        for cal in calendars:
            primary = " [bold](primary)[/bold]" if cal.get("primary") else ""
            raw_name = cal.get("summary", cal.get("id", "unknown"))
            cal_name = _sanitize(str(raw_name))
            console.print(f"  - {cal_name}{primary}")


def cmd_create(cmd_args: CreateArgs) -> None:
    """Create a new event.

    Args:
        cmd_args: Typed command arguments.
    """
    console = _get_console()
    account_name = cmd_args["account"]
    token = _get_token(account_name)
    if not token:
        console.print(f"[{STYLE_ERROR}]Unknown account: {account_name}[/]")
        return

    date = cmd_args["date"]
    start_time = _parse_time(cmd_args["time"])

    duration = cmd_args["duration"]
    start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
    end_dt = start_dt + timedelta(minutes=duration)

    start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
    end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%S")

    calendar_id = "primary"
    for acc in ACCOUNTS:
        if acc.name.lower() == account_name.lower():
            calendar_id = acc.default_calendar

    title = cmd_args["title"]
    location = cmd_args["location"]

    _create_event(
        token,
        calendar_id,
        title,
        start_iso,
        end_iso,
        location=location,
    )

    console.print(f"[{STYLE_SUCCESS}]Created:[/] {title}")
    console.print(f"  [dim]Date:[/dim] {date}")
    console.print(f"  [dim]Time:[/dim] {start_time} - {end_dt.strftime('%H:%M')}")
    if location:
        console.print(f"  [dim]Location:[/dim] {location}")


def _show_events_for_delete(all_events: list[EventInfo], date: str) -> None:
    """Display events with numbers for deletion selection.

    Args:
        all_events: List of events to display.
        date: Date string for header.
    """
    console = _get_console()
    console.print(f"\n[{STYLE_HEADER}]Events for {date}:[/]\n")
    for i, ev in enumerate(all_events, 1):
        line = f"  [{STYLE_DIM}]{i}.[/] [{STYLE_TIME}]{ev.time_str}[/] "
        line += f"{ev.summary} [{STYLE_DIM}]({ev.account})[/]"
        console.print(line)


def _get_delete_choice(all_events: list[EventInfo]) -> int | None:
    """Get user's choice for which event to delete.

    Args:
        all_events: List of events to choose from.

    Returns:
        Index of the selected event, or None if cancelled or invalid.
    """
    console = _get_console()
    console.print()
    choice = _prompt_ask("Enter event number to delete (or 'q' to cancel)")

    if choice.lower() == "q":
        return None

    # Validate input is a positive integer without try/except
    if not choice.isdigit():
        console.print(f"[{STYLE_ERROR}]Invalid input[/]")
        return None

    idx = int(choice) - 1
    if idx < 0 or idx >= len(all_events):
        console.print(f"[{STYLE_ERROR}]Invalid number[/]")
        return None
    return idx


def cmd_delete(cmd_args: DeleteArgs) -> None:
    """Delete an event by listing and selecting.

    Args:
        cmd_args: Typed command arguments.
    """
    console = _get_console()
    date = cmd_args["date"]
    all_events = _collect_events(date)

    if not all_events:
        console.print("[dim]No events to delete.[/dim]")
        return

    _show_events_for_delete(all_events, date)
    idx = _get_delete_choice(all_events)
    if idx is None:
        return

    ev = all_events[idx]

    if not _confirm_ask(f"Delete '[bold]{ev.summary}[/bold]' at {ev.time_str}?"):
        return

    token = _get_token(ev.account)
    if not token:
        console.print(f"[{STYLE_ERROR}]No token for {ev.account}[/]")
        return

    # Handle recurring event instances (ID contains underscore)
    base_event_id = ev.event_id.split("_")[0] if "_" in ev.event_id else ev.event_id

    _delete_event(token, ev.cal_id, base_event_id)
    console.print(f"[{STYLE_SUCCESS}]Deleted:[/] {ev.summary}")


def cmd_tomorrow() -> None:
    """Show tomorrow's events."""
    tomorrow = (_get_now() + timedelta(days=1)).strftime("%Y-%m-%d")
    cmd_list(ListArgs(date=tomorrow))


def cmd_week() -> None:
    """Show this week's events."""
    console = _get_console()
    today = _get_now()

    for i in range(7):
        date = (today + timedelta(days=i)).strftime("%Y-%m-%d")
        cmd_list(ListArgs(date=date))
        console.print()


# =============================================================================
# Main
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Calendar CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # list
    list_parser = subparsers.add_parser("list", aliases=["ls", "l"], help="List events")
    list_parser.add_argument("date", nargs="?", help="Date (YYYY-MM-DD)")

    # calendars
    subparsers.add_parser("calendars", aliases=["cals"], help="List all calendars")

    # create
    create_parser = subparsers.add_parser("create", aliases=["add", "new"], help="Create event")
    create_parser.add_argument("title", help="Event title")
    create_parser.add_argument("time", help="Start time (e.g., 14:00 or 2pm)")
    create_parser.add_argument("-d", "--date", help="Date (YYYY-MM-DD, default today)")
    create_parser.add_argument("-D", "--duration", type=int, default=60, help="Duration in minutes")
    create_parser.add_argument("-l", "--location", help="Location")
    create_parser.add_argument("-a", "--account", default="Personal", help="Account name")

    # delete
    delete_parser = subparsers.add_parser("delete", aliases=["rm", "del"], help="Delete event")
    delete_parser.add_argument("date", nargs="?", help="Date (YYYY-MM-DD)")

    # tomorrow
    subparsers.add_parser("tomorrow", aliases=["tm"], help="Show tomorrow's events")

    # week
    subparsers.add_parser("week", aliases=["w"], help="Show this week's events")

    return parser


def _dispatch_command(command_str: str, args: argparse.Namespace) -> None:
    """Dispatch command to appropriate handler.

    Args:
        command_str: Command name.
        args: Parsed arguments.
    """
    if command_str in ("list", "ls", "l", ""):
        cmd_list(decode_list_args(args))
    elif command_str in ("calendars", "cals"):
        cmd_calendars()
    elif command_str in ("create", "add", "new"):
        cmd_create(decode_create_args(args))
    elif command_str in ("delete", "rm", "del"):
        cmd_delete(decode_delete_args(args))
    elif command_str in ("tomorrow", "tm"):
        cmd_tomorrow()
    else:
        cmd_week()


def main() -> None:
    """Main entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    command_str = _extract_str(args, "command", "")
    _dispatch_command(command_str, args)
