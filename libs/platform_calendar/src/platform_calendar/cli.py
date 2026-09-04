#!/usr/bin/env python
"""Calendar CLI - Manage events across all accounts and calendars."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import TypedDict

from platform_core.cli_args import namespace_int, namespace_str, namespace_str_or_none
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from platform_calendar.cli_api import (
    EventInfo,
    _collect_events,
    _create_event,
    _delete_event,
    _fetch_calendars,
    _parse_time,
    _sanitize,
)
from platform_calendar.cli_auth import (
    ACCOUNTS,
    _get_now,
    _get_token,
    _get_valid_token_for_account,
)
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


def decode_list_args(args: argparse.Namespace) -> ListArgs:
    """Decode list arguments from argparse.Namespace.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Typed ListArgs structure.
    """
    raw_date = namespace_str_or_none(args, "date")
    date = raw_date if raw_date is not None else _get_now().strftime("%Y-%m-%d")
    return ListArgs(date=date)


def decode_create_args(args: argparse.Namespace) -> CreateArgs:
    """Decode create arguments from argparse.Namespace.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Typed CreateArgs structure.
    """
    title = namespace_str(args, "title", "")
    time_val = namespace_str(args, "time", "")
    raw_date = namespace_str_or_none(args, "date")
    date = raw_date if raw_date is not None else _get_now().strftime("%Y-%m-%d")
    duration = namespace_int(args, "duration", 60)
    raw_location = namespace_str_or_none(args, "location")
    location = raw_location if raw_location is not None else ""
    account = namespace_str(args, "account", "Personal")

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
    raw_date = namespace_str_or_none(args, "date")
    date = raw_date if raw_date is not None else _get_now().strftime("%Y-%m-%d")
    return DeleteArgs(date=date)


# =============================================================================
# Token Refresh Types
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
    command_str = namespace_str(args, "command", "")
    _dispatch_command(command_str, args)
