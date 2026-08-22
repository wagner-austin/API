"""Graph API access and event formatting for the calendar CLI.

The commands live in :mod:`platform_calendar.cli`.
"""

from __future__ import annotations

import urllib.parse

from platform_core.json_utils import JSONObject, narrow_json_to_dict

from platform_calendar.cli_auth import ACCOUNTS, _get_valid_token_for_account
from platform_calendar.testing import hooks


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
