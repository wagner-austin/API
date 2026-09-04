"""Google Calendar API client implementation."""

from __future__ import annotations

import urllib.parse

from platform_core.errors import AppError, CalendarErrorCode
from platform_core.json_utils import (
    InvalidJsonError,
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)
from platform_core.oauth_types import (
    OAuthTokens,
)

from platform_calendar.config import GOOGLE_CALENDAR_API_BASE
from platform_calendar.testing import CalendarClientProtocol, HTTPErrorProtocol, hooks
from platform_calendar.types import (
    CalendarEvent,
    CalendarListItem,
    EventDateTime,
    decode_calendar_event,
    decode_calendar_list_item,
)

# =============================================================================
# Internal Client Implementation
# =============================================================================


class _GoogleCalendarClient(CalendarClientProtocol):
    """Google Calendar API client."""

    def __init__(self, *, access_token: str) -> None:
        """Initialize the client with an access token.

        Args:
            access_token: OAuth access token for API requests.
        """
        self._access_token = access_token

    def _get_headers(self) -> dict[str, str]:
        """Get authorization headers."""
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

    def _handle_error(self, status_code: int, response_body: str, context: str) -> None:
        """Handle API error response.

        Args:
            status_code: HTTP status code.
            response_body: Response body from API.
            context: Context string for error message.

        Raises:
            AppError[CalendarErrorCode]: Always raises with appropriate error code.
        """
        if status_code == 404:
            # Check for events first since event endpoints also contain "calendar"
            if "events" in context.lower():
                raise AppError(
                    CalendarErrorCode.EVENT_NOT_FOUND,
                    f"Event not found: {context}",
                    http_status=404,
                )
            if "calendar" in context.lower():
                raise AppError(
                    CalendarErrorCode.CALENDAR_NOT_FOUND,
                    f"Calendar not found: {context}",
                    http_status=404,
                )
            raise AppError(
                CalendarErrorCode.EVENT_NOT_FOUND,
                f"Not found: {context}",
                http_status=404,
            )
        raise AppError(
            CalendarErrorCode.CALENDAR_API_ERROR,
            f"API error ({status_code}): {context}",
            http_status=status_code,
        )

    def _api_get(self, endpoint: str) -> JSONObject:
        """Make GET request to Calendar API.

        Args:
            endpoint: API endpoint path.

        Returns:
            Parsed JSON response as dict.

        Raises:
            AppError[CalendarErrorCode]: On API errors.
        """
        url = GOOGLE_CALENDAR_API_BASE + endpoint
        try:
            response = hooks.http_get(url, self._get_headers())
        except ConnectionError as e:
            raise AppError(
                CalendarErrorCode.CALENDAR_API_ERROR,
                f"Request failed: {e}",
                http_status=502,
            ) from e
        except OSError as e:
            # Check for HTTP errors using Protocol
            if isinstance(e, HTTPErrorProtocol):
                status_code: int = e.code
                body: str = e.read().decode("utf-8")
                self._handle_error(status_code, body, endpoint)
            raise AppError(
                CalendarErrorCode.CALENDAR_API_ERROR,
                f"Request failed: {e}",
                http_status=502,
            ) from e

        try:
            raw_value = load_json_str(response)
            data = narrow_json_to_dict(raw_value)
        except (InvalidJsonError, JSONTypeError) as e:
            raise AppError(
                CalendarErrorCode.CALENDAR_API_ERROR,
                f"Invalid response: {e}",
                http_status=502,
            ) from e

        return data

    def _api_post(self, endpoint: str, body: JSONObject) -> JSONObject:
        """Make POST request to Calendar API.

        Args:
            endpoint: API endpoint path.
            body: Request body as dict.

        Returns:
            Parsed JSON response as dict.

        Raises:
            AppError[CalendarErrorCode]: On API errors.
        """
        url = GOOGLE_CALENDAR_API_BASE + endpoint
        body_json = dump_json_str(body)
        try:
            response = hooks.http_post(url, self._get_headers(), body_json)
        except ConnectionError as e:
            raise AppError(
                CalendarErrorCode.CALENDAR_API_ERROR,
                f"Request failed: {e}",
                http_status=502,
            ) from e
        except OSError as e:
            if isinstance(e, HTTPErrorProtocol):
                status_code = e.code
                resp_body: str = e.read().decode("utf-8")
                self._handle_error(status_code, resp_body, endpoint)
            raise AppError(
                CalendarErrorCode.CALENDAR_API_ERROR,
                f"Request failed: {e}",
                http_status=502,
            ) from e

        try:
            raw_value = load_json_str(response)
            data = narrow_json_to_dict(raw_value)
        except (InvalidJsonError, JSONTypeError) as e:
            raise AppError(
                CalendarErrorCode.CALENDAR_API_ERROR,
                f"Invalid response: {e}",
                http_status=502,
            ) from e

        return data

    def _api_patch(self, endpoint: str, body: JSONObject) -> JSONObject:
        """Make PATCH request to Calendar API.

        Args:
            endpoint: API endpoint path.
            body: Request body as dict (partial update).

        Returns:
            Parsed JSON response as dict.

        Raises:
            AppError[CalendarErrorCode]: On API errors.
        """
        url = GOOGLE_CALENDAR_API_BASE + endpoint
        body_json = dump_json_str(body)
        try:
            response = hooks.http_patch(url, self._get_headers(), body_json)
        except ConnectionError as e:
            raise AppError(
                CalendarErrorCode.CALENDAR_API_ERROR,
                f"Request failed: {e}",
                http_status=502,
            ) from e
        except OSError as e:
            if isinstance(e, HTTPErrorProtocol):
                status_code = e.code
                resp_body: str = e.read().decode("utf-8")
                self._handle_error(status_code, resp_body, endpoint)
            raise AppError(
                CalendarErrorCode.CALENDAR_API_ERROR,
                f"Request failed: {e}",
                http_status=502,
            ) from e

        try:
            raw_value = load_json_str(response)
            data = narrow_json_to_dict(raw_value)
        except (InvalidJsonError, JSONTypeError) as e:
            raise AppError(
                CalendarErrorCode.CALENDAR_API_ERROR,
                f"Invalid response: {e}",
                http_status=502,
            ) from e

        return data

    def _normalize_event_response(self, data: JSONObject) -> None:
        """Normalize API response by adding missing optional fields.

        Args:
            data: JSON response to normalize in-place.
        """
        data.setdefault("description", "")
        data.setdefault("status", "confirmed")
        data.setdefault("location", "")
        data.setdefault("recurrence", [])
        if "reminders" not in data:
            data["reminders"] = {"useDefault": True, "overrides": []}
        reminders = data.get("reminders")
        if isinstance(reminders, dict):
            reminders.setdefault("useDefault", True)
            reminders.setdefault("overrides", [])

    def list_calendars(self) -> tuple[CalendarListItem, ...]:
        """List all calendars for the authenticated user."""
        calendars: list[CalendarListItem] = []
        page_token: str | None = None

        while True:
            endpoint = "/users/me/calendarList"
            if page_token:
                endpoint += f"?pageToken={urllib.parse.quote(page_token)}"

            data = self._api_get(endpoint)
            items = data.get("items", [])
            if not isinstance(items, list):
                break

            for item in items:
                if isinstance(item, dict):
                    # Handle optional fields with defaults
                    item.setdefault("description", "")
                    item.setdefault("primary", False)
                    calendars.append(decode_calendar_list_item(item))

            next_token = data.get("nextPageToken")
            if isinstance(next_token, str) and next_token:
                page_token = next_token
            else:
                break

        return tuple(calendars)

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
        encoded_cal_id = urllib.parse.quote(calendar_id, safe="")
        encoded_event_id = urllib.parse.quote(event_id, safe="")
        endpoint = f"/calendars/{encoded_cal_id}/events/{encoded_event_id}"

        data = self._api_get(endpoint)
        self._normalize_event_response(data)
        return decode_calendar_event(data)

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
            time_min: Start of time range (RFC 3339).
            time_max: End of time range (RFC 3339).

        Returns:
            Tuple of CalendarEvent within the time range.
        """
        events: list[CalendarEvent] = []
        page_token: str | None = None
        encoded_id = urllib.parse.quote(calendar_id, safe="")

        while True:
            params = {
                "timeMin": time_min,
                "timeMax": time_max,
                "singleEvents": "true",
                "orderBy": "startTime",
            }
            if page_token:
                params["pageToken"] = page_token

            endpoint = f"/calendars/{encoded_id}/events?" + urllib.parse.urlencode(params)
            data = self._api_get(endpoint)

            items = data.get("items", [])
            if not isinstance(items, list):
                break

            for item in items:
                if isinstance(item, dict):
                    self._normalize_event_response(item)
                    events.append(decode_calendar_event(item))

            next_token = data.get("nextPageToken")
            if isinstance(next_token, str) and next_token:
                page_token = next_token
            else:
                break

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
            reminders: Reminder times in minutes before event.
            location: Event location string.
            recurrence: RRULE strings for recurring events.

        Returns:
            The created CalendarEvent.
        """
        encoded_id = urllib.parse.quote(calendar_id, safe="")
        endpoint = f"/calendars/{encoded_id}/events"

        overrides: list[JSONValue] = []
        for minutes in reminders:
            override_obj: JSONObject = {"method": "popup", "minutes": minutes}
            overrides.append(override_obj)

        body: JSONObject = {
            "summary": summary,
            "description": description,
            "start": {
                "dateTime": start["dateTime"],
                "timeZone": start["timeZone"],
            },
            "end": {
                "dateTime": end["dateTime"],
                "timeZone": end["timeZone"],
            },
            "reminders": {
                "useDefault": False,
                "overrides": overrides,
            },
        }

        # Add optional fields
        if location:
            body["location"] = location
        if recurrence:
            recurrence_list: list[JSONValue] = list(recurrence)
            body["recurrence"] = recurrence_list

        data = self._api_post(endpoint, body)
        self._normalize_event_response(data)
        return decode_calendar_event(data)

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
        """Update an existing calendar event using PATCH.

        Only provided fields are updated; None values keep existing values.

        Args:
            calendar_id: Calendar containing the event.
            event_id: ID of the event to update.
            summary: New event title (None to keep existing).
            description: New description (None to keep existing).
            start: New start time (None to keep existing).
            end: New end time (None to keep existing).
            reminders: New reminder times in minutes (None to keep existing).
            location: New location (None to keep existing).
            recurrence: New recurrence rules (None to keep existing).

        Returns:
            The updated CalendarEvent.

        Raises:
            AppError[CalendarErrorCode]: If event not found.
        """
        encoded_cal_id = urllib.parse.quote(calendar_id, safe="")
        encoded_event_id = urllib.parse.quote(event_id, safe="")
        endpoint = f"/calendars/{encoded_cal_id}/events/{encoded_event_id}"

        # Build partial update body - only include fields that are being updated
        body: JSONObject = {}

        if summary is not None:
            body["summary"] = summary
        if description is not None:
            body["description"] = description
        if start is not None:
            body["start"] = {
                "dateTime": start["dateTime"],
                "timeZone": start["timeZone"],
            }
        if end is not None:
            body["end"] = {
                "dateTime": end["dateTime"],
                "timeZone": end["timeZone"],
            }
        if reminders is not None:
            overrides: list[JSONValue] = []
            for minutes in reminders:
                override_obj: JSONObject = {"method": "popup", "minutes": minutes}
                overrides.append(override_obj)
            body["reminders"] = {
                "useDefault": False,
                "overrides": overrides,
            }
        if location is not None:
            body["location"] = location
        if recurrence is not None:
            recurrence_list: list[JSONValue] = list(recurrence)
            body["recurrence"] = recurrence_list

        data = self._api_patch(endpoint, body)
        self._normalize_event_response(data)
        return decode_calendar_event(data)

    def delete_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
    ) -> None:
        """Delete a calendar event.

        Args:
            calendar_id: The calendar ID (use "primary" for main calendar).
            event_id: The event ID to delete.

        Raises:
            AppError[CalendarErrorCode]: If the event is not found or API fails.
        """
        encoded_cal_id = urllib.parse.quote(calendar_id, safe="")
        encoded_event_id = urllib.parse.quote(event_id, safe="")
        endpoint = f"/calendars/{encoded_cal_id}/events/{encoded_event_id}"
        url = GOOGLE_CALENDAR_API_BASE + endpoint

        try:
            hooks.http_delete(url, self._get_headers())
        except ConnectionError as e:
            raise AppError(
                CalendarErrorCode.CALENDAR_API_ERROR,
                f"Delete request failed: {e}",
                http_status=502,
            ) from e
        except OSError as e:
            if isinstance(e, HTTPErrorProtocol):
                status_code: int = e.code
                body: str = e.read().decode("utf-8")
                self._handle_error(status_code, body, endpoint)
            raise AppError(
                CalendarErrorCode.CALENDAR_API_ERROR,
                f"Delete request failed: {e}",
                http_status=502,
            ) from e


# =============================================================================
# Factory Function
# =============================================================================


def google_calendar_client(*, tokens: OAuthTokens) -> CalendarClientProtocol:
    """Create a Google Calendar client.

    Args:
        tokens: OAuth tokens with access_token.

    Returns:
        CalendarClientProtocol implementation.
    """
    return _GoogleCalendarClient(access_token=tokens["access_token"])
