"""Competition tracking and calendar sync logic."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from platform_core.errors import AppError, CalendarErrorCode
from platform_core.json_utils import (
    InvalidJsonError,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from platform_calendar.config import get_competitions_path
from platform_calendar.testing import CalendarClientProtocol, hooks
from platform_calendar.types import (
    DEFAULT_REMINDERS,
    CalendarEvent,
    CompetitionsFile,
    EventDateTime,
    TrackedCompetition,
    decode_competitions_file,
    encode_competitions_file,
)

# =============================================================================
# File I/O
# =============================================================================


def load_competitions(path: Path | None = None) -> tuple[TrackedCompetition, ...]:
    """Load tracked competitions from JSON file.

    Args:
        path: Path to competitions file. Defaults to ~/.competitions/tracked.json.

    Returns:
        Tuple of TrackedCompetition.

    Raises:
        AppError[CalendarErrorCode]: If file is invalid.
    """
    file_path = str(path) if path else str(get_competitions_path())

    if not hooks.file_exists(file_path):
        return ()

    try:
        content = hooks.read_file(file_path)
    except PermissionError as e:
        msg = f"Failed to read competitions file: {e}"
        raise AppError(CalendarErrorCode.COMPETITIONS_FILE_ERROR, msg, http_status=500) from e
    except OSError as e:
        msg = f"Failed to read competitions file: {e}"
        raise AppError(CalendarErrorCode.COMPETITIONS_FILE_ERROR, msg, http_status=500) from e

    try:
        raw_value = load_json_str(content)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError) as e:
        msg = f"Invalid JSON in competitions file: {e}"
        raise AppError(CalendarErrorCode.COMPETITIONS_FILE_ERROR, msg, http_status=500) from e

    competitions_file = decode_competitions_file(data)
    return competitions_file["competitions"]


def save_competitions(
    competitions: tuple[TrackedCompetition, ...],
    path: Path | None = None,
) -> None:
    """Save tracked competitions to JSON file.

    Args:
        competitions: Competitions to save.
        path: Path to competitions file. Defaults to ~/.competitions/tracked.json.

    Raises:
        AppError[CalendarErrorCode]: If write fails.
    """
    file_path = str(path) if path else str(get_competitions_path())

    competitions_file = CompetitionsFile(competitions=competitions)
    content = dump_json_str(encode_competitions_file(competitions_file), indent=2)

    try:
        hooks.write_file(file_path, content)
    except PermissionError as e:
        msg = f"Failed to write competitions file: {e}"
        raise AppError(CalendarErrorCode.COMPETITIONS_FILE_ERROR, msg, http_status=500) from e
    except OSError as e:
        msg = f"Failed to write competitions file: {e}"
        raise AppError(CalendarErrorCode.COMPETITIONS_FILE_ERROR, msg, http_status=500) from e


# =============================================================================
# Competition Management
# =============================================================================


def add_competition(
    competitions: tuple[TrackedCompetition, ...],
    competition: TrackedCompetition,
) -> tuple[TrackedCompetition, ...]:
    """Add a competition to the list.

    Args:
        competitions: Existing competitions.
        competition: Competition to add.

    Returns:
        Updated tuple with new competition.

    Raises:
        AppError[CalendarErrorCode]: If competition ID already exists.
    """
    for c in competitions:
        if c["id"] == competition["id"]:
            msg = f"Competition '{competition['id']}' already exists"
            raise AppError(CalendarErrorCode.COMPETITION_ALREADY_EXISTS, msg, http_status=409)

    return (*competitions, competition)


def remove_competition(
    competitions: tuple[TrackedCompetition, ...],
    competition_id: str,
) -> tuple[TrackedCompetition, ...]:
    """Remove a competition from the list.

    Args:
        competitions: Existing competitions.
        competition_id: ID of competition to remove.

    Returns:
        Updated tuple without the competition.

    Raises:
        AppError[CalendarErrorCode]: If competition not found.
    """
    found = False
    result: list[TrackedCompetition] = []
    for c in competitions:
        if c["id"] == competition_id:
            found = True
        else:
            result.append(c)

    if not found:
        msg = f"Competition '{competition_id}' not found"
        raise AppError(CalendarErrorCode.COMPETITION_NOT_FOUND, msg, http_status=404)

    return tuple(result)


def get_competition(
    competitions: tuple[TrackedCompetition, ...],
    competition_id: str,
) -> TrackedCompetition:
    """Get a competition by ID.

    Args:
        competitions: Competitions to search.
        competition_id: ID to find.

    Returns:
        The TrackedCompetition.

    Raises:
        AppError[CalendarErrorCode]: If not found.
    """
    for c in competitions:
        if c["id"] == competition_id:
            return c

    msg = f"Competition '{competition_id}' not found"
    raise AppError(CalendarErrorCode.COMPETITION_NOT_FOUND, msg, http_status=404)


def update_competition(
    competitions: tuple[TrackedCompetition, ...],
    competition_id: str,
    *,
    name: str | None = None,
    deadline: str | None = None,
    url: str | None = None,
    project_path: str | None = None,
    calendar_event_id: str | None = None,
    reminders: tuple[int, ...] | None = None,
) -> tuple[TrackedCompetition, ...]:
    """Update a competition's fields.

    Args:
        competitions: Existing competitions.
        competition_id: ID of competition to update.
        name: New name (optional).
        deadline: New deadline (optional).
        url: New URL (optional).
        project_path: New project path (optional).
        calendar_event_id: New calendar event ID (optional).
        reminders: New reminders (optional).

    Returns:
        Updated tuple with modified competition.

    Raises:
        AppError[CalendarErrorCode]: If competition not found.
    """
    found = False
    result: list[TrackedCompetition] = []

    for c in competitions:
        if c["id"] == competition_id:
            found = True
            updated = TrackedCompetition(
                id=c["id"],
                source=c["source"],
                name=name if name is not None else c["name"],
                deadline=deadline if deadline is not None else c["deadline"],
                url=url if url is not None else c["url"],
                project_path=project_path if project_path is not None else c["project_path"],
                calendar_event_id=(
                    calendar_event_id if calendar_event_id is not None else c["calendar_event_id"]
                ),
                reminders=reminders if reminders is not None else c["reminders"],
            )
            result.append(updated)
        else:
            result.append(c)

    if not found:
        msg = f"Competition '{competition_id}' not found"
        raise AppError(CalendarErrorCode.COMPETITION_NOT_FOUND, msg, http_status=404)

    return tuple(result)


# =============================================================================
# Calendar Sync
# =============================================================================


def _parse_deadline(deadline: str) -> datetime:
    """Parse ISO 8601 deadline string to datetime."""
    # Handle common formats
    if deadline.endswith("Z"):
        deadline = deadline[:-1] + "+00:00"
    return datetime.fromisoformat(deadline)


def _format_datetime(dt: datetime) -> str:
    """Format datetime to RFC 3339 string."""
    return dt.isoformat()


def create_competition_event(
    client: CalendarClientProtocol,
    *,
    competition: TrackedCompetition,
    calendar_id: str = "primary",
) -> CalendarEvent:
    """Create a calendar event for a competition deadline.

    Args:
        client: Calendar client.
        competition: Competition to create event for.
        calendar_id: Calendar to create event in.

    Returns:
        Created CalendarEvent.
    """
    deadline_dt = _parse_deadline(competition["deadline"])

    # Event starts 1 hour before deadline, ends at deadline
    start_dt = deadline_dt - timedelta(hours=1)

    start = EventDateTime(
        dateTime=_format_datetime(start_dt),
        timeZone="UTC",
    )
    end = EventDateTime(
        dateTime=_format_datetime(deadline_dt),
        timeZone="UTC",
    )

    # Build description
    description_parts = [
        f"Competition: {competition['name']}",
        f"Source: {competition['source']}",
        f"URL: {competition['url']}",
    ]
    if competition["project_path"]:
        description_parts.append(f"Project: {competition['project_path']}")

    description = "\n".join(description_parts)

    return client.create_event(
        calendar_id=calendar_id,
        summary=f"DEADLINE: {competition['name']}",
        description=description,
        start=start,
        end=end,
        reminders=competition["reminders"],
    )


def sync_competition(
    client: CalendarClientProtocol,
    *,
    competition: TrackedCompetition,
    calendar_id: str = "primary",
) -> TrackedCompetition:
    """Sync a single competition to calendar.

    Creates event if not synced, returns updated competition with event ID.

    Args:
        client: Calendar client.
        competition: Competition to sync.
        calendar_id: Calendar to sync to.

    Returns:
        Updated TrackedCompetition with calendar_event_id set.
    """
    if competition["calendar_event_id"] is not None:
        # Already synced
        return competition

    event = create_competition_event(client, competition=competition, calendar_id=calendar_id)

    return TrackedCompetition(
        id=competition["id"],
        source=competition["source"],
        name=competition["name"],
        deadline=competition["deadline"],
        url=competition["url"],
        project_path=competition["project_path"],
        calendar_event_id=event["id"],
        reminders=competition["reminders"],
    )


def sync_all_competitions(
    client: CalendarClientProtocol,
    *,
    competitions: tuple[TrackedCompetition, ...],
    calendar_id: str = "primary",
) -> tuple[TrackedCompetition, ...]:
    """Sync all unsynced competitions to calendar.

    Args:
        client: Calendar client.
        competitions: Competitions to sync.
        calendar_id: Calendar to sync to.

    Returns:
        Updated tuple with calendar_event_id set for synced competitions.
    """
    result: list[TrackedCompetition] = []

    for c in competitions:
        synced = sync_competition(client, competition=c, calendar_id=calendar_id)
        result.append(synced)

    return tuple(result)


# =============================================================================
# Factory Helpers
# =============================================================================


def make_competition(
    *,
    competition_id: str,
    source: str,
    name: str,
    deadline: str,
    url: str,
    project_path: str | None = None,
    reminders: tuple[int, ...] | None = None,
) -> TrackedCompetition:
    """Create a new TrackedCompetition.

    Args:
        competition_id: Unique ID for the competition.
        source: Source platform (kaggle, devpost, manual).
        name: Competition name.
        deadline: ISO 8601 deadline datetime.
        url: URL to competition page.
        project_path: Optional path to associated project.
        reminders: Optional custom reminders (defaults to 1 day + 1 hour).

    Returns:
        TrackedCompetition ready to be added.

    Raises:
        ValueError: If source is invalid.
    """
    # Validate source
    if source not in ("kaggle", "devpost", "manual"):
        msg = f"Invalid source: {source}"
        raise ValueError(msg)

    actual_source: str = source

    return TrackedCompetition(
        id=competition_id,
        source=(
            "kaggle"
            if actual_source == "kaggle"
            else ("devpost" if actual_source == "devpost" else "manual")
        ),
        name=name,
        deadline=deadline,
        url=url,
        project_path=project_path,
        calendar_event_id=None,
        reminders=reminders if reminders is not None else DEFAULT_REMINDERS,
    )
