"""Competition persistence and calendar sync."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, CalendarErrorCode
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

from platform_calendar.competitions import (
    create_competition_event,
    load_competitions,
    make_competition,
    save_competitions,
    sync_all_competitions,
    sync_competition,
)
from platform_calendar.fakes import (
    FakeCalendarClient,
    make_fake_file_system,
)
from platform_calendar.testing import (
    hooks,
)
from platform_calendar.types import TrackedCompetition


class TestLoadSaveCompetitions:
    def test_load_nonexistent_file(self) -> None:
        read_hook, write_hook, exists_hook = make_fake_file_system({})
        hooks.read_file = read_hook
        hooks.write_file = write_hook
        hooks.file_exists = exists_hook

        result = load_competitions()
        assert result == ()

    def test_load_existing_file(self) -> None:
        comp_data = {
            "competitions": [
                {
                    "id": "test",
                    "source": "manual",
                    "name": "Test",
                    "deadline": "2025-12-26T22:00:00Z",
                    "url": "https://example.com",
                    "project_path": None,
                    "calendar_event_id": None,
                    "reminders": [1440, 60],
                }
            ]
        }
        # Override with proper file content
        file_content = dump_json_str(comp_data)

        def custom_read(path: str) -> str:
            return file_content

        def custom_exists(path: str) -> bool:
            return True

        hooks.read_file = custom_read
        hooks.file_exists = custom_exists

        result = load_competitions()
        assert len(result) == 1
        assert result[0]["id"] == "test"

    def test_load_invalid_json(self) -> None:
        def custom_read(path: str) -> str:
            return "not valid json"

        def custom_exists(path: str) -> bool:
            return True

        hooks.read_file = custom_read
        hooks.file_exists = custom_exists

        with pytest.raises(AppError) as exc_info:
            load_competitions()
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.COMPETITIONS_FILE_ERROR
        assert "Invalid JSON" in error.message

    def test_load_not_object(self) -> None:
        def custom_read(path: str) -> str:
            return "[]"

        def custom_exists(path: str) -> bool:
            return True

        hooks.read_file = custom_read
        hooks.file_exists = custom_exists

        with pytest.raises(AppError) as exc_info:
            load_competitions()
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.COMPETITIONS_FILE_ERROR
        assert "Expected JSON object" in error.message

    def test_load_read_error(self) -> None:
        def custom_read(path: str) -> str:
            raise PermissionError("Access denied")

        def custom_exists(path: str) -> bool:
            return True

        hooks.read_file = custom_read
        hooks.file_exists = custom_exists

        with pytest.raises(AppError) as exc_info:
            load_competitions()
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.COMPETITIONS_FILE_ERROR
        assert "Failed to read" in error.message

    def test_load_read_os_error(self) -> None:
        def custom_read(path: str) -> str:
            raise OSError("Disk error")

        def custom_exists(path: str) -> bool:
            return True

        hooks.read_file = custom_read
        hooks.file_exists = custom_exists

        with pytest.raises(AppError) as exc_info:
            load_competitions()
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.COMPETITIONS_FILE_ERROR
        assert "Failed to read" in error.message

    def test_save_competitions(self) -> None:
        saved_content: list[str] = []

        def custom_write(path: str, content: str) -> None:
            saved_content.append(content)

        hooks.write_file = custom_write

        comp = make_competition(
            competition_id="test",
            source="manual",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
        )
        save_competitions((comp,))

        assert len(saved_content) == 1
        # Decode the saved JSON and verify the structure
        from platform_calendar.types import decode_competitions_file

        raw_value = load_json_str(saved_content[0])
        data = narrow_json_to_dict(raw_value)
        decoded = decode_competitions_file(data)
        assert len(decoded["competitions"]) == 1

    def test_save_write_error(self) -> None:
        def custom_write(path: str, content: str) -> None:
            raise PermissionError("Access denied")

        hooks.write_file = custom_write

        comp = make_competition(
            competition_id="test",
            source="manual",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
        )

        with pytest.raises(AppError) as exc_info:
            save_competitions((comp,))
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.COMPETITIONS_FILE_ERROR
        assert "Failed to write" in error.message

    def test_save_write_os_error(self) -> None:
        def custom_write(path: str, content: str) -> None:
            raise OSError("Disk full")

        hooks.write_file = custom_write

        comp = make_competition(
            competition_id="test",
            source="manual",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
        )

        with pytest.raises(AppError) as exc_info:
            save_competitions((comp,))
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.COMPETITIONS_FILE_ERROR
        assert "Failed to write" in error.message


class TestCreateCompetitionEvent:
    def test_create_event(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        comp = make_competition(
            competition_id="test",
            source="devpost",
            name="Test Competition",
            deadline="2025-12-26T22:00:00Z",
            url="https://devpost.com/test",
            project_path="libs/test",
            reminders=(1440, 60),
        )

        event = create_competition_event(client, competition=comp)

        assert "DEADLINE: Test Competition" in event["summary"]
        assert "devpost" in event["description"]
        assert "libs/test" in event["description"]
        assert len(event["reminders"]["overrides"]) == 2

    def test_create_event_no_project(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        comp = make_competition(
            competition_id="test",
            source="kaggle",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://kaggle.com/test",
        )

        event = create_competition_event(client, competition=comp)
        assert "Project:" not in event["description"]

    def test_create_event_with_timezone_offset(self) -> None:
        # Test deadline with timezone offset instead of "Z"
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        comp = make_competition(
            competition_id="test",
            source="manual",
            name="Test",
            deadline="2025-12-26T22:00:00+00:00",  # Uses +00:00 instead of Z
            url="https://example.com",
        )

        event = create_competition_event(client, competition=comp)
        assert "DEADLINE: Test" in event["summary"]


class TestSyncCompetition:
    def test_sync_unsynced(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        comp = make_competition(
            competition_id="test",
            source="manual",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
        )

        synced = sync_competition(client, competition=comp)
        # The event ID should be set to a fake event ID
        event_id = synced["calendar_event_id"]
        assert event_id is not None and event_id.startswith("fake_event_")

    def test_sync_already_synced(self) -> None:
        client = FakeCalendarClient()

        comp = TrackedCompetition(
            id="test",
            source="manual",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
            project_path=None,
            calendar_event_id="existing_event",
            reminders=(1440,),
        )

        synced = sync_competition(client, competition=comp)
        assert synced["calendar_event_id"] == "existing_event"
        # No new events should be created
        assert len(client.get_created_events()) == 0


class TestSyncAllCompetitions:
    def test_sync_multiple(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        comp1 = make_competition(
            competition_id="comp1",
            source="manual",
            name="Comp 1",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
        )
        comp2 = make_competition(
            competition_id="comp2",
            source="manual",
            name="Comp 2",
            deadline="2025-12-27T22:00:00Z",
            url="https://example.com",
        )

        synced = sync_all_competitions(client, competitions=(comp1, comp2))

        assert len(synced) == 2
        # Both events should have fake event IDs assigned
        event_id_0 = synced[0]["calendar_event_id"]
        event_id_1 = synced[1]["calendar_event_id"]
        assert event_id_0 is not None and event_id_0.startswith("fake_event_")
        assert event_id_1 is not None and event_id_1.startswith("fake_event_")
        assert len(client.get_created_events()) == 2

    def test_sync_mixed_synced_unsynced(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        comp1 = TrackedCompetition(
            id="comp1",
            source="manual",
            name="Already Synced",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
            project_path=None,
            calendar_event_id="existing",
            reminders=(1440,),
        )
        comp2 = make_competition(
            competition_id="comp2",
            source="manual",
            name="Not Synced",
            deadline="2025-12-27T22:00:00Z",
            url="https://example.com",
        )

        synced = sync_all_competitions(client, competitions=(comp1, comp2))

        assert synced[0]["calendar_event_id"] == "existing"
        # The unsynced competition should now have a fake event ID
        event_id_1 = synced[1]["calendar_event_id"]
        assert event_id_1 is not None and event_id_1.startswith("fake_event_")
        assert len(client.get_created_events()) == 1
