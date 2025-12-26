"""Tests for platform_calendar.competitions module."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, CalendarErrorCode
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

from platform_calendar.competitions import (
    add_competition,
    create_competition_event,
    get_competition,
    load_competitions,
    make_competition,
    remove_competition,
    save_competitions,
    sync_all_competitions,
    sync_competition,
    update_competition,
)
from platform_calendar.testing import (
    FakeCalendarClient,
    hooks,
    make_fake_file_system,
)
from platform_calendar.types import DEFAULT_REMINDERS, TrackedCompetition


class TestMakeCompetition:
    def test_make_competition_defaults(self) -> None:
        comp = make_competition(
            competition_id="test-comp",
            source="devpost",
            name="Test Competition",
            deadline="2025-12-26T22:00:00Z",
            url="https://devpost.com/test",
        )
        assert comp["id"] == "test-comp"
        assert comp["source"] == "devpost"
        assert comp["project_path"] is None
        assert comp["calendar_event_id"] is None
        assert comp["reminders"] == DEFAULT_REMINDERS

    def test_make_competition_all_sources(self) -> None:
        for source in ("kaggle", "devpost", "manual"):
            comp = make_competition(
                competition_id="test",
                source=source,
                name="Test",
                deadline="2025-12-26T22:00:00Z",
                url="https://example.com",
            )
            assert comp["source"] == source

    def test_make_competition_invalid_source(self) -> None:
        with pytest.raises(ValueError, match="Invalid source"):
            make_competition(
                competition_id="test",
                source="invalid",
                name="Test",
                deadline="2025-12-26T22:00:00Z",
                url="https://example.com",
            )

    def test_make_competition_with_options(self) -> None:
        comp = make_competition(
            competition_id="test",
            source="kaggle",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://kaggle.com/test",
            project_path="libs/test",
            reminders=(60,),
        )
        assert comp["project_path"] == "libs/test"
        assert comp["reminders"] == (60,)


class TestAddCompetition:
    def test_add_to_empty(self) -> None:
        comp = make_competition(
            competition_id="test",
            source="manual",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
        )
        result = add_competition((), comp)
        assert len(result) == 1
        assert result[0]["id"] == "test"

    def test_add_to_existing(self) -> None:
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
        result = add_competition((comp1,), comp2)
        assert len(result) == 2

    def test_add_duplicate_id(self) -> None:
        comp = make_competition(
            competition_id="test",
            source="manual",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
        )
        with pytest.raises(AppError) as exc_info:
            add_competition((comp,), comp)
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.COMPETITION_ALREADY_EXISTS
        assert "already exists" in error.message


class TestRemoveCompetition:
    def test_remove_existing(self) -> None:
        comp = make_competition(
            competition_id="test",
            source="manual",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
        )
        result = remove_competition((comp,), "test")
        assert len(result) == 0

    def test_remove_from_multiple(self) -> None:
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
        result = remove_competition((comp1, comp2), "comp1")
        assert len(result) == 1
        assert result[0]["id"] == "comp2"

    def test_remove_not_found(self) -> None:
        with pytest.raises(AppError) as exc_info:
            remove_competition((), "nonexistent")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.COMPETITION_NOT_FOUND
        assert "not found" in error.message


class TestGetCompetition:
    def test_get_existing(self) -> None:
        comp = make_competition(
            competition_id="test",
            source="manual",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
        )
        result = get_competition((comp,), "test")
        assert result["id"] == "test"

    def test_get_finds_target_after_iterating(self) -> None:
        """Test get_competition when target is not the first item (covers loop branch)."""
        comp1 = make_competition(
            competition_id="first",
            source="manual",
            name="First",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com/1",
        )
        comp2 = make_competition(
            competition_id="second",
            source="manual",
            name="Second",
            deadline="2025-12-27T22:00:00Z",
            url="https://example.com/2",
        )
        comp3 = make_competition(
            competition_id="third",
            source="manual",
            name="Third",
            deadline="2025-12-28T22:00:00Z",
            url="https://example.com/3",
        )
        # Find the second item - requires iterating past the first
        result = get_competition((comp1, comp2, comp3), "second")
        assert result["id"] == "second"
        assert result["name"] == "Second"

    def test_get_not_found(self) -> None:
        with pytest.raises(AppError) as exc_info:
            get_competition((), "nonexistent")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.COMPETITION_NOT_FOUND
        assert "not found" in error.message


class TestUpdateCompetition:
    def test_update_name(self) -> None:
        comp = make_competition(
            competition_id="test",
            source="manual",
            name="Original",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
        )
        result = update_competition((comp,), "test", name="Updated")
        assert result[0]["name"] == "Updated"

    def test_update_multiple_fields(self) -> None:
        comp = make_competition(
            competition_id="test",
            source="manual",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
        )
        result = update_competition(
            (comp,),
            "test",
            deadline="2025-12-27T22:00:00Z",
            url="https://new-url.com",
            project_path="libs/new",
        )
        assert result[0]["deadline"] == "2025-12-27T22:00:00Z"
        assert result[0]["url"] == "https://new-url.com"
        assert result[0]["project_path"] == "libs/new"

    def test_update_calendar_event_id(self) -> None:
        comp = make_competition(
            competition_id="test",
            source="manual",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
        )
        result = update_competition((comp,), "test", calendar_event_id="event123")
        assert result[0]["calendar_event_id"] == "event123"

    def test_update_reminders(self) -> None:
        comp = make_competition(
            competition_id="test",
            source="manual",
            name="Test",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com",
        )
        result = update_competition((comp,), "test", reminders=(60, 120))
        assert result[0]["reminders"] == (60, 120)

    def test_update_not_found(self) -> None:
        with pytest.raises(AppError) as exc_info:
            update_competition((), "nonexistent", name="Test")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.COMPETITION_NOT_FOUND
        assert "not found" in error.message

    def test_update_preserves_other_competitions(self) -> None:
        # Test that updating one competition preserves others (hits else branch)
        comp1 = make_competition(
            competition_id="comp1",
            source="manual",
            name="Competition 1",
            deadline="2025-12-26T22:00:00Z",
            url="https://example.com/1",
        )
        comp2 = make_competition(
            competition_id="comp2",
            source="manual",
            name="Competition 2",
            deadline="2025-12-27T22:00:00Z",
            url="https://example.com/2",
        )
        result = update_competition((comp1, comp2), "comp1", name="Updated Name")

        # Check that comp1 was updated
        assert result[0]["id"] == "comp1"
        assert result[0]["name"] == "Updated Name"
        # Check that comp2 was preserved unchanged (tests the else branch)
        assert result[1]["id"] == "comp2"
        assert result[1]["name"] == "Competition 2"


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
