"""Competition add/remove/get/update."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, CalendarErrorCode

from platform_calendar.competitions import (
    add_competition,
    get_competition,
    make_competition,
    remove_competition,
    update_competition,
)
from platform_calendar.types import DEFAULT_REMINDERS


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
