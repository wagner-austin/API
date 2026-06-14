"""Tests for built-in-radar coverage-grid primitives."""

from __future__ import annotations

from tankpit_bot.bot.ai.scan_coverage import (
    FORAGE_CELL_SIZE,
    FORAGE_COVERAGE_TTL_MS,
    cell_center,
    is_cell_covered,
    local_scan_cell_key,
    record_local_scan,
)


class TestCellGeometry:
    """Tests for cell-key and cell-center math."""

    def test_cell_key_floors_to_block_index(self) -> None:
        """Tiles in the same 5x5 block share a cell key."""
        assert local_scan_cell_key(0, 0) == "0,0"
        assert local_scan_cell_key(4, 4) == "0,0"
        assert local_scan_cell_key(5, 0) == "1,0"
        assert local_scan_cell_key(132, 180) == "26,36"

    def test_cell_center_is_block_midpoint(self) -> None:
        """A cell center sits at the middle tile of its 5x5 block."""
        assert cell_center(0, 0) == (2, 2)
        assert cell_center(26, 36) == (132, 182)

    def test_scan_at_cell_center_marks_that_cell(self) -> None:
        """A scan centered in a cell keys back to the same cell.

        This is the invariant that makes the grid sweep exact: the
        forager navigates to a cell center, scans, and the mark lands
        on the cell it was clearing.
        """
        center_x, center_y = cell_center(26, 36)
        assert local_scan_cell_key(center_x, center_y) == "26,36"


class TestCoverageMemory:
    """Tests for record_local_scan and is_cell_covered."""

    def test_record_marks_the_scan_cell(self) -> None:
        """Recording a scan marks the cell containing its center."""
        grid = record_local_scan({}, 132, 180, 100000)

        assert grid == {"26,36": 100000}
        assert is_cell_covered(grid, 26, 36, 100000) is True

    def test_uncovered_cell_reports_false(self) -> None:
        """A cell with no mark is never covered."""
        assert is_cell_covered({}, 26, 36, 100000) is False

    def test_coverage_expires_with_ttl(self) -> None:
        """A mark past the coverage TTL no longer counts as covered."""
        grid = {"26,36": 100000}

        assert is_cell_covered(grid, 26, 36, 100000 + FORAGE_COVERAGE_TTL_MS) is True
        assert is_cell_covered(grid, 26, 36, 100000 + FORAGE_COVERAGE_TTL_MS + 1) is False

    def test_record_prunes_expired_cells(self) -> None:
        """Recording drops marks aged past the TTL in the same pass."""
        stale = {"1,1": 100000 - FORAGE_COVERAGE_TTL_MS - 1}
        fresh = {"2,2": 100000 - 1000}

        grid = record_local_scan({**stale, **fresh}, 132, 180, 100000)

        assert grid == {"2,2": 100000 - 1000, "26,36": 100000}

    def test_record_does_not_mutate_input(self) -> None:
        """Recording returns a new grid and leaves the input untouched."""
        original = {"1,1": 99000}

        record_local_scan(original, 10, 10, 100000)

        assert original == {"1,1": 99000}

    def test_cell_size_is_radar_diameter(self) -> None:
        """The cell size equals the built-in radar's 5-tile diameter."""
        assert FORAGE_CELL_SIZE == 5
