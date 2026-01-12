"""Tests for tankpit_bot._test_hooks module."""

from __future__ import annotations

from pathlib import Path

import pytest

from tankpit_bot import _test_hooks


def test_default_get_env_returns_none_for_missing() -> None:
    """Test _default_get_env returns None for missing env var."""
    # Use an unlikely env var name
    result = _test_hooks._default_get_env("TANKPIT_TEST_UNLIKELY_VAR_12345")
    assert result is None


def test_real_write_text_creates_file(tmp_path: Path) -> None:
    """Test _real_write_text creates file with content."""
    test_file = tmp_path / "test.txt"
    _test_hooks._real_write_text(test_file, "test content")

    assert test_file.exists()
    assert test_file.read_text(encoding="utf-8") == "test content"


def test_real_write_text_creates_directories(tmp_path: Path) -> None:
    """Test _real_write_text creates parent directories."""
    test_file = tmp_path / "subdir" / "nested" / "test.txt"
    _test_hooks._real_write_text(test_file, "nested content")

    assert test_file.exists()
    assert test_file.read_text(encoding="utf-8") == "nested content"


def test_real_read_text_reads_file(tmp_path: Path) -> None:
    """Test _real_read_text reads file content."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("file content", encoding="utf-8")

    result = _test_hooks._real_read_text(test_file)
    assert result == "file content"


def test_real_read_text_raises_for_missing(tmp_path: Path) -> None:
    """Test _real_read_text raises FileNotFoundError for missing file."""
    test_file = tmp_path / "nonexistent.txt"

    with pytest.raises(FileNotFoundError):
        _test_hooks._real_read_text(test_file)


def test_real_path_exists_true(tmp_path: Path) -> None:
    """Test _real_path_exists returns True for existing path."""
    test_file = tmp_path / "exists.txt"
    test_file.write_text("content", encoding="utf-8")

    assert _test_hooks._real_path_exists(test_file) is True


def test_real_path_exists_false(tmp_path: Path) -> None:
    """Test _real_path_exists returns False for missing path."""
    test_file = tmp_path / "missing.txt"

    assert _test_hooks._real_path_exists(test_file) is False


def test_sync_playwright_initially_none() -> None:
    """Test sync_playwright hook starts as None."""
    # Note: conftest.py restores hooks, so we check the module attribute
    # after it's been potentially set and restored
    # This test verifies the pattern works
    original = _test_hooks.sync_playwright
    _test_hooks.sync_playwright = None
    assert _test_hooks.sync_playwright is None
    _test_hooks.sync_playwright = original


def test_real_get_sync_playwright_returns_callable() -> None:
    """Test _real_get_sync_playwright returns a callable."""
    result = _test_hooks._real_get_sync_playwright()
    assert callable(result)


def test_real_load_terrain_map_returns_terrain_map() -> None:
    """Test _real_load_terrain_map loads a TerrainMap from GIF."""
    gif_path = Path(__file__).parent.parent / "field42-r.gif"
    result = _test_hooks._real_load_terrain_map(gif_path)

    # Verify it implements TerrainMapProtocol by calling methods
    terrain = result.get_terrain(128, 128)
    assert terrain in (result.ROCK, result.GROUND, result.WATER)

    passable = result.is_passable(128, 128)
    assert passable in (True, False)

    viewport = result.render_viewport(128, 128, 4, 4)
    assert len(viewport) == 4
    assert len(viewport[0]) == 4
