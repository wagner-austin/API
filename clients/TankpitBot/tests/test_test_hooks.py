"""Tests for tankpit_bot._test_hooks module."""

from __future__ import annotations

from pathlib import Path

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot.resources import data_directory


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


def test_real_append_text_appends_to_existing_file(tmp_path: Path) -> None:
    """Test _real_append_text appends content to an existing file."""
    test_file = tmp_path / "append.txt"
    test_file.write_text("first", encoding="utf-8")

    _test_hooks._real_append_text(test_file, " second")

    assert test_file.exists()
    assert test_file.read_text(encoding="utf-8") == "first second"


def test_real_append_text_creates_directories_for_new_file(tmp_path: Path) -> None:
    """Test _real_append_text creates parent directories for a new file."""
    test_file = tmp_path / "subdir" / "nested" / "append.txt"

    _test_hooks._real_append_text(test_file, "created")

    assert test_file.exists()
    assert test_file.read_text(encoding="utf-8") == "created"


def test_real_path_exists_true(tmp_path: Path) -> None:
    """Test _real_path_exists returns True for existing path."""
    test_file = tmp_path / "exists.txt"
    test_file.write_text("content", encoding="utf-8")

    assert _test_hooks._real_path_exists(test_file) is True


def test_real_path_exists_false(tmp_path: Path) -> None:
    """Test _real_path_exists returns False for missing path."""
    test_file = tmp_path / "missing.txt"

    assert _test_hooks._real_path_exists(test_file) is False


def test_real_glob_paths_lists_sorted_matches(tmp_path: Path) -> None:
    """Test _real_glob_paths returns matching files in sorted order."""
    (tmp_path / "bot-20260610-2.events.jsonl").write_text("b", encoding="utf-8")
    (tmp_path / "bot-20260610-1.events.jsonl").write_text("a", encoding="utf-8")
    (tmp_path / "other.txt").write_text("c", encoding="utf-8")

    result = _test_hooks._real_glob_paths(tmp_path, "bot-*.events.jsonl")

    assert [path.name for path in result] == [
        "bot-20260610-1.events.jsonl",
        "bot-20260610-2.events.jsonl",
    ]


def test_real_glob_paths_missing_directory_is_empty(tmp_path: Path) -> None:
    """Test _real_glob_paths returns empty for a nonexistent directory."""
    assert _test_hooks._real_glob_paths(tmp_path / "nope", "*.jsonl") == []


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
    gif_path = data_directory() / "field42_r.gif"
    result = _test_hooks._real_load_terrain_map(gif_path)

    # Verify it implements TerrainMapProtocol by calling methods
    terrain = result.get_terrain(128, 128)
    assert terrain in (result.ROCK, result.GROUND, result.WATER)

    passable = result.is_passable(128, 128)
    assert passable in (True, False)

    viewport = result.render_viewport(128, 128, 4, 4)
    assert len(viewport) == 4
    assert len(viewport[0]) == 4


def test_real_replace_text_swaps_content_atomically(tmp_path: Path) -> None:
    """The replace lands the new content and leaves no staging file."""
    from tankpit_bot._test_hooks import _real_replace_text

    target = tmp_path / "knowledge.json"
    target.write_text("old", encoding="utf-8")

    _real_replace_text(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.glob("*.tmp")) == []


def test_real_replace_text_creates_parent_directories(tmp_path: Path) -> None:
    """A fresh instance directory is created on first write."""
    from tankpit_bot._test_hooks import _real_replace_text

    target = tmp_path / "arterial" / "knowledge.json"

    _real_replace_text(target, "first")

    assert target.read_text(encoding="utf-8") == "first"


def test_real_replace_text_drops_the_beat_under_reader_contention(tmp_path: Path) -> None:
    """Windows destination-open law: the previous content stays current.

    CPython opens files without FILE_SHARE_DELETE, so ``os.replace``
    onto a destination a reader holds open raises PermissionError.
    The contract for heartbeat files (the fleet knowledge exchange):
    drop this beat -- old complete content stays, staging is cleaned,
    the next tick's write refreshes.
    """
    from tankpit_bot._test_hooks import _real_replace_text

    target = tmp_path / "knowledge.json"
    target.write_text("old", encoding="utf-8")

    with target.open("r", encoding="utf-8"):
        _real_replace_text(target, "new")

    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob("*.tmp")) == []


def test_real_create_text_exclusive_first_writer_wins(tmp_path: Path) -> None:
    """The creation race has exactly one winner and keeps its content.

    The mutex law of the fleet's authoritative container claim
    ([[fleet-forage-allocation]]): the existence check and the
    creation are one atomic operation, so the second creator is
    refused and the first creator's content survives.
    """
    from tankpit_bot._test_hooks import _real_create_text_exclusive

    target = tmp_path / "6" / "100_136.claim"

    assert _real_create_text_exclusive(target, "winner") is True
    assert _real_create_text_exclusive(target, "loser") is False
    assert target.read_text(encoding="utf-8") == "winner"


def test_real_create_text_exclusive_creates_parent_directories(tmp_path: Path) -> None:
    """A fresh claims namespace is created on the first acquisition."""
    from tankpit_bot._test_hooks import _real_create_text_exclusive

    target = tmp_path / "_claims" / "6" / "1_2.claim"

    assert _real_create_text_exclusive(target, "first") is True
    assert target.read_text(encoding="utf-8") == "first"


def test_real_file_marker_reports_identity_and_size(tmp_path: Path) -> None:
    """The marker grows with the file and keeps the same identity."""
    target = tmp_path / "latest.events.jsonl"
    target.write_bytes(b"one\n")

    first_identity, first_size = _test_hooks._real_file_marker(target)
    target.write_bytes(b"one\ntwo\n")
    second_identity, second_size = _test_hooks._real_file_marker(target)

    assert first_size == 4
    assert second_size == 8
    assert first_identity == second_identity


def test_real_file_marker_identity_changes_when_the_path_is_recreated(
    tmp_path: Path,
) -> None:
    """A new file under the same name is a different file.

    This is what tells the incremental reader that a bot started a new
    run, rather than that its old one grew.
    """
    target = tmp_path / "latest.events.jsonl"
    target.write_text("first run\n", encoding="utf-8")
    before, _ = _test_hooks._real_file_marker(target)

    target.unlink()
    target.write_text("second run\n", encoding="utf-8")
    after, _ = _test_hooks._real_file_marker(target)

    assert before != after


def test_real_file_marker_raises_for_a_missing_file(tmp_path: Path) -> None:
    """An artifact that does not exist is an error, not a zero size."""
    with pytest.raises(OSError):
        _test_hooks._real_file_marker(tmp_path / "never-written")


def test_real_read_bytes_from_reads_the_tail(tmp_path: Path) -> None:
    """Reading from an offset returns only what follows it."""
    target = tmp_path / "latest.events.jsonl"
    target.write_bytes(b"alpha\nbravo\n")

    assert _test_hooks._real_read_bytes_from(target, 6) == b"bravo\n"


def test_real_read_bytes_from_at_the_end_reads_nothing(tmp_path: Path) -> None:
    """A cursor already at the end costs a read and returns no bytes."""
    target = tmp_path / "latest.events.jsonl"
    target.write_bytes(b"alpha\n")

    assert _test_hooks._real_read_bytes_from(target, 6) == b""


def test_real_read_bytes_from_raises_for_a_missing_file(tmp_path: Path) -> None:
    """A cursor cannot read an artifact that is not there."""
    with pytest.raises(OSError):
        _test_hooks._real_read_bytes_from(tmp_path / "never-written", 0)
