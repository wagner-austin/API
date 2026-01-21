"""Tests for run_probe function."""

from __future__ import annotations

from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks
from tankpit_bot.probe import DEFAULT_MOUSE_POSITIONS, DEFAULT_PROBE_KEYS, run_probe
from tankpit_bot.types import decode_probe_session
from tests.conftest import FakeFileSystem
from tests.fakes import fake_sync_playwright_probe


def test_run_probe_saves_to_file(fake_fs: FakeFileSystem) -> None:
    """Test run_probe saves probe session to file."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe

    session = run_probe(
        "https://tankpit.com/play",
        "probe_output.json",
        headless=True,
        probe_keys=["s"],  # Use key with known command mapping
        probe_mouse_positions=[],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    written_files = fake_fs.get_written_files()
    content = written_files["probe_output.json"]
    parsed = load_json_str(content)
    parsed_dict = narrow_json_to_dict(parsed)
    decoded = decode_probe_session(parsed_dict)
    assert decoded["session_id"] == session["session_id"]


def test_run_probe_uses_defaults(fake_fs: FakeFileSystem) -> None:
    """Test run_probe uses default keys and positions when not specified."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe

    session = run_probe(
        "https://tankpit.com/play",
        "probe_output.json",
        headless=True,
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    # Should have results for all default keys + mouse positions
    expected_count = len(DEFAULT_PROBE_KEYS) + len(DEFAULT_MOUSE_POSITIONS)
    assert len(session["results"]) == expected_count


def test_run_probe_multiple_mouse_positions_with_messages(
    fake_fs: FakeFileSystem,
) -> None:
    """Test run_probe with multiple mouse positions that emit messages.

    This covers the branch 573->567 (loop continuation after mouse result).
    """
    from tests.fakes import fake_sync_playwright_probe_mouse_emits

    _test_hooks.sync_playwright = fake_sync_playwright_probe_mouse_emits

    session = run_probe(
        "https://tankpit.com/play",
        "probe_output.json",
        headless=True,
        probe_keys=[],
        probe_mouse_positions=[(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    # All 3 mouse positions should have generated messages
    results_with_messages = [r for r in session["results"] if len(r["messages_after"]) > 0]
    assert len(results_with_messages) == 3
    for r in results_with_messages:
        assert r["input"]["input_type"] == "mouse"
