"""Tests for scripts._test_hooks module."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from scripts import _test_hooks

from tankpit_bot import _test_hooks as core_hooks


@pytest.fixture(autouse=True)
def _restore_real_hooks() -> Generator[None, None, None]:
    """Force real script/core hooks for these integration-style hook tests."""
    _test_hooks.path_exists = _test_hooks._real_path_exists
    _test_hooks.read_text = _test_hooks._real_read_text
    _test_hooks.load_and_decode_session = _test_hooks._real_load_and_decode_session
    _test_hooks.setup_rich_logging = _test_hooks._real_setup_rich_logging
    core_hooks.path_exists = core_hooks._real_path_exists
    core_hooks.read_text = core_hooks._real_read_text
    yield
    _test_hooks.path_exists = _test_hooks._real_path_exists
    _test_hooks.read_text = _test_hooks._real_read_text
    _test_hooks.load_and_decode_session = _test_hooks._real_load_and_decode_session
    _test_hooks.setup_rich_logging = _test_hooks._real_setup_rich_logging
    core_hooks.path_exists = core_hooks._real_path_exists
    core_hooks.read_text = core_hooks._real_read_text


def test_real_path_exists_true(tmp_path: Path) -> None:
    """_real_path_exists returns True for existing path."""
    test_file = tmp_path / "exists.txt"
    test_file.write_text("content", encoding="utf-8")
    assert _test_hooks._real_path_exists(test_file) is True


def test_real_path_exists_false(tmp_path: Path) -> None:
    """_real_path_exists returns False for missing path."""
    test_file = tmp_path / "missing.txt"
    assert _test_hooks._real_path_exists(test_file) is False


def test_real_read_text_reads_file(tmp_path: Path) -> None:
    """_real_read_text reads file content."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("file content", encoding="utf-8")
    result = _test_hooks._real_read_text(test_file)
    assert result == "file content"


def test_real_load_and_decode_session_loads_session(tmp_path: Path) -> None:
    """_real_load_and_decode_session loads a valid session."""
    from platform_core.json_utils import dump_json_str

    session_file = tmp_path / "session.json"
    session_data = {
        "session_id": "test-123",
        "start_timestamp_ms": 1000,
        "end_timestamp_ms": 2000,
        "base_url": "https://tankpit.com",
        "messages": [],
        "magic": "testmagic123456789",
    }
    session_file.write_text(dump_json_str(session_data), encoding="utf-8")

    decoder = _test_hooks._real_load_and_decode_session(session_file)
    assert len(decoder.commands) == 0
    assert len(decoder.lobby_messages) == 0


def test_real_setup_rich_logging() -> None:
    """_real_setup_rich_logging sets up logging without error."""
    _test_hooks._real_setup_rich_logging("INFO")


def test_path_exists_hook_is_callable() -> None:
    """path_exists hook is callable with Path argument."""
    result = _test_hooks.path_exists(Path(__file__))
    assert result is True


def test_read_text_hook_is_callable() -> None:
    """read_text hook is callable with Path argument."""
    result = _test_hooks.read_text(Path(__file__))
    assert "test_read_text_hook_is_callable" in result


def test_load_and_decode_session_hook_is_callable(tmp_path: Path) -> None:
    """load_and_decode_session hook is callable."""
    from platform_core.json_utils import dump_json_str

    session_file = tmp_path / "session.json"
    session_data = {
        "session_id": "test-hook",
        "start_timestamp_ms": 1000,
        "end_timestamp_ms": 2000,
        "base_url": "https://tankpit.com",
        "messages": [],
        "magic": "testmagic123456789",
    }
    session_file.write_text(dump_json_str(session_data), encoding="utf-8")

    decoder = _test_hooks.load_and_decode_session(session_file)
    assert len(decoder.commands) == 0


def test_setup_rich_logging_hook_is_callable() -> None:
    """setup_rich_logging hook is callable."""
    _test_hooks.setup_rich_logging("WARNING")


def test_real_http_get_returns_response() -> None:
    """_real_http_get returns a response with status_code and content."""
    response = _test_hooks._real_http_get("https://tankpit.com/play/field01.gif")
    assert response.status_code == 200
    assert response.content != b""
