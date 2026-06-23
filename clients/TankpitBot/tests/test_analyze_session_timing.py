"""Tests for scripts.analyze_session_timing module."""

from __future__ import annotations

import runpy
import warnings
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str
from scripts.analyze_session_timing import analyze_timing, render_timing_report

from scripts import _test_hooks
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.types import encode_captured_message


def _make_capture(messages_data: list[JSONObject]) -> JSONObject:
    """Build a minimal capture session dict."""
    msgs: list[JSONValue] = list(messages_data)
    return {
        "session_id": "test-timing",
        "start_timestamp_ms": 1000,
        "end_timestamp_ms": 9000,
        "base_url": "https://tankpit.com",
        "messages": msgs,
        "magic": "testmagic123456789",
    }


def _sent(ts: int, label: str) -> JSONObject:
    """Build a sent message dict."""
    return encode_captured_message(
        {
            "timestamp_ms": ts,
            "direction": "sent",
            "payload": "binary_data",
            "ws_url": "wss://test",
            "sent_label": label,
        }
    )


def _received(ts: int) -> JSONObject:
    """Build a received message dict."""
    return encode_captured_message(
        {
            "timestamp_ms": ts,
            "direction": "received",
            "payload": "binary_response",
            "ws_url": "wss://test",
        }
    )


@pytest.fixture(autouse=True)
def _isolate_hooks() -> Generator[None, None, None]:
    """Save and restore hooks around each test."""
    orig_setup = _test_hooks.setup_rich_logging
    orig_argv = core_hooks.get_argv
    yield
    _test_hooks.setup_rich_logging = orig_setup
    core_hooks.get_argv = orig_argv


def test_analyze_timing_extracts_latencies(tmp_path: Path) -> None:
    """Extracts per-command latencies from sent/received pairs."""
    capture = _make_capture(
        [
            _sent(1000, "shoot(100,100,id=50)"),
            _received(1350),
            _sent(3000, "shoot(100,100,id=50)"),
            _received(3100),
        ]
    )
    path = tmp_path / "test.capture_session.json"
    path.write_text(dump_json_str(capture), encoding="utf-8")

    report = analyze_timing(path)

    assert report["total_commands"] == 2
    assert report["timings"][0]["latency_ms"] == 350
    assert report["timings"][1]["latency_ms"] == 100


def test_analyze_timing_computes_shoot_gaps(tmp_path: Path) -> None:
    """Computes gaps between consecutive shoot commands."""
    capture = _make_capture(
        [
            _sent(1000, "shoot(100,100,id=50)"),
            _received(1100),
            _sent(3000, "shoot(100,100,id=50)"),
            _received(3100),
            _sent(5100, "shoot(100,100,id=50)"),
            _received(5200),
        ]
    )
    path = tmp_path / "test.capture_session.json"
    path.write_text(dump_json_str(capture), encoding="utf-8")

    report = analyze_timing(path)

    assert report["shoot_gaps_ms"] == [2000, 2100]
    assert report["avg_shoot_gap_ms"] == 2050


def test_analyze_timing_skips_unlabeled_commands(tmp_path: Path) -> None:
    """Sent messages without labels are ignored."""
    capture = _make_capture(
        [
            {"timestamp_ms": 1000, "direction": "sent", "payload": "x", "ws_url": "wss://t"},
            _received(1100),
        ]
    )
    path = tmp_path / "test.capture_session.json"
    path.write_text(dump_json_str(capture), encoding="utf-8")

    report = analyze_timing(path)

    assert report["total_commands"] == 0


def test_render_timing_report_produces_output(tmp_path: Path) -> None:
    """Render produces readable output."""
    capture = _make_capture(
        [
            _sent(1000, "shoot(100,100,id=50)"),
            _received(1200),
            _sent(3000, "shoot(100,100,id=50)"),
            _received(3200),
        ]
    )
    path = tmp_path / "test.capture_session.json"
    path.write_text(dump_json_str(capture), encoding="utf-8")

    report = analyze_timing(path)
    rendered = render_timing_report(report)

    assert "SESSION TIMING" in rendered
    assert "avg latency: 200ms" in rendered


def test_analyze_rejects_non_object_json(tmp_path: Path) -> None:
    """Non-object JSON raises ValueError."""
    path = tmp_path / "bad.json"
    path.write_text('"just a string"', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        analyze_timing(path)


def test_analyze_rejects_non_list_messages(tmp_path: Path) -> None:
    """Non-list messages field raises ValueError."""
    path = tmp_path / "bad.json"
    path.write_text('{"messages": "not_a_list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="messages must be a list"):
        analyze_timing(path)


def test_analyze_skips_non_dict_message_entries(tmp_path: Path) -> None:
    """Non-dict entries in messages list are skipped."""
    data: JSONObject = {
        "messages": ["not_a_dict", 42],
    }
    path = tmp_path / "mixed.json"
    path.write_text(dump_json_str(data), encoding="utf-8")
    report = analyze_timing(path)
    assert report["total_commands"] == 0


def test_analyze_handles_no_response_for_command(tmp_path: Path) -> None:
    """A sent command with no subsequent received message gets latency 0."""
    capture = _make_capture([_sent(1000, "shoot(100,100,id=50)")])
    path = tmp_path / "no_response.json"
    path.write_text(dump_json_str(capture), encoding="utf-8")
    report = analyze_timing(path)
    assert report["total_commands"] == 1
    assert report["timings"][0]["latency_ms"] == 0


def test_analyze_skips_sent_messages_looking_for_response(tmp_path: Path) -> None:
    """Consecutive sent messages are skipped when searching for a response."""
    capture = _make_capture(
        [
            _sent(1000, "shoot(100,100,id=50)"),
            _sent(1010, "move(105,100)"),
            _received(1100),
        ]
    )
    path = tmp_path / "consecutive.json"
    path.write_text(dump_json_str(capture), encoding="utf-8")
    report = analyze_timing(path)
    assert report["timings"][0]["latency_ms"] == 100


def test_main_uses_default_path_when_no_args(tmp_path: Path) -> None:
    """main() uses runs/bot/latest.capture_session.json when no args given.

    Resolves the default path under ``tmp_path`` by swapping CWD for
    the duration of ``main()`` -- the original test wrote directly
    into the live ``runs/bot/latest.capture_session.json``, which
    clobbered real session captures whenever the suite ran on a
    machine that had just executed ``make bot``.
    """
    import os

    from scripts.analyze_session_timing import main

    capture = _make_capture([_sent(1000, "move(105,100)"), _received(1050)])
    default_path = tmp_path / "runs" / "bot" / "latest.capture_session.json"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    default_path.write_text(dump_json_str(capture), encoding="utf-8")

    called: list[str] = []

    def _fake_setup(level: _test_hooks.LogLevel) -> None:
        called.append(level)

    _test_hooks.setup_rich_logging = _fake_setup
    core_hooks.get_argv = lambda: ["analyze_session_timing"]
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = main()
    finally:
        os.chdir(original_cwd)
    assert result == 0


def test_main_runs_with_explicit_path(tmp_path: Path) -> None:
    """main() succeeds when given an explicit capture path."""
    from scripts.analyze_session_timing import main

    capture = _make_capture(
        [
            _sent(1000, "move(105,100)"),
            _received(1050),
        ]
    )
    path = tmp_path / "test.capture_session.json"
    path.write_text(dump_json_str(capture), encoding="utf-8")

    called: list[str] = []

    def _fake_setup(level: _test_hooks.LogLevel) -> None:
        called.append(level)

    _test_hooks.setup_rich_logging = _fake_setup
    core_hooks.get_argv = lambda: ["analyze_session_timing", str(path)]
    result = main()
    assert result == 0


def test_main_module_entry(tmp_path: Path) -> None:
    """Running as __main__ invokes main() and exits 0."""
    import sys

    capture = _make_capture(
        [
            _sent(1000, "move(105,100)"),
            _received(1050),
        ]
    )
    runs_dir = tmp_path / "runs" / "bot"
    runs_dir.mkdir(parents=True)
    target = runs_dir / "latest.capture_session.json"
    target.write_text(dump_json_str(capture), encoding="utf-8")

    called: list[str] = []

    def _fake_setup(level: _test_hooks.LogLevel) -> None:
        called.append(level)

    _test_hooks.setup_rich_logging = _fake_setup
    core_hooks.get_argv = lambda: ["analyze_session_timing", str(target)]
    sys.modules.pop("scripts.analyze_session_timing", None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("scripts.analyze_session_timing", run_name="__main__")
    assert exc_info.value.code == 0
