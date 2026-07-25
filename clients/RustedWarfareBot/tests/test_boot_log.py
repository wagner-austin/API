"""Boot-log decoding, exercised against the real archived engine logs.

The fixtures under ``wiki/sources/`` are genuine output from headless runs of
build 1.15, not hand-written samples. Parsing them is the only assurance that
the decoder matches the format the engine actually emits.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rw_bot.harness.boot_log import (
    BootLogError,
    find_subsystem,
    parse_boot_log,
    strip_timestamp,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CLEAN_LOG = _PROJECT_ROOT / "wiki" / "sources" / "m0-probe" / "nodisplay-boot.log"
_CRASH_LOG = _PROJECT_ROOT / "wiki" / "sources" / "m1-sandbox" / "sandbox-crash.log"

_HEADER = (
    "2026-07-25 01:31:20.979: Build Number: #28",
    "2026-07-25 01:31:20.979: Game Version: 1.15",
    "2026-07-25 01:31:20.979: Game Code: 176",
)


def _read(path: Path) -> tuple[str, ...]:
    """Read an archived engine log.

    Args:
        path: Log file to read.

    Returns:
        The log's lines without trailing newlines.
    """
    return tuple(path.read_text(encoding="utf-8").splitlines())


def test_strip_timestamp_removes_the_engine_prefix() -> None:
    assert strip_timestamp("2026-07-25 01:31:22.555: --Now loading:GameEngine") == (
        "--Now loading:GameEngine"
    )


def test_strip_timestamp_leaves_untimestamped_lines_alone() -> None:
    frame = "\tat com.corrodinggames.rts.java.d.a.EnableScissorRegion(SourceFile:650)"
    assert strip_timestamp(frame) == frame


def test_strip_timestamp_leaves_short_lines_alone() -> None:
    assert strip_timestamp("short") == "short"


def test_strip_timestamp_leaves_slick_lines_alone() -> None:
    slick = "Sat Jul 25 01:31:21 PDT 2026 INFO:Slick Build #84"
    assert strip_timestamp(slick) == slick


def test_parses_the_real_clean_boot_log() -> None:
    log = parse_boot_log(_read(_CLEAN_LOG))
    assert log["version"] == {
        "version": "1.15",
        "game_code": 176,
        "build_number": "#28",
    }
    assert log["crashes"] == ()


def test_real_boot_log_records_the_command_controller() -> None:
    log = parse_boot_log(_read(_CLEAN_LOG))
    assert find_subsystem(log, "CommandController") == {
        "line_number": 183,
        "name": "CommandController",
    }


def test_real_boot_log_recovers_the_game_engine_class_mapping() -> None:
    log = parse_boot_log(_read(_CLEAN_LOG))
    mappings = {item["subsystem"]: item["java_class"] for item in log["class_mappings"]}
    assert mappings["gameEngine"] == "com.corrodinggames.rts.game.i"


def test_real_boot_log_records_the_menu_background_map() -> None:
    log = parse_boot_log(_read(_CLEAN_LOG))
    assert [item["map_file"] for item in log["maps"]] == ["assets/maps/menu_background/menu2.tmx"]


def test_find_subsystem_returns_none_for_a_subsystem_that_never_loaded() -> None:
    log = parse_boot_log(_read(_CLEAN_LOG))
    assert find_subsystem(log, "NoSuchEngine") is None


def test_parses_the_real_sandbox_crash_log() -> None:
    log = parse_boot_log(_read(_CRASH_LOG))
    assert len(log["crashes"]) == 1
    crash = log["crashes"][0]
    assert crash["exception_type"] == "java.lang.NullPointerException"
    assert crash["top_frame"] == (
        "com.corrodinggames.rts.java.d.a.EnableScissorRegion(SourceFile:650)"
    )


def test_sandbox_crash_log_reached_the_skirmish_map() -> None:
    log = parse_boot_log(_read(_CRASH_LOG))
    loaded = [item["map_file"] for item in log["maps"]]
    assert "assets/maps/skirmish/[z;p10]Crossing Large (10p).tmx" in loaded


def test_missing_version_header_is_rejected() -> None:
    with pytest.raises(BootLogError) as caught:
        parse_boot_log(("2026-07-25 01:31:22.555: --Now loading:GameEngine",))
    assert caught.value.code == "RW-BOOTLOG-001"
    assert "Game Version" in caught.value.message
    assert "Build Number" in caught.value.message
    assert "Game Code" in caught.value.message


def test_partially_missing_version_header_names_only_what_is_absent() -> None:
    lines = (
        "2026-07-25 01:31:20.979: Build Number: #28",
        "2026-07-25 01:31:20.979: Game Version: 1.15",
    )
    with pytest.raises(BootLogError) as caught:
        parse_boot_log(lines)
    assert caught.value.message.endswith("Game Code not found")


def test_non_numeric_game_code_is_rejected() -> None:
    lines = (
        "2026-07-25 01:31:20.979: Build Number: #28",
        "2026-07-25 01:31:20.979: Game Version: 1.15",
        "2026-07-25 01:31:20.979: Game Code: beta",
    )
    with pytest.raises(BootLogError) as caught:
        parse_boot_log(lines)
    assert caught.value.code == "RW-BOOTLOG-003"
    assert caught.value.message == "Game Code must be an integer, got 'beta'"


def test_crash_marker_without_a_stack_frame_is_rejected() -> None:
    lines = (*_HEADER, "2026-07-25 02:22:26.809: uncaughtException start")
    with pytest.raises(BootLogError) as caught:
        parse_boot_log(lines)
    assert caught.value.code == "RW-BOOTLOG-002"
    assert caught.value.message == "crash marker at line 4 has no stack frame following it"


def test_a_second_crash_marker_terminates_the_scan_for_the_first() -> None:
    lines = (
        *_HEADER,
        "2026-07-25 02:22:26.809: uncaughtException start",
        "2026-07-25 02:22:26.809: uncaughtException start",
        "java.lang.IllegalStateException",
        "\tat com.example.Thing.run(Thing.java:12)",
    )
    with pytest.raises(BootLogError) as caught:
        parse_boot_log(lines)
    assert caught.value.message == "crash marker at line 4 has no stack frame following it"


def test_created_line_without_the_infix_is_not_a_class_mapping() -> None:
    lines = (*_HEADER, "2026-07-25 01:31:22.559: Created new thing without a class")
    assert parse_boot_log(lines)["class_mappings"] == ()
