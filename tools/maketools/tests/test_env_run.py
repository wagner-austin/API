"""``env``: assignments, defaults, unsets, draws, @NAME@ markers and --then."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError

from maketools.env_run import (
    apply_assignments,
    parse_draw,
    parse_env_arguments,
    run_env,
    substitute,
)
from tests.conftest import World


def test_parse_splits_assignments_draws_command_and_then() -> None:
    request = parse_env_arguments(
        [
            "A=1",
            "--draw",
            "PORT=27600-27999",
            "B?=2",
            "C=",
            "--then",
            "poetry run report x",
            "--",
            "poetry",
            "run",
            "bot",
        ]
    )
    assert request["assignments"] == [("A", "1", False), ("B", "2", True), ("C", "", False)]
    assert request["draws"] == [("PORT", 27600, 27999)]
    assert request["command"] == ["poetry", "run", "bot"]
    assert request["then"] == ["poetry", "run", "report", "x"]


def test_parse_refuses_a_missing_separator_command_or_name() -> None:
    with pytest.raises(AppError, match=r"env needs '--' between"):
        parse_env_arguments(["A=1", "poetry"])
    with pytest.raises(AppError, match=r"env has no command after '--'"):
        parse_env_arguments(["A=1", "--"])
    with pytest.raises(AppError, match=r"is not NAME=VALUE or NAME\?=VALUE"):
        parse_env_arguments(["A", "--", "x"])
    with pytest.raises(AppError, match=r"'=1' has no name"):
        parse_env_arguments(["=1", "--", "x"])
    with pytest.raises(AppError, match=r"--then needs a quoted command"):
        parse_env_arguments(["--then", "--", "x"])
    with pytest.raises(AppError, match=r"--draw needs NAME=LOW-HIGH"):
        parse_env_arguments(["--draw", "--", "x"])


def test_parse_draw_reads_the_bounds_and_refuses_bad_shapes() -> None:
    assert parse_draw("PORT=1-5") == ("PORT", 1, 5)
    assert parse_draw("PORT=7-7") == ("PORT", 7, 7)
    for bad in ("PORT", "=1-5", "PORT=15", "PORT=a-5", "PORT=1-b", "PORT=-5"):
        with pytest.raises(AppError, match=r"is not NAME=LOW-HIGH") as caught:
            parse_draw(bad)
        assert caught.value.code is MaketoolsErrorCode.USAGE
    with pytest.raises(AppError, match=r"'PORT=9-8' has an empty range"):
        parse_draw("PORT=9-8")


def test_apply_sets_defaults_and_unsets() -> None:
    result = apply_assignments(
        {"KEEP": "k", "GONE": "g", "SET": "old", "BLANK": "  "},
        [("SET", "new", False), ("GONE", "", False), ("KEEP", "x", True), ("BLANK", "d", True)],
    )
    assert result == {"KEEP": "k", "SET": "new", "BLANK": "d"}


def test_substitute_replaces_every_marker_and_blanks_an_unset_name() -> None:
    argv = ["--port", "@PORT@", "@PORT@/@GONE@/x", "plain", "@OTHER@"]
    result = substitute(argv, ["PORT", "GONE"], {"PORT": "27650"})
    assert result == ["--port", "27650", "27650//x", "plain", "@OTHER@"]


def test_run_env_hands_the_child_the_assembled_environment(world: World, tmp_path: Path) -> None:
    world.environment = {"PATH": "/bin", "OLD": "1"}
    assert run_env(["NEW=2", "OLD=", "--", "poetry", "run", "tool"], tmp_path) == 0
    call = world.inheriting_calls[0]
    assert call["argv"] == ("poetry", "run", "tool")
    assert call["env"] == {"PATH": "/bin", "NEW": "2"}
    assert call["cwd"] == tmp_path
    assert call["new_session"] is False


def test_run_env_draws_once_and_substitutes_the_draw_into_both_commands(
    world: World, tmp_path: Path
) -> None:
    world.drawn = 27650
    code = run_env(
        [
            "--draw",
            "PORT=27600-27999",
            "LOG=runs/play.log",
            "--then",
            "poetry run report --port @PORT@ @LOG@",
            "--",
            "poetry",
            "run",
            "play",
            "--port",
            "@PORT@",
        ],
        tmp_path,
    )
    assert code == 0
    assert world.draws == [(27600, 27999)]
    first, second = world.inheriting_calls
    assert first["argv"] == ("poetry", "run", "play", "--port", "27650")
    assert first["env"] == {"PATH": "/bin", "LOG": "runs/play.log", "PORT": "27650"}
    assert second["argv"] == ("poetry", "run", "report", "--port", "27650", "runs/play.log")
    assert second["env"] == first["env"]


def test_run_env_runs_the_then_command_after_a_failure_and_keeps_the_failure(
    world: World, tmp_path: Path
) -> None:
    world.inheriting_code = 3
    code = run_env(["--then", "poetry run report", "--", "poetry", "run", "bot"], tmp_path)
    assert code == 3
    assert [c["argv"] for c in world.inheriting_calls] == [
        ("poetry", "run", "bot"),
        ("poetry", "run", "report"),
    ]


def test_run_env_reports_the_then_commands_failure_when_the_first_passed(
    world: World, tmp_path: Path
) -> None:
    world.inheriting_codes = [0, 5]
    assert run_env(["--then", "poetry run report", "--", "poetry", "run", "bot"], tmp_path) == 5
