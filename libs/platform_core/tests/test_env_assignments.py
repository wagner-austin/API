"""The strict reader of the untracked credentials file (MCPs board task 94ac1c4f)."""

from __future__ import annotations

import pytest

from platform_core.env_assignments import parse_env_assignments
from platform_core.errors import AppError, ErrorCode

SOURCE = "runs/env.ps1"


def test_assignments_come_back_in_file_order_past_comments_and_blank_lines() -> None:
    text = (
        "# hpc-wake credentials. UNTRACKED.\n"
        "$env:TASKBOARD_MCP_API_KEY = 'board-key'\n"
        "\n"
        "   # an indented comment\n"
        "$env:FLEET_MCP_API_KEY='fleet-key'\n"
        "  $env:CORVIS_TENANT_ID = ''  \n"
    )
    assert parse_env_assignments(text, source=SOURCE) == (
        ("TASKBOARD_MCP_API_KEY", "board-key"),
        ("FLEET_MCP_API_KEY", "fleet-key"),
        ("CORVIS_TENANT_ID", ""),
    )


def test_a_byte_order_mark_is_not_part_of_the_first_line() -> None:
    text = "\N{BYTE ORDER MARK}$env:ONLY = 'value'\r\n"
    assert text.encode("utf-8").startswith(b"\xef\xbb\xbf")
    assert parse_env_assignments(text, source=SOURCE) == (("ONLY", "value"),)


def test_an_empty_file_holds_no_assignments() -> None:
    assert parse_env_assignments("", source=SOURCE) == ()


@pytest.mark.parametrize(
    "line",
    [
        '$env:KEY = "double-quoted-secret"',
        "$KEY = 'not an env assignment'",
        "Set-Item env:KEY 'value'",
        "$env:KEY = 'unterminated",
    ],
)
def test_any_other_statement_is_refused_by_line_number_without_echoing_it(line: str) -> None:
    text = f"# header\n$env:FIRST = 'one'\n{line}\n"
    with pytest.raises(AppError) as caught:
        parse_env_assignments(text, source=SOURCE)
    assert caught.value.code is ErrorCode.CONFIG_ERROR
    assert caught.value.message.startswith(f"line 3 of {SOURCE} is not a plain $env:NAME")
    assert line not in caught.value.message
    assert "secret" not in caught.value.message


def test_a_name_assigned_twice_is_refused_naming_it() -> None:
    text = "$env:KEY = 'first'\n$env:OTHER = 'x'\n$env:KEY = 'second'\n"
    with pytest.raises(AppError) as caught:
        parse_env_assignments(text, source=SOURCE)
    assert caught.value.code is ErrorCode.CONFIG_ERROR
    assert caught.value.message == (
        f"line 3 of {SOURCE} assigns KEY a second time; each name is set once"
    )
