"""The one reader of the compose stack's ``.env`` (MCPs board task af8b5662)."""

from __future__ import annotations

import pytest

from platform_core.env_file import parse_env_file
from platform_core.errors import AppError, ErrorCode

SOURCE = "MCPs/.env"


def test_each_key_maps_to_its_raw_value_past_comments_and_blank_lines() -> None:
    text = (
        "# the stack\n"
        "MCP_INTERNAL_KEY=internal-key\n"
        "\n"
        "   # an indented comment\n"
        "  DATABASE_URL = postgresql://u:p@127.0.0.1/db?sslmode=disable  \r\n"
        "EMPTY=\n"
    )
    assert parse_env_file(text, source=SOURCE) == {
        "MCP_INTERNAL_KEY": "internal-key",
        # Split at the first '=': the value keeps the rest, '=' signs and all.
        "DATABASE_URL": " postgresql://u:p@127.0.0.1/db?sslmode=disable",
        "EMPTY": "",
    }


def test_a_quoted_value_is_kept_as_written_as_the_owners_reader_keeps_it() -> None:
    assert parse_env_file("KEY=\"quoted\"\nOTHER='single'\n", source=SOURCE) == {
        "KEY": '"quoted"',
        "OTHER": "'single'",
    }


def test_the_last_assignment_of_a_key_wins() -> None:
    assert parse_env_file("KEY=first\nOTHER=x\nKEY=second\n", source=SOURCE) == {
        "KEY": "second",
        "OTHER": "x",
    }


def test_an_empty_file_holds_no_keys() -> None:
    assert parse_env_file("", source=SOURCE) == {}


def test_a_line_with_no_equals_sign_is_refused_by_number_without_echoing_it() -> None:
    text = "# header\nKEY=one\nexport tskey-secret-value\n"
    with pytest.raises(AppError) as caught:
        parse_env_file(text, source=SOURCE)
    assert caught.value.code is ErrorCode.CONFIG_ERROR
    assert caught.value.message == (
        f"line 3 of {SOURCE} is not a KEY=VALUE line; only those, blank lines and # comments "
        "are read, and skipping one would drop whichever key it sets"
    )
