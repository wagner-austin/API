"""Tests for project naming and directory derivation.

HPC3 is shared: 102 distinct users had jobs running when this was measured,
and every one of them sees every job name in ``squeue``. These rules are what
keep our rows legible to us and inoffensive to them.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.layout import (
    MAX_PROJECT_LENGTH,
    log_dir,
    qualified_name,
    require_project,
    require_root,
    script_dir,
)


def _obj(value: JSONValue) -> dict[str, JSONValue]:
    """Wrap a value as the field a decoder would read.

    Args:
        value: The project value under test.

    Returns:
        A one-field object.
    """
    return {"project": value}


class TestRequireProject:
    def test_the_real_project_names_are_accepted(self) -> None:
        """The bodies of work this is meant to carry."""
        for name in ("abl", "turkic-lstm", "cleargbm", "covenant-radar", "sirius", "zodiac"):
            assert require_project(_obj(name), "project") == name

    def test_an_empty_project_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            require_project(_obj(""), "project")

    def test_a_dot_is_refused(self) -> None:
        """The dot separates project from name; one inside either half is ambiguous."""
        with pytest.raises(JSONTypeError, match=r"\['\.'\]"):
            require_project(_obj("abl.v2"), "project")

    def test_a_slash_is_refused(self) -> None:
        """It becomes a directory component."""
        with pytest.raises(JSONTypeError):
            require_project(_obj("abl/v2"), "project")

    def test_uppercase_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            require_project(_obj("ABL"), "project")

    def test_a_space_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            require_project(_obj("my project"), "project")

    def test_exactly_the_length_limit_is_accepted(self) -> None:
        limit = "a" * MAX_PROJECT_LENGTH
        assert require_project(_obj(limit), "project") == limit

    def test_over_the_length_limit_is_refused(self) -> None:
        """squeue truncates the name column; a long prefix eats the useful half."""
        with pytest.raises(JSONTypeError, match="at most"):
            require_project(_obj("a" * (MAX_PROJECT_LENGTH + 1)), "project")

    def test_a_non_string_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            require_project(_obj(7), "project")


class TestQualifiedName:
    def test_it_prefixes_the_project(self) -> None:
        assert qualified_name("abl", "armB-s42") == "abl.armB-s42"

    def test_two_projects_cannot_collide_on_one_name(self) -> None:
        assert qualified_name("abl", "train") != qualified_name("turkic-lstm", "train")


class TestDirectoryDerivation:
    def test_scripts_and_logs_separate_per_project(self) -> None:
        assert script_dir("/pub/wagnera3", "abl") == "/pub/wagnera3/abl/scripts"
        assert log_dir("/pub/wagnera3", "abl") == "/pub/wagnera3/abl/logs"

    def test_two_projects_do_not_share_a_directory(self) -> None:
        assert log_dir("/pub/w", "abl") != log_dir("/pub/w", "sirius")

    def test_a_trailing_slash_on_the_root_does_not_double_up(self) -> None:
        assert script_dir("/pub/wagnera3/", "abl") == "/pub/wagnera3/abl/scripts"


class TestRequireRoot:
    def test_an_absolute_root_is_accepted(self) -> None:
        assert require_root("/pub/wagnera3") == "/pub/wagnera3"

    def test_a_trailing_slash_is_trimmed(self) -> None:
        assert require_root("/pub/wagnera3/") == "/pub/wagnera3"

    def test_the_filesystem_root_survives_trimming(self) -> None:
        assert require_root("/") == "/"

    def test_a_relative_root_is_refused(self) -> None:
        with pytest.raises(ValueError, match="absolute POSIX path"):
            require_root("pub/wagnera3")

    def test_a_windows_root_is_refused(self) -> None:
        """The cluster is POSIX; a backslash would become part of a filename."""
        with pytest.raises(ValueError, match="forward-slashed"):
            require_root("/pub\\wagnera3")

    def test_an_escaping_root_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"must not contain"):
            require_root("/pub/../etc")
