"""A registered project's section is a heading in the registered part.

The check this replaced passed whenever the name appeared anywhere in the
file, which the generated table made true of every registered project. These
tests pin the property that replaced it, including the two near-misses a
looser reading would let through: a section under ``## Not registered
anywhere``, and one project's heading standing in for a longer name.
"""

from __future__ import annotations

import pytest

from hpc3.core.index_sections import (
    REGISTERED_HEADING,
    projects_without_section,
    registered_part,
    section_heading,
)

_INDEX = "\n".join(
    [
        "# Research index",
        "",
        REGISTERED_HEADING,
        "",
        "| `ghost` | free | cpu |",
        "",
        "### `mi` — Model-Trainer probes",
        "",
        "### `bare`",
        "",
        "## Not registered anywhere",
        "",
        "### `sirius` — declared as an example, never run",
    ]
)


class TestTheRegisteredPart:
    """Where the part starts and stops."""

    def test_it_stops_at_the_next_second_level_heading(self) -> None:
        part = registered_part(_INDEX)

        assert "### `mi` — Model-Trainer probes" in part
        assert "sirius" not in part

    def test_it_runs_to_the_end_when_no_heading_follows(self) -> None:
        text = f"{REGISTERED_HEADING}\n\n### `last` — at the end"

        assert registered_part(text) == "\n### `last` — at the end"

    def test_an_index_without_the_heading_is_refused(self) -> None:
        with pytest.raises(ValueError, match="has no '## Registered with the hpc3 CLI' heading"):
            _ = registered_part("# Research index\n\n### `mi` — probes")


class TestFindingASection:
    """A heading line, matched by whole name, inside the registered part."""

    def test_the_heading_is_spelled_with_the_name_in_backticks(self) -> None:
        assert section_heading("tankpit") == "### `tankpit`"

    def test_a_heading_with_a_title_counts(self) -> None:
        assert projects_without_section(_INDEX, ["mi"]) == []

    def test_a_heading_with_no_title_counts(self) -> None:
        assert projects_without_section(_INDEX, ["bare"]) == []

    def test_a_table_row_is_not_a_section(self) -> None:
        """The generated table names every project; that is what made the old check vacuous."""
        assert projects_without_section(_INDEX, ["ghost"]) == ["ghost"]

    def test_a_shorter_name_does_not_stand_in_for_a_longer_one(self) -> None:
        assert projects_without_section(_INDEX, ["mi-cu128"]) == ["mi-cu128"]

    def test_a_section_under_not_registered_does_not_count(self) -> None:
        assert projects_without_section(_INDEX, ["sirius"]) == ["sirius"]

    def test_every_missing_project_is_named_sorted(self) -> None:
        assert projects_without_section(_INDEX, ["zeta", "mi", "alpha"]) == ["alpha", "zeta"]
