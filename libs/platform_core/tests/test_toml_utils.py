"""Parsing TOML into the value type the JSON validators already understand.

The interesting cases are the ones where TOML is NOT a subset of JSON. TOML has
first-class date, time and datetime literals; ``tomllib`` returns real
``datetime`` objects for them, and any function annotated as returning JSON
values while passing those through is lying about its own type.
"""

from __future__ import annotations

import pytest

from platform_core.json_utils import JSONTypeError, require_int, require_str
from platform_core.toml_utils import loads_toml


class TestParsing:
    def test_a_table_becomes_a_mapping(self) -> None:
        document = loads_toml('[tool]\nname = "fleet"\ncount = 3\n')

        assert document == {"tool": {"name": "fleet", "count": 3}}

    def test_the_result_validates_with_the_json_helpers(self) -> None:
        """The whole point of narrowing to JSONValue: one family of
        validators covers both formats."""
        document = loads_toml('name = "fleet"\ncount = 3\n')

        assert require_str(document, "name") == "fleet"
        assert require_int(document, "count") == 3

    def test_an_empty_document_is_an_empty_mapping(self) -> None:
        assert loads_toml("") == {}

    def test_arrays_of_tables_survive(self) -> None:
        document = loads_toml('[[dep]]\nname = "a"\n\n[[dep]]\nname = "b"\n')

        assert document == {"dep": [{"name": "a"}, {"name": "b"}]}

    def test_invalid_toml_carries_the_line_and_column(self) -> None:
        """Propagated rather than translated -- the parser's own message is
        the whole diagnostic."""
        with pytest.raises(ValueError, match="line 1"):
            loads_toml("this is not toml\n")


class TestTemporalValuesAreRefused:
    def test_a_date_at_the_top_level_is_refused(self) -> None:
        with pytest.raises(JSONTypeError) as refusal:
            loads_toml("released = 2026-09-04\n")

        assert "'released'" in str(refusal.value)
        assert "date" in str(refusal.value)

    def test_a_datetime_nested_in_a_table_names_its_path(self) -> None:
        with pytest.raises(JSONTypeError) as refusal:
            loads_toml("[build]\nstamped = 2026-09-04T12:00:00\n")

        assert "'build.stamped'" in str(refusal.value)

    def test_a_time_inside_an_array_names_its_index(self) -> None:
        with pytest.raises(JSONTypeError) as refusal:
            loads_toml("windows = [07:30:00, 09:00:00]\n")

        assert "'windows[0]'" in str(refusal.value)

    def test_a_date_deep_in_an_array_of_tables_is_still_found(self) -> None:
        with pytest.raises(JSONTypeError) as refusal:
            loads_toml('[[run]]\nname = "a"\n\n[[run]]\nstarted = 2026-09-04\n')

        assert "'run[1].started'" in str(refusal.value)

    def test_the_refusal_says_what_to_do_instead(self) -> None:
        with pytest.raises(JSONTypeError, match="declare it as a string"):
            loads_toml("released = 2026-09-04\n")

    def test_a_string_that_looks_like_a_date_is_fine(self) -> None:
        """Quoting is the escape the refusal names, so it has to work."""
        assert loads_toml('released = "2026-09-04"\n') == {"released": "2026-09-04"}
