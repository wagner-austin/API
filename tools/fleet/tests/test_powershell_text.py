"""The shared PowerShell text helpers (fleet.core.powershell_text).

The helpers' output is executed by the Pester suites inside the renders that
use it; this pins the shapes those suites cannot tell apart from outside: a
compiled type is guarded by its own name, a here-string's terminator stays
at column 0 under any nesting, and what cannot be embedded is refused.
"""

from __future__ import annotations

import pytest

from fleet.core.powershell_text import add_type_lines, indented, system32_parameter


def test_a_type_is_compiled_only_when_the_session_lacks_it() -> None:
    assert add_type_lines("Fleet.Probe", "namespace Fleet {\n\n    public class Probe {}\n}\n") == (
        "if ($null -eq ('Fleet.Probe' -as [type])) {",
        "    Add-Type -TypeDefinition @'",
        "namespace Fleet {",
        "",
        "    public class Probe {}",
        "}",
        "'@",
        "}",
    )


@pytest.mark.parametrize("name", ["Probe", "fleet.Probe", "Fleet.Probe.Inner", "Fleet.Pro be"])
def test_a_type_name_not_of_the_namespace_dot_type_shape_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match=r"Namespace\.Type"):
        add_type_lines(name, "")


@pytest.mark.parametrize("source", ["'@\n", "class A {}\n'@ trailing\n"])
def test_a_source_holding_the_here_string_terminator_is_refused(source: str) -> None:
    with pytest.raises(ValueError, match="here-string early"):
        add_type_lines("Fleet.Probe", source)


def test_nesting_indents_every_line_but_a_blank_one_and_the_terminator() -> None:
    lines = ("if ($a) {", "", "    Add-Type -TypeDefinition @'", "class A {}", "'@", "}")

    assert indented(lines, depth=2) == (
        "        if ($a) {",
        "",
        "            Add-Type -TypeDefinition @'",
        "        class A {}",
        "'@",
        "        }",
    )


def test_a_system32_binary_is_named_by_its_absolute_path() -> None:
    named = system32_parameter("Cmd", "cmd.exe")

    assert named == '[string]$Cmd = "$env:SystemRoot\\System32\\cmd.exe"'
