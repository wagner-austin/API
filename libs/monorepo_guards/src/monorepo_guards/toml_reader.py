from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

# A dependency table: [tool.poetry.dependencies] or a group's equivalent.
_DEPENDENCY_SECTION = re.compile(r"^\[tool\.poetry(?:\.group\.[^\]]+)?\.dependencies\]\s*$")
_ANY_SECTION = re.compile(r"^\[[^\]]+\]\s*$")

# name = { ..., path = "..." }, with the name optionally quoted. Poetry writes
# inline tables on one line, so one line is the unit this reads.
_PATH_DEPENDENCY = re.compile(
    r'^\s*(?:"(?P<quoted>[^"]+)"|(?P<bare>[A-Za-z0-9_.\-]+))\s*=\s*\{'
    r"""[^}]*\bpath\s*=\s*["'](?P<path>[^"']+)["']"""
)


# An entry of the [tool.poetry] `packages` array, written as an inline table.
# Not anchored, so a single-line `packages = [{...}, {...}]` yields both.
_PACKAGE_INCLUDE = re.compile(
    r"""\{\s*include\s*=\s*["'](?P<include>[^"']+)["'](?P<rest>[^}]*)\}"""
)
_POETRY_SECTION = re.compile(r"^\[tool\.poetry\]\s*$")
_FROM_KEY = re.compile(r"\bfrom\s*=")


class PathDependency(NamedTuple):
    """One dependency declared as a filesystem path.

    Attributes:
        name (str): Dependency name as declared.
        path (str): The path string, exactly as written.
        line_no (int): One-based line the declaration sits on.
    """

    name: str
    path: str
    line_no: int


def extract_path_dependencies(toml_content: str) -> list[PathDependency]:
    """Find every path dependency a pyproject declares.

    Both the main dependency table and every dependency group are read, so a
    path dependency cannot hide in a dev group.

    Parsed with a regular expression rather than a TOML library because
    reading TOML through ``tomllib`` is banned everywhere but the guard
    configuration loader, and because reporting a violation needs the line
    number, which a decoded document does not carry.

    Args:
        toml_content (str): Full contents of a ``pyproject.toml``.

    Returns:
        list[PathDependency]: One entry per path dependency, in file order.
    """
    found: list[PathDependency] = []
    in_dependencies = False
    for line_no, line in enumerate(toml_content.splitlines(), start=1):
        if _ANY_SECTION.match(line):
            in_dependencies = bool(_DEPENDENCY_SECTION.match(line))
            continue
        if not in_dependencies:
            continue
        match = _PATH_DEPENDENCY.match(line)
        if match is None:
            continue
        quoted = match.group("quoted")
        bare = match.group("bare")
        found.append(
            PathDependency(
                name=quoted if isinstance(quoted, str) else str(bare),
                path=str(match.group("path")),
                line_no=line_no,
            )
        )
    return found


class PackageInclude(NamedTuple):
    """One entry of the ``[tool.poetry]`` ``packages`` array.

    Attributes:
        include (str): The directory name the entry ships.
        has_from (bool): Whether the entry names a source root with ``from``.
        line_no (int): One-based line the entry sits on.
    """

    include: str
    has_from: bool
    line_no: int


def extract_package_includes(toml_content: str) -> list[PackageInclude]:
    """Find every package a pyproject declares it ships.

    Read line by line for the same reason as
    :func:`extract_path_dependencies`: reporting a violation needs the line
    number, which a decoded TOML document does not carry, and ``tomllib`` is
    banned outside the guard configuration loader.

    Args:
        toml_content (str): Full contents of a ``pyproject.toml``.

    Returns:
        list[PackageInclude]: One entry per declared package, in file order.
    """
    found: list[PackageInclude] = []
    in_poetry = False
    for line_no, line in enumerate(toml_content.splitlines(), start=1):
        if _ANY_SECTION.match(line):
            in_poetry = bool(_POETRY_SECTION.match(line))
            continue
        if not in_poetry:
            continue
        for match in _PACKAGE_INCLUDE.finditer(line):
            found.append(
                PackageInclude(
                    include=str(match.group("include")),
                    has_from=_FROM_KEY.search(str(match.group("rest"))) is not None,
                    line_no=line_no,
                )
            )
    return found


def extract_mypy_files(toml_content: str) -> list[str] | None:
    """Extract files list from [tool.mypy] section."""
    pattern = r"\[tool\.mypy\].*?^files\s*=\s*\[(.*?)\]"
    match = re.search(pattern, toml_content, re.MULTILINE | re.DOTALL)
    if not match:
        return None

    array_content: str = match.group(1)
    items: list[str] = []
    found_items: list[str] = re.findall(r'["\']([^"\']+)["\']', array_content)
    for item in found_items:
        items.append(item)
    return items if items else None


def extract_mypy_bool(toml_content: str, key: str) -> bool | None:
    """Extract a boolean value from [tool.mypy] section."""
    pattern = rf"\[tool\.mypy\].*?^{re.escape(key)}\s*=\s*(true|false)"
    match = re.search(pattern, toml_content, re.MULTILINE | re.DOTALL)
    if not match:
        return None
    value: str = match.group(1)
    return value == "true"


def extract_ruff_src(toml_content: str) -> list[str] | None:
    """Extract src list from [tool.ruff] section."""
    pattern = r"\[tool\.ruff\].*?^src\s*=\s*\[(.*?)\]"
    match = re.search(pattern, toml_content, re.MULTILINE | re.DOTALL)
    if not match:
        return None

    array_content: str = match.group(1)
    items: list[str] = []
    found_items: list[str] = re.findall(r'["\']([^"\']+)["\']', array_content)
    for item in found_items:
        items.append(item)
    return items if items else None


def check_banned_api(toml_content: str, api_name: str) -> bool:
    """Check if an API is banned in [tool.ruff.lint.flake8-tidy-imports.banned-api]."""
    section = r"\[tool\.ruff\.lint\.flake8-tidy-imports\.banned-api\]"
    key_pattern = rf'^["\']?{re.escape(api_name)}["\']?\s*='
    pattern = rf"{section}.*?{key_pattern}"
    return bool(re.search(pattern, toml_content, re.MULTILINE | re.DOTALL))


def read_pyproject(path: Path) -> str:
    """Read pyproject.toml content as string."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"Failed to read pyproject.toml at {path}: {e}") from e


# A coverage carve-out: a key whose value narrows what the 100% threshold is
# measured against. Read line by line rather than with a section-spanning
# regular expression so the reported line is the one the reader has to edit.
_COVERAGE_RUN_SECTION = "[tool.coverage.run]"
_COVERAGE_REPORT_SECTION = "[tool.coverage.report]"
_CARVE_OUT_KEYS = {
    _COVERAGE_RUN_SECTION: ("omit",),
    _COVERAGE_REPORT_SECTION: ("exclude_lines",),
}
_ARRAY_ASSIGNMENT = re.compile(r"^\s*(?P<key>[A-Za-z_]+)\s*=\s*\[(?P<inline>[^\]]*)\]?\s*$")
_QUOTED_ENTRY = re.compile(r'["\']([^"\']+)["\']')
_FAIL_UNDER = re.compile(r"^\s*fail_under\s*=\s*(?P<value>[0-9.]+)\s*$")


class CoverageCarveOut(NamedTuple):
    """One coverage setting that narrows what the threshold measures.

    Attributes:
        key (str): The setting's name, as written.
        detail (str): What it excludes, for the violation message.
        line_no (int): One-based line the setting sits on.
    """

    key: str
    detail: str
    line_no: int


def _array_entries(lines: list[str], start: int, inline: str) -> tuple[list[str], int]:
    """Collect a TOML array's string entries, whether inline or spread.

    Args:
        lines (list[str]): The manifest's lines.
        start (int): Zero-based index of the line holding the assignment.
        inline (str): Whatever followed the opening bracket on that line.

    Returns:
        tuple[list[str], int]: The quoted entries, and the zero-based index of
        the last line consumed.
    """
    if lines[start].rstrip().endswith("]"):
        inline_entries: list[str] = _QUOTED_ENTRY.findall(inline)
        return inline_entries, start
    collected = inline
    index = start
    while index + 1 < len(lines):
        index += 1
        collected += lines[index]
        if lines[index].rstrip().endswith("]"):
            break
    spread_entries: list[str] = _QUOTED_ENTRY.findall(collected)
    return spread_entries, index


def coverage_carve_outs(toml_content: str) -> list[CoverageCarveOut]:
    """Find every coverage setting that narrows what 100% is measured against.

    Three shapes, all of which let a package report full coverage over less
    code than it ships: ``omit`` drops files from measurement, ``exclude_lines``
    drops statements from the files that are measured, and a ``fail_under``
    below 100 lowers the bar itself. A blank or absent setting is not a
    carve-out and is not reported.

    Args:
        toml_content (str): Full contents of a ``pyproject.toml``.

    Returns:
        list[CoverageCarveOut]: One entry per carve-out, in file order.
    """
    lines = toml_content.splitlines()
    found: list[CoverageCarveOut] = []
    section = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        if _ANY_SECTION.match(line):
            section = line.strip()
            index += 1
            continue
        fail_under = _FAIL_UNDER.match(line)
        if fail_under is not None and section == _COVERAGE_REPORT_SECTION:
            value = fail_under.group("value")
            if float(value) != 100.0:
                found.append(
                    CoverageCarveOut(
                        key="fail_under",
                        detail=f"threshold is {value}, not 100",
                        line_no=index + 1,
                    )
                )
            index += 1
            continue
        assignment = _ARRAY_ASSIGNMENT.match(line)
        if assignment is None or assignment.group("key") not in _CARVE_OUT_KEYS.get(section, ()):
            index += 1
            continue
        entries, last = _array_entries(lines, index, assignment.group("inline"))
        if entries:
            found.append(
                CoverageCarveOut(
                    key=assignment.group("key"),
                    detail=", ".join(entries),
                    line_no=index + 1,
                )
            )
        index = last + 1
    return found


__all__ = [
    "CoverageCarveOut",
    "PackageInclude",
    "PathDependency",
    "check_banned_api",
    "coverage_carve_outs",
    "extract_mypy_bool",
    "extract_mypy_files",
    "extract_package_includes",
    "extract_path_dependencies",
    "extract_ruff_src",
    "read_pyproject",
]
