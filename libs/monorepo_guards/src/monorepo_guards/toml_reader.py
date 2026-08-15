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


__all__ = [
    "PathDependency",
    "check_banned_api",
    "extract_mypy_bool",
    "extract_mypy_files",
    "extract_path_dependencies",
    "extract_ruff_src",
    "read_pyproject",
]
