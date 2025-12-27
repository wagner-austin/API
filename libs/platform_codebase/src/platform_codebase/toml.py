"""TOML parsing utilities using regex.

This module provides regex-based TOML parsing to avoid using the tomllib
module which is banned by the monorepo guards.
"""

from __future__ import annotations

import re
from pathlib import Path


def extract_poetry_name(content: str) -> str:
    """Extract name from [tool.poetry] section using regex.

    Args:
        content: TOML file content as string.

    Returns:
        Package name or empty string if not found.
    """
    pattern = r'\[tool\.poetry\].*?^name\s*=\s*["\']([^"\']+)["\']'
    matches: list[str] = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
    if matches:
        return matches[0]
    return ""


def extract_poetry_dependencies(content: str) -> list[str]:
    """Extract dependencies from [tool.poetry.dependencies] section.

    Args:
        content: TOML file content as string.

    Returns:
        List of dependency names.
    """
    # Find the dependencies section using findall
    section_pattern = r"\[tool\.poetry\.dependencies\](.*?)(?:\n\[|\Z)"
    section_matches: list[str] = re.findall(section_pattern, content, re.DOTALL)
    if not section_matches:
        return []

    section_content: str = section_matches[0]
    dependencies: list[str] = []

    # Match dependency lines: name = "version" or name = { ... }
    dep_pattern = r"^([a-zA-Z0-9_-]+)\s*="
    lines: list[str] = section_content.split("\n")
    for line in lines:
        stripped: str = line.strip()
        line_matches: list[str] = re.findall(dep_pattern, stripped)
        if line_matches:
            dep_name: str = line_matches[0]
            if dep_name != "python":
                dependencies.append(dep_name)

    return dependencies


def parse_pyproject(path: Path) -> tuple[str, tuple[str, ...]]:
    """Parse pyproject.toml and extract name and dependencies.

    Args:
        path: Path to pyproject.toml file.

    Returns:
        Tuple of (name, dependencies).
    """
    content = path.read_text(encoding="utf-8")
    return parse_pyproject_content(content)


def parse_pyproject_content(content: str) -> tuple[str, tuple[str, ...]]:
    """Parse pyproject.toml content and extract name and dependencies.

    Args:
        content: TOML file content as string.

    Returns:
        Tuple of (name, dependencies).
    """
    name = extract_poetry_name(content)
    dependencies = extract_poetry_dependencies(content)
    return name, tuple(dependencies)
