from __future__ import annotations

import re
from pathlib import Path


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
    "check_banned_api",
    "extract_mypy_bool",
    "extract_mypy_files",
    "extract_ruff_src",
    "read_pyproject",
]
