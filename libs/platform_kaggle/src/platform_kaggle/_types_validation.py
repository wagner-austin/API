"""types: CompetitionCategory and related definitions."""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_list,
    require_str,
)

# -----------------------------------------------------------------------------
# Kaggle-Specific Literal Types
# -----------------------------------------------------------------------------

CompetitionCategory = Literal[
    "Featured", "Research", "Playground", "Getting Started", "Masters", "Kudos"
]


# -----------------------------------------------------------------------------
# Internal Validation Helpers
# -----------------------------------------------------------------------------


def _require_list_str(obj: JSONObject, key: str) -> list[str]:
    """Extract required list of strings from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Field key.

    Returns:
        List of strings.

    Raises:
        JSONTypeError: If field is missing or contains non-strings.
    """
    items = require_list(obj, key)
    result: list[str] = []
    for i, item in enumerate(items):
        if not isinstance(item, str):
            raise JSONTypeError(f"Field '{key}[{i}]' must be a string, got {type(item).__name__}")
        result.append(item)
    return result


def _require_dict_value(value: JSONValue, context: str) -> JSONObject:
    """Require value to be a dict.

    Args:
        value: JSON value to check.
        context: Context for error message.

    Returns:
        The value as JSONObject.

    Raises:
        JSONTypeError: If value is not a dict.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"{context} must be an object, got {type(value).__name__}")
    return value


def _require_category(obj: JSONObject, key: str) -> CompetitionCategory:
    """Extract and validate CompetitionCategory from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Field key.

    Returns:
        Validated CompetitionCategory.

    Raises:
        JSONTypeError: If field is missing or not a valid category.
    """
    value = require_str(obj, key)
    if value == "Featured":
        return "Featured"
    if value == "Research":
        return "Research"
    if value == "Playground":
        return "Playground"
    if value == "Getting Started":
        return "Getting Started"
    if value == "Masters":
        return "Masters"
    if value == "Kudos":
        return "Kudos"
    raise JSONTypeError(f"Field '{key}' must be a valid category, got '{value}'")


def _require_category_value(value: JSONValue, context: str) -> CompetitionCategory:
    """Require value to be a valid CompetitionCategory.

    Args:
        value: JSON value to check.
        context: Context for error message.

    Returns:
        Validated CompetitionCategory.

    Raises:
        JSONTypeError: If value is not a valid category.
    """
    if not isinstance(value, str):
        raise JSONTypeError(f"{context} must be a string, got {type(value).__name__}")
    if value == "Featured":
        return "Featured"
    if value == "Research":
        return "Research"
    if value == "Playground":
        return "Playground"
    if value == "Getting Started":
        return "Getting Started"
    if value == "Masters":
        return "Masters"
    if value == "Kudos":
        return "Kudos"
    raise JSONTypeError(f"{context} must be a valid category, got '{value}'")
