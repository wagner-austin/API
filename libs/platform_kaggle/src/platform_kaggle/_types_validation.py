"""types: CompetitionCategory and related definitions."""

from __future__ import annotations

from enum import StrEnum

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
)
from platform_core.members import as_member


class CompetitionCategory(StrEnum):
    """A Kaggle competition's host segment, spelled as ``ApiCompetition.category`` spells it.

    The values are the strings the live ``competitions_list`` returned on
    2026-09-26 across every category filter the SDK admits and both the
    general and community groups. The SDK's ``HostSegment`` also declares
    Analytics, which no listing returned, so it is not admitted: a competition
    carrying it is refused by name rather than filed under a guess.
    """

    FEATURED = "Featured"
    RESEARCH = "Research"
    RECRUITMENT = "Recruitment"
    GETTING_STARTED = "Getting Started"
    MASTERS = "Masters"
    PLAYGROUND = "Playground"
    COMMUNITY = "Community"


# -----------------------------------------------------------------------------
# Internal Validation Helpers
# -----------------------------------------------------------------------------


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


def _require_category_value(value: JSONValue, context: str) -> CompetitionCategory:
    """Require value to be a valid CompetitionCategory.

    Args:
        value: JSON value to check.
        context: Context for error message.

    Returns:
        The member whose value is ``value``.

    Raises:
        JSONTypeError: If value is not a string, or names no category.
    """
    if not isinstance(value, str):
        raise JSONTypeError(f"{context} must be a string, got {type(value).__name__}")
    return as_member(value, context, CompetitionCategory)
