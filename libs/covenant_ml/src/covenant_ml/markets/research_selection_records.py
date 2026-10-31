"""Frozen selection settings owned by the existing research admission policy."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONValue, require_list
from typing_extensions import ReadOnly, TypedDict

from .campaign_records import require_digest
from .validation import require_bounded_int, require_nonempty


class ResearchSelection(TypedDict):
    """The primary comparison, complete calendar and mandatory baseline candidates."""

    primary: ReadOnly[str]
    reference: ReadOnly[str]
    cost_bps: ReadOnly[int]
    first_month: ReadOnly[int]
    last_month: ReadOnly[int]
    final_seed: ReadOnly[int]
    required_candidates: ReadOnly[tuple[str, ...]]


def encode_research_selection(value: ResearchSelection) -> JSONObject:
    """Encode every selection choice for the journal's immutable policy.

    Args:
        value: Validated selection settings.

    Returns:
        Portable settings without a result or an executable command.
    """
    candidates: list[JSONValue] = list(value["required_candidates"])
    return {
        "primary": value["primary"],
        "reference": value["reference"],
        "cost_bps": value["cost_bps"],
        "first_month": value["first_month"],
        "last_month": value["last_month"],
        "final_seed": value["final_seed"],
        "required_candidates": candidates,
    }


def decode_research_selection(value: JSONObject) -> ResearchSelection:
    """Refuse ambiguous arms, incomplete choices and duplicate baseline identities.

    Args:
        value: Untrusted policy member.

    Returns:
        Exact immutable selection settings.

    Raises:
        ValueError: RESEARCH_SELECTION_POLICY for malformed selection settings.
    """
    if set(value) != set(ResearchSelection.__annotations__):
        raise ValueError("RESEARCH_SELECTION_POLICY: exact selection fields required")
    primary = require_nonempty(value, "primary")
    reference = require_nonempty(value, "reference")
    if primary not in {"cleargbm", "lightgbm", "xgboost"} or reference != "momentum":
        raise ValueError("RESEARCH_SELECTION_POLICY: require a model against fixed momentum")
    first = require_bounded_int(value, "first_month", 1, 119987)
    last = require_bounded_int(value, "last_month", first, 119987)
    candidates = tuple(
        require_digest({"candidate": item}, "candidate")
        for item in require_list(value, "required_candidates")
    )
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("RESEARCH_SELECTION_POLICY: require unique baseline candidates")
    return ResearchSelection(
        primary=primary,
        reference=reference,
        cost_bps=require_bounded_int(value, "cost_bps", 0, 10000),
        first_month=first,
        last_month=last,
        final_seed=require_bounded_int(value, "final_seed", 0, 2**31 - 1),
        required_candidates=candidates,
    )


__all__ = ["ResearchSelection", "decode_research_selection", "encode_research_selection"]
