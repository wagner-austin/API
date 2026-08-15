"""Core types for platform_devpost.

The TypedDicts and their codecs live in the private _types_* modules, grouped by
the entity they describe; this module is the public surface that re-exports them
alongside the shared types from platform_codebase.
"""

from __future__ import annotations

from platform_codebase import (
    CapabilityStrength,
    CodebaseCapability,
    CodebaseProfile,
    MatchRecommendation,
    decode_capability,
    decode_profile,
    encode_capability,
    encode_profile,
    require_recommendation,
    require_strength,
)

from platform_devpost._types_hackathon import (
    Hackathon,
    decode_hackathon,
    encode_hackathon,
)
from platform_devpost._types_listing import (
    HackathonListMeta,
    HackathonListResponse,
    decode_list_meta,
    decode_list_response,
    encode_list_meta,
    encode_list_response,
)
from platform_devpost._types_match import (
    HackathonMatch,
    InterestFilter,
    decode_filter,
    decode_match,
    encode_filter,
    encode_match,
)
from platform_devpost._types_protocols import (
    DevpostApiProtocol,
    DevpostClientProtocol,
)
from platform_devpost._types_theme import (
    DisplayedLocation,
    Theme,
    decode_displayed_location,
    decode_theme,
    encode_displayed_location,
    encode_theme,
)
from platform_devpost._types_validation import HackathonState

__all__ = [
    "CapabilityStrength",
    "CodebaseCapability",
    "CodebaseProfile",
    "DevpostApiProtocol",
    "DevpostClientProtocol",
    "DisplayedLocation",
    "Hackathon",
    "HackathonListMeta",
    "HackathonListResponse",
    "HackathonMatch",
    "HackathonState",
    "InterestFilter",
    "MatchRecommendation",
    "Theme",
    "decode_capability",
    "decode_displayed_location",
    "decode_filter",
    "decode_hackathon",
    "decode_list_meta",
    "decode_list_response",
    "decode_match",
    "decode_profile",
    "decode_theme",
    "encode_capability",
    "encode_displayed_location",
    "encode_filter",
    "encode_hackathon",
    "encode_list_meta",
    "encode_list_response",
    "encode_match",
    "encode_profile",
    "encode_theme",
    "require_recommendation",
    "require_strength",
]
