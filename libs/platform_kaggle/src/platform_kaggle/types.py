"""Core types for platform_kaggle.

The payload classes and their codecs live in the private _types_* modules,
grouped by the entity they describe; this module is the public surface that
re-exports them alongside the shared types from platform_codebase."""

from __future__ import annotations

from platform_codebase import (
    CapabilityStrength,
    CodebaseCapability,
    CodebaseProfile,
    LibInfo,
    MatchRecommendation,
    ServiceInfo,
    decode_capability,
    decode_profile,
    encode_capability,
    encode_profile,
    require_recommendation,
    require_strength,
)

from platform_kaggle._types_competition import (
    Competition,
    decode_competition,
    encode_competition,
)
from platform_kaggle._types_match import (
    CompetitionMatch,
    InterestFilter,
    decode_filter,
    decode_match,
    encode_filter,
    encode_match,
)
from platform_kaggle._types_pages import (
    CompetitionPage,
    CompetitionPages,
    decode_competition_page,
    decode_competition_pages,
    encode_competition_page,
    encode_competition_pages,
)
from platform_kaggle._types_protocols import (
    CompetitionsResponseProtocol,
    KaggleApiClassProtocol,
    KaggleApiFactoryProtocol,
    KaggleApiProtocol,
    KaggleClientProtocol,
    KaggleCompetitionProtocol,
    KaggleModuleProtocol,
    KagglePageFetcherProtocol,
    KagglePreAuthModuleProtocol,
    KaggleTagProtocol,
)
from platform_kaggle._types_validation import CompetitionCategory

__all__ = [
    "CapabilityStrength",
    "CodebaseCapability",
    "CodebaseProfile",
    "Competition",
    "CompetitionCategory",
    "CompetitionMatch",
    "CompetitionPage",
    "CompetitionPages",
    "CompetitionsResponseProtocol",
    "InterestFilter",
    "KaggleApiClassProtocol",
    "KaggleApiFactoryProtocol",
    "KaggleApiProtocol",
    "KaggleClientProtocol",
    "KaggleCompetitionProtocol",
    "KaggleModuleProtocol",
    "KagglePageFetcherProtocol",
    "KagglePreAuthModuleProtocol",
    "KaggleTagProtocol",
    "LibInfo",
    "MatchRecommendation",
    "ServiceInfo",
    "decode_capability",
    "decode_competition",
    "decode_competition_page",
    "decode_competition_pages",
    "decode_filter",
    "decode_match",
    "decode_profile",
    "encode_capability",
    "encode_competition",
    "encode_competition_page",
    "encode_competition_pages",
    "encode_filter",
    "encode_match",
    "encode_profile",
    "require_recommendation",
    "require_strength",
]
