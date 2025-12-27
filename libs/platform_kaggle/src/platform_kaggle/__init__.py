"""Kaggle competition discovery with codebase capability matching.

This library provides tools for discovering Kaggle competitions that match
your codebase capabilities and personal interests.

Example usage::

    from platform_kaggle import find_competitions, make_interest_filter

    # Find ML/linguistics competitions that fit the codebase
    matches = find_competitions(
        interests=make_interest_filter(
            include_tags=("tabular", "nlp", "classification"),
            exclude_tags=("computer-vision", "image"),
        ),
        match_codebase=True,
        min_match_score=0.3,
    )

    for match in matches:
        title = match.competition.title
        score = f"{match.match_score:.0%}"
        recommendation = match.recommendation
        # Use title, score, recommendation as needed
"""

from __future__ import annotations

from pathlib import Path

from .capabilities import build_profile, scan_codebase
from .client import KaggleClient
from .filters import filter_competitions, make_interest_filter
from .matcher import match_competition, match_competitions
from .testing import (
    FakeKaggleApi,
    FakeKaggleClient,
    FakeKaggleCompetition,
    hooks,
    make_fake_capability,
    make_fake_competition,
    make_fake_kaggle_competition,
    make_fake_profile,
    reset_hooks,
)
from .types import (
    CapabilityStrength,
    CodebaseCapability,
    CodebaseProfile,
    Competition,
    CompetitionCategory,
    CompetitionMatch,
    InterestFilter,
    KaggleClientProtocol,
    LibInfo,
    MatchRecommendation,
    ServiceInfo,
    decode_capability,
    decode_competition,
    decode_filter,
    decode_match,
    decode_profile,
    encode_capability,
    encode_competition,
    encode_filter,
    encode_match,
    encode_profile,
)

# -----------------------------------------------------------------------------
# High-Level API
# -----------------------------------------------------------------------------


def find_competitions(
    *,
    interests: InterestFilter | None = None,
    match_codebase: bool = True,
    min_match_score: float = 0.0,
    codebase_root: Path | None = None,
) -> tuple[CompetitionMatch, ...]:
    """Find competitions matching interests and codebase capabilities.

    This is the main entry point for discovering competitions. It:
    1. Fetches competitions from Kaggle API
    2. Filters by user interests (if provided)
    3. Matches against codebase capabilities (if enabled)
    4. Returns sorted results

    Args:
        interests: Optional interest filter to apply.
        match_codebase: Whether to match against codebase capabilities.
        min_match_score: Minimum match score to include (0.0 to 1.0).
        codebase_root: Path to monorepo root (auto-detected if None).

    Returns:
        Tuple of CompetitionMatch, sorted by score descending.
    """
    # Get client via hook
    client = hooks.kaggle_client()

    # Fetch all competitions
    competitions = client.list_competitions()

    # Apply interest filter if provided
    if interests is not None:
        competitions = filter_competitions(competitions, interests)

    # Match against codebase capabilities
    if match_codebase:
        root = codebase_root
        if root is None:
            # Default to parent of libs directory
            root = Path(__file__).parent.parent.parent.parent.parent
        profile = scan_codebase(root)
        return match_competitions(competitions, profile, min_score=min_match_score)

    # Return as matches with default score
    result: list[CompetitionMatch] = []
    for comp in competitions:
        result.append(
            CompetitionMatch(
                competition=comp,
                match_score=0.5,
                matched_capabilities=(),
                missing_capabilities=(),
                recommendation="good_fit",
            )
        )
    return tuple(result)


def get_codebase_profile(root: Path | None = None) -> CodebaseProfile:
    """Get the capability profile of this codebase.

    Args:
        root: Path to monorepo root (auto-detected if None).

    Returns:
        CodebaseProfile with detected capabilities.
    """
    if root is None:
        root = Path(__file__).parent.parent.parent.parent.parent
    return scan_codebase(root)


__all__ = [
    "CapabilityStrength",
    "CodebaseCapability",
    "CodebaseProfile",
    "Competition",
    "CompetitionCategory",
    "CompetitionMatch",
    "FakeKaggleApi",
    "FakeKaggleClient",
    "FakeKaggleCompetition",
    "InterestFilter",
    "KaggleClient",
    "KaggleClientProtocol",
    "LibInfo",
    "MatchRecommendation",
    "ServiceInfo",
    "build_profile",
    "decode_capability",
    "decode_competition",
    "decode_filter",
    "decode_match",
    "decode_profile",
    "encode_capability",
    "encode_competition",
    "encode_filter",
    "encode_match",
    "encode_profile",
    "filter_competitions",
    "find_competitions",
    "get_codebase_profile",
    "hooks",
    "make_fake_capability",
    "make_fake_competition",
    "make_fake_kaggle_competition",
    "make_fake_profile",
    "make_interest_filter",
    "match_competition",
    "match_competitions",
    "reset_hooks",
    "scan_codebase",
]
