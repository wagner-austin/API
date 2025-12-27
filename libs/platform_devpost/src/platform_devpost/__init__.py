"""Devpost hackathon discovery with codebase capability matching.

This library provides tools for discovering Devpost hackathons that match
your codebase capabilities and personal interests.
"""

from __future__ import annotations

from pathlib import Path

from platform_devpost.capabilities import scan_codebase
from platform_devpost.client import DevpostClient
from platform_devpost.filters import filter_hackathons
from platform_devpost.matcher import match_hackathon, match_hackathons
from platform_devpost.testing import (
    FakeDevpostApi,
    FakeDevpostClient,
    hooks,
    make_fake_capability,
    make_fake_displayed_location,
    make_fake_hackathon,
    make_fake_profile,
    make_fake_theme,
    make_interest_filter,
    reset_hooks,
)
from platform_devpost.types import (
    CapabilityStrength,
    CodebaseCapability,
    CodebaseProfile,
    DevpostApiProtocol,
    DevpostClientProtocol,
    DisplayedLocation,
    Hackathon,
    HackathonListMeta,
    HackathonListResponse,
    HackathonMatch,
    HackathonState,
    InterestFilter,
    MatchRecommendation,
    Theme,
    decode_capability,
    decode_displayed_location,
    decode_filter,
    decode_hackathon,
    decode_list_meta,
    decode_list_response,
    decode_match,
    decode_profile,
    decode_theme,
    encode_capability,
    encode_displayed_location,
    encode_filter,
    encode_hackathon,
    encode_list_meta,
    encode_list_response,
    encode_match,
    encode_profile,
    encode_theme,
)


def _find_monorepo_root(start: Path) -> Path:
    """Find the monorepo root by looking for libs directory.

    Args:
        start: Starting path to search from.

    Returns:
        Path to monorepo root.

    Raises:
        RuntimeError: If monorepo root not found.
    """
    current = start
    while True:
        if (current / "libs").is_dir():
            return current
        if current.parent == current:
            raise RuntimeError("monorepo root with 'libs' directory not found")
        current = current.parent


def find_hackathons(
    *,
    interests: InterestFilter | None = None,
    match_codebase: bool = True,
    min_match_score: float = 0.0,
    root: Path | None = None,
) -> tuple[HackathonMatch, ...]:
    """Find hackathons matching interests and codebase capabilities.

    Args:
        interests: Optional interest filter.
        match_codebase: Whether to score against codebase capabilities.
        min_match_score: Minimum match score to include (0.0 to 1.0).
        root: Optional monorepo root path (auto-detected if not provided).

    Returns:
        Tuple of HackathonMatch sorted by score (highest first).
    """
    # Get hackathons from client
    client = hooks.devpost_client()
    hackathons = client.list_hackathons()

    # Apply interest filter
    if interests is not None:
        hackathons = filter_hackathons(hackathons, interests)

    # Match against codebase
    if match_codebase:
        if root is None:
            root = _find_monorepo_root(Path(__file__).resolve())
        profile = hooks.profile_scanner(root)
        matches = match_hackathons(hackathons, profile)

        # Filter by minimum score
        if min_match_score > 0.0:
            matches = tuple(m for m in matches if m.match_score >= min_match_score)

        return matches

    # Return without matching (all scores = 0)
    return tuple(
        HackathonMatch(
            hackathon=h,
            match_score=0.0,
            matched_capabilities=(),
            missing_capabilities=(),
            recommendation="new_territory",
        )
        for h in hackathons
    )


def get_codebase_profile(root: Path | None = None) -> CodebaseProfile:
    """Get the capability profile of this codebase.

    Args:
        root: Optional monorepo root path (auto-detected if not provided).

    Returns:
        CodebaseProfile with detected capabilities.
    """
    if root is None:
        root = _find_monorepo_root(Path(__file__).resolve())
    return hooks.profile_scanner(root)


__all__ = [
    "CapabilityStrength",
    "CodebaseCapability",
    "CodebaseProfile",
    "DevpostApiProtocol",
    "DevpostClient",
    "DevpostClientProtocol",
    "DisplayedLocation",
    "FakeDevpostApi",
    "FakeDevpostClient",
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
    "filter_hackathons",
    "find_hackathons",
    "get_codebase_profile",
    "hooks",
    "make_fake_capability",
    "make_fake_displayed_location",
    "make_fake_hackathon",
    "make_fake_profile",
    "make_fake_theme",
    "make_interest_filter",
    "match_hackathon",
    "match_hackathons",
    "reset_hooks",
    "scan_codebase",
]
