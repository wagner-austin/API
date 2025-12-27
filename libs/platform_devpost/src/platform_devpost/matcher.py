"""Hackathon-to-capability matching.

This module provides functions to score hackathons against codebase
capabilities and determine match recommendations.
"""

from __future__ import annotations

from platform_devpost.types import (
    CodebaseProfile,
    Hackathon,
    HackathonMatch,
    MatchRecommendation,
)

# Theme to capability tag mapping
THEME_TAG_MAPPING: dict[str, tuple[str, ...]] = {
    "machine learning": (
        "ml",
        "ai",
        "data-science",
        "tabular",
        "classification",
        "regression",
        "deep-learning",
        "neural-network",
        "nlp",
        "xgboost",
        "lightgbm",
        "pytorch",
        "sklearn",
        "transformers",
    ),
    "artificial intelligence": (
        "ai",
        "ml",
        "deep-learning",
        "neural-network",
        "nlp",
        "transformers",
        "text-generation",
        "llm",
    ),
    "web": ("web", "frontend", "backend"),
    "api": ("api", "backend", "http"),
    "mobile": ("mobile", "ios", "android"),
    "blockchain": ("blockchain", "web3", "crypto"),
    "gaming": ("gaming", "game-dev"),
    "iot": ("iot", "hardware", "embedded"),
    "fintech": ("fintech", "finance", "banking"),
    "healthcare": ("healthcare", "medical", "health"),
    "education": ("education", "edtech", "learning"),
    "social": ("social", "community"),
    "productivity": ("productivity", "tools"),
    "data": ("data", "analytics", "visualization"),
    "security": ("security", "cybersecurity"),
    "cloud": ("cloud", "infrastructure", "devops"),
    "nlp": ("nlp", "language", "text"),
    "computer vision": ("cv", "image", "video"),
    "python": ("python", "backend"),
    "javascript": ("javascript", "frontend", "web"),
}


def _get_hackathon_tags(hackathon: Hackathon) -> set[str]:
    """Extract tags from hackathon themes.

    Args:
        hackathon: Hackathon to extract tags from.

    Returns:
        Set of tags derived from hackathon themes.
    """
    tags: set[str] = set()

    for theme in hackathon.themes:
        theme_lower = theme.name.lower()

        # Add mapped tags if we have a mapping, otherwise add raw theme
        has_mapping = False
        for key, mapped_tags in THEME_TAG_MAPPING.items():
            if key in theme_lower:
                tags.update(mapped_tags)
                has_mapping = True

        if not has_mapping:
            tags.add(theme_lower)

    return tags


def _get_profile_tags(profile: CodebaseProfile) -> set[str]:
    """Extract tags from codebase profile.

    Args:
        profile: Codebase profile to extract tags from.

    Returns:
        Set of tags from profile capabilities.
    """
    tags: set[str] = set()

    for cap in profile.capabilities:
        tags.update(cap.tags)

    # Add technologies and frameworks as tags
    tags.update(profile.technologies)
    tags.update(profile.frameworks)

    return tags


def _calculate_match_score(
    hackathon_tags: set[str],
    profile_tags: set[str],
) -> float:
    """Calculate match score based on tag overlap.

    Args:
        hackathon_tags: Tags from hackathon.
        profile_tags: Tags from codebase profile.

    Returns:
        Score from 0.0 to 1.0.
    """
    if len(hackathon_tags) == 0:
        return 0.0

    overlap = hackathon_tags & profile_tags
    score = len(overlap) / len(hackathon_tags)
    return min(1.0, score)


def _determine_recommendation(score: float) -> MatchRecommendation:
    """Determine recommendation based on match score.

    Args:
        score: Match score from 0.0 to 1.0.

    Returns:
        Match recommendation level.
    """
    if score >= 0.7:
        return "strong_fit"
    if score >= 0.4:
        return "good_fit"
    if score >= 0.2:
        return "stretch"
    return "new_territory"


def _get_matched_capabilities(
    hackathon_tags: set[str],
    profile: CodebaseProfile,
) -> tuple[str, ...]:
    """Get names of capabilities that match hackathon tags.

    Args:
        hackathon_tags: Tags from hackathon.
        profile: Codebase profile.

    Returns:
        Tuple of matched capability names.
    """
    matched: list[str] = []

    for cap in profile.capabilities:
        cap_tags = set(cap.tags)
        if cap_tags & hackathon_tags:
            matched.append(cap.name)

    return tuple(sorted(set(matched)))


def _get_missing_capabilities(
    hackathon_tags: set[str],
    profile_tags: set[str],
) -> tuple[str, ...]:
    """Get tags that hackathon needs but profile doesn't have.

    Args:
        hackathon_tags: Tags from hackathon.
        profile_tags: Tags from codebase profile.

    Returns:
        Tuple of missing tag names.
    """
    missing = hackathon_tags - profile_tags
    return tuple(sorted(missing))


def match_hackathon(
    hackathon: Hackathon,
    profile: CodebaseProfile,
) -> HackathonMatch:
    """Score a hackathon against codebase capabilities.

    Args:
        hackathon: Hackathon to score.
        profile: Codebase capability profile.

    Returns:
        HackathonMatch with score and recommendation.
    """
    hackathon_tags = _get_hackathon_tags(hackathon)
    profile_tags = _get_profile_tags(profile)

    score = _calculate_match_score(hackathon_tags, profile_tags)
    recommendation = _determine_recommendation(score)
    matched = _get_matched_capabilities(hackathon_tags, profile)
    missing = _get_missing_capabilities(hackathon_tags, profile_tags)

    return HackathonMatch(
        hackathon=hackathon,
        match_score=score,
        matched_capabilities=matched,
        missing_capabilities=missing,
        recommendation=recommendation,
    )


def match_hackathons(
    hackathons: tuple[Hackathon, ...],
    profile: CodebaseProfile,
) -> tuple[HackathonMatch, ...]:
    """Score multiple hackathons and sort by match score.

    Args:
        hackathons: Tuple of hackathons to score.
        profile: Codebase capability profile.

    Returns:
        Tuple of HackathonMatch sorted by score (highest first).
    """
    matches = [match_hackathon(h, profile) for h in hackathons]
    matches.sort(key=lambda m: m.match_score, reverse=True)
    return tuple(matches)
