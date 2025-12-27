"""Competition matching logic.

Scores competitions against codebase capabilities to determine fit.
"""

from __future__ import annotations

from .types import (
    CodebaseProfile,
    Competition,
    CompetitionMatch,
    MatchRecommendation,
)

# -----------------------------------------------------------------------------
# Tag Matching
# -----------------------------------------------------------------------------


def _normalize_tag(tag: str) -> str:
    """Normalize tag for comparison.

    Args:
        tag: Tag to normalize.

    Returns:
        Lowercase, hyphen-normalized tag.
    """
    return tag.lower().replace("_", "-")


def _tags_overlap(
    competition_tags: tuple[str, ...],
    capability_tags: tuple[str, ...],
) -> tuple[str, ...]:
    """Find overlapping tags between competition and capability.

    Args:
        competition_tags: Tags from competition.
        capability_tags: Tags from capability.

    Returns:
        Tuple of matching tags.
    """
    comp_normalized = {_normalize_tag(t) for t in competition_tags}
    cap_normalized = {_normalize_tag(t) for t in capability_tags}

    overlap = comp_normalized & cap_normalized
    return tuple(sorted(overlap))


def _infer_competition_requirements(competition: Competition) -> tuple[str, ...]:
    """Infer capability requirements from competition tags.

    Args:
        competition: Competition to analyze.

    Returns:
        Tuple of inferred capability names that would help.
    """
    requirements: list[str] = []
    tags_lower = {_normalize_tag(t) for t in competition.tags}

    # Tabular data competitions
    if "tabular" in tags_lower or "structured" in tags_lower:
        requirements.append("xgboost_tabular")
        requirements.append("lightgbm_tabular")

    # Classification tasks
    if "classification" in tags_lower or "binary-classification" in tags_lower:
        requirements.append("sklearn_ml")

    # Time series
    if "time-series" in tags_lower or "forecasting" in tags_lower:
        requirements.append("pytorch_deep_learning")

    # NLP tasks
    if "nlp" in tags_lower or "text" in tags_lower:
        requirements.append("language_identification")

    # Speech tasks
    if "speech" in tags_lower or "audio" in tags_lower:
        requirements.append("speech_to_text")

    # Optimization/AutoML
    if "hyperparameter" in tags_lower or "optimization" in tags_lower:
        requirements.append("hyperparameter_optimization")

    return tuple(requirements)


# -----------------------------------------------------------------------------
# Score Calculation
# -----------------------------------------------------------------------------


def _calculate_match_score(
    matched_caps: tuple[str, ...],
    missing_caps: tuple[str, ...],
) -> float:
    """Calculate match score from matched and missing capabilities.

    Args:
        matched_caps: Capabilities that match.
        missing_caps: Capabilities that are missing.

    Returns:
        Score from 0.0 to 1.0.
    """
    total = len(matched_caps) + len(missing_caps)
    if total == 0:
        # No requirements inferred, assume moderate fit
        return 0.5

    return len(matched_caps) / total


def _determine_recommendation(score: float) -> MatchRecommendation:
    """Determine recommendation level from score.

    Args:
        score: Match score from 0.0 to 1.0.

    Returns:
        MatchRecommendation based on score.
    """
    if score >= 0.8:
        return "strong_fit"
    if score >= 0.5:
        return "good_fit"
    if score >= 0.2:
        return "stretch"
    return "new_territory"


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def match_competition(
    competition: Competition,
    profile: CodebaseProfile,
) -> CompetitionMatch:
    """Score a competition against codebase capabilities.

    Analyzes competition tags to infer requirements, then matches against
    the codebase's detected capabilities.

    Args:
        competition: Competition to evaluate.
        profile: Codebase capability profile.

    Returns:
        CompetitionMatch with score and recommendation.
    """
    # Get all capability names and tags from profile
    cap_names: set[str] = set()
    all_cap_tags: set[str] = set()
    for cap in profile.capabilities:
        cap_names.add(cap.name)
        for tag in cap.tags:
            all_cap_tags.add(_normalize_tag(tag))

    # Infer requirements from competition
    requirements = _infer_competition_requirements(competition)

    # Find matched and missing capabilities
    matched: list[str] = []
    missing: list[str] = []

    for req in requirements:
        if req in cap_names:
            matched.append(req)
        else:
            missing.append(req)

    # Also check for direct tag overlap
    comp_tags_normalized = {_normalize_tag(t) for t in competition.tags}
    tag_overlap = comp_tags_normalized & all_cap_tags

    # Boost score if there's tag overlap but no explicit requirement matching
    matched_tuple = tuple(matched)
    missing_tuple = tuple(missing)

    score = _calculate_match_score(matched_tuple, missing_tuple)

    # Boost score slightly for tag overlap
    if tag_overlap and score < 1.0:
        boost = min(0.1 * len(tag_overlap), 0.2)
        score = min(score + boost, 1.0)

    recommendation = _determine_recommendation(score)

    return CompetitionMatch(
        competition=competition,
        match_score=score,
        matched_capabilities=matched_tuple,
        missing_capabilities=missing_tuple,
        recommendation=recommendation,
    )


def match_competitions(
    competitions: tuple[Competition, ...],
    profile: CodebaseProfile,
    *,
    min_score: float = 0.0,
) -> tuple[CompetitionMatch, ...]:
    """Match multiple competitions against codebase capabilities.

    Args:
        competitions: Competitions to evaluate.
        profile: Codebase capability profile.
        min_score: Minimum match score to include (default 0.0).

    Returns:
        Tuple of CompetitionMatch, sorted by score descending.
    """
    matches: list[CompetitionMatch] = []

    for comp in competitions:
        match = match_competition(comp, profile)
        if match.match_score >= min_score:
            matches.append(match)

    # Sort by score descending
    matches.sort(key=lambda m: m.match_score, reverse=True)

    return tuple(matches)


__all__ = [
    "match_competition",
    "match_competitions",
]
