"""Competition matching logic.

Scores competitions against codebase capabilities to determine fit.
"""

from __future__ import annotations

import re

from .types import (
    CodebaseProfile,
    Competition,
    CompetitionMatch,
    CompetitionPages,
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


# Tag to capability mapping: tag -> (capability_names, related_tags)
_TAG_CAPABILITY_MAP: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "tabular": (("xgboost_tabular", "lightgbm_tabular"), ("tabular", "structured")),
    "structured": (("xgboost_tabular", "lightgbm_tabular"), ("tabular", "structured")),
    "classification": (("sklearn_ml",), ("classification", "binary-classification")),
    "binary-classification": (("sklearn_ml",), ("classification", "binary-classification")),
    "regression": (("sklearn_ml",), ("regression",)),
    "time-series": (("pytorch_deep_learning",), ("time-series", "forecasting")),
    "forecasting": (("pytorch_deep_learning",), ("time-series", "forecasting")),
    "nlp": (("huggingface_transformers",), ("nlp", "text")),
    "text": (("huggingface_transformers",), ("nlp", "text")),
    "computer-vision": (("torchvision_cv",), ("computer-vision", "image")),
    "image": (("torchvision_cv",), ("computer-vision", "image")),
    "speech": (("speech_to_text",), ("speech", "audio")),
    "audio": (("speech_to_text",), ("speech", "audio")),
    "hyperparameter": (("hyperparameter_optimization",), ("hyperparameter", "optimization")),
    "optimization": (("hyperparameter_optimization",), ("hyperparameter", "optimization")),
    "deep-learning": (("pytorch_deep_learning",), ("deep-learning", "neural-network")),
    "neural-network": (("pytorch_deep_learning",), ("deep-learning", "neural-network")),
    "llm": (("huggingface_transformers",), ("llm", "large-language-model")),
    "large-language-model": (("huggingface_transformers",), ("llm", "large-language-model")),
}


# -----------------------------------------------------------------------------
# Description-Based Keyword Matching
# -----------------------------------------------------------------------------

# Keywords that indicate specific framework/tool requirements
# Maps keyword pattern -> (capability_name, is_hard_requirement)
# Hard requirements mean competition EXPLICITLY MANDATES this tool
# Soft requirements mean it's mentioned/useful but not mandatory
_KEYWORD_REQUIREMENTS: tuple[tuple[str, str, bool], ...] = (
    # Explicit model requirements (hard - only when clearly mandated)
    (r"\bmust\s+use\s+gemma\b", "gemma_model", True),
    (r"\busing\s+gemma\b", "gemma_model", True),
    (r"\bgemma\s+3n\b", "gemma_model", True),  # Specific version = requirement
    # Mobile/Edge deployment (hard - these are structural requirements)
    (r"\bon-device\b", "mobile_development", True),
    (r"\bmobile[- ]first\b", "mobile_development", True),
    (r"\bmobile\s+app\b", "mobile_development", True),
    (r"\bedge[- ]ai\b", "edge_deployment", True),
    (r"\bedge\s+deployment\b", "edge_deployment", True),
    (r"\bjetson\b", "nvidia_jetson", True),
    # LLM mentions (soft - typically just examples or suggestions)
    (r"\bgemma\b", "gemma_model", False),
    (r"\bllama\b", "llama_model", False),
    (r"\bmistral\b", "mistral_model", False),
    (r"\bgpt-4\b", "openai_gpt4", False),
    (r"\bclaude\b", "anthropic_claude", False),
    (r"\bollama\b", "ollama_runtime", False),
    # Frameworks (soft - indicates useful skills)
    (r"\bpytorch\b", "pytorch_deep_learning", False),
    (r"\btensorflow\b", "tensorflow", False),
    (r"\bjax\b", "jax", False),
    (r"\bxgboost\b", "xgboost_tabular", False),
    (r"\blightgbm\b", "lightgbm_tabular", False),
    (r"\bcatboost\b", "catboost", False),
    (r"\bscikit-learn\b", "sklearn_ml", False),
    (r"\bsklearn\b", "sklearn_ml", False),
    (r"\bhugg?ing\s*face\b", "huggingface_transformers", False),
    (r"\btransformers\b", "huggingface_transformers", False),
    (r"\bopencv\b", "opencv_cv", False),
    (r"\bwhisper\b", "speech_to_text", False),
    # Hackathon indicators (hard only when it's clearly a hackathon format)
    (r"\bsubmit\s+a\s+video\b", "video_production", True),
    (r"\bvideo\s+demo\s+required\b", "video_production", True),
    (r"\bhackathon\s+format\b", "hackathon_skills", True),
)


def _extract_requirements_from_description(
    pages: CompetitionPages,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract capability requirements from competition description.

    Analyzes the description, evaluation, and rules for specific
    framework/tool mentions.

    Args:
        pages: Competition pages with description content.

    Returns:
        Tuple of (hard_requirements, soft_requirements).
    """
    # Combine all text for analysis
    text = " ".join(
        [
            pages.description,
            pages.evaluation,
            pages.rules,
        ]
    ).lower()

    hard_reqs: set[str] = set()
    soft_reqs: set[str] = set()

    for pattern, capability, is_hard in _KEYWORD_REQUIREMENTS:
        if re.search(pattern, text, re.IGNORECASE):
            if is_hard:
                hard_reqs.add(capability)
            else:
                soft_reqs.add(capability)

    return tuple(sorted(hard_reqs)), tuple(sorted(soft_reqs))


def _infer_competition_requirements(
    competition: Competition,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Infer capability requirements from competition tags.

    Args:
        competition: Competition to analyze.

    Returns:
        Tuple of (mapped_requirements, unmapped_tags).
    """
    requirements: set[str] = set()
    mapped_tags: set[str] = set()
    tags_lower = {_normalize_tag(t) for t in competition.tags}

    for tag in tags_lower:
        if tag in _TAG_CAPABILITY_MAP:
            caps, related = _TAG_CAPABILITY_MAP[tag]
            requirements.update(caps)
            mapped_tags.update(related)

    unmapped = tags_lower - mapped_tags

    return tuple(sorted(requirements)), tuple(sorted(unmapped))


# -----------------------------------------------------------------------------
# Score Calculation
# -----------------------------------------------------------------------------


def _calculate_match_score(
    matched_caps: tuple[str, ...],
    missing_caps: tuple[str, ...],
    missing_hard_caps: tuple[str, ...] = (),
) -> float:
    """Calculate match score from matched and missing capabilities.

    Hard requirements (from description analysis) are weighted more heavily.
    Missing a hard requirement significantly reduces the score.

    Args:
        matched_caps: Capabilities that match.
        missing_caps: Capabilities that are missing (soft).
        missing_hard_caps: Hard requirements that are missing.

    Returns:
        Score from 0.0 to 1.0.
    """
    # If missing hard requirements, cap score at 0.3 max
    if missing_hard_caps:
        # Each missing hard req reduces max score further
        hard_penalty = min(len(missing_hard_caps) * 0.15, 0.3)
        max_score = 0.3 - hard_penalty

        # Calculate base score from soft requirements
        total = len(matched_caps) + len(missing_caps)
        base_score = 0.5 if total == 0 else len(matched_caps) / total

        return min(base_score, max_score)

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


_EMPTY_PAGES: CompetitionPages = CompetitionPages(
    competition_id=0,
    pages=(),
    description="",
    evaluation="",
    timeline="",
    rules="",
)


def match_competition(
    competition: Competition,
    profile: CodebaseProfile,
    pages: CompetitionPages = _EMPTY_PAGES,
) -> CompetitionMatch:
    """Score a competition against codebase capabilities.

    Analyzes competition tags and description to infer requirements,
    then matches against the codebase's detected capabilities.

    Args:
        competition: Competition to evaluate.
        profile: Codebase capability profile.
        pages: Competition pages for description-based matching.

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

    # Infer requirements from competition tags
    requirements, unmapped_tags = _infer_competition_requirements(competition)

    # Extract requirements from description
    hard_reqs, soft_reqs = _extract_requirements_from_description(pages)

    # Combine all requirements
    all_requirements = set(requirements) | set(soft_reqs)

    # Find matched and missing capabilities
    matched: list[str] = []
    missing: list[str] = []
    missing_hard: list[str] = []

    # Check hard requirements first
    for req in hard_reqs:
        if req in cap_names:
            matched.append(req)
        else:
            missing_hard.append(req)

    # Check soft/tag-based requirements
    for req in all_requirements:
        if req in cap_names:
            matched.append(req)
        else:
            missing.append(req)

    # Unmapped tags count as missing (we don't have capabilities for them)
    # But check if any unmapped tag matches our capability tags directly
    for tag in unmapped_tags:
        if tag not in all_cap_tags:
            missing.append(tag)

    matched_tuple = tuple(sorted(set(matched)))
    missing_tuple = tuple(sorted(set(missing)))
    missing_hard_tuple = tuple(sorted(set(missing_hard)))

    score = _calculate_match_score(matched_tuple, missing_tuple, missing_hard_tuple)
    recommendation = _determine_recommendation(score)

    return CompetitionMatch(
        competition=competition,
        match_score=score,
        matched_capabilities=matched_tuple,
        missing_capabilities=missing_tuple + missing_hard_tuple,
        recommendation=recommendation,
    )


_EMPTY_PAGES_MAP: dict[str, CompetitionPages] = {}


def match_competitions(
    competitions: tuple[Competition, ...],
    profile: CodebaseProfile,
    *,
    min_score: float = 0.0,
    pages_map: dict[str, CompetitionPages] = _EMPTY_PAGES_MAP,
) -> tuple[CompetitionMatch, ...]:
    """Match multiple competitions against codebase capabilities.

    Args:
        competitions: Competitions to evaluate.
        profile: Codebase capability profile.
        min_score: Minimum match score to include (default 0.0).
        pages_map: Mapping of competition ref to pages for description matching.

    Returns:
        Tuple of CompetitionMatch, sorted by score descending.
    """
    matches: list[CompetitionMatch] = []

    for comp in competitions:
        pages = pages_map.get(comp.ref, _EMPTY_PAGES)
        match = match_competition(comp, profile, pages)
        if match.match_score >= min_score:
            matches.append(match)

    # Sort by score descending
    matches.sort(key=lambda m: m.match_score, reverse=True)

    return tuple(matches)


__all__ = [
    "match_competition",
    "match_competitions",
]
