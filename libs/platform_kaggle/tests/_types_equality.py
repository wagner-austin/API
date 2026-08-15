"""Structural equality helpers shared by the platform_kaggle type tests.

Each payload class compares by identity, so these compare field by field."""

from __future__ import annotations

from platform_kaggle.types import (
    CodebaseCapability,
    CodebaseProfile,
    Competition,
    CompetitionMatch,
    CompetitionPage,
    CompetitionPages,
    InterestFilter,
    LibInfo,
    ServiceInfo,
)

# -----------------------------------------------------------------------------
# Comparison Helpers (classes use __slots__, so no built-in equality)
# -----------------------------------------------------------------------------


def _competitions_equal(a: Competition, b: Competition) -> bool:
    """Check if two Competition instances are equal."""
    return (
        a.ref == b.ref
        and a.title == b.title
        and a.category == b.category
        and a.reward == b.reward
        and a.deadline == b.deadline
        and a.team_count == b.team_count
        and a.tags == b.tags
        and a.description == b.description
        and a.url == b.url
    )


def _capabilities_equal(a: CodebaseCapability, b: CodebaseCapability) -> bool:
    """Check if two CodebaseCapability instances are equal."""
    return (
        a.name == b.name
        and a.strength == b.strength
        and a.tags == b.tags
        and a.description == b.description
    )


def _profiles_equal(a: CodebaseProfile, b: CodebaseProfile) -> bool:
    """Check if two CodebaseProfile instances are equal."""
    if len(a.capabilities) != len(b.capabilities):
        return False
    for cap_a, cap_b in zip(a.capabilities, b.capabilities, strict=True):
        if not _capabilities_equal(cap_a, cap_b):
            return False
    return (
        a.ml_backends == b.ml_backends
        and a.data_formats == b.data_formats
        and a.task_types == b.task_types
    )


def _matches_equal(a: CompetitionMatch, b: CompetitionMatch) -> bool:
    """Check if two CompetitionMatch instances are equal."""
    return (
        _competitions_equal(a.competition, b.competition)
        and a.match_score == b.match_score
        and a.matched_capabilities == b.matched_capabilities
        and a.missing_capabilities == b.missing_capabilities
        and a.recommendation == b.recommendation
    )


def _filters_equal(a: InterestFilter, b: InterestFilter) -> bool:
    """Check if two InterestFilter instances are equal."""
    return (
        a.include_tags == b.include_tags
        and a.exclude_tags == b.exclude_tags
        and a.min_reward == b.min_reward
        and a.categories == b.categories
    )


def _libinfos_equal(a: LibInfo, b: LibInfo) -> bool:
    """Check if two LibInfo instances are equal."""
    return a.name == b.name and a.path == b.path and a.dependencies == b.dependencies


def _serviceinfos_equal(a: ServiceInfo, b: ServiceInfo) -> bool:
    """Check if two ServiceInfo instances are equal."""
    return (
        a.name == b.name
        and a.path == b.path
        and a.dependencies == b.dependencies
        and a.has_rules_files == b.has_rules_files
    )


def _pages_equal(a: CompetitionPage, b: CompetitionPage) -> bool:
    """Check if two CompetitionPage instances are equal."""
    return a.id == b.id and a.name == b.name and a.content == b.content


def _competition_pages_equal(a: CompetitionPages, b: CompetitionPages) -> bool:
    """Check if two CompetitionPages instances are equal."""
    if len(a.pages) != len(b.pages):
        return False
    for page_a, page_b in zip(a.pages, b.pages, strict=True):
        if not _pages_equal(page_a, page_b):
            return False
    return (
        a.competition_id == b.competition_id
        and a.description == b.description
        and a.evaluation == b.evaluation
        and a.timeline == b.timeline
        and a.rules == b.rules
    )
