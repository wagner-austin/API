from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict


class StatsRequest(TypedDict, total=True):
    """Request for user stats card.

    Attributes:
        username: GitHub username to fetch stats for.
        theme: Color theme for the card.
        hide_border: Whether to hide the card border.
        show_icons: Whether to show icons.
        include_all_commits: Include all commits, not just current year.
        hide: List of stats to hide (stars, commits, prs, issues, contribs).
        disable_animations: Whether to disable CSS animations.
    """

    username: str
    theme: Literal[
        "default",
        "dark",
        "dracula",
        "github_dark",
        "transparent",
    ]
    hide_border: bool
    show_icons: bool
    include_all_commits: bool
    hide: tuple[str, ...]
    disable_animations: bool


class LangsRequest(TypedDict, total=True):
    """Request for top languages card.

    Attributes:
        username: GitHub username to fetch languages for.
        theme: Color theme for the card.
        hide_border: Whether to hide the card border.
        layout: Layout style (default, compact, donut, pie).
        langs_count: Number of languages to show (1-20).
        hide: Languages to hide from the card.
        disable_animations: Whether to disable CSS animations.
    """

    username: str
    theme: Literal[
        "default",
        "dark",
        "dracula",
        "github_dark",
        "transparent",
    ]
    hide_border: bool
    layout: Literal["default", "compact", "donut", "pie"]
    langs_count: int
    hide: tuple[str, ...]
    disable_animations: bool


class UserStats(TypedDict, total=True):
    """GitHub user statistics.

    Attributes:
        username: GitHub username.
        name: Display name.
        total_commits: Total commit count.
        total_prs: Total pull request count.
        total_issues: Total issue count.
        total_stars: Total stars received.
        total_contributions: Total contributions.
        rank: Calculated rank (S+, S, A+, A, B+, B, C).
        rank_percentile: Percentile ranking (0-100).
    """

    username: str
    name: str
    total_commits: int
    total_prs: int
    total_issues: int
    total_stars: int
    total_contributions: int
    rank: Literal["S+", "S", "A+", "A", "B+", "B", "C"]
    rank_percentile: float


class LanguageStats(TypedDict, total=True):
    """Language usage statistics.

    Attributes:
        name: Language name.
        size: Total bytes of code.
        percentage: Percentage of total code.
        color: Language color (hex).
    """

    name: str
    size: int
    percentage: float
    color: str


class LangsResponse(TypedDict, total=True):
    """Response containing language statistics.

    Attributes:
        username: GitHub username.
        languages: List of language statistics.
        total_size: Total bytes of code across all languages.
    """

    username: str
    languages: list[LanguageStats]
    total_size: int


# Theme literal type for reuse
ThemeName = Literal[
    "default",
    "dark",
    "dracula",
    "github_dark",
    "transparent",
]

# Capability strength literal type (matches platform_codebase)
CapabilityStrength = Literal["strong", "moderate", "basic"]


class CapabilitiesRequest(TypedDict, total=True):
    """Request for codebase capabilities card.

    Attributes:
        repo: GitHub repository in 'owner/repo' format.
        theme: Color theme for the card.
        hide_border: Whether to hide the card border.
        disable_animations: Whether to disable CSS animations.
    """

    repo: str
    theme: ThemeName
    hide_border: bool
    disable_animations: bool


class Capability(TypedDict, total=True):
    """A detected codebase capability.

    Attributes:
        name: Capability identifier (e.g., 'xgboost_tabular').
        strength: Capability strength level.
        tags: Tuple of tags this capability matches.
        description: Human-readable description.
    """

    name: str
    strength: CapabilityStrength
    tags: tuple[str, ...]
    description: str


class CapabilitiesResponse(TypedDict, total=True):
    """Response containing codebase capabilities.

    Attributes:
        repo: GitHub repository scanned.
        capabilities: List of detected capabilities.
        ml_backends: Tuple of ML backend names.
        frameworks: Tuple of framework names.
        data_formats: Tuple of supported data formats.
        task_types: Tuple of supported task types.
    """

    repo: str
    capabilities: tuple[Capability, ...]
    ml_backends: tuple[str, ...]
    frameworks: tuple[str, ...]
    data_formats: tuple[str, ...]
    task_types: tuple[str, ...]


__all__ = [
    "CapabilitiesRequest",
    "CapabilitiesResponse",
    "Capability",
    "CapabilityStrength",
    "LangsRequest",
    "LangsResponse",
    "LanguageStats",
    "StatsRequest",
    "ThemeName",
    "UserStats",
]
