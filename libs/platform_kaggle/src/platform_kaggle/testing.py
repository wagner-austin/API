"""Public test utilities for platform_kaggle consumers.

This module provides hooks for dependency injection and fake implementations
for testing. Production code sets hooks at startup; tests set them to fakes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from platform_kaggle.types import (
    CapabilityStrength,
    CodebaseCapability,
    CodebaseProfile,
    Competition,
    CompetitionCategory,
    CompetitionsResponseProtocol,
    KaggleApiClassProtocol,
    KaggleApiFactoryProtocol,
    KaggleApiProtocol,
    KaggleClientProtocol,
    KaggleCompetitionProtocol,
    KaggleModuleProtocol,
)

# -----------------------------------------------------------------------------
# Hook Types
# -----------------------------------------------------------------------------

KaggleClientHook = Callable[[], KaggleClientProtocol]
ProfileScannerHook = Callable[[Path], CodebaseProfile]
KaggleModuleHook = Callable[[], KaggleModuleProtocol]


# -----------------------------------------------------------------------------
# Hooks Container
# -----------------------------------------------------------------------------


class HooksContainer:
    """Container for dependency injection hooks.

    Attributes:
        kaggle_api_factory: Factory for low-level Kaggle API.
        kaggle_client: Factory for Kaggle client.
        profile_scanner: Factory for codebase profile scanner.
        kaggle_module: Factory for kaggle module import.
    """

    kaggle_api_factory: KaggleApiFactoryProtocol
    kaggle_client: KaggleClientHook
    profile_scanner: ProfileScannerHook
    kaggle_module: KaggleModuleHook


hooks = HooksContainer()


def _init_production_hooks() -> None:
    """Initialize hooks with production implementations."""
    # Import here to avoid circular dependency
    from platform_kaggle._production import default_kaggle_api_factory, make_kaggle_client
    from platform_kaggle.capabilities import scan_codebase

    hooks.kaggle_api_factory = default_kaggle_api_factory
    hooks.kaggle_client = make_kaggle_client
    hooks.profile_scanner = scan_codebase


def _default_kaggle_module() -> KaggleModuleProtocol:
    """Default kaggle module importer."""
    mod: KaggleModuleProtocol = __import__("kaggle.api.kaggle_api_extended", fromlist=["KaggleApi"])
    return mod


def _init_minimal_hooks() -> None:
    """Initialize only low-level hooks (for module import time)."""
    from platform_kaggle._production import default_kaggle_api_factory

    hooks.kaggle_api_factory = default_kaggle_api_factory
    hooks.kaggle_module = _default_kaggle_module


def reset_hooks() -> None:
    """Reset hooks to production implementations (for test teardown)."""
    _init_production_hooks()


# Initialize minimal hooks on module load to break circular import
_init_minimal_hooks()


# -----------------------------------------------------------------------------
# Fake Kaggle API Implementation
# -----------------------------------------------------------------------------


class FakeApiTag:
    """Fake Kaggle API tag object matching ApiCategory structure.

    Attributes:
        ref: Tag reference slug.
    """

    __slots__ = ("ref",)

    def __init__(self, ref: str) -> None:
        """Initialize fake tag.

        Args:
            ref: Tag reference slug.
        """
        self.ref = ref


class FakeKaggleCompetition:
    """Fake Kaggle competition object matching ApiCompetition structure.

    Attributes:
        ref: Competition reference slug.
        title: Competition title.
        category: Competition category string.
        reward: Prize description.
        deadline: Deadline as ISO 8601 date string.
        team_count: Number of teams.
        tags: Sequence of FakeApiTag objects.
        description: Short description.
        url: Full Kaggle URL.
    """

    __slots__ = (
        "category",
        "deadline",
        "description",
        "ref",
        "reward",
        "tags",
        "team_count",
        "title",
        "url",
    )

    def __init__(
        self,
        *,
        ref: str,
        title: str,
        category: str,
        reward: str,
        deadline: str,
        team_count: int,
        tags: Sequence[FakeApiTag],
        description: str,
        url: str,
    ) -> None:
        """Initialize fake competition.

        Args:
            ref: Competition reference slug.
            title: Competition title.
            category: Competition category string.
            reward: Prize description.
            deadline: Deadline as ISO 8601 date string.
            team_count: Number of teams.
            tags: Sequence of FakeApiTag objects.
            description: Short description.
            url: Full Kaggle URL.
        """
        self.ref = ref
        self.title = title
        self.category = category
        self.reward = reward
        self.deadline = deadline
        self.team_count = team_count
        self.tags = tags
        self.description = description
        self.url = url


class FakeCompetitionsResponse:
    """Fake response wrapper for competitions_list (matches new Kaggle API format)."""

    __slots__ = ("_competitions",)

    def __init__(self, competitions: Sequence[KaggleCompetitionProtocol]) -> None:
        """Initialize fake response.

        Args:
            competitions: Sequence of competition objects.
        """
        self._competitions = competitions

    @property
    def competitions(self) -> Sequence[KaggleCompetitionProtocol]:
        """Get competitions sequence."""
        return self._competitions


class FakeKaggleApi:
    """Fake Kaggle API for testing.

    Attributes:
        _competitions: Configured competitions to return.
        _list_calls: Record of calls to competitions_list.
        _authenticated: Whether authenticate() was called.
    """

    __slots__ = ("_authenticated", "_competitions", "_list_calls")

    def __init__(
        self,
        competitions: Sequence[KaggleCompetitionProtocol] = (),
    ) -> None:
        """Initialize fake API.

        Args:
            competitions: Competitions to return from competitions_list.
        """
        self._competitions: list[KaggleCompetitionProtocol] = list(competitions)
        self._list_calls: list[dict[str, str]] = []
        self._authenticated = False

    def authenticate(self) -> None:
        """Mark API as authenticated."""
        self._authenticated = True

    def competitions_list(
        self,
        search: str = "",
        category: str = "",
    ) -> CompetitionsResponseProtocol:
        """Return configured competitions, optionally filtered.

        Args:
            search: Optional search query.
            category: Optional category filter.

        Returns:
            Response wrapper with competitions list (matches new Kaggle API format).
        """
        self._list_calls.append({"search": search, "category": category})
        result: list[KaggleCompetitionProtocol] = list(self._competitions)
        if search:
            search_lower = search.lower()
            result = [
                c
                for c in result
                if search_lower in c.title.lower() or search_lower in c.ref.lower()
            ]
        if category:
            result = [c for c in result if c.category.lower() == category.lower()]

        return FakeCompetitionsResponse(result)


# -----------------------------------------------------------------------------
# Fake Module Implementation
# -----------------------------------------------------------------------------


class _FakeKaggleApiClass:
    """Fake KaggleApi class that creates FakeKaggleApi instances.

    This class is used as the KaggleApi attribute of FakeKaggleModule.
    When called (instantiated), it returns the configured FakeKaggleApi.
    """

    __slots__ = ("_api",)

    def __init__(self, api: FakeKaggleApi) -> None:
        """Initialize with the API instance to return.

        Args:
            api: The FakeKaggleApi instance to return when called.
        """
        self._api = api

    def __call__(self) -> KaggleApiProtocol:
        """Create and return the fake API instance.

        Returns:
            The configured FakeKaggleApi instance.
        """
        return self._api


class FakeKaggleModule:
    """Fake kaggle module for testing.

    This class mimics the kaggle.api.kaggle_api_extended module structure.
    It has a KaggleApi attribute that is a class-like callable.

    Attributes:
        KaggleApi: Callable that returns FakeKaggleApi instances.
    """

    __slots__ = ("KaggleApi",)

    KaggleApi: KaggleApiClassProtocol

    def __init__(self, api: FakeKaggleApi) -> None:
        """Initialize fake module.

        Args:
            api: The FakeKaggleApi instance to return from KaggleApi().
        """
        self.KaggleApi = _FakeKaggleApiClass(api)


# -----------------------------------------------------------------------------
# Fake Client Implementation
# -----------------------------------------------------------------------------


class FakeKaggleClient:
    """Fake Kaggle client for testing.

    Attributes:
        _competitions: Configured competitions to return.
        _list_calls: Record of calls to list_competitions.
        _get_calls: Record of calls to get_competition.
    """

    __slots__ = ("_competitions", "_get_calls", "_list_calls")

    def __init__(self, competitions: tuple[Competition, ...] = ()) -> None:
        """Initialize fake client.

        Args:
            competitions: Competitions to return from list_competitions.
        """
        self._competitions = competitions
        self._list_calls: list[dict[str, str | CompetitionCategory | None]] = []
        self._get_calls: list[str] = []

    def list_competitions(
        self,
        *,
        search: str | None = None,
        category: CompetitionCategory | None = None,
    ) -> tuple[Competition, ...]:
        """Return configured competitions, optionally filtered.

        Args:
            search: Optional search query (filters by title).
            category: Optional category filter.

        Returns:
            Tuple of matching competitions.
        """
        self._list_calls.append({"search": search, "category": category})
        result = self._competitions
        if search:
            search_lower = search.lower()
            result = tuple(c for c in result if search_lower in c.title.lower())
        if category:
            result = tuple(c for c in result if c.category == category)
        return result

    def get_competition(self, ref: str) -> Competition | None:
        """Get competition by ref.

        Args:
            ref: Competition reference slug.

        Returns:
            Competition if found, None otherwise.
        """
        self._get_calls.append(ref)
        for c in self._competitions:
            if c.ref == ref:
                return c
        return None


# -----------------------------------------------------------------------------
# Factory Functions for Tests
# -----------------------------------------------------------------------------


def make_fake_competition(
    *,
    ref: str = "test-competition",
    title: str = "Test Competition",
    category: CompetitionCategory = "Playground",
    reward: str = "Knowledge",
    deadline: str = "2025-12-31",
    team_count: int = 100,
    tags: tuple[str, ...] = ("tabular",),
    description: str = "Test description",
) -> Competition:
    """Factory for creating test Competition instances.

    Args:
        ref: Competition reference slug.
        title: Competition title.
        category: Competition category.
        reward: Prize description.
        deadline: Deadline as ISO 8601 date string.
        team_count: Number of teams.
        tags: Tuple of tags.
        description: Short description.

    Returns:
        Competition instance.
    """
    return Competition(
        ref=ref,
        title=title,
        category=category,
        reward=reward,
        deadline=deadline,
        team_count=team_count,
        tags=tags,
        description=description,
        url=f"https://www.kaggle.com/competitions/{ref}",
    )


def make_fake_kaggle_competition(
    *,
    ref: str = "test-competition",
    title: str = "Test Competition",
    category: str = "Playground",
    reward: str = "Knowledge",
    deadline: str = "2025-12-31",
    team_count: int = 100,
    tags: tuple[str, ...] = ("tabular",),
    description: str = "Test description",
) -> FakeKaggleCompetition:
    """Factory for creating test FakeKaggleCompetition instances.

    Matches Kaggle API 1.8.3 format where ref is a full URL.

    Args:
        ref: Competition reference slug (converted to full URL internally).
        title: Competition title.
        category: Competition category string.
        reward: Prize description.
        deadline: Deadline as ISO 8601 date string.
        team_count: Number of teams.
        tags: Tuple of tag strings (converted to FakeApiTag objects).
        description: Short description.

    Returns:
        FakeKaggleCompetition instance with ref as full URL.
    """
    url = f"https://www.kaggle.com/competitions/{ref}"
    return FakeKaggleCompetition(
        ref=url,  # Kaggle API 1.8.3 returns full URL in ref field
        title=title,
        category=category,
        reward=reward,
        deadline=deadline,
        team_count=team_count,
        tags=[FakeApiTag(t) for t in tags],
        description=description,
        url=url,
    )


def make_fake_capability(
    *,
    name: str = "test_capability",
    strength: CapabilityStrength = "moderate",
    tags: tuple[str, ...] = ("test",),
    description: str = "Test capability",
) -> CodebaseCapability:
    """Factory for creating test CodebaseCapability instances.

    Args:
        name: Capability identifier.
        strength: Capability strength level.
        tags: Tuple of tags.
        description: Human-readable description.

    Returns:
        CodebaseCapability instance.
    """
    return CodebaseCapability(
        name=name,
        strength=strength,
        tags=tags,
        description=description,
    )


def make_fake_profile(
    *,
    capabilities: tuple[CodebaseCapability, ...] = (),
    ml_backends: tuple[str, ...] = ("xgboost",),
    data_formats: tuple[str, ...] = ("csv",),
    task_types: tuple[str, ...] = ("binary_classification",),
) -> CodebaseProfile:
    """Factory for creating test CodebaseProfile instances.

    Args:
        capabilities: Tuple of capabilities.
        ml_backends: Tuple of ML backend names.
        data_formats: Tuple of data format names.
        task_types: Tuple of task type names.

    Returns:
        CodebaseProfile instance.
    """
    return CodebaseProfile(
        capabilities=capabilities,
        ml_backends=ml_backends,
        data_formats=data_formats,
        task_types=task_types,
    )


__all__ = [
    "FakeApiTag",
    "FakeCompetitionsResponse",
    "FakeKaggleApi",
    "FakeKaggleClient",
    "FakeKaggleCompetition",
    "FakeKaggleModule",
    "HooksContainer",
    "KaggleApiFactoryProtocol",
    "KaggleApiProtocol",
    "KaggleClientHook",
    "KaggleModuleHook",
    "ProfileScannerHook",
    "_FakeKaggleApiClass",
    "hooks",
    "make_fake_capability",
    "make_fake_competition",
    "make_fake_kaggle_competition",
    "make_fake_profile",
    "reset_hooks",
]
