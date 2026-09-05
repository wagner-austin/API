"""Public test utilities for platform_devpost consumers.

This module provides hooks for dependency injection and fake implementations
for testing. Production code sets hooks at startup; tests set them to fakes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

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
    HackathonState,
    InterestFilter,
    Theme,
)


class DevpostApiFactoryProtocol(Protocol):
    """Protocol for Devpost API factory.

    This is a callable that returns a DevpostApiProtocol.
    """

    def __call__(self) -> DevpostApiProtocol:
        """Create Devpost API instance.

        Returns:
            DevpostApiProtocol instance.
        """
        ...


# -----------------------------------------------------------------------------
# Hooks Container
# -----------------------------------------------------------------------------


class HooksContainer:
    """Container for dependency injection hooks.

    Attributes:
        devpost_api_factory: Factory for low-level Devpost API.
        devpost_client: Factory for Devpost client.
        profile_scanner: Factory for codebase profile scanner.
    """

    devpost_api_factory: DevpostApiFactoryProtocol
    devpost_client: Callable[[], DevpostClientProtocol]
    profile_scanner: Callable[[Path], CodebaseProfile]

    def reset(self) -> None:
        """Restore every hook to its production implementation.

        The restoration `reset_hooks()` performs, exposed as a method so an
        autouse fixture can name the container it protects.
        """
        reset_hooks()


hooks = HooksContainer()


def _init_production_hooks() -> None:
    """Initialize hooks with production implementations."""
    from platform_devpost._production import create_devpost_api, make_devpost_client
    from platform_devpost.capabilities import scan_codebase

    hooks.devpost_api_factory = create_devpost_api
    hooks.devpost_client = make_devpost_client
    hooks.profile_scanner = scan_codebase


def reset_hooks() -> None:
    """Reset hooks to production implementations (for test teardown)."""
    _init_production_hooks()


# Initialize on module load
_init_production_hooks()


# -----------------------------------------------------------------------------
# Fake Devpost API Implementation
# -----------------------------------------------------------------------------


class FakeDevpostApi:
    """Fake Devpost API for testing.

    Attributes:
        _hackathons: Configured hackathons to return.
        _fetch_calls: Record of calls to fetch_hackathons.
    """

    __slots__ = ("_fetch_calls", "_hackathons")

    def __init__(
        self,
        hackathons: tuple[Hackathon, ...] = (),
    ) -> None:
        """Initialize fake API.

        Args:
            hackathons: Hackathons to return from fetch_hackathons.
        """
        self._hackathons = hackathons
        self._fetch_calls: list[dict[str, int | str | None]] = []

    def fetch_hackathons(
        self,
        *,
        page: int = 1,
        search: str | None = None,
    ) -> HackathonListResponse:
        """Return configured hackathons.

        Args:
            page: Page number (1-indexed).
            search: Optional search query.

        Returns:
            HackathonListResponse with configured hackathons.
        """
        self._fetch_calls.append({"page": page, "search": search})
        result = self._hackathons
        if search is not None:
            search_lower = search.lower()
            result = tuple(h for h in result if search_lower in h.title.lower())
        return HackathonListResponse(
            hackathons=result,
            meta=HackathonListMeta(
                total_count=len(result),
                per_page=10,
            ),
        )


# -----------------------------------------------------------------------------
# Fake Client Implementation
# -----------------------------------------------------------------------------


class FakeDevpostClient:
    """Fake Devpost client for testing.

    Attributes:
        _hackathons: Configured hackathons to return.
        _list_calls: Record of calls to list_hackathons.
        _get_calls: Record of calls to get_hackathon.
    """

    __slots__ = ("_get_calls", "_hackathons", "_list_calls")

    def __init__(self, hackathons: tuple[Hackathon, ...] = ()) -> None:
        """Initialize fake client.

        Args:
            hackathons: Hackathons to return from list_hackathons.
        """
        self._hackathons = hackathons
        self._list_calls: list[dict[str, str | HackathonState | None]] = []
        self._get_calls: list[int] = []

    def list_hackathons(
        self,
        *,
        search: str | None = None,
        state: HackathonState | None = None,
    ) -> tuple[Hackathon, ...]:
        """Return configured hackathons, optionally filtered.

        Args:
            search: Optional search query (filters by title).
            state: Optional state filter.

        Returns:
            Tuple of matching hackathons.
        """
        self._list_calls.append({"search": search, "state": state})
        result = self._hackathons
        if search is not None:
            search_lower = search.lower()
            result = tuple(h for h in result if search_lower in h.title.lower())
        if state is not None:
            result = tuple(h for h in result if h.open_state == state)
        return result

    def get_hackathon(self, hackathon_id: int) -> Hackathon | None:
        """Get hackathon by ID.

        Args:
            hackathon_id: Hackathon identifier.

        Returns:
            Hackathon if found, None otherwise.
        """
        self._get_calls.append(hackathon_id)
        for h in self._hackathons:
            if h.id == hackathon_id:
                return h
        return None


# -----------------------------------------------------------------------------
# Factory Functions for Tests
# -----------------------------------------------------------------------------


def make_fake_theme(
    *,
    id: int = 1,
    name: str = "Test Theme",
) -> Theme:
    """Factory for creating test Theme instances.

    Args:
        id: Theme identifier.
        name: Theme display name.

    Returns:
        Theme instance.
    """
    return Theme(id=id, name=name)


def make_fake_displayed_location(
    *,
    icon: str = "globe",
    location: str = "Online",
) -> DisplayedLocation:
    """Factory for creating test DisplayedLocation instances.

    Args:
        icon: Icon name.
        location: Location text.

    Returns:
        DisplayedLocation instance.
    """
    return DisplayedLocation(icon=icon, location=location)


def make_fake_hackathon(
    *,
    id: int = 1,
    title: str = "Test Hackathon",
    url: str = "https://test.devpost.com/",
    thumbnail_url: str = "https://example.com/thumb.jpg",
    organization_name: str = "Test Org",
    displayed_location: DisplayedLocation | None = None,
    open_state: HackathonState = "open",
    time_left_to_submission: str = "5 days left",
    submission_period_dates: str = "Jan 01 - Feb 01, 2025",
    themes: tuple[Theme, ...] = (),
    prize_amount: str = "$1,000",
    registrations_count: int = 100,
    featured: bool = False,
    winners_announced: bool = False,
    invite_only: bool = False,
) -> Hackathon:
    """Factory for creating test Hackathon instances.

    Args:
        id: Unique hackathon identifier.
        title: Hackathon title.
        url: Full Devpost URL.
        thumbnail_url: URL for thumbnail image.
        organization_name: Name of organizing organization.
        displayed_location: Location information (default: Online).
        open_state: Current state.
        time_left_to_submission: Human-readable time remaining.
        submission_period_dates: Date range string.
        themes: Tuple of hackathon themes.
        prize_amount: Prize amount string.
        registrations_count: Number of registrations.
        featured: Whether hackathon is featured.
        winners_announced: Whether winners have been announced.
        invite_only: Whether hackathon is invite-only.

    Returns:
        Hackathon instance.
    """
    if displayed_location is None:
        displayed_location = make_fake_displayed_location()

    return Hackathon(
        id=id,
        title=title,
        url=url,
        thumbnail_url=thumbnail_url,
        organization_name=organization_name,
        displayed_location=displayed_location,
        open_state=open_state,
        time_left_to_submission=time_left_to_submission,
        submission_period_dates=submission_period_dates,
        themes=themes,
        prize_amount=prize_amount,
        registrations_count=registrations_count,
        featured=featured,
        winners_announced=winners_announced,
        invite_only=invite_only,
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
    technologies: tuple[str, ...] = ("python",),
    frameworks: tuple[str, ...] = ("flask",),
) -> CodebaseProfile:
    """Factory for creating test CodebaseProfile instances.

    Args:
        capabilities: Tuple of capabilities.
        technologies: Tuple of technology names.
        frameworks: Tuple of framework names.

    Returns:
        CodebaseProfile instance.
    """
    return CodebaseProfile(
        capabilities=capabilities,
        technologies=technologies,
        frameworks=frameworks,
    )


def make_interest_filter(
    *,
    include_themes: tuple[str, ...] = (),
    exclude_themes: tuple[str, ...] = (),
    states: tuple[HackathonState, ...] | None = None,
    featured_only: bool = False,
) -> InterestFilter:
    """Factory for creating InterestFilter instances.

    Args:
        include_themes: Theme names to include.
        exclude_themes: Theme names to exclude.
        states: Allowed states (None = all).
        featured_only: If True, only return featured hackathons.

    Returns:
        InterestFilter instance.
    """
    return InterestFilter(
        include_themes=include_themes,
        exclude_themes=exclude_themes,
        states=states,
        featured_only=featured_only,
    )


__all__ = [
    "DevpostApiFactoryProtocol",
    "FakeDevpostApi",
    "FakeDevpostClient",
    "HooksContainer",
    "hooks",
    "make_fake_capability",
    "make_fake_displayed_location",
    "make_fake_hackathon",
    "make_fake_profile",
    "make_fake_theme",
    "make_interest_filter",
    "reset_hooks",
]
