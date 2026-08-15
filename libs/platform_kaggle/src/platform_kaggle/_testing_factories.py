"""testing: make_fake_competition and related definitions."""

from __future__ import annotations

from datetime import datetime

from platform_kaggle._testing_fakes import FakeApiTag, FakeKaggleCompetition
from platform_kaggle._testing_hooks import _FAR_FUTURE_DEADLINE, _FAR_FUTURE_DEADLINE_DT
from platform_kaggle.types import (
    CapabilityStrength,
    CodebaseCapability,
    CodebaseProfile,
    Competition,
    CompetitionCategory,
    CompetitionPage,
    CompetitionPages,
)

# -----------------------------------------------------------------------------
# Factory Functions for Tests
# -----------------------------------------------------------------------------


def make_fake_competition(
    *,
    ref: str = "test-competition",
    title: str = "Test Competition",
    category: CompetitionCategory = "Playground",
    reward: str = "Knowledge",
    deadline: str = _FAR_FUTURE_DEADLINE,
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
    deadline: datetime | None = None,
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
        deadline: Deadline as datetime (defaults to the far-future date).
        team_count: Number of teams.
        tags: Tuple of tag strings (converted to FakeApiTag objects).
        description: Short description.

    Returns:
        FakeKaggleCompetition instance with ref as full URL.
    """
    url = f"https://www.kaggle.com/competitions/{ref}"
    if deadline is None:
        deadline = _FAR_FUTURE_DEADLINE_DT
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


def make_fake_competition_page(
    *,
    id: int = 1,
    name: str = "Description",
    content: str = "Test content",
) -> CompetitionPage:
    """Factory for creating test CompetitionPage instances.

    Args:
        id: Page ID.
        name: Page name (e.g., "Description", "Evaluation").
        content: Markdown content.

    Returns:
        CompetitionPage instance.
    """
    return CompetitionPage(
        id=id,
        name=name,
        content=content,
    )


def make_fake_competition_pages(
    *,
    competition_id: int = 12345,
    pages: tuple[CompetitionPage, ...] | None = None,
    description: str = "Test description",
    evaluation: str = "Test evaluation",
    timeline: str = "Test timeline",
    rules: str = "Test rules",
) -> CompetitionPages:
    """Factory for creating test CompetitionPages instances.

    Args:
        competition_id: Numeric competition ID.
        pages: Tuple of pages. If None, creates default pages from content args.
        description: Description page content.
        evaluation: Evaluation page content.
        timeline: Timeline page content.
        rules: Rules page content.

    Returns:
        CompetitionPages instance.
    """
    if pages is None:
        pages = (
            CompetitionPage(id=1, name="Description", content=description),
            CompetitionPage(id=2, name="Evaluation", content=evaluation),
            CompetitionPage(id=3, name="Timeline", content=timeline),
            CompetitionPage(id=4, name="Rules", content=rules),
        )
    return CompetitionPages(
        competition_id=competition_id,
        pages=pages,
        description=description,
        evaluation=evaluation,
        timeline=timeline,
        rules=rules,
    )
