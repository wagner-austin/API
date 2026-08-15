"""types: CompetitionPage and related definitions."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    require_int,
    require_list,
    require_str,
)

from platform_kaggle._types_validation import _require_dict_value

# -----------------------------------------------------------------------------
# CompetitionPage
# -----------------------------------------------------------------------------


class CompetitionPage:
    """A single page of competition content from Kaggle's internal API.

    Attributes:
        id: Numeric page ID.
        name: Page name (e.g., "Description", "Evaluation", "Timeline").
        content: Markdown content of the page.
    """

    __slots__ = ("content", "id", "name")

    def __init__(
        self,
        *,
        id: int,
        name: str,
        content: str,
    ) -> None:
        """Initialize competition page.

        Args:
            id: Numeric page ID.
            name: Page name.
            content: Markdown content.
        """
        self.id = id
        self.name = name
        self.content = content


def encode_competition_page(page: CompetitionPage) -> JSONObject:
    """Encode CompetitionPage to JSON-serializable dict.

    Args:
        page: CompetitionPage to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "id": page.id,
        "name": page.name,
        "content": page.content,
    }
    return result


def decode_competition_page(data: JSONObject) -> CompetitionPage:
    """Decode CompetitionPage from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CompetitionPage.

    Raises:
        JSONTypeError: If validation fails.
    """
    return CompetitionPage(
        id=require_int(data, "id"),
        name=require_str(data, "name"),
        content=require_str(data, "content"),
    )


# -----------------------------------------------------------------------------
# CompetitionPages
# -----------------------------------------------------------------------------


class CompetitionPages:
    """Collection of competition pages with convenient accessors.

    Provides quick access to common pages (description, evaluation, etc.)
    while also exposing the full list of pages.

    Attributes:
        competition_id: Numeric Kaggle competition ID.
        pages: Tuple of all pages.
        description: Content of the Description page (empty if not found).
        evaluation: Content of the Evaluation page (empty if not found).
        timeline: Content of the Timeline page (empty if not found).
        rules: Content of the Rules page (empty if not found).
    """

    __slots__ = (
        "competition_id",
        "description",
        "evaluation",
        "pages",
        "rules",
        "timeline",
    )

    def __init__(
        self,
        *,
        competition_id: int,
        pages: tuple[CompetitionPage, ...],
        description: str,
        evaluation: str,
        timeline: str,
        rules: str,
    ) -> None:
        """Initialize competition pages collection.

        Args:
            competition_id: Numeric Kaggle competition ID.
            pages: Tuple of all pages.
            description: Content of the Description page.
            evaluation: Content of the Evaluation page.
            timeline: Content of the Timeline page.
            rules: Content of the Rules page.
        """
        self.competition_id = competition_id
        self.pages = pages
        self.description = description
        self.evaluation = evaluation
        self.timeline = timeline
        self.rules = rules


def encode_competition_pages(pages: CompetitionPages) -> JSONObject:
    """Encode CompetitionPages to JSON-serializable dict.

    Args:
        pages: CompetitionPages to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "competition_id": pages.competition_id,
        "pages": [encode_competition_page(p) for p in pages.pages],
        "description": pages.description,
        "evaluation": pages.evaluation,
        "timeline": pages.timeline,
        "rules": pages.rules,
    }
    return result


def decode_competition_pages(data: JSONObject) -> CompetitionPages:
    """Decode CompetitionPages from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CompetitionPages.

    Raises:
        JSONTypeError: If validation fails.
    """
    pages_raw = require_list(data, "pages")
    pages: list[CompetitionPage] = []
    for i, page_data in enumerate(pages_raw):
        pages.append(decode_competition_page(_require_dict_value(page_data, f"pages[{i}]")))

    return CompetitionPages(
        competition_id=require_int(data, "competition_id"),
        pages=tuple(pages),
        description=require_str(data, "description"),
        evaluation=require_str(data, "evaluation"),
        timeline=require_str(data, "timeline"),
        rules=require_str(data, "rules"),
    )
