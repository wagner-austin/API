"""types: Competition and related definitions."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    require_int,
    require_str,
    require_str_list,
)

from platform_kaggle._types_validation import (
    CompetitionCategory,
    _require_category,
)

# -----------------------------------------------------------------------------
# Competition
# -----------------------------------------------------------------------------


class Competition:
    """Kaggle competition metadata.

    Attributes:
        ref: Competition reference slug (e.g., "amex-default-prediction").
        title: Competition title.
        category: Competition category.
        reward: Prize description (e.g., "$100,000" or "Knowledge").
        deadline: Deadline as ISO 8601 date string.
        team_count: Number of teams participating.
        tags: Tuple of competition tags.
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
        category: CompetitionCategory,
        reward: str,
        deadline: str,
        team_count: int,
        tags: tuple[str, ...],
        description: str,
        url: str,
    ) -> None:
        """Initialize competition.

        Args:
            ref: Competition reference slug.
            title: Competition title.
            category: Competition category.
            reward: Prize description.
            deadline: Deadline as ISO 8601 date string.
            team_count: Number of teams participating.
            tags: Tuple of competition tags.
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


def encode_competition(comp: Competition) -> JSONObject:
    """Encode Competition to JSON-serializable dict.

    Args:
        comp: Competition to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "ref": comp.ref,
        "title": comp.title,
        "category": comp.category,
        "reward": comp.reward,
        "deadline": comp.deadline,
        "team_count": comp.team_count,
        "tags": list(comp.tags),
        "description": comp.description,
        "url": comp.url,
    }
    return result


def decode_competition(data: JSONObject) -> Competition:
    """Decode Competition from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated Competition.

    Raises:
        JSONTypeError: If validation fails.
    """
    return Competition(
        ref=require_str(data, "ref"),
        title=require_str(data, "title"),
        category=_require_category(data, "category"),
        reward=require_str(data, "reward"),
        deadline=require_str(data, "deadline"),
        team_count=require_int(data, "team_count"),
        tags=tuple(require_str_list(data, "tags")),
        description=require_str(data, "description"),
        url=require_str(data, "url"),
    )
