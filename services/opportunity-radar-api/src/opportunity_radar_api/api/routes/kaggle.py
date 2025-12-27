"""Kaggle competition routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from platform_core.json_utils import JSONObject
from platform_kaggle import (
    encode_competition,
    encode_match,
    filter_competitions,
    make_interest_filter,
    match_competitions,
)

from opportunity_radar_api.api.container import ServiceContainer

# Query parameter defaults as module-level constants
_TAGS_QUERY: list[str] = []
_EXCLUDE_QUERY: list[str] = []


def build_router(container: ServiceContainer) -> APIRouter:
    """Build Kaggle router.

    Args:
        container: Service container with dependencies.

    Returns:
        Configured APIRouter with Kaggle endpoints.
    """
    router = APIRouter(prefix="/kaggle", tags=["kaggle"])

    def _list_competitions(
        tags: Annotated[list[str], Query()] = _TAGS_QUERY,
        exclude: Annotated[list[str], Query()] = _EXCLUDE_QUERY,
        min_score: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
        match_codebase: bool = True,
    ) -> list[JSONObject]:
        """Find Kaggle competitions matching criteria.

        Args:
            tags: Tags to include (must have at least one).
            exclude: Tags to exclude (must not have any).
            min_score: Minimum match score (0.0-1.0).
            match_codebase: Whether to score against codebase capabilities.

        Returns:
            List of competition matches as JSON.
        """
        client = container.get_kaggle_client()
        competitions = client.list_competitions()

        # Apply interest filter if tags provided
        if tags:
            interests = make_interest_filter(
                include_tags=tuple(tags),
                exclude_tags=tuple(exclude),
                min_reward=None,
                categories=None,
            )
            competitions = filter_competitions(competitions, interests)

        # Match against codebase capabilities
        if match_codebase:
            profile = container.get_codebase_profile()
            matches = match_competitions(competitions, profile)
            # Filter by minimum score
            matches = tuple(m for m in matches if m.match_score >= min_score)
            return [encode_match(m) for m in matches]

        # Return unscored competitions
        return [encode_competition(c) for c in competitions]

    def _get_competition(ref: str) -> JSONObject:
        """Get a specific competition by reference.

        Args:
            ref: Competition reference slug.

        Returns:
            Competition as JSON.

        Raises:
            HTTPException: If competition not found.
        """
        client = container.get_kaggle_client()
        comp = client.get_competition(ref)
        if comp is None:
            raise HTTPException(status_code=404, detail=f"Competition {ref} not found")
        return encode_competition(comp)

    router.add_api_route("/competitions", _list_competitions, methods=["GET"], response_model=None)
    router.add_api_route(
        "/competitions/{ref}", _get_competition, methods=["GET"], response_model=None
    )
    return router
