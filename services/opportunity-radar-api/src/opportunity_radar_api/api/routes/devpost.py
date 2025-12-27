"""Devpost hackathon routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from platform_core.json_utils import JSONObject
from platform_devpost import (
    HackathonState,
    encode_hackathon,
    encode_match,
    filter_hackathons,
    make_interest_filter,
    match_hackathons,
)

from opportunity_radar_api.api.container import ServiceContainer

# Query parameter defaults as module-level constants
_THEMES_QUERY: list[str] = []
_EXCLUDE_QUERY: list[str] = []
_STATES_QUERY: list[str] = ["open"]


def _parse_states(states: list[str]) -> tuple[HackathonState, ...]:
    """Parse state strings to HackathonState tuple.

    Args:
        states: List of state strings.

    Returns:
        Tuple of valid HackathonState values.
    """
    result: list[HackathonState] = []
    for s in states:
        if s == "open":
            result.append("open")
        elif s == "upcoming":
            result.append("upcoming")
        elif s == "ended":
            result.append("ended")
        elif s == "submissions":
            result.append("submissions")
    return tuple(result)


def build_router(container: ServiceContainer) -> APIRouter:
    """Build Devpost router.

    Args:
        container: Service container with dependencies.

    Returns:
        Configured APIRouter with Devpost endpoints.
    """
    router = APIRouter(prefix="/devpost", tags=["devpost"])

    def _list_hackathons(
        themes: Annotated[list[str], Query()] = _THEMES_QUERY,
        exclude: Annotated[list[str], Query()] = _EXCLUDE_QUERY,
        states: Annotated[list[str], Query()] = _STATES_QUERY,
        min_score: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
        match_codebase: bool = True,
        featured_only: bool = False,
    ) -> list[JSONObject]:
        """Find Devpost hackathons matching criteria.

        Args:
            themes: Theme names to include (must have at least one).
            exclude: Theme names to exclude (must not have any).
            states: Allowed states (open, upcoming, ended, submissions).
            min_score: Minimum match score (0.0-1.0).
            match_codebase: Whether to score against codebase capabilities.
            featured_only: Only return featured hackathons.

        Returns:
            List of hackathon matches as JSON.
        """
        client = container.get_devpost_client()
        hackathons = client.list_hackathons()

        # Convert state strings to HackathonState
        valid_states = _parse_states(states)

        # Apply interest filter
        interests = make_interest_filter(
            include_themes=tuple(themes),
            exclude_themes=tuple(exclude),
            states=valid_states if valid_states else None,
            featured_only=featured_only,
        )
        hackathons = filter_hackathons(hackathons, interests)

        # Match against codebase capabilities
        if match_codebase:
            profile = container.get_codebase_profile()
            matches = match_hackathons(hackathons, profile)
            # Filter by minimum score
            matches = tuple(m for m in matches if m.match_score >= min_score)
            return [encode_match(m) for m in matches]

        # Return unscored hackathons
        return [encode_hackathon(h) for h in hackathons]

    def _get_hackathon(hackathon_id: int) -> JSONObject:
        """Get a specific hackathon by ID.

        Args:
            hackathon_id: Hackathon identifier.

        Returns:
            Hackathon as JSON.

        Raises:
            HTTPException: If hackathon not found.
        """
        client = container.get_devpost_client()
        hackathon = client.get_hackathon(hackathon_id)
        if hackathon is None:
            raise HTTPException(status_code=404, detail=f"Hackathon {hackathon_id} not found")
        return encode_hackathon(hackathon)

    router.add_api_route("/hackathons", _list_hackathons, methods=["GET"], response_model=None)
    router.add_api_route(
        "/hackathons/{hackathon_id}", _get_hackathon, methods=["GET"], response_model=None
    )
    return router
