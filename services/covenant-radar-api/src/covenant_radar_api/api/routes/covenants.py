"""CRUD endpoints for Covenant resources."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from covenant_domain import Covenant, CovenantId, DealId, encode_covenant
from covenant_persistence import CovenantRepository
from fastapi import APIRouter, Request, Response, status
from platform_core.json_utils import JSONValue, dump_json_str

from ..decode import parse_covenant_request


class ContainerProtocol(Protocol):
    """Protocol for service container with covenant_repo method."""

    def covenant_repo(self) -> CovenantRepository: ...


def build_router(get_container: ContainerProtocol) -> APIRouter:
    """Build FastAPI router for covenant CRUD operations.

    Args:
        get_container: Container instance with covenant_repo method.

    Returns:
        Configured router with covenant endpoints.
    """
    router = APIRouter(prefix="/covenants", tags=["covenants"])

    def _list_covenants_for_deal(deal_id: str) -> Response:
        """List all covenants attached to a specific deal.

        Returns array of Covenant objects with id, deal_id, name, formula,
        threshold_value_scaled, threshold_direction, and frequency.
        """
        repo = get_container.covenant_repo()
        id_obj = DealId(value=deal_id)
        covenants: Sequence[Covenant] = repo.list_for_deal(id_obj)
        body: list[dict[str, JSONValue]] = [encode_covenant(c) for c in covenants]
        return Response(
            content=dump_json_str(body),
            media_type="application/json",
        )

    async def _create_covenant(request: Request) -> Response:
        """Create a new covenant rule for a deal.

        Request body requires: id, deal_id, name, formula (e.g. "total_debt / ebitda"),
        threshold_value_scaled, threshold_direction ("<=" or ">="), frequency.
        Returns 201 with the created Covenant object.
        """
        body_bytes = await request.body()
        covenant = parse_covenant_request(body_bytes)
        repo = get_container.covenant_repo()
        repo.create(covenant)
        return Response(
            content=dump_json_str(encode_covenant(covenant)),
            media_type="application/json",
            status_code=201,
        )

    def _get_covenant(covenant_id: str) -> Response:
        """Get a covenant by its UUID.

        Returns the Covenant object or 404 if not found.
        """
        repo = get_container.covenant_repo()
        id_obj = CovenantId(value=covenant_id)
        covenant = repo.get(id_obj)
        return Response(
            content=dump_json_str(encode_covenant(covenant)),
            media_type="application/json",
        )

    def _delete_covenant(covenant_id: str) -> Response:
        """Delete a covenant by UUID.

        Returns 204 No Content on success.
        """
        repo = get_container.covenant_repo()
        id_obj = CovenantId(value=covenant_id)
        repo.delete(id_obj)
        return Response(status_code=204)

    router.add_api_route(
        "/by-deal/{deal_id}", _list_covenants_for_deal, methods=["GET"], response_model=None
    )
    router.add_api_route(
        "",
        _create_covenant,
        methods=["POST"],
        status_code=status.HTTP_201_CREATED,
        response_model=None,
    )
    router.add_api_route("/{covenant_id}", _get_covenant, methods=["GET"], response_model=None)
    router.add_api_route(
        "/{covenant_id}", _delete_covenant, methods=["DELETE"], response_model=None
    )

    return router


__all__ = ["build_router"]
