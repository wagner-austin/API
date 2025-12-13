"""CRUD endpoints for Deal resources."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from covenant_domain import Deal, DealId, encode_deal
from covenant_persistence import DealRepository
from fastapi import APIRouter, Request, Response, status
from platform_core.json_utils import JSONValue, dump_json_str

from ..decode import parse_deal_request, parse_update_deal_request

# OpenAPI response schemas (no type annotation for FastAPI compatibility)
_DEAL_EXAMPLE: dict[str, JSONValue] = {
    "id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
    "name": "TechCorp Senior Credit Facility",
    "borrower": "TechCorp Inc",
    "sector": "Technology",
    "region": "North America",
    "commitment_amount_cents": 50000000000,
    "currency": "USD",
    "maturity_date_iso": "2027-12-31",
}

_DEAL_EXAMPLE_ARRAY: list[JSONValue] = [_DEAL_EXAMPLE]
_DEAL_EXAMPLE_WRAPPED: dict[str, JSONValue] = {"example": _DEAL_EXAMPLE}

_LIST_DEALS_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    200: {
        "description": "Array of Deal objects",
        "content": {"application/json": {"example": _DEAL_EXAMPLE_ARRAY}},
    },
}

_CREATE_DEAL_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    201: {
        "description": "Created Deal object",
        "content": {"application/json": _DEAL_EXAMPLE_WRAPPED},
    },
}

_GET_DEAL_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    200: {
        "description": "Deal object",
        "content": {"application/json": _DEAL_EXAMPLE_WRAPPED},
    },
    404: {"description": "Deal not found"},
}

_UPDATE_DEAL_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    200: {
        "description": "Updated Deal object",
        "content": {"application/json": _DEAL_EXAMPLE_WRAPPED},
    },
    404: {"description": "Deal not found"},
}

_DELETE_DEAL_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    204: {"description": "Deal deleted successfully"},
}


class ContainerProtocol(Protocol):
    """Protocol for service container with deal_repo method."""

    def deal_repo(self) -> DealRepository: ...


def build_router(get_container: ContainerProtocol) -> APIRouter:
    """Build FastAPI router for deal CRUD operations.

    Args:
        get_container: Container instance with deal_repo method.

    Returns:
        Configured router with deal endpoints.
    """
    router = APIRouter(prefix="/deals", tags=["deals"])

    def _list_deals() -> Response:
        """List all deals in the system.

        Returns array of Deal objects with id, name, borrower, sector, region,
        commitment_amount_cents, currency, and maturity_date_iso.
        """
        repo = get_container.deal_repo()
        deals: Sequence[Deal] = repo.list_all()
        body: list[dict[str, JSONValue]] = [encode_deal(d) for d in deals]
        return Response(
            content=dump_json_str(body),
            media_type="application/json",
        )

    async def _create_deal(request: Request) -> Response:
        """Create a new loan deal.

        Request body requires: id, name, borrower, sector, region,
        commitment_amount_cents, currency, maturity_date_iso.
        Returns 201 with the created Deal object.
        """
        body_bytes = await request.body()
        deal = parse_deal_request(body_bytes)
        repo = get_container.deal_repo()
        repo.create(deal)
        return Response(
            content=dump_json_str(encode_deal(deal)),
            media_type="application/json",
            status_code=201,
        )

    def _get_deal(deal_id: str) -> Response:
        """Get a deal by its UUID.

        Returns the Deal object or 404 if not found.
        """
        repo = get_container.deal_repo()
        id_obj = DealId(value=deal_id)
        deal = repo.get(id_obj)
        return Response(
            content=dump_json_str(encode_deal(deal)),
            media_type="application/json",
        )

    async def _update_deal(deal_id: str, request: Request) -> Response:
        """Update an existing deal by UUID.

        Request body requires all deal fields (full replacement).
        Returns the updated Deal object or 404 if not found.
        """
        body_bytes = await request.body()
        id_obj = DealId(value=deal_id)
        deal = parse_update_deal_request(body_bytes, id_obj)
        repo = get_container.deal_repo()
        repo.update(deal)
        return Response(
            content=dump_json_str(encode_deal(deal)),
            media_type="application/json",
        )

    def _delete_deal(deal_id: str) -> Response:
        """Delete a deal by UUID.

        Returns 204 No Content on success. Also deletes associated covenants and measurements.
        """
        repo = get_container.deal_repo()
        id_obj = DealId(value=deal_id)
        repo.delete(id_obj)
        return Response(status_code=204)

    router.add_api_route(
        "",
        _list_deals,
        methods=["GET"],
        response_model=None,
        summary="List all deals",
        description="List all loan deals in the system with full metadata.",
        response_description="Array of Deal objects",
        responses=_LIST_DEALS_RESPONSES,
    )
    router.add_api_route(
        "",
        _create_deal,
        methods=["POST"],
        status_code=status.HTTP_201_CREATED,
        response_model=None,
        summary="Create a deal",
        description=(
            "Create a new loan deal with borrower, sector, region, "
            "commitment amount, currency, and maturity date."
        ),
        response_description="Created Deal object",
        responses=_CREATE_DEAL_RESPONSES,
    )
    router.add_api_route(
        "/{deal_id}",
        _get_deal,
        methods=["GET"],
        response_model=None,
        summary="Get a deal",
        description="Get a specific deal by its UUID.",
        response_description="Deal object",
        responses=_GET_DEAL_RESPONSES,
    )
    router.add_api_route(
        "/{deal_id}",
        _update_deal,
        methods=["PUT"],
        response_model=None,
        summary="Update a deal",
        description="Update an existing deal by UUID. Requires all fields (full replacement).",
        response_description="Updated Deal object",
        responses=_UPDATE_DEAL_RESPONSES,
    )
    router.add_api_route(
        "/{deal_id}",
        _delete_deal,
        methods=["DELETE"],
        response_model=None,
        status_code=204,
        summary="Delete a deal",
        description="Delete a deal by UUID. Also removes associated covenants and measurements.",
        response_description="No content",
        responses=_DELETE_DEAL_RESPONSES,
    )

    return router


__all__ = ["build_router"]
