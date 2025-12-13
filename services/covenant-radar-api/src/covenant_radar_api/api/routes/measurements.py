"""CRUD endpoints for Measurement resources."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from covenant_domain import DealId, Measurement, encode_measurement
from covenant_persistence import MeasurementRepository
from fastapi import APIRouter, Request, Response, status
from platform_core.json_utils import JSONValue, dump_json_str

from ..decode import parse_measurements_request


class ContainerProtocol(Protocol):
    """Protocol for service container with measurement_repo method."""

    def measurement_repo(self) -> MeasurementRepository: ...


def build_router(get_container: ContainerProtocol) -> APIRouter:
    """Build FastAPI router for measurement CRUD operations.

    Args:
        get_container: Container instance with measurement_repo method.

    Returns:
        Configured router with measurement endpoints.
    """
    router = APIRouter(prefix="/measurements", tags=["measurements"])

    def _list_measurements_for_deal(deal_id: str) -> Response:
        """List all financial measurements for a deal.

        Returns array of Measurement objects with deal_id, period_start_iso,
        period_end_iso, metric_name, and metric_value_scaled.
        """
        repo = get_container.measurement_repo()
        id_obj = DealId(value=deal_id)
        measurements: Sequence[Measurement] = repo.list_for_deal(id_obj)
        body: list[dict[str, JSONValue]] = [encode_measurement(m) for m in measurements]
        return Response(
            content=dump_json_str(body),
            media_type="application/json",
        )

    def _list_measurements_for_deal_and_period(
        deal_id: str,
        period_start: str,
        period_end: str,
    ) -> Response:
        """List measurements for a deal within a specific period.

        Query params: period_start (YYYY-MM-DD), period_end (YYYY-MM-DD).
        Returns measurements where the period matches exactly.
        """
        repo = get_container.measurement_repo()
        id_obj = DealId(value=deal_id)
        measurements: Sequence[Measurement] = repo.list_for_deal_and_period(
            id_obj, period_start, period_end
        )
        body: list[dict[str, JSONValue]] = [encode_measurement(m) for m in measurements]
        return Response(
            content=dump_json_str(body),
            media_type="application/json",
        )

    async def _add_measurements(request: Request) -> Response:
        """Add financial measurements for deals.

        Request body: {"measurements": [...]} with each measurement containing
        deal_id, period_start_iso, period_end_iso, metric_name, metric_value_scaled.
        Returns 201 with {"count": N} indicating measurements added.
        """
        body_bytes = await request.body()
        measurements = parse_measurements_request(body_bytes)
        repo = get_container.measurement_repo()
        count = repo.add_many(measurements)
        response_body: dict[str, JSONValue] = {"count": count}
        return Response(
            content=dump_json_str(response_body),
            media_type="application/json",
            status_code=201,
        )

    router.add_api_route(
        "/by-deal/{deal_id}", _list_measurements_for_deal, methods=["GET"], response_model=None
    )
    router.add_api_route(
        "/by-deal/{deal_id}/period",
        _list_measurements_for_deal_and_period,
        methods=["GET"],
        response_model=None,
    )
    router.add_api_route(
        "",
        _add_measurements,
        methods=["POST"],
        status_code=status.HTTP_201_CREATED,
        response_model=None,
    )

    return router


__all__ = ["build_router"]
