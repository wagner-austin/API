"""Covenant evaluation endpoint."""

from __future__ import annotations

from typing import Protocol

from covenant_domain import (
    DealId,
    encode_covenant_result,
    evaluate_all_covenants_for_period,
)
from covenant_persistence import CovenantRepository, CovenantResultRepository, MeasurementRepository
from fastapi import APIRouter, Request, Response
from platform_core.json_utils import JSONValue, dump_json_str

from ..decode import parse_evaluate_request


class ContainerProtocol(Protocol):
    """Protocol for service container with evaluation repositories."""

    def covenant_repo(self) -> CovenantRepository: ...

    def measurement_repo(self) -> MeasurementRepository: ...

    def covenant_result_repo(self) -> CovenantResultRepository: ...


def build_router(get_container: ContainerProtocol) -> APIRouter:
    """Build FastAPI router for covenant evaluation.

    Args:
        get_container: Container instance with repository methods.

    Returns:
        Configured router with evaluation endpoint.
    """
    router = APIRouter(prefix="/evaluate", tags=["evaluate"])

    async def _evaluate(request: Request) -> Response:
        """Evaluate all covenants for a deal and period.

        Request body:
            deal_id: Deal UUID string
            period_start_iso: ISO 8601 period start date
            period_end_iso: ISO 8601 period end date
            tolerance_ratio_scaled: Tolerance ratio as scaled int (1_000_000 = 100%)

        Returns:
            JSON array of CovenantResult objects.

        Raises:
            KeyError: Required metric missing from measurements
            FormulaParseError: Invalid formula in covenant
            FormulaEvalError: Division by zero during evaluation
            ValueError: Duplicate metrics for same period
        """
        body_bytes = await request.body()
        req = parse_evaluate_request(body_bytes)

        deal_id = DealId(value=req["deal_id"])

        covenant_repo = get_container.covenant_repo()
        measurement_repo = get_container.measurement_repo()
        result_repo = get_container.covenant_result_repo()

        covenants = covenant_repo.list_for_deal(deal_id)
        measurements = measurement_repo.list_for_deal_and_period(
            deal_id,
            req["period_start_iso"],
            req["period_end_iso"],
        )

        results = evaluate_all_covenants_for_period(
            covenants=list(covenants),
            period_start_iso=req["period_start_iso"],
            period_end_iso=req["period_end_iso"],
            measurements=list(measurements),
            tolerance_ratio_scaled=req["tolerance_ratio_scaled"],
        )

        # Store results
        result_repo.save_many(results)

        # Encode and return
        body: list[dict[str, JSONValue]] = [encode_covenant_result(r) for r in results]
        return Response(
            content=dump_json_str(body),
            media_type="application/json",
        )

    router.add_api_route("", _evaluate, methods=["POST"], response_model=None)

    return router


__all__ = ["build_router"]
