"""Background job for batch covenant evaluation."""

from __future__ import annotations

from typing import Protocol

from covenant_domain import DealId, encode_covenant_result, evaluate_all_covenants_for_period
from covenant_persistence import (
    CovenantRepository,
    CovenantResultRepository,
    MeasurementRepository,
)
from platform_core.json_utils import JSONTypeError, JSONValue


class RepositoryProvider(Protocol):
    """Protocol for providing repositories to the job."""

    def covenant_repo(self) -> CovenantRepository: ...

    def measurement_repo(self) -> MeasurementRepository: ...

    def covenant_result_repo(self) -> CovenantResultRepository: ...


def run_batch_evaluation(
    deal_ids_json: str,
    period_start_iso: str,
    period_end_iso: str,
    tolerance_ratio_scaled: int,
    repo_provider: RepositoryProvider,
) -> dict[str, JSONValue]:
    """Run batch evaluation for multiple deals.

    This function is designed to be called from an RQ worker.

    Args:
        deal_ids_json: JSON array of deal ID strings
        period_start_iso: ISO 8601 period start date
        period_end_iso: ISO 8601 period end date
        tolerance_ratio_scaled: Tolerance ratio as scaled int
        repo_provider: Provider for repository instances

    Returns:
        Job result with evaluated deal count and total result count.

    Raises:
        KeyError: Deal not found or metric missing
        FormulaParseError: Invalid formula in covenant
        FormulaEvalError: Division by zero during evaluation
        ValueError: Duplicate metrics
    """
    from platform_core.json_utils import load_json_str, narrow_json_to_list

    raw_ids = narrow_json_to_list(load_json_str(deal_ids_json))
    deal_id_strs: list[str] = []
    for raw_id in raw_ids:
        if not isinstance(raw_id, str):
            raise JSONTypeError("Each deal_id must be a string")
        deal_id_strs.append(raw_id)

    covenant_repo = repo_provider.covenant_repo()
    measurement_repo = repo_provider.measurement_repo()
    result_repo = repo_provider.covenant_result_repo()

    total_results_count = 0
    all_results: list[dict[str, JSONValue]] = []

    for deal_id_str in deal_id_strs:
        deal_id = DealId(value=deal_id_str)

        covenants = covenant_repo.list_for_deal(deal_id)
        measurements = measurement_repo.list_for_deal_and_period(
            deal_id, period_start_iso, period_end_iso
        )

        results = evaluate_all_covenants_for_period(
            covenants=list(covenants),
            period_start_iso=period_start_iso,
            period_end_iso=period_end_iso,
            measurements=list(measurements),
            tolerance_ratio_scaled=tolerance_ratio_scaled,
        )

        result_repo.save_many(results)
        total_results_count += len(results)

        for r in results:
            all_results.append(encode_covenant_result(r))

    results_as_json: list[JSONValue] = list(all_results)
    return {
        "status": "complete",
        "deals_evaluated": len(deal_id_strs),
        "results_count": total_results_count,
        "results": results_as_json,
    }


__all__ = ["RepositoryProvider", "run_batch_evaluation"]
