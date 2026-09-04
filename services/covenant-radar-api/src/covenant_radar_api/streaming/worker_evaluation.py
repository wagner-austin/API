"""Streaming worker evaluation layer: data loading, features, prediction, alerting."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Literal

from covenant_domain import (
    Covenant,
    CovenantResult,
    Deal,
    DealId,
    Measurement,
    evaluate_all_covenants_for_period,
)
from covenant_domain.features import (
    LoanFeatures,
    classify_risk_tier,
    extract_features,
)
from covenant_ml.predictor import predict_probabilities
from platform_core.logging import get_logger

from covenant_radar_api.streaming.worker_buffers import _StreamingWorkerBuffers
from covenant_radar_api.streaming.worker_events import (
    _covenant_result_period_end_key,
)

from .schemas import (
    EvaluationStatus,
)

_log = get_logger(__name__)


class _StreamingWorkerEvaluation(_StreamingWorkerBuffers):
    """Evaluation layer of the streaming worker.

    Loads deal, covenant, result, and historical-metric data, builds ML
    features, runs covenant evaluation and risk prediction, and decides
    when a prediction warrants an alert.
    """

    def _load_deal(self, deal_id: str) -> Deal:
        """Load deal from repository.

        Args:
            deal_id: Deal identifier.

        Returns:
            Deal data.

        Raises:
            KeyError: If deal does not exist.
        """
        deal_id_typed = DealId(value=deal_id)
        return self._deal_repo.get(deal_id_typed)

    def _load_covenants(self, deal_id: str) -> tuple[Covenant, ...]:
        """Load covenants for a deal.

        Args:
            deal_id: Deal identifier.

        Returns:
            Tuple of covenants for the deal.
        """
        deal_id_typed = DealId(value=deal_id)
        covenants_seq = self._covenant_repo.list_for_deal(deal_id_typed)
        return tuple(covenants_seq)

    def _load_recent_results(
        self,
        deal_id: str,
        limit: int = 4,
    ) -> tuple[CovenantResult, ...]:
        """Load recent covenant results for a deal.

        Args:
            deal_id: Deal identifier.
            limit: Maximum results to load.

        Returns:
            Tuple of recent covenant results, sorted by period_end descending.
        """
        deal_id_typed = DealId(value=deal_id)
        all_results = self._result_repo.list_for_deal(deal_id_typed)
        # Sort by period_end descending (most recent first)
        sorted_results = sorted(
            all_results,
            key=_covenant_result_period_end_key,
            reverse=True,
        )
        return tuple(sorted_results[:limit])

    def _load_historical_metrics(
        self,
        deal_id: str,
        periods_back: int,
    ) -> dict[str, dict[str, int]]:
        """Load historical metrics for feature extraction.

        Args:
            deal_id: Deal identifier.
            periods_back: Number of most-recent periods to keep.

        Returns:
            Dict mapping period_end to metrics dict, holding at most
            `periods_back` periods.
        """
        deal_id_typed = DealId(value=deal_id)
        measurements = self._measurement_repo.list_for_deal(deal_id_typed)

        # Group by period_end, convert to scaled ints
        by_period: dict[str, dict[str, int]] = defaultdict(dict)
        for m in measurements:
            period_key = m["period_end_iso"]
            by_period[period_key][m["metric_name"]] = m["metric_value_scaled"]

        # Honour periods_back. _build_features reads only the most recent
        # period and the fourth most recent, so retaining everything grew the
        # returned mapping with the deal's whole history for no benefit.
        newest_first = sorted(by_period.keys(), reverse=True)[:periods_back]
        return {period: by_period[period] for period in newest_first}

    def _build_features(
        self,
        deal: Deal,
        current_metrics: dict[str, int],
        historical: dict[str, dict[str, int]],
        recent_results: tuple[CovenantResult, ...],
    ) -> LoanFeatures:
        """Build feature vector for ML prediction.

        Args:
            deal: Deal data.
            current_metrics: Current period metrics (scaled).
            historical: Historical metrics by period.
            recent_results: Recent covenant results.

        Returns:
            LoanFeatures for ML prediction.
        """
        # Sort periods to get 1p and 4p ago
        sorted_periods = sorted(historical.keys(), reverse=True)
        metrics_1p = historical.get(sorted_periods[0], {}) if len(sorted_periods) > 0 else {}
        metrics_4p = historical.get(sorted_periods[3], {}) if len(sorted_periods) > 3 else {}

        return extract_features(
            deal=deal,
            metrics_current=current_metrics,
            metrics_1p_ago=metrics_1p,
            metrics_4p_ago=metrics_4p,
            recent_results=list(recent_results),
            sector_encoder=self._sector_encoder,
            region_encoder=self._region_encoder,
        )

    def _run_evaluation(
        self,
        deal_id: DealId,
        covenants: tuple[Covenant, ...],
        period_start: str,
        period_end: str,
        metrics_scaled: dict[str, int],
    ) -> tuple[CovenantResult, ...]:
        """Run covenant evaluation.

        Args:
            deal_id: Deal identifier.
            covenants: Covenants to evaluate.
            period_start: Period start date.
            period_end: Period end date.
            metrics_scaled: Metrics for evaluation (scaled).

        Returns:
            Tuple of covenant results.
        """
        # Convert metrics dict to Measurement list
        measurements: list[Measurement] = []
        for name, value in metrics_scaled.items():
            measurements.append(
                {
                    "deal_id": deal_id,
                    "period_start_iso": period_start,
                    "period_end_iso": period_end,
                    "metric_name": name,
                    "metric_value_scaled": value,
                }
            )

        results = evaluate_all_covenants_for_period(
            covenants=list(covenants),
            period_start_iso=period_start,
            period_end_iso=period_end,
            measurements=measurements,
            tolerance_ratio_scaled=self._config["tolerance_ratio_scaled"],
        )
        return tuple(results)

    def _run_prediction(
        self,
        features: LoanFeatures,
    ) -> tuple[float, Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"], int]:
        """Run ML prediction and return results with timing.

        Args:
            features: Feature vector for prediction.

        Returns:
            Tuple of (probability, risk_tier, latency_ms).
        """
        start_time = time.perf_counter()
        features_list: list[LoanFeatures] = [features]
        probabilities = predict_probabilities(self._model, features_list)
        probability = probabilities[0]
        risk_tier = classify_risk_tier(probability)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return probability, risk_tier, latency_ms

    def _should_alert(
        self,
        evaluation_status: EvaluationStatus,
        risk_probability: float,
    ) -> bool:
        """Determine if an alert should be generated.

        Args:
            evaluation_status: Deterministic evaluation result.
            risk_probability: ML-predicted probability.

        Returns:
            True if alert should be generated.
        """
        if evaluation_status == "BREACH":
            return True
        return risk_probability >= self._config["alert_threshold"]
