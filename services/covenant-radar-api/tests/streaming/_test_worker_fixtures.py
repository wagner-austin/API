"""Shared fixtures and factories for streaming worker tests.

Provides test data factories and helper functions used across worker test modules.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from typing import Literal

from covenant_domain import (
    Covenant,
    CovenantId,
    CovenantResult,
    Deal,
    DealId,
    Measurement,
)

from covenant_radar_api.integrations.datadog.metrics import MetricsClient
from covenant_radar_api.streaming._test_hooks import (
    FakeCovenantRepository,
    FakeCovenantResultRepository,
    FakeDealRepository,
    FakeKafkaConsumer,
    FakeKafkaProducer,
    FakeMeasurementRepository,
    FakeMetricsSink,
    FakePredictor,
)
from covenant_radar_api.streaming.consumer import StreamingConsumer
from covenant_radar_api.streaming.producer import StreamingProducer
from covenant_radar_api.streaming.schemas import MeasurementEventV1
from covenant_radar_api.streaming.worker import StreamingWorker
from covenant_radar_api.streaming.worker_events import WorkerConfig

# Required metrics for extract_features
REQUIRED_METRICS: dict[str, float] = {
    "total_debt": 5000000.0,  # $5M
    "ebitda": 1000000.0,  # $1M
    "interest_expense": 200000.0,  # $200K
    "current_assets": 2000000.0,  # $2M
    "current_liabilities": 1000000.0,  # $1M
}


def make_deal(deal_id: str = "deal-001", name: str = "Test Deal") -> Deal:
    """Create a test deal."""
    return {
        "id": DealId(value=deal_id),
        "name": name,
        "borrower": "Test Borrower",
        "sector": "Technology",
        "region": "North America",
        "commitment_amount_cents": 10_000_000_00,  # $10M
        "currency": "USD",
        "maturity_date_iso": "2029-01-01",
    }


def make_covenant(
    covenant_id: str = "cov-001",
    deal_id: str = "deal-001",
    formula: str = "total_debt / ebitda",  # Uses metrics from REQUIRED_METRICS
    threshold_value_scaled: int = 10_000_000,  # 10.0 (debt/ebitda ratio)
    threshold_direction: Literal["<=", ">="] = "<=",
) -> Covenant:
    """Create a test covenant."""
    return {
        "id": CovenantId(value=covenant_id),
        "deal_id": DealId(value=deal_id),
        "name": "Debt to EBITDA Ratio",
        "formula": formula,
        "threshold_value_scaled": threshold_value_scaled,
        "threshold_direction": threshold_direction,
        "frequency": "QUARTERLY",
    }


def make_measurement(
    deal_id: str = "deal-001",
    metric_name: str = "total_debt",
    value_scaled: int = 1_500_000,
    period_start: str = "2024-01-01",
    period_end: str = "2024-03-31",
) -> Measurement:
    """Create a test measurement."""
    return {
        "deal_id": DealId(value=deal_id),
        "metric_name": metric_name,
        "metric_value_scaled": value_scaled,
        "period_start_iso": period_start,
        "period_end_iso": period_end,
    }


def make_covenant_result(
    covenant_id: str = "cov-001",
    status: Literal["OK", "NEAR_BREACH", "BREACH"] = "OK",
    period_start: str = "2024-01-01",
    period_end: str = "2024-03-31",
) -> CovenantResult:
    """Create a test covenant result."""
    return {
        "covenant_id": CovenantId(value=covenant_id),
        "status": status,
        "period_start_iso": period_start,
        "period_end_iso": period_end,
        "calculated_value_scaled": 1_500_000,
    }


def make_measurement_event(
    deal_id: str = "deal-001",
    metric_name: str = "debt_to_equity",
    metric_value: float = 1.5,
    period_start: str = "2024-01-01",
    period_end: str = "2024-03-31",
) -> MeasurementEventV1:
    """Create a test measurement event."""
    return {
        "type": "covenant.measurement.v1",
        "event_id": "evt-001",
        "deal_id": deal_id,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "period_start": period_start,
        "period_end": period_end,
        "timestamp": "2024-04-01T00:00:00Z",
    }


def make_worker_config() -> WorkerConfig:
    """Create a test worker config."""
    return {
        "model_version": "test-v1",
        "poll_timeout_seconds": 0.1,
        "alert_threshold": 0.80,
        "commit_interval": 5,
        "buffer_timeout_seconds": 0.1,  # Short for testing
        "min_metrics_per_period": 5,  # Need all 5 required metrics
        "tolerance_ratio_scaled": 100_000,
    }


def make_streaming_worker() -> tuple[
    StreamingWorker,
    FakeKafkaConsumer,
    FakeKafkaProducer,
    FakeMetricsSink,
    FakePredictor,
    FakeDealRepository,
    FakeCovenantRepository,
    FakeMeasurementRepository,
    FakeCovenantResultRepository,
]:
    """Create a StreamingWorker with all fake dependencies."""
    # Create fake Kafka components
    fake_consumer = FakeKafkaConsumer()
    fake_producer = FakeKafkaProducer()

    # Wrap in high-level classes
    consumer = StreamingConsumer(fake_consumer, "measurements")
    producer = StreamingProducer(fake_producer, "predictions", "alerts", "dlq")

    # Create fake metrics
    fake_sink = FakeMetricsSink()
    metrics = MetricsClient(fake_sink)

    # Create fake model
    fake_predictor = FakePredictor(default_probability=0.25)

    # Create fake repositories
    deal_repo = FakeDealRepository()
    covenant_repo = FakeCovenantRepository()
    measurement_repo = FakeMeasurementRepository()
    result_repo = FakeCovenantResultRepository()

    # Create worker
    worker = StreamingWorker(
        consumer=consumer,
        producer=producer,
        metrics=metrics,
        model=fake_predictor,
        deal_repo=deal_repo,
        covenant_repo=covenant_repo,
        measurement_repo=measurement_repo,
        result_repo=result_repo,
        sector_encoder={"Technology": 0, "Healthcare": 1},
        region_encoder={"North America": 0, "Europe": 1},
        config=make_worker_config(),
    )

    return (
        worker,
        fake_consumer,
        fake_producer,
        fake_sink,
        fake_predictor,
        deal_repo,
        covenant_repo,
        measurement_repo,
        result_repo,
    )


__all__ = [
    "REQUIRED_METRICS",
    "make_covenant",
    "make_covenant_result",
    "make_deal",
    "make_measurement",
    "make_measurement_event",
    "make_streaming_worker",
    "make_worker_config",
]
