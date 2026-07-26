"""Shared fixtures and factories for generic streaming worker tests.

Provides test data factories and helper functions for GenericStreamingWorker.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from covenant_radar_api.domains.base_schemas import (
    BaseInputEventV1,
    encode_base_input_event,
    make_base_input_event,
)
from covenant_radar_api.domains.protocols import (
    DomainConfig,
    make_domain_config,
)
from covenant_radar_api.streaming._test_hooks import (
    FakeKafkaConsumer,
    FakeKafkaProducer,
)
from covenant_radar_api.streaming._test_hooks_generic_worker import FakeTextGenerator
from covenant_radar_api.streaming._test_hooks_model import FakePredictor
from covenant_radar_api.streaming.generic_worker import (
    GenericStreamingWorker,
    GenericWorkerConfig,
    make_generic_worker_config,
)
from tests.domains._test_domain_fixtures import FakeDomain

# =============================================================================
# Factory Functions
# =============================================================================


def make_test_domain_config() -> DomainConfig:
    """Create a test domain config."""
    return make_domain_config(
        name="test",
        display_name="Test Domain",
        input_topic="test-input",
        prediction_topic="test-predictions",
        alert_topic="test-alerts",
        alert_threshold=0.80,
    )


def make_test_generic_worker_config() -> GenericWorkerConfig:
    """Create a test generic worker config."""
    return make_generic_worker_config(
        model_version="test-v1",
        poll_timeout_seconds=0.1,
    )


def make_base_input_payload(
    entity_id: str = "entity-001",
    timestamp: str = "2026-01-15T10:00:00Z",
) -> str:
    """Create a serialized base input event payload.

    Args:
        entity_id: Entity identifier.
        timestamp: Event timestamp.

    Returns:
        JSON string of BaseInputEventV1.
    """
    event: BaseInputEventV1 = make_base_input_event(
        type="test.input.v1",
        event_id="evt-001",
        entity_id=entity_id,
        timestamp=timestamp,
    )
    return encode_base_input_event(event)


def make_generic_streaming_worker(
    probability: float = 0.25,
    alert_threshold: float = 0.80,
) -> tuple[
    GenericStreamingWorker,
    FakeDomain,
    FakeKafkaConsumer,
    FakeKafkaProducer,
    FakePredictor,
    FakeTextGenerator,
]:
    """Create a GenericStreamingWorker with all fake dependencies.

    Args:
        probability: Default prediction probability.
        alert_threshold: Domain alert threshold.

    Returns:
        Tuple of (worker, domain, consumer, producer, predictor, text_generator).
    """
    domain_config: DomainConfig = make_domain_config(
        name="test",
        display_name="Test Domain",
        input_topic="test-input",
        prediction_topic="test-predictions",
        alert_topic="test-alerts",
        alert_threshold=alert_threshold,
    )
    domain = FakeDomain(domain_config)
    consumer = FakeKafkaConsumer()
    producer = FakeKafkaProducer()
    predictor = FakePredictor(default_probability=probability)
    text_generator = FakeTextGenerator()

    config: GenericWorkerConfig = make_test_generic_worker_config()

    worker = GenericStreamingWorker(
        domain=domain,
        consumer=consumer,
        producer=producer,
        model=predictor,
        text_generator=text_generator,
        config=config,
    )

    return (worker, domain, consumer, producer, predictor, text_generator)


__all__ = [
    "FakeDomain",
    "make_base_input_payload",
    "make_generic_streaming_worker",
    "make_test_domain_config",
    "make_test_generic_worker_config",
]
