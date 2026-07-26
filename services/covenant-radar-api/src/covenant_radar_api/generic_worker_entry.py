"""Entry point for the domain-agnostic streaming worker.

Runs GenericStreamingWorker against one registered domain, selected by name.
The worker itself knows nothing about weather or covenants; everything
domain-specific arrives through DomainProtocol.

Usage:
    poetry run covenant-streaming-worker

Environment variables:
    STREAMING__ENABLED: Must be true, or the process exits non-zero.
    STREAMING__DOMAIN: Which registered domain to run (default: weather).
    CONFLUENT__BOOTSTRAP_SERVERS: Kafka bootstrap servers.
    CONFLUENT__SECURITY_PROTOCOL: SASL_SSL (default) or PLAINTEXT.
    CONFLUENT__API_KEY / CONFLUENT__API_SECRET: SASL credentials.
    MODEL_PATH: Path to the saved model file.
    MODEL_VERSION: Version string reported on every prediction event.
    WEATHER__STATE_PATH: Fitted temporal feature state, JSON.
    WEATHER__STATION_MAP_PATH: station_id to location index, JSON.
    GEMINI_API_KEY / GEMINI_MODEL: Alert summary generation.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

from covenant_ml.types import PredictorProtocol
from platform_core.config import _parse_float, _parse_str, _require_env_str
from platform_core.logging import setup_logging

from . import generic_worker_entry_hooks as _hooks
from .domains.protocols import DomainProtocol
from .domains.registry import DomainRegistry
from .domains.weather.domain import WEATHER_ALERT_THRESHOLD, make_weather_domain
from .generic_worker_entry_hooks import LoggerProtocol
from .streaming._test_hooks import consumer_factory, producer_factory
from .streaming._test_hooks_generic_worker import TextGeneratorProtocol
from .streaming.config import StreamingConfig, load_streaming_config
from .streaming.generic_worker import (
    GenericStreamingWorker,
    GenericWorkerConfig,
    make_generic_worker_config,
)

# =============================================================================
# Dependencies
# =============================================================================


class GenericWorkerDeps(TypedDict, total=True):
    """Everything GenericStreamingWorker needs, resolved from configuration.

    Fields:
        domain: The registered domain the worker runs.
        model: ML model exposing predict_proba.
        text_generator: Generator for alert summaries.
        worker_config: Model version and poll timeout.
    """

    domain: DomainProtocol
    model: PredictorProtocol
    text_generator: TextGeneratorProtocol
    worker_config: GenericWorkerConfig


# =============================================================================
# Domain Construction
# =============================================================================


def build_domain_registry() -> DomainRegistry:
    """Build the registry of domains this deployment can run.

    Weather is constructed from a fitted temporal state and a station map on
    disk, because the seasonal cycle and tail thresholds come from training
    data and cannot be derived at startup.

    Returns:
        Registry with every available domain registered.

    Raises:
        RuntimeError: If a required environment variable is unset.
        FileNotFoundError: If a referenced state or map file is missing.
    """
    state_path = Path(_require_env_str("WEATHER__STATE_PATH"))
    station_map_path = Path(_require_env_str("WEATHER__STATION_MAP_PATH"))
    alert_threshold = _parse_float("WEATHER__ALERT_THRESHOLD", WEATHER_ALERT_THRESHOLD)

    registry = DomainRegistry()
    registry.register(
        make_weather_domain(
            state=_hooks.temporal_state_loader(state_path),
            station_to_location=_hooks.station_map_loader(station_map_path),
            alert_threshold=alert_threshold,
        )
    )
    return registry


def build_dependencies() -> GenericWorkerDeps:
    """Resolve the worker's dependencies from the environment.

    Returns:
        GenericWorkerDeps ready to construct a worker from.

    Raises:
        RuntimeError: If a required environment variable is unset.
        KeyError: If STREAMING__DOMAIN names a domain that is not registered.
        FileNotFoundError: If a referenced model, state or map file is missing.
    """
    registry = build_domain_registry()
    domain_name = _parse_str("STREAMING__DOMAIN", "weather")
    domain = registry.get(domain_name)

    model_path = _require_env_str("MODEL_PATH")
    model_version = _parse_str("MODEL_VERSION", "v1.0.0")
    poll_timeout = _parse_float("STREAMING__POLL_TIMEOUT_SECONDS", 1.0)

    gemini_api_key = _require_env_str("GEMINI_API_KEY")
    gemini_model = _parse_str("GEMINI_MODEL", "gemini-2.0-flash")

    return {
        "domain": domain,
        "model": _hooks.model_loader(model_path),
        "text_generator": _hooks.text_generator_factory(gemini_api_key, gemini_model),
        "worker_config": make_generic_worker_config(
            model_version=model_version,
            poll_timeout_seconds=poll_timeout,
        ),
    }


def create_worker(
    streaming_config: StreamingConfig,
    deps: GenericWorkerDeps,
) -> GenericStreamingWorker:
    """Construct the worker from streaming configuration and dependencies.

    The domain's own input topic is what the consumer subscribes to, so
    adding a domain does not require touching Kafka configuration.

    Args:
        streaming_config: Kafka connection and consumer/producer settings.
        deps: Resolved worker dependencies.

    Returns:
        A worker ready to run.
    """
    return GenericStreamingWorker(
        domain=deps["domain"],
        consumer=consumer_factory(streaming_config["confluent"], streaming_config["consumer"]),
        producer=producer_factory(streaming_config["confluent"], streaming_config["producer"]),
        model=deps["model"],
        text_generator=deps["text_generator"],
        config=deps["worker_config"],
    )


# =============================================================================
# Entry Point
# =============================================================================


def main(
    streaming_config: StreamingConfig | None = None,
    deps: GenericWorkerDeps | None = None,
    logger: LoggerProtocol | None = None,
    max_iterations: int | None = None,
) -> int:
    """Start the generic streaming worker.

    Args:
        streaming_config: Kafka configuration. Loaded from the environment
            when None.
        deps: Worker dependencies. Built from the environment when None.
        logger: Logger. Created from the hook when None.
        max_iterations: Stop after this many poll cycles. None runs until
            shutdown, which is what the container does; a bound is for
            smoke-testing a deployment without leaving a daemon behind.

    Returns:
        Exit code: 0 on a clean shutdown, 1 if streaming is disabled.
    """
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="covenant-generic-streaming-worker",
        instance_id=None,
        extra_fields=None,
    )
    resolved_logger: LoggerProtocol = (
        logger if logger is not None else _hooks.logger_factory(__name__)
    )

    resolved_config = streaming_config if streaming_config is not None else load_streaming_config()
    if not resolved_config["enabled"]:
        resolved_logger.error("Streaming is disabled. Set STREAMING__ENABLED=true")
        return 1

    resolved_deps = deps if deps is not None else build_dependencies()
    worker = create_worker(resolved_config, resolved_deps)

    resolved_logger.info(
        f"Starting generic streaming worker for domain "
        f"'{resolved_deps['domain'].config['name']}' on topic "
        f"'{resolved_deps['domain'].config['input_topic']}'"
    )
    worker.run(max_iterations)
    worker.shutdown()
    resolved_logger.info("Generic streaming worker stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
