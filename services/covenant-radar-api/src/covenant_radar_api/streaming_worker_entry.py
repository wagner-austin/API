"""Streaming worker entry point for covenant breach prediction.

This module starts the Kafka streaming worker that consumes measurement
events, runs ML prediction, and produces prediction/alert events.

Usage:
    poetry run python -m covenant_radar_api.streaming_worker_entry

Environment variables:
    CONFLUENT__BOOTSTRAP_SERVERS: Kafka bootstrap servers
    CONFLUENT__API_KEY: SASL username
    CONFLUENT__API_SECRET: SASL password
    DATABASE_URL: PostgreSQL connection URL
    DD_AGENT_HOST: Datadog agent host (default: localhost)
    DD_DOGSTATSD_PORT: Datadog agent port (default: 8125)
    MODEL_PATH: Path to ML model file
    MODEL_TYPE: Model type (xgboost, lightgbm, logreg, random_forest)
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Literal, TypedDict

from covenant_ml.types import PredictorProtocol
from covenant_persistence import (
    CovenantRepository,
    CovenantResultRepository,
    DealRepository,
    MeasurementRepository,
)
from covenant_persistence.protocols import ConnectionProtocol
from platform_core.config import _parse_int, _parse_str, _require_env_str
from platform_core.logging import setup_logging

from . import streaming_worker_entry_hooks as _hooks
from .integrations.datadog import MetricsClient, MetricsConfig, create_metrics_client
from .streaming.config import StreamingConfig, load_streaming_config
from .streaming.consumer import StreamingConsumer, create_streaming_consumer
from .streaming.producer import StreamingProducer, create_streaming_producer
from .streaming.worker import StreamingWorker, WorkerConfig, make_default_worker_config
from .streaming_worker_entry_hooks import LoggerProtocol

# =============================================================================
# Types
# =============================================================================


ModelType = Literal["xgboost", "lightgbm", "logreg", "random_forest", "mlp"]


class StreamingWorkerDeps(TypedDict, total=True):
    """Dependencies for streaming worker.

    Fields:
        consumer: Kafka consumer for measurements.
        producer: Kafka producer for predictions/alerts.
        metrics: Datadog metrics client.
        model: ML model for predictions.
        deal_repo: Repository for deal data.
        covenant_repo: Repository for covenant data.
        measurement_repo: Repository for historical measurements.
        result_repo: Repository for covenant results.
        sector_encoder: Sector to integer mapping.
        region_encoder: Region to integer mapping.
        config: Worker configuration.
        db_conn: Database connection (kept open for worker lifetime).
    """

    consumer: StreamingConsumer
    producer: StreamingProducer
    metrics: MetricsClient
    model: PredictorProtocol
    deal_repo: DealRepository
    covenant_repo: CovenantRepository
    measurement_repo: MeasurementRepository
    result_repo: CovenantResultRepository
    sector_encoder: dict[str, int]
    region_encoder: dict[str, int]
    config: WorkerConfig
    db_conn: ConnectionProtocol


# =============================================================================
# Configuration Loading
# =============================================================================


def _load_metrics_config() -> MetricsConfig:
    """Load Datadog metrics configuration from environment.

    Returns:
        MetricsConfig with host, port, and namespace.
    """
    return {
        "host": _parse_str("DD_AGENT_HOST", "localhost"),
        "port": _parse_int("DD_DOGSTATSD_PORT", 8125),
        "namespace": "covenant",
    }


def _parse_model_type(value: str) -> ModelType:
    """Parse model type from string.

    Args:
        value: Model type string.

    Returns:
        Validated ModelType literal.

    Raises:
        ValueError: If value is not valid.
    """
    if value == "xgboost":
        return "xgboost"
    if value == "lightgbm":
        return "lightgbm"
    if value == "logreg":
        return "logreg"
    if value == "random_forest":
        return "random_forest"
    if value == "mlp":
        return "mlp"
    raise ValueError(
        f"Invalid MODEL_TYPE: '{value}'. "
        "Must be one of: xgboost, lightgbm, logreg, random_forest, mlp"
    )


def _load_model(model_path: Path, model_type: ModelType) -> PredictorProtocol:
    """Load ML model from path.

    Args:
        model_path: Path to model file.
        model_type: Type of model to load.

    Returns:
        Loaded model implementing PredictorProtocol.

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Import loaders from worker module
    from .worker._model_loaders import (
        load_lightgbm_model,
        load_logreg_model,
        load_random_forest_model,
    )

    if model_type == "lightgbm":
        return load_lightgbm_model(model_path)
    if model_type == "logreg":
        return load_logreg_model(model_path)
    if model_type == "random_forest":
        return load_random_forest_model(model_path)
    if model_type == "xgboost":
        return _hooks.xgboost_loader(str(model_path))

    # model_type == "mlp"
    from .worker._model_loaders import load_mlp_model

    meta_path = model_path.with_suffix(".meta.json")
    return load_mlp_model(model_path, meta_path)


def _load_encoders() -> tuple[dict[str, int], dict[str, int]]:
    """Load sector and region encoders.

    Returns:
        Tuple of (sector_encoder, region_encoder).
    """
    # Standard encoders matching training data
    sector_encoder: dict[str, int] = {
        "Technology": 0,
        "Healthcare": 1,
        "Finance": 2,
        "Manufacturing": 3,
        "Retail": 4,
        "Energy": 5,
        "Real Estate": 6,
        "Other": 7,
    }
    region_encoder: dict[str, int] = {
        "North America": 0,
        "Europe": 1,
        "Asia Pacific": 2,
        "Latin America": 3,
        "Middle East": 4,
        "Africa": 5,
    }
    return sector_encoder, region_encoder


# =============================================================================
# Dependency Creation
# =============================================================================


def _create_connection() -> ConnectionProtocol:
    """Create database connection from environment.

    Uses connection_factory hook for testability.

    Returns:
        Database connection.

    Raises:
        RuntimeError: If DATABASE_URL environment variable is not set.
    """
    database_url = _require_env_str("DATABASE_URL")
    return _hooks.connection_factory(database_url)


def _create_repositories(
    conn: ConnectionProtocol,
) -> tuple[DealRepository, CovenantRepository, MeasurementRepository, CovenantResultRepository]:
    """Create database repositories from connection.

    Uses repository_factory hook for testability.

    Args:
        conn: Database connection.

    Returns:
        Tuple of (deal_repo, covenant_repo, measurement_repo, result_repo).
    """
    return _hooks.repository_factory(conn)


def _build_dependencies(
    streaming_config: StreamingConfig,
) -> StreamingWorkerDeps:
    """Build all dependencies for streaming worker.

    Args:
        streaming_config: Kafka streaming configuration.

    Returns:
        StreamingWorkerDeps with all dependencies.
    """
    # Create Kafka consumer/producer
    consumer = create_streaming_consumer(streaming_config)
    producer = create_streaming_producer(streaming_config)

    # Create metrics client
    metrics_config = _load_metrics_config()
    metrics = create_metrics_client(metrics_config)

    # Load ML model
    model_path_str = _parse_str("MODEL_PATH", "models/model.json")
    model_type_str = _parse_str("MODEL_TYPE", "xgboost")
    model_path = Path(model_path_str)
    model_type = _parse_model_type(model_type_str)
    model = _load_model(model_path, model_type)

    # Create database connection and repositories
    db_conn = _create_connection()
    deal_repo, covenant_repo, measurement_repo, result_repo = _create_repositories(db_conn)

    # Load encoders
    sector_encoder, region_encoder = _load_encoders()

    # Worker config
    worker_config = make_default_worker_config()

    return {
        "consumer": consumer,
        "producer": producer,
        "metrics": metrics,
        "model": model,
        "deal_repo": deal_repo,
        "covenant_repo": covenant_repo,
        "measurement_repo": measurement_repo,
        "result_repo": result_repo,
        "sector_encoder": sector_encoder,
        "region_encoder": region_encoder,
        "config": worker_config,
        "db_conn": db_conn,
    }


# =============================================================================
# Worker Runner
# =============================================================================


def _create_worker(deps: StreamingWorkerDeps) -> StreamingWorker:
    """Create streaming worker from dependencies.

    Args:
        deps: All worker dependencies.

    Returns:
        Configured StreamingWorker instance.
    """
    return StreamingWorker(
        consumer=deps["consumer"],
        producer=deps["producer"],
        metrics=deps["metrics"],
        model=deps["model"],
        deal_repo=deps["deal_repo"],
        covenant_repo=deps["covenant_repo"],
        measurement_repo=deps["measurement_repo"],
        result_repo=deps["result_repo"],
        sector_encoder=deps["sector_encoder"],
        region_encoder=deps["region_encoder"],
        config=deps["config"],
    )


def _run_worker(
    worker: StreamingWorker,
    logger: LoggerProtocol,
) -> int:
    """Run the streaming worker with signal handling.

    Args:
        worker: Configured streaming worker.
        logger: Logger for status messages.

    Returns:
        Exit code (0 = success).
    """
    # Set up signal handlers for graceful shutdown
    shutdown_requested = False

    def handle_signal(signum: int, frame: FrameType | None) -> None:
        nonlocal shutdown_requested
        shutdown_requested = True
        logger.info("Shutdown signal received, stopping worker...")
        worker.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("Starting streaming worker...")

    total_messages, total_periods = worker.run()
    logger.info(
        "Worker stopped",
        extra={
            "messages_consumed": str(total_messages),
            "periods_processed": str(total_periods),
        },
    )
    return 0


# =============================================================================
# Main Entry Point
# =============================================================================


def main(
    streaming_config: StreamingConfig | None = None,
    deps: StreamingWorkerDeps | None = None,
    logger: LoggerProtocol | None = None,
) -> int:
    """Start the streaming worker.

    Args:
        streaming_config: Kafka configuration. If None, loads from environment.
        deps: Worker dependencies. If None, builds from environment.
        logger: Logger instance. If None, uses default logger.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="covenant-streaming-worker",
        instance_id=None,
        extra_fields=None,
    )

    resolved_logger: LoggerProtocol = (
        logger if logger is not None else _hooks.logger_factory(__name__)
    )

    # Load configuration
    resolved_config = streaming_config if streaming_config is not None else load_streaming_config()

    # Check if streaming is enabled
    if not resolved_config["enabled"]:
        resolved_logger.error("Streaming is disabled. Set STREAMING__ENABLED=true")
        return 1

    # Build or use provided dependencies
    resolved_deps = deps if deps is not None else _build_dependencies(resolved_config)

    # Create and run worker
    worker = _create_worker(resolved_deps)
    exit_code = _run_worker(worker, resolved_logger)

    # Clean up database connection
    resolved_deps["db_conn"].close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
