"""Service container for dependency injection in covenant-radar-api.

Provides centralized access to shared resources like Redis connections
and database pools. Routes and workers access dependencies through
the container rather than creating their own connections.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from covenant_ml.predictor import load_model
from covenant_ml.types import XGBModelProtocol
from covenant_persistence import (
    ConnectionProtocol,
    CovenantRepository,
    CovenantResultRepository,
    DealRepository,
    MeasurementRepository,
    PostgresCovenantRepository,
    PostgresCovenantResultRepository,
    PostgresDealRepository,
    PostgresMeasurementRepository,
)
from platform_core.queues import COVENANT_QUEUE
from platform_workers.redis import RedisStrProto
from platform_workers.rq_harness import RQClientQueue, _RedisBytesClient

from . import _test_hooks
from .config import Settings


class ModelInfo(TypedDict, total=True):
    """Information about the active ML model."""

    model_id: str
    model_path: str
    is_loaded: bool


class ServiceContainer:
    """Container holding shared service dependencies.

    Attributes:
        settings: Application configuration loaded from environment.
        redis: Redis client for job queue and health checks.
        db_conn: Database connection for repository operations.
        _redis_rq: Redis client for RQ operations.
        _model: Cached XGBoost model (lazy loaded).
        _model_info: Information about current model.
    """

    settings: Settings
    redis: RedisStrProto
    db_conn: ConnectionProtocol
    _redis_rq: _RedisBytesClient
    _model: XGBModelProtocol | None
    _model_info: ModelInfo
    _sector_encoder: dict[str, int]
    _region_encoder: dict[str, int]
    _model_output_dir: Path

    def __init__(
        self: ServiceContainer,
        settings: Settings,
        redis: RedisStrProto,
        db_conn: ConnectionProtocol,
        redis_rq: _RedisBytesClient,
        model_path: str,
        model_output_dir: Path,
        sector_encoder: dict[str, int],
        region_encoder: dict[str, int],
    ) -> None:
        """Initialize container with dependencies.

        Args:
            settings: Application configuration.
            redis: Redis client instance.
            db_conn: Database connection instance.
            redis_rq: Redis client for RQ operations.
            model_path: Path to active model file.
            model_output_dir: Directory for new model output.
            sector_encoder: Sector to int encoding.
            region_encoder: Region to int encoding.
        """
        self.settings = settings
        self.redis = redis
        self.db_conn = db_conn
        self._redis_rq = redis_rq
        self._model = None
        self._model_info = ModelInfo(
            model_id="default",
            model_path=model_path,
            is_loaded=False,
        )
        self._model_output_dir = model_output_dir
        self._sector_encoder = sector_encoder
        self._region_encoder = region_encoder

    @classmethod
    def from_settings(
        cls: type[ServiceContainer],
        settings: Settings,
        model_path: str = "",
        model_output_dir: Path | None = None,
        sector_encoder: dict[str, int] | None = None,
        region_encoder: dict[str, int] | None = None,
    ) -> ServiceContainer:
        """Create container from settings, instantiating all dependencies.

        Args:
            settings: Application configuration.
            model_path: Path to active model file.
            model_output_dir: Directory for new model output.
            sector_encoder: Sector to int encoding.
            region_encoder: Region to int encoding.

        Returns:
            Fully initialized service container.
        """
        redis: RedisStrProto = _test_hooks.kv_factory(settings["redis_url"])
        db_conn = _test_hooks.connection_factory(settings["database_url"])
        redis_rq = _test_hooks.rq_client_factory(settings["redis_url"])
        output_dir = model_output_dir if model_output_dir is not None else Path("./models")
        default_sector_encoder: dict[str, int] = (
            sector_encoder if sector_encoder is not None else {}
        )
        default_region_encoder: dict[str, int] = (
            region_encoder if region_encoder is not None else {}
        )
        return cls(
            settings=settings,
            redis=redis,
            db_conn=db_conn,
            redis_rq=redis_rq,
            model_path=model_path,
            model_output_dir=output_dir,
            sector_encoder=default_sector_encoder,
            region_encoder=default_region_encoder,
        )

    def close(self: ServiceContainer) -> None:
        """Close all resources held by the container."""
        self.redis.close()
        self.db_conn.close()
        self._redis_rq.close()

    def deal_repo(self: ServiceContainer) -> DealRepository:
        """Get deal repository bound to container's connection."""
        repo: DealRepository = PostgresDealRepository(self.db_conn)
        return repo

    def covenant_repo(self: ServiceContainer) -> CovenantRepository:
        """Get covenant repository bound to container's connection."""
        repo: CovenantRepository = PostgresCovenantRepository(self.db_conn)
        return repo

    def measurement_repo(self: ServiceContainer) -> MeasurementRepository:
        """Get measurement repository bound to container's connection."""
        repo: MeasurementRepository = PostgresMeasurementRepository(self.db_conn)
        return repo

    def covenant_result_repo(self: ServiceContainer) -> CovenantResultRepository:
        """Get covenant result repository bound to container's connection."""
        repo: CovenantResultRepository = PostgresCovenantResultRepository(self.db_conn)
        return repo

    def rq_queue(self: ServiceContainer) -> RQClientQueue:
        """Get RQ queue client for enqueueing jobs."""
        return _test_hooks.queue_factory(COVENANT_QUEUE, self._redis_rq)

    def get_model(self: ServiceContainer) -> XGBModelProtocol:
        """Get the XGBoost model, loading it if necessary.

        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        if self._model is None:
            self._model = load_model(self._model_info["model_path"])
            self._model_info = ModelInfo(
                model_id=self._model_info["model_id"],
                model_path=self._model_info["model_path"],
                is_loaded=True,
            )
        return self._model

    def get_model_info(self: ServiceContainer) -> ModelInfo:
        """Get information about the current model."""
        return self._model_info

    def get_sector_encoder(self: ServiceContainer) -> dict[str, int]:
        """Get sector to int encoding."""
        return self._sector_encoder

    def get_region_encoder(self: ServiceContainer) -> dict[str, int]:
        """Get region to int encoding."""
        return self._region_encoder

    def get_model_output_dir(self: ServiceContainer) -> Path:
        """Get directory for model output."""
        return self._model_output_dir


__all__ = ["ModelInfo", "ServiceContainer"]
