"""Service container for dependency injection in covenant-radar-api.

Provides centralized access to shared resources like Redis connections
and database pools. Routes and workers access dependencies through
the container rather than creating their own connections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

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
    ensure_schema,
)
from platform_core.json_utils import JSONValue
from platform_core.queues import COVENANT_QUEUE
from platform_workers.redis import RedisStrProto
from platform_workers.rq_harness import (
    RQClientQueue,
    _RedisBytesClient,
    load_no_such_job_error,
    rq_fetch_job,
)

from . import _test_hooks
from .config import Settings

# Default encoders used for ML feature extraction.
# These map categorical values to integer indices.
DEFAULT_SECTOR_ENCODER: dict[str, int] = {
    "Technology": 0,
    "Finance": 1,
    "Healthcare": 2,
}

DEFAULT_REGION_ENCODER: dict[str, int] = {
    "North America": 0,
    "Europe": 1,
    "Asia": 2,
}


class JobStatus(TypedDict, total=True):
    """Status of a background job."""

    job_id: str
    status: Literal["queued", "started", "finished", "failed", "not_found"]
    result: JSONValue | None


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
        eager_load_model: bool = False,
    ) -> ServiceContainer:
        """Create container from settings, instantiating all dependencies.

        Args:
            settings: Application configuration.
            model_path: Path to active model file.
            model_output_dir: Directory for new model output.
            sector_encoder: Sector to int encoding.
            region_encoder: Region to int encoding.
            eager_load_model: If True, load ML model immediately at startup.
                This ensures fast first predictions and validates model exists.

        Returns:
            Fully initialized service container.
        """
        redis_url = settings["redis"]["url"]
        database_url = settings["database_url"]
        redis: RedisStrProto = _test_hooks.kv_factory(redis_url)
        db_conn = _test_hooks.connection_factory(database_url)
        # Ensure database schema exists (safe to call multiple times)
        ensure_schema(db_conn)
        redis_rq = _test_hooks.rq_client_factory(redis_url)
        output_dir = (
            model_output_dir
            if model_output_dir is not None
            else Path(settings["app"]["models_root"])
        )
        resolved_model_path = model_path if model_path else settings["app"]["active_model_path"]
        default_sector_encoder: dict[str, int] = (
            sector_encoder if sector_encoder is not None else DEFAULT_SECTOR_ENCODER
        )
        default_region_encoder: dict[str, int] = (
            region_encoder if region_encoder is not None else DEFAULT_REGION_ENCODER
        )
        container = cls(
            settings=settings,
            redis=redis,
            db_conn=db_conn,
            redis_rq=redis_rq,
            model_path=resolved_model_path,
            model_output_dir=output_dir,
            sector_encoder=default_sector_encoder,
            region_encoder=default_region_encoder,
        )

        if eager_load_model:
            container.load_model_now()

        return container

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

    def load_model_now(self: ServiceContainer) -> bool:
        """Eagerly load the ML model into memory.

        Call this at startup to ensure the model is loaded and ready
        for predictions. If the model file doesn't exist yet (e.g., no
        training has been done), logs a warning and returns False.

        Returns:
            True if model was loaded successfully, False if file not found.
        """
        from pathlib import Path

        from platform_core.logging import get_logger

        log = get_logger(__name__)
        model_path = Path(self._model_info["model_path"])

        if not model_path.exists():
            log.warning(
                "Model file not found, predictions will fail until model is trained",
                extra={"model_path": str(model_path)},
            )
            return False

        self._model = load_model(str(model_path))
        self._model_info = ModelInfo(
            model_id=self._model_info["model_id"],
            model_path=self._model_info["model_path"],
            is_loaded=True,
        )
        log.info(
            "ML model loaded successfully",
            extra={"model_path": str(model_path)},
        )
        return True

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

    def get_job_status(self: ServiceContainer, job_id: str) -> JobStatus:
        """Get status of a background job.

        Args:
            job_id: The job UUID string

        Returns:
            JobStatus with job_id, status, and result if available.
        """
        from platform_core.logging import get_logger

        log = get_logger(__name__)
        no_such_job_error = load_no_such_job_error()
        try:
            job = rq_fetch_job(job_id, self._redis_rq)
        except no_such_job_error:
            log.debug("job not found: %s", job_id)
            return JobStatus(job_id=job_id, status="not_found", result=None)

        # Map RQ status to our status enum
        rq_status = job.get_status()
        status: Literal["queued", "started", "finished", "failed", "not_found"]
        if rq_status == "queued":
            status = "queued"
        elif rq_status == "started":
            status = "started"
        elif rq_status == "finished":
            status = "finished"
        elif rq_status == "failed":
            status = "failed"
        else:
            status = "not_found"

        # Get result if job is finished
        result: JSONValue | None = None
        if status == "finished":
            raw_result = job.return_value()
            if isinstance(raw_result, dict):
                result = raw_result

        return JobStatus(job_id=job_id, status=status, result=result)


__all__ = [
    "DEFAULT_REGION_ENCODER",
    "DEFAULT_SECTOR_ENCODER",
    "JobStatus",
    "ModelInfo",
    "ServiceContainer",
]
