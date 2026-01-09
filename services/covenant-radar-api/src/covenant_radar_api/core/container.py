"""Service container for dependency injection in covenant-radar-api.

Provides centralized access to shared resources like Redis connections
and database pools. Routes and workers access dependencies through
the container rather than creating their own connections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

from covenant_ml.predictor import load_model
from covenant_ml.types import PredictorProtocol
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
from platform_core.config import MLBackend
from platform_core.data_bank_client import (
    DataBankClientError,
)
from platform_core.data_bank_client import (
    NotFoundError as DataBankNotFoundError,
)
from platform_core.json_utils import JSONValue
from platform_core.logging import get_logger
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

_log = get_logger(__name__)

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
        _model: Cached model (lazy loaded, backend-aware).
        _model_info: Information about current model.
        _ml_backend: ML backend type for inference (xgboost, mlp, lstm, or lightgbm).
        _data_bank_url: URL for data-bank-api (empty if not configured).
        _data_bank_key: API key for data-bank-api (empty if not configured).
        _data_bank_model_file_id: Model file_id to download from data-bank.
    """

    settings: Settings
    redis: RedisStrProto
    db_conn: ConnectionProtocol
    _redis_rq: _RedisBytesClient
    _model: PredictorProtocol | None
    _model_info: ModelInfo
    _sector_encoder: dict[str, int]
    _region_encoder: dict[str, int]
    _model_output_dir: Path
    _ml_backend: MLBackend
    _data_bank_url: str
    _data_bank_key: str
    _data_bank_model_file_id: str

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
        ml_backend: MLBackend,
        data_bank_url: str = "",
        data_bank_key: str = "",
        data_bank_model_file_id: str = "",
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
            ml_backend: ML backend type for inference (xgboost, mlp, lstm, or lightgbm).
            data_bank_url: URL for data-bank-api (empty if not configured).
            data_bank_key: API key for data-bank-api (empty if not configured).
            data_bank_model_file_id: Model file_id to download from data-bank.
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
        self._ml_backend = ml_backend
        self._data_bank_url = data_bank_url
        self._data_bank_key = data_bank_key
        self._data_bank_model_file_id = data_bank_model_file_id

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
        # Get ML backend from settings (defaults to xgboost)
        ml_backend: MLBackend = settings["app"]["ml_backend"]
        # active_model_path is pre-resolved by config loader based on ml_backend
        resolved_model_path = model_path if model_path else settings["app"]["active_model_path"]
        default_sector_encoder: dict[str, int] = (
            sector_encoder if sector_encoder is not None else DEFAULT_SECTOR_ENCODER
        )
        default_region_encoder: dict[str, int] = (
            region_encoder if region_encoder is not None else DEFAULT_REGION_ENCODER
        )
        # Get data-bank config
        data_bank_url = settings["app"]["data_bank_api_url"]
        data_bank_key = settings["app"]["data_bank_api_key"]
        data_bank_model_file_id = settings["app"]["data_bank_model_file_id"]

        container = cls(
            settings=settings,
            redis=redis,
            db_conn=db_conn,
            redis_rq=redis_rq,
            model_path=resolved_model_path,
            model_output_dir=output_dir,
            sector_encoder=default_sector_encoder,
            region_encoder=default_region_encoder,
            ml_backend=ml_backend,
            data_bank_url=data_bank_url,
            data_bank_key=data_bank_key,
            data_bank_model_file_id=data_bank_model_file_id,
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

    def _load_xgboost_model(self: ServiceContainer, model_path: str) -> PredictorProtocol:
        """Load XGBoost model from file.

        Args:
            model_path: Path to the XGBoost .ubj model file.

        Returns:
            Loaded XGBoost model as PredictorProtocol.
        """
        model: PredictorProtocol = load_model(model_path)
        return model

    def _load_mlp_model(self: ServiceContainer, model_path: str) -> PredictorProtocol:
        """Load MLP model from file with architecture metadata.

        Args:
            model_path: Path to the PyTorch .pt model file.

        Returns:
            Loaded MLP model as PredictorProtocol.

        Raises:
            FileNotFoundError: If model or metadata file is missing.
        """
        from covenant_radar_api.worker import _test_hooks as worker_hooks

        model_p = Path(model_path)
        meta_p = model_p.parent / "active_mlp_meta.json"
        return worker_hooks.mlp_loader(model_p, meta_p)

    def _load_lstm_model(self: ServiceContainer, model_path: str) -> PredictorProtocol:
        """Load LSTM model from file with architecture metadata.

        Args:
            model_path: Path to the PyTorch .pt model file.

        Returns:
            Loaded LSTM model as PredictorProtocol.

        Raises:
            FileNotFoundError: If model or metadata file is missing.
        """
        from covenant_radar_api.worker import _test_hooks as worker_hooks

        model_p = Path(model_path)
        meta_p = model_p.parent / "active_lstm_meta.json"
        return worker_hooks.lstm_loader(model_p, meta_p)

    def _load_lightgbm_model(self: ServiceContainer, model_path: str) -> PredictorProtocol:
        """Load LightGBM model from file.

        Args:
            model_path: Path to the LightGBM .txt model file.

        Returns:
            Loaded LightGBM model as PredictorProtocol.

        Raises:
            FileNotFoundError: If model file is missing.
        """
        from covenant_radar_api.worker import _test_hooks as worker_hooks

        model_p = Path(model_path)
        return worker_hooks.lightgbm_loader(model_p)

    def _get_model_file_id(self: ServiceContainer) -> str:
        """Get the file_id to download from data-bank.

        Returns the configured DATA_BANK_MODEL_FILE_ID if set, otherwise
        returns an empty string indicating no model should be downloaded.

        Returns:
            File ID string (SHA256 hash) or empty string if not configured.
        """
        return self._data_bank_model_file_id

    def _download_model_from_data_bank(self: ServiceContainer, dest_path: Path) -> bool:
        """Download model from data-bank-api if configured.

        Args:
            dest_path: Local path where model should be saved.

        Returns:
            True if model was downloaded successfully, False if not configured
            or model not found in data-bank.
        """
        if not self._data_bank_url or not self._data_bank_key:
            _log.debug("Data-bank not configured, skipping download")
            return False

        file_id = self._get_model_file_id()
        if not file_id:
            _log.debug("No model file_id configured, skipping download")
            return False

        _log.info(
            "Attempting to download model from data-bank",
            extra={"file_id": file_id, "dest_path": str(dest_path)},
        )

        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        client = _test_hooks.data_bank_client_factory(
            self._data_bank_url,
            self._data_bank_key,
        )

        try:
            head_info = client.download_to_path(file_id, dest_path)
            _log.info(
                "Downloaded model from data-bank",
                extra={
                    "file_id": file_id,
                    "size": head_info["size"],
                    "dest_path": str(dest_path),
                },
            )
            return True
        except DataBankNotFoundError:
            _log.warning(
                "Model not found in data-bank",
                extra={"file_id": file_id},
            )
            return False
        except DataBankClientError as exc:
            _log.warning(
                "Failed to download model from data-bank",
                extra={"file_id": file_id, "error": str(exc)},
            )
            return False

    def load_model_now(self: ServiceContainer) -> bool:
        """Eagerly load the ML model into memory.

        Call this at startup to ensure the model is loaded and ready
        for predictions. If the model file doesn't exist locally and
        data-bank is configured, attempts to download from data-bank.
        If the model file doesn't exist after download attempt, logs
        a warning and returns False.

        Returns:
            True if model was loaded successfully, False if file not found.
        """
        model_path = Path(self._model_info["model_path"])

        # If model doesn't exist locally, try downloading from data-bank
        if not model_path.exists():
            # Use /tmp for downloaded models since /data may not exist
            if self._data_bank_url and self._data_bank_key:
                tmp_model_path = Path("/tmp/models") / model_path.name
                downloaded = self._download_model_from_data_bank(tmp_model_path)
                if downloaded:
                    model_path = tmp_model_path
            else:
                downloaded = False
            if not downloaded:
                _log.warning(
                    "Model file not found, predictions will fail until model is trained",
                    extra={"model_path": str(model_path), "backend": self._ml_backend},
                )
                return False

        # Load model based on configured backend
        if self._ml_backend == "xgboost":
            self._model = self._load_xgboost_model(str(model_path))
        elif self._ml_backend == "mlp":
            self._model = self._load_mlp_model(str(model_path))
        elif self._ml_backend == "lstm":
            self._model = self._load_lstm_model(str(model_path))
        else:  # lightgbm
            self._model = self._load_lightgbm_model(str(model_path))

        self._model_info = ModelInfo(
            model_id=self._model_info["model_id"],
            model_path=self._model_info["model_path"],
            is_loaded=True,
        )
        _log.info(
            "ML model loaded successfully",
            extra={"model_path": str(model_path), "backend": self._ml_backend},
        )
        return True

    def get_model(self: ServiceContainer) -> PredictorProtocol:
        """Get the ML model, loading it if necessary.

        If the model file doesn't exist locally and data-bank is configured,
        attempts to download from data-bank first.

        Returns:
            Loaded model implementing PredictorProtocol.

        Raises:
            FileNotFoundError: If model file doesn't exist locally or in data-bank.
        """
        if self._model is None:
            model_path = Path(self._model_info["model_path"])

            # If model doesn't exist locally, try downloading from data-bank
            if not model_path.exists():
                # Use /tmp for downloaded models since /data may not exist
                if self._data_bank_url and self._data_bank_key:
                    tmp_model_path = Path("/tmp/models") / model_path.name
                    downloaded = self._download_model_from_data_bank(tmp_model_path)
                    if downloaded:
                        model_path = tmp_model_path
                else:
                    downloaded = False
                if not downloaded:
                    raise FileNotFoundError(
                        f"Model file not found: {model_path}. "
                        "Train a model first or configure data-bank integration."
                    )

            if self._ml_backend == "xgboost":
                self._model = self._load_xgboost_model(str(model_path))
            elif self._ml_backend == "mlp":
                self._model = self._load_mlp_model(str(model_path))
            elif self._ml_backend == "lstm":
                self._model = self._load_lstm_model(str(model_path))
            else:  # lightgbm
                self._model = self._load_lightgbm_model(str(model_path))
            self._model_info = ModelInfo(
                model_id=self._model_info["model_id"],
                model_path=str(model_path),
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
