"""ML prediction and training endpoints."""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

from covenant_domain import DealId
from covenant_domain.features import (
    LoanFeatures,
    classify_risk_tier,
    extract_features,
)
from covenant_ml.predictor import predict_probabilities
from covenant_ml.types import TrainConfig, XGBModelProtocol
from covenant_persistence import (
    CovenantResultRepository,
    DealRepository,
    MeasurementRepository,
)
from fastapi import APIRouter, Request, Response
from platform_core.json_utils import JSONValue, dump_json_str
from platform_workers.rq_harness import RQClientQueue

from ..decode import PredictResponse, TrainResponse, parse_predict_request, parse_train_request


class ModelInfo(TypedDict, total=True):
    """Information about the active ML model."""

    model_id: str
    model_path: str
    is_loaded: bool


class JobStatus(TypedDict, total=True):
    """Status of a background job."""

    job_id: str
    status: Literal["queued", "started", "finished", "failed", "not_found"]
    result: JSONValue | None


class ContainerProtocol(Protocol):
    """Protocol for service container with ML dependencies."""

    def deal_repo(self) -> DealRepository: ...

    def measurement_repo(self) -> MeasurementRepository: ...

    def covenant_result_repo(self) -> CovenantResultRepository: ...

    def rq_queue(self) -> RQClientQueue: ...

    def get_model(self) -> XGBModelProtocol: ...

    def get_model_info(self) -> ModelInfo: ...

    def get_sector_encoder(self) -> dict[str, int]: ...

    def get_region_encoder(self) -> dict[str, int]: ...

    def get_job_status(self, job_id: str) -> JobStatus: ...


def build_router(get_container: ContainerProtocol) -> APIRouter:
    """Build FastAPI router for ML operations.

    Args:
        get_container: Container instance with ML dependencies.

    Returns:
        Configured router with prediction and training endpoints.
    """
    router = APIRouter(prefix="/ml", tags=["ml"])

    async def _predict(request: Request) -> Response:
        """Predict breach risk for a deal.

        Request body:
            deal_id: Deal UUID string

        Returns:
            JSON object with probability and risk_tier.

        Raises:
            KeyError: Deal not found or required metrics missing
        """
        body_bytes = await request.body()
        req = parse_predict_request(body_bytes)

        deal_id = DealId(value=req["deal_id"])

        deal_repo = get_container.deal_repo()
        measurement_repo = get_container.measurement_repo()
        result_repo = get_container.covenant_result_repo()

        deal = deal_repo.get(deal_id)
        measurements = measurement_repo.list_for_deal(deal_id)

        # Get recent covenant results for near-breach count
        recent_results = result_repo.list_for_deal(deal_id)

        # Build metric dictionaries from measurements
        # Group measurements by period and find current/historical periods
        periods: dict[str, dict[str, int]] = {}
        for m in measurements:
            period_key = f"{m['period_start_iso']}_{m['period_end_iso']}"
            if period_key not in periods:
                periods[period_key] = {}
            periods[period_key][m["metric_name"]] = m["metric_value_scaled"]

        # Sort periods and get current, 1 period ago, 4 periods ago
        sorted_periods = sorted(periods.keys(), reverse=True)

        metrics_current = periods[sorted_periods[0]] if len(sorted_periods) > 0 else {}
        metrics_1p = periods[sorted_periods[1]] if len(sorted_periods) > 1 else {}
        metrics_4p = periods[sorted_periods[4]] if len(sorted_periods) > 4 else {}

        features = extract_features(
            deal=deal,
            metrics_current=metrics_current,
            metrics_1p_ago=metrics_1p,
            metrics_4p_ago=metrics_4p,
            recent_results=list(recent_results),
            sector_encoder=get_container.get_sector_encoder(),
            region_encoder=get_container.get_region_encoder(),
        )

        model = get_container.get_model()
        features_list: list[LoanFeatures] = [features]
        probabilities = predict_probabilities(model, features_list)
        probability = probabilities[0]

        risk_tier: Literal["LOW", "MEDIUM", "HIGH"] = classify_risk_tier(probability)

        response = PredictResponse(
            deal_id=req["deal_id"],
            probability=probability,
            risk_tier=risk_tier,
        )

        body: dict[str, JSONValue] = {
            "deal_id": response["deal_id"],
            "probability": response["probability"],
            "risk_tier": response["risk_tier"],
        }
        return Response(
            content=dump_json_str(body),
            media_type="application/json",
        )

    async def _train(request: Request) -> Response:
        """Enqueue XGBoost model training job on internal deal data.

        Supports GPU training via device parameter. Class imbalance is handled
        automatically: if scale_pos_weight is omitted, it's calculated as
        (n_negative / n_positive) from the training set.

        Request body:
            learning_rate: float - Learning rate (e.g. 0.1)
            max_depth: int - Max tree depth (e.g. 6)
            n_estimators: int - Number of boosting rounds (e.g. 100)
            subsample: float - Row subsample ratio (e.g. 0.8)
            colsample_bytree: float - Column subsample ratio (e.g. 0.8)
            random_state: int - Random seed for reproducibility
            train_ratio: float - Training set ratio (e.g. 0.7)
            val_ratio: float - Validation set ratio (e.g. 0.15)
            test_ratio: float - Test set ratio (e.g. 0.15)
            early_stopping_rounds: int - Stop if no improvement (e.g. 10)
            device: "cpu" | "cuda" | "auto" - "cpu" forces CPU, "cuda" forces GPU,
                "auto" uses GPU if available (default "auto")
            reg_alpha: float - L1 regularization (default 0.0)
            reg_lambda: float - L2 regularization (default 1.0)
            scale_pos_weight: float (optional) - Auto-calculated as
                (n_negative / n_positive) if omitted

        Returns:
            202 with {job_id, status: "queued"}. Poll /ml/jobs/{job_id} for results.
        """
        body_bytes = await request.body()
        config: TrainConfig = parse_train_request(body_bytes)

        queue = get_container.rq_queue()
        payload: dict[str, JSONValue] = {
            "learning_rate": config["learning_rate"],
            "max_depth": config["max_depth"],
            "n_estimators": config["n_estimators"],
            "subsample": config["subsample"],
            "colsample_bytree": config["colsample_bytree"],
            "random_state": config["random_state"],
            "train_ratio": config["train_ratio"],
            "val_ratio": config["val_ratio"],
            "test_ratio": config["test_ratio"],
            "early_stopping_rounds": config["early_stopping_rounds"],
            "reg_alpha": config["reg_alpha"],
            "reg_lambda": config["reg_lambda"],
            "device": config["device"],
        }
        scale_pos_weight = config.get("scale_pos_weight")
        if scale_pos_weight is not None:
            payload["scale_pos_weight"] = scale_pos_weight

        config_json = dump_json_str(payload)
        job = queue.enqueue(
            "covenant_radar_api.worker.train_job.process_train_job",
            config_json,
            job_timeout=3600,
            result_ttl=86400,
            failure_ttl=86400,
            description="Covenant ML model training",
        )

        response = TrainResponse(job_id=job.get_id(), status="queued")
        body: dict[str, JSONValue] = {"job_id": response["job_id"], "status": response["status"]}
        return Response(
            content=dump_json_str(body),
            media_type="application/json",
            status_code=202,
        )

    def _get_model_info() -> Response:
        """Get information about the active ML model.

        Returns:
            JSON object with model_id, model_path, is_loaded.
        """
        info = get_container.get_model_info()
        body: dict[str, JSONValue] = {
            "model_id": info["model_id"],
            "model_path": info["model_path"],
            "is_loaded": info["is_loaded"],
        }
        return Response(
            content=dump_json_str(body),
            media_type="application/json",
        )

    def _get_job_status(job_id: str) -> Response:
        """Get status of a background job.

        Args:
            job_id: The job UUID string

        Returns:
            JSON object with job_id, status, and result (if finished).
        """
        status = get_container.get_job_status(job_id)
        body: dict[str, JSONValue] = {
            "job_id": status["job_id"],
            "status": status["status"],
        }
        if status["result"] is not None:
            body["result"] = status["result"]
        return Response(
            content=dump_json_str(body),
            media_type="application/json",
        )

    async def _train_external(request: Request) -> Response:
        """Train XGBoost on external bankruptcy datasets with automatic feature selection.

        Supports GPU training via device parameter. XGBoost trains on ALL columns
        and ranks features by importance. Class imbalance is handled automatically:
        if scale_pos_weight is omitted, it's calculated as (n_negative / n_positive).

        Available datasets:
            - taiwan: 6,819 samples, 95 financial ratio features
            - us: 78,682 samples, 18 features
            - polish: 7,027 samples, 64 financial ratio features

        Request body:
            dataset: "taiwan" | "us" | "polish" - Which dataset to train on
            learning_rate: float - Learning rate (e.g. 0.1)
            max_depth: int - Max tree depth (e.g. 6)
            n_estimators: int - Number of boosting rounds (e.g. 100)
            subsample: float - Row subsample ratio (e.g. 0.8)
            colsample_bytree: float - Column subsample ratio (e.g. 0.8)
            random_state: int - Random seed for reproducibility
            device: "cpu" | "cuda" | "auto" - "cpu" forces CPU, "cuda" forces GPU,
                "auto" uses GPU if available (default "auto")
            reg_alpha: float (optional, default 0.0) - L1 regularization
            reg_lambda: float (optional, default 1.0) - L2 regularization
            scale_pos_weight: float (optional) - Auto-calculated as
                (n_negative / n_positive) if omitted

        Returns:
            202 with {job_id, status: "queued"}. Poll /ml/jobs/{job_id} for results
            including feature_importances ranking and scale_pos_weight used.
        """
        body_bytes = await request.body()
        config_json = body_bytes.decode("utf-8")

        queue = get_container.rq_queue()
        job = queue.enqueue(
            "covenant_radar_api.worker.train_external_job.process_external_train_job",
            config_json,
            job_timeout=3600,
            result_ttl=86400,
            failure_ttl=86400,
            description="External data ML training with automatic feature selection",
        )

        body: dict[str, JSONValue] = {"job_id": job.get_id(), "status": "queued"}
        return Response(
            content=dump_json_str(body),
            media_type="application/json",
            status_code=202,
        )

    router.add_api_route("/predict", _predict, methods=["POST"], response_model=None)
    router.add_api_route("/train", _train, methods=["POST"], response_model=None)
    router.add_api_route("/train-external", _train_external, methods=["POST"], response_model=None)
    router.add_api_route("/models/active", _get_model_info, methods=["GET"], response_model=None)
    router.add_api_route("/jobs/{job_id}", _get_job_status, methods=["GET"], response_model=None)

    return router


__all__ = ["JobStatus", "ModelInfo", "build_router"]
