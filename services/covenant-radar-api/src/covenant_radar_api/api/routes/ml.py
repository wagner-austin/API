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
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str
from platform_workers.rq_harness import RQClientQueue

from ..decode import (
    PredictResponse,
    TrainResponse,
    parse_external_train_request,
    parse_predict_request,
    parse_train_request,
)

# OpenAPI response schemas (no type annotation for FastAPI compatibility)
_PREDICT_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    200: {
        "description": "Successful prediction",
        "content": {
            "application/json": {
                "example": {
                    "deal_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
                    "probability": 0.23,
                    "risk_tier": "LOW",
                },
            },
        },
    },
}

_TRAIN_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    202: {
        "description": "Training job queued",
        "content": {
            "application/json": {
                "example": {"job_id": "train-job-uuid", "status": "queued"},
            },
        },
    },
}

_TRAIN_EXTERNAL_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    202: {
        "description": "Training job queued",
        "content": {
            "application/json": {
                "example": {"job_id": "train-job-uuid", "status": "queued"},
            },
        },
    },
    400: {
        "description": "Invalid configuration",
        "content": {
            "application/json": {
                "example": {
                    "error": {
                        "code": "INVALID_INPUT",
                        "message": "Split ratios must sum to 1.0",
                    }
                }
            }
        },
    },
}

_MODEL_INFO_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    200: {
        "description": "Active model info",
        "content": {
            "application/json": {
                "example": {
                    "model_id": "model-2024-01-15",
                    "model_path": "/data/models/active.ubj",
                    "is_loaded": True,
                },
            },
        },
    },
}

_JOB_STATUS_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    200: {
        "description": "Job status with optional result",
        "content": {
            "application/json": {
                "example": {
                    "job_id": "train-job-uuid",
                    "status": "finished",
                    "result": {
                        "model_id": "model-2024-01-15",
                        "best_val_auc": 0.94,
                        "feature_importances": [{"name": "X6", "importance": 0.18, "rank": 1}],
                    },
                },
            },
        },
    },
}


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


def _register_predict(router: APIRouter, get_container: ContainerProtocol) -> None:
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
        return Response(content=dump_json_str(body), media_type="application/json")

    router.add_api_route(
        "/predict",
        _predict,
        methods=["POST"],
        response_model=None,
        summary="Predict breach risk",
        description=(
            "Predict covenant breach probability for a deal based on financial metrics. "
            "Returns probability score (0.0-1.0) and risk tier (LOW/MEDIUM/HIGH)."
        ),
        response_description="Prediction with probability and risk tier",
        responses=_PREDICT_RESPONSES,
    )


def _register_train(router: APIRouter, get_container: ContainerProtocol) -> None:
    async def _train(request: Request) -> Response:
        """Enqueue XGBoost model training job on internal deal data.

        Supports GPU training via device parameter. Class imbalance is handled
        automatically: if scale_pos_weight is omitted, it's calculated as
        (n_negative / n_positive) from the training set.
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

    router.add_api_route(
        "/train",
        _train,
        methods=["POST"],
        response_model=None,
        status_code=202,
        summary="Train model on internal data",
        description=(
            "Enqueue XGBoost model training job using internal deal/measurement data. "
            "Supports GPU training via device parameter ('cpu', 'cuda', 'auto'). "
            "Class imbalance is handled automatically if scale_pos_weight is omitted."
        ),
        response_description="Job ID for polling status",
        responses=_TRAIN_RESPONSES,
    )


def _register_train_external(router: APIRouter, get_container: ContainerProtocol) -> None:
    async def _train_external(request: Request) -> Response:
        """Train model on external bankruptcy datasets with pluggable backend.

        Supports both XGBoost and MLP (neural network) backends via the 'backend'
        field. Performs automatic feature selection using model importance.
        """
        body_bytes = await request.body()
        # Validate request at the API edge to prevent bad jobs from entering the queue
        try:
            _ = parse_external_train_request(body_bytes)
        except ValueError as exc:
            raise AppError(code=ErrorCode.INVALID_INPUT, message=str(exc), http_status=400) from exc
        except JSONTypeError as exc:
            raise AppError(code=ErrorCode.INVALID_INPUT, message=str(exc), http_status=400) from exc
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

    router.add_api_route(
        "/train-external",
        _train_external,
        methods=["POST"],
        response_model=None,
        status_code=202,
        summary="Train model on external datasets",
        description=(
            "Train on external bankruptcy datasets (taiwan, us, polish) with pluggable "
            "ML backends. Supports XGBoost (gradient boosting with feature importance) "
            "and MLP (neural network). GPU training supported via device parameter. "
            "XGBoost returns ranked feature importances; MLP does not."
        ),
        response_description="Job ID for polling status",
        responses=_TRAIN_EXTERNAL_RESPONSES,
    )


def _register_model_info(router: APIRouter, get_container: ContainerProtocol) -> None:
    def _get_model_info() -> Response:
        info = get_container.get_model_info()
        body: dict[str, JSONValue] = {
            "model_id": info["model_id"],
            "model_path": info["model_path"],
            "is_loaded": info["is_loaded"],
        }
        return Response(content=dump_json_str(body), media_type="application/json")

    router.add_api_route(
        "/models/active",
        _get_model_info,
        methods=["GET"],
        response_model=None,
        summary="Get active model info",
        description=(
            "Get information about the currently loaded ML model "
            "including model ID, path, and load status."
        ),
        response_description="Active model information",
        responses=_MODEL_INFO_RESPONSES,
    )


def _register_job_status(router: APIRouter, get_container: ContainerProtocol) -> None:
    def _get_job_status(job_id: str) -> Response:
        status_obj = get_container.get_job_status(job_id)
        body: dict[str, JSONValue] = {
            "job_id": status_obj["job_id"],
            "status": status_obj["status"],
        }
        if status_obj["result"] is not None:
            body["result"] = status_obj["result"]
        return Response(content=dump_json_str(body), media_type="application/json")

    router.add_api_route(
        "/jobs/{job_id}",
        _get_job_status,
        methods=["GET"],
        response_model=None,
        summary="Get job status",
        description=(
            "Get status of a background training job. Status can be: queued, started, "
            "finished, failed, or not_found. When finished, includes full training "
            "results with metrics and feature importances."
        ),
        response_description="Job status and result",
        responses=_JOB_STATUS_RESPONSES,
    )


def build_router(get_container: ContainerProtocol) -> APIRouter:
    """Build FastAPI router for ML operations."""
    router = APIRouter(prefix="/ml", tags=["ml"])
    _register_predict(router, get_container)
    _register_train(router, get_container)
    _register_train_external(router, get_container)
    _register_model_info(router, get_container)
    _register_job_status(router, get_container)
    return router


__all__ = ["JobStatus", "ModelInfo", "build_router"]
