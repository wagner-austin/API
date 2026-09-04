"""ML prediction and training endpoints."""

from __future__ import annotations

from typing import Literal

from covenant_domain import DealId
from covenant_domain.features import (
    LoanFeatures,
    classify_risk_tier,
    extract_features,
)
from covenant_ml.predictor import predict_probabilities
from covenant_ml.types import TrainConfig
from fastapi import APIRouter, Request, Response
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str

from covenant_radar_api.api.routes.ml_analysis import (
    _register_explain,
    _register_explain_regression,
    _register_job_status,
    _register_model_info,
    _register_optimize,
    _register_optimize_regression,
    _register_predict_regression,
)
from covenant_radar_api.api.routes.ml_types import (
    ContainerProtocol,
    JobStatus,
    ModelInfo,
)

from ..decode import (
    PredictResponse,
    TrainResponse,
    parse_external_regression_train_request,
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


def _register_predict(router: APIRouter, get_container: ContainerProtocol) -> None:
    async def _predict(request: Request) -> Response:
        """Predict breach risk for a deal.

        Request body:
            deal_id: Deal UUID string

        Returns:
            JSON object with probability and risk_tier.

        Raises:
            AppError: NOT_FOUND (404) if the deal does not exist.
            KeyError: If a required metric is missing from the measurements,
                which is a data defect and surfaces as a 500.
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

        risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = classify_risk_tier(probability)

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
            "Returns probability score (0.0-1.0) and risk tier (LOW/MEDIUM/HIGH/CRITICAL)."
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

        Supports all 7 classifier backends via the 'backend' field:
        xgboost, lightgbm, cleargbm, logreg, random_forest, mlp, lstm.
        Performs automatic feature selection using model importance.
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
            "ML backends. Supports xgboost, lightgbm, cleargbm, logreg, random_forest, "
            "mlp, and lstm. GPU training supported via device parameter for tree "
            "boosters and neural network backends."
        ),
        response_description="Job ID for polling status",
        responses=_TRAIN_EXTERNAL_RESPONSES,
    )


_TRAIN_EXTERNAL_REGRESSION_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    202: {
        "description": "Regression training job queued",
        "content": {
            "application/json": {
                "example": {"job_id": "train-reg-job-uuid", "status": "queued"},
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


def _register_train_external_regression(
    router: APIRouter, get_container: ContainerProtocol
) -> None:
    async def _train_external_regression(request: Request) -> Response:
        """Train regressor on external regression datasets.

        Supports xgboost_reg and lightgbm_reg backends via the 'backend' field.
        Performs regression training on continuous target variables.
        """
        body_bytes = await request.body()
        # Validate request at the API edge
        try:
            _ = parse_external_regression_train_request(body_bytes)
        except ValueError as exc:
            raise AppError(code=ErrorCode.INVALID_INPUT, message=str(exc), http_status=400) from exc
        except JSONTypeError as exc:
            raise AppError(code=ErrorCode.INVALID_INPUT, message=str(exc), http_status=400) from exc
        config_json = body_bytes.decode("utf-8")

        queue = get_container.rq_queue()
        job = queue.enqueue(
            "covenant_radar_api.worker.train_external_regression_job."
            "process_external_regression_train_job",
            config_json,
            job_timeout=3600,
            result_ttl=86400,
            failure_ttl=86400,
            description="External regression data ML training",
        )

        body: dict[str, JSONValue] = {
            "job_id": job.get_id(),
            "status": "queued",
        }
        return Response(
            content=dump_json_str(body),
            media_type="application/json",
            status_code=202,
        )

    router.add_api_route(
        "/train-external-regression",
        _train_external_regression,
        methods=["POST"],
        response_model=None,
        status_code=202,
        summary="Train regressor on external datasets",
        description=(
            "Train on external regression datasets with pluggable "
            "regressor backends. Supports xgboost_reg and lightgbm_reg. "
            "Uses continuous target variables (not classification)."
        ),
        response_description="Job ID for polling status",
        responses=_TRAIN_EXTERNAL_REGRESSION_RESPONSES,
    )


def build_router(get_container: ContainerProtocol) -> APIRouter:
    """Build FastAPI router for ML operations."""
    router = APIRouter(prefix="/ml", tags=["ml"])
    _register_predict(router, get_container)
    _register_predict_regression(router, get_container)
    _register_train(router, get_container)
    _register_train_external(router, get_container)
    _register_train_external_regression(router, get_container)
    _register_optimize(router, get_container)
    _register_optimize_regression(router, get_container)
    _register_explain(router, get_container)
    _register_explain_regression(router, get_container)
    _register_model_info(router, get_container)
    _register_job_status(router, get_container)
    return router


__all__ = ["JobStatus", "ModelInfo", "build_router"]
