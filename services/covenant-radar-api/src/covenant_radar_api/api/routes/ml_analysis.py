"""ML route registrations: optimize, explain, model info, regression."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str

from covenant_radar_api.api.decode_ml import (
    ExplainResponse,
    OptimizeResponse,
    parse_explain_request,
    parse_optimize_request,
)
from covenant_radar_api.api.decode_regression import (
    RegressionExplainResponse,
    RegressionPredictResponse,
    parse_regression_explain_request,
    parse_regression_optimize_request,
    parse_regression_predict_request,
)
from covenant_radar_api.api.routes.ml_types import ContainerProtocol

from ...core.model_paths import resolve_model_path

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

_EXPLAIN_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    202: {
        "description": "Explanation job queued",
        "content": {
            "application/json": {
                "example": {"job_id": "explain-job-uuid", "status": "queued"},
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
                        "message": "explainer must be one of: permutation, gradient, "
                        "integrated_gradients, shap_tree",
                    }
                }
            }
        },
    },
}


_OPTIMIZE_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    202: {
        "description": "Optimization job queued",
        "content": {
            "application/json": {
                "example": {"job_id": "optimize-job-uuid", "status": "queued"},
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
                        "message": "dataset must be one of: taiwan, us, polish",
                    }
                }
            }
        },
    },
}


def _register_optimize(router: APIRouter, get_container: ContainerProtocol) -> None:
    async def _optimize(request: Request) -> Response:
        """Enqueue hyperparameter optimization job using Optuna TPE.

        Validates common fields at the API edge, then forwards the raw
        JSON body to the unified worker job. Backend-specific fields are
        parsed by the worker.
        """
        body_bytes = await request.body()
        # Validate common fields at the API edge
        try:
            parsed = parse_optimize_request(body_bytes)
        except ValueError as exc:
            raise AppError(code=ErrorCode.INVALID_INPUT, message=str(exc), http_status=400) from exc
        except JSONTypeError as exc:
            raise AppError(code=ErrorCode.INVALID_INPUT, message=str(exc), http_status=400) from exc

        # Forward raw JSON to unified worker job
        config_json = body_bytes.decode("utf-8")
        queue = get_container.rq_queue()
        job = queue.enqueue(
            "covenant_radar_api.worker.optimize_job.process_optimize_job",
            config_json,
            job_timeout=7200,
            result_ttl=86400,
            failure_ttl=86400,
            description=f"{parsed['backend']} hyperparameter optimization with Optuna TPE",
        )

        response = OptimizeResponse(job_id=job.get_id(), status="queued")
        body: dict[str, JSONValue] = {"job_id": response["job_id"], "status": response["status"]}
        return Response(
            content=dump_json_str(body),
            media_type="application/json",
            status_code=202,
        )

    router.add_api_route(
        "/optimize",
        _optimize,
        methods=["POST"],
        response_model=None,
        status_code=202,
        summary="Optimize hyperparameters with Optuna TPE",
        description=(
            "Run Bayesian hyperparameter optimization using Optuna's Tree-structured "
            "Parzen Estimator (TPE) on external bankruptcy datasets.\n\n"
            "**Supported Backends:**\n"
            "- `xgboost`: XGBoost gradient boosting (default)\n"
            "- `mlp`: Multi-layer perceptron neural network\n"
            "- `lightgbm`: LightGBM gradient boosting\n"
            "- `lstm`: Long short-term memory recurrent network\n"
            "- `cleargbm`: ClearGBM interpretable boosting\n"
            "- `logreg`: Logistic regression baseline\n"
            "- `random_forest`: Random forest ensemble\n\n"
            "**Supported Datasets:**\n"
            "- `taiwan`: Taiwan Economic Journal bankruptcy data\n"
            "- `us`: American bankruptcy dataset\n"
            "- `polish`: Polish companies dataset\n\n"
            "**Feature Engineering Presets:**\n"
            "- `none`: Original features only (default)\n"
            "- `log_only`: Original + signed log transforms\n"
            "- `ratios_only`: Original + pairwise ratios (Xi/Xj)\n"
            "- `full`: Original + ratios + products + log transforms\n\n"
            "**Job Result:**\n"
            "When complete, the job result includes:\n"
            "- Best hyperparameters found\n"
            "- Validation AUC achieved\n"
            "- Feature preset used\n"
            "- Recommended config for use with /train-external\n\n"
            "Poll /ml/jobs/{job_id} for status and results."
        ),
        response_description="Job ID for polling status",
        responses=_OPTIMIZE_RESPONSES,
    )


def _register_optimize_regression(router: APIRouter, get_container: ContainerProtocol) -> None:
    async def _optimize_regression(request: Request) -> Response:
        """Enqueue regression hyperparameter optimization job using Optuna TPE.

        Validates common fields at the API edge, then forwards the raw
        JSON body to the regression worker job.
        """
        body_bytes = await request.body()
        try:
            parsed = parse_regression_optimize_request(body_bytes)
        except ValueError as exc:
            raise AppError(code=ErrorCode.INVALID_INPUT, message=str(exc), http_status=400) from exc
        except JSONTypeError as exc:
            raise AppError(code=ErrorCode.INVALID_INPUT, message=str(exc), http_status=400) from exc

        config_json = body_bytes.decode("utf-8")
        queue = get_container.rq_queue()
        job = queue.enqueue(
            "covenant_radar_api.worker.optimize_regression_job.process_regression_optimize_job",
            config_json,
            job_timeout=7200,
            result_ttl=86400,
            failure_ttl=86400,
            description=f"{parsed['backend']} regression HPO with Optuna TPE",
        )

        response = OptimizeResponse(job_id=job.get_id(), status="queued")
        body: dict[str, JSONValue] = {"job_id": response["job_id"], "status": response["status"]}
        return Response(
            content=dump_json_str(body),
            media_type="application/json",
            status_code=202,
        )

    router.add_api_route(
        "/optimize-regression",
        _optimize_regression,
        methods=["POST"],
        response_model=None,
        status_code=202,
        summary="Optimize regression hyperparameters with Optuna TPE",
        description=(
            "Run Bayesian hyperparameter optimization using Optuna's Tree-structured "
            "Parzen Estimator (TPE) on regression datasets.\n\n"
            "**Supported Backends:**\n"
            "- `xgboost_reg`: XGBoost regressor (default)\n"
            "- `lightgbm_reg`: LightGBM regressor\n\n"
            "**Supported Datasets:**\n"
            "- `financial_distress`: Financial distress regression dataset\n\n"
            "**Job Result:**\n"
            "When complete, the job result includes:\n"
            "- Best hyperparameters found\n"
            "- Negative RMSE achieved (higher = better)\n"
            "- Feature preset used\n\n"
            "Poll /ml/jobs/{job_id} for status and results."
        ),
        response_description="Job ID for polling status",
        responses=_OPTIMIZE_RESPONSES,
    )


def _register_explain(router: APIRouter, get_container: ContainerProtocol) -> None:
    async def _explain(request: Request) -> Response:
        """Enqueue feature importance explanation job.

        Computes feature importances using pluggable explainers on trained models.
        Supports permutation, gradient, integrated_gradients, and shap_tree explainers.
        """
        body_bytes = await request.body()
        # Validate request at the API edge
        try:
            parsed = parse_explain_request(body_bytes)
        except ValueError as exc:
            raise AppError(code=ErrorCode.INVALID_INPUT, message=str(exc), http_status=400) from exc
        except JSONTypeError as exc:
            raise AppError(code=ErrorCode.INVALID_INPUT, message=str(exc), http_status=400) from exc

        # Pass the raw JSON to the worker
        config_json = body_bytes.decode("utf-8")

        queue = get_container.rq_queue()
        job = queue.enqueue(
            "covenant_radar_api.worker.explain_job.process_explain_job",
            config_json,
            job_timeout=3600,  # 1 hour for explanations
            result_ttl=86400,
            failure_ttl=86400,
            description=f"Feature importance explanation ({parsed['explainer']})",
        )

        response = ExplainResponse(job_id=job.get_id(), status="queued")
        body: dict[str, JSONValue] = {"job_id": response["job_id"], "status": response["status"]}
        return Response(
            content=dump_json_str(body),
            media_type="application/json",
            status_code=202,
        )

    router.add_api_route(
        "/explain",
        _explain,
        methods=["POST"],
        response_model=None,
        status_code=202,
        summary="Compute feature importance explanations",
        description=(
            "Compute feature importances for a trained model using pluggable explainers. "
            "Supported explainers depend on the backend:\n\n"
            "**XGBoost/LightGBM backends:**\n"
            "- `permutation`: Shuffles features and measures prediction change\n"
            "- `shap_tree`: TreeSHAP values (fast, exact for tree models)\n\n"
            "**MLP/LSTM backends:**\n"
            "- `gradient`: Input gradients (fast)\n"
            "- `integrated_gradients`: Path-integrated gradients (more accurate)\n"
            "- `permutation`: Feature permutation (model-agnostic)\n\n"
            "**Job Result:**\n"
            "When complete, the job result includes:\n"
            "- Ranked feature importance scores\n"
            "- Number of samples used\n"
            "- Computation time\n\n"
            "Poll /ml/jobs/{job_id} for status and results."
        ),
        response_description="Job ID for polling status",
        responses=_EXPLAIN_RESPONSES,
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


_PREDICT_REGRESSION_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    200: {
        "description": "Successful regression prediction",
        "content": {
            "application/json": {
                "example": {
                    "backend": "xgboost_reg",
                    "predictions": [0.45, 0.82, 0.12],
                    "n_samples": 3,
                },
            },
        },
    },
    400: {
        "description": "Invalid request",
        "content": {
            "application/json": {
                "example": {
                    "error": {
                        "code": "INVALID_INPUT",
                        "message": "backend must be one of: xgboost_reg, lightgbm_reg, "
                        "mlp_reg, lstm_reg",
                    }
                }
            }
        },
    },
}


def _register_predict_regression(router: APIRouter, get_container: ContainerProtocol) -> None:
    async def _predict_regression(request: Request) -> Response:
        """Predict continuous values using a trained regressor model.

        Loads the specified regressor model from disk and runs inference
        on the provided feature matrix. Returns predicted continuous values.
        """
        import numpy as np
        from numpy.typing import NDArray

        from covenant_radar_api.worker import _regression_hooks as hooks

        body_bytes = await request.body()
        try:
            req = parse_regression_predict_request(body_bytes)
        except ValueError as exc:
            raise AppError(code=ErrorCode.INVALID_INPUT, message=str(exc), http_status=400) from exc
        except JSONTypeError as exc:
            raise AppError(code=ErrorCode.INVALID_INPUT, message=str(exc), http_status=400) from exc

        # Confine the caller-supplied path before it reaches any loader: this
        # route loads in the API process, so an unconstrained path would open
        # an arbitrary host file here.
        try:
            model_path = resolve_model_path(req["model_path"], get_container.models_root())
        except ValueError as exc:
            raise AppError(code=ErrorCode.INVALID_INPUT, message=str(exc), http_status=400) from exc

        # Load the regressor backend and model
        registry = hooks.regressor_registry_factory()
        backend = registry.get(req["backend"])
        model = backend.load(path=str(model_path))

        # Convert features to numpy array and run inference
        x: NDArray[np.float64] = np.array(req["features"], dtype=np.float64)
        predictions_array: NDArray[np.float64] = model.predict(x)
        predictions: list[float] = predictions_array.tolist()
        preds_json: list[JSONValue] = [float(v) for v in predictions]

        response = RegressionPredictResponse(
            backend=req["backend"],
            predictions=predictions,
            n_samples=len(predictions),
        )

        body: dict[str, JSONValue] = {
            "backend": response["backend"],
            "predictions": preds_json,
            "n_samples": response["n_samples"],
        }
        return Response(content=dump_json_str(body), media_type="application/json")

    router.add_api_route(
        "/predict-regression",
        _predict_regression,
        methods=["POST"],
        response_model=None,
        summary="Predict with regressor model",
        description=(
            "Predict continuous values using a trained regressor model. "
            "Provide the backend, model path, and feature matrix. "
            "Supports xgboost_reg, lightgbm_reg, mlp_reg, and lstm_reg backends."
        ),
        response_description="Predictions with continuous values",
        responses=_PREDICT_REGRESSION_RESPONSES,
    )


_EXPLAIN_REGRESSION_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    202: {
        "description": "Regression explanation job queued",
        "content": {
            "application/json": {
                "example": {
                    "job_id": "explain-reg-job-uuid",
                    "status": "queued",
                },
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
                        "message": "explainer must be one of: permutation, "
                        "gradient, integrated_gradients, shap_tree",
                    }
                }
            }
        },
    },
}


def _register_explain_regression(
    router: APIRouter,
    get_container: ContainerProtocol,
) -> None:
    async def _explain_regression(request: Request) -> Response:
        """Enqueue regression feature importance explanation job.

        Computes feature importances for trained regressor models.
        Supports permutation, gradient, integrated_gradients, and
        shap_tree explainers depending on backend.
        """
        body_bytes = await request.body()
        try:
            parsed = parse_regression_explain_request(body_bytes)
        except ValueError as exc:
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message=str(exc),
                http_status=400,
            ) from exc
        except JSONTypeError as exc:
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message=str(exc),
                http_status=400,
            ) from exc

        config_json = body_bytes.decode("utf-8")

        queue = get_container.rq_queue()
        job = queue.enqueue(
            "covenant_radar_api.worker.explain_regression_job.process_regression_explain_job",
            config_json,
            job_timeout=3600,
            result_ttl=86400,
            failure_ttl=86400,
            description=(f"Regression feature importance ({parsed['explainer']})"),
        )

        response = RegressionExplainResponse(
            job_id=job.get_id(),
            status="queued",
        )
        body: dict[str, JSONValue] = {
            "job_id": response["job_id"],
            "status": response["status"],
        }
        return Response(
            content=dump_json_str(body),
            media_type="application/json",
            status_code=202,
        )

    router.add_api_route(
        "/explain-regression",
        _explain_regression,
        methods=["POST"],
        response_model=None,
        status_code=202,
        summary="Compute regression feature importance explanations",
        description=(
            "Compute feature importances for a trained regressor model "
            "using pluggable explainers. Supported explainers depend on "
            "the backend:\n\n"
            "**XGBoost_reg/LightGBM_reg backends:**\n"
            "- `permutation`: Feature permutation (MSE change)\n"
            "- `shap_tree`: TreeSHAP values (fast, exact)\n\n"
            "**MLP_reg/LSTM_reg backends:**\n"
            "- `gradient`: Input gradients (fast)\n"
            "- `integrated_gradients`: Path-integrated gradients\n"
            "- `permutation`: Feature permutation (model-agnostic)\n\n"
            "**Job Result:**\n"
            "When complete, the job result includes:\n"
            "- Ranked feature importance scores\n"
            "- Number of samples used\n"
            "- Computation time\n\n"
            "Poll /ml/jobs/{job_id} for status and results."
        ),
        response_description="Job ID for polling status",
        responses=_EXPLAIN_REGRESSION_RESPONSES,
    )
