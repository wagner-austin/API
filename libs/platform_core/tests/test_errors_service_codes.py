"""Tests for errors: ServiceErrorCode."""

from __future__ import annotations

import pytest

from platform_core.errors import (
    AppError,
    ErrorCodeBase,
    _JSONResponseProto,
    install_exception_handlers,
)
from platform_core.logging import stdlib_logging
from platform_core.request_context import request_id_var
from tests._error_helpers import (
    FakeFastAPIApp,
    FakeRequest,
    parse_response_body,
)


class ServiceErrorCode(ErrorCodeBase):
    ITEM_MISSING = "ITEM_MISSING"


def test_app_error_supports_custom_error_code_base() -> None:
    """AppError accepts custom ErrorCodeBase implementations."""
    err = AppError(ServiceErrorCode.ITEM_MISSING, "missing item")
    assert err.code is ServiceErrorCode.ITEM_MISSING
    assert err.message == "missing item"
    assert err.http_status == 500


def test_install_exception_handlers_defaults_to_global_request_id_var(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Default request_id_var is platform_core.request_context.request_id_var."""
    token = request_id_var.set("global-req-123")
    caplog.set_level(stdlib_logging.ERROR)
    try:
        app = FakeFastAPIApp()

        install_exception_handlers(app, logger_name="test_errors_global_default")

        handler = app.handlers[Exception]
        request = FakeRequest(path="/api/crash", method="DELETE")

        import asyncio

        async def _run_handler() -> _JSONResponseProto:
            return await handler(request, RuntimeError("boom"))

        response: _JSONResponseProto = asyncio.run(_run_handler())

        content = parse_response_body(response)
        assert content["request_id"] == "global-req-123"
    finally:
        request_id_var.reset(token)


def test_install_exception_handlers_custom_internal_error_code() -> None:
    """Unhandled exceptions use provided internal_error_code."""

    class CustomInternal(ErrorCodeBase):
        INTERNAL = "CUSTOM_INTERNAL_ERROR"

    app = FakeFastAPIApp()
    install_exception_handlers(
        app,
        request_id_var=None,
        logger_name="test_errors_custom_internal",
        internal_error_code=CustomInternal.INTERNAL,
    )

    handler = app.handlers[Exception]
    request = FakeRequest(path="/api/custom", method="GET")

    import asyncio

    async def _run_handler() -> _JSONResponseProto:
        return await handler(request, RuntimeError("unexpected"))

    response: _JSONResponseProto = asyncio.run(_run_handler())

    assert response.status_code == 500
    content = parse_response_body(response)
    assert content["code"] == "CUSTOM_INTERNAL_ERROR"


def test_error_body() -> None:
    """Test error_body creates correct payload structure."""
    from platform_core.errors import error_body

    result = error_body("TEST_CODE", "test message", "req-123")
    assert result == {"code": "TEST_CODE", "message": "test message", "request_id": "req-123"}

    result_no_id = error_body("ERR", "msg", None)
    assert result_no_id == {"code": "ERR", "message": "msg", "request_id": None}


def test_handwriting_status_for() -> None:
    """Test handwriting_status_for returns correct HTTP status codes for domain-specific codes."""
    from platform_core.errors import HandwritingErrorCode, handwriting_status_for

    assert handwriting_status_for(HandwritingErrorCode.invalid_image) == 400
    assert handwriting_status_for(HandwritingErrorCode.bad_dimensions) == 400
    assert handwriting_status_for(HandwritingErrorCode.preprocessing_failed) == 400
    assert handwriting_status_for(HandwritingErrorCode.malformed_multipart) == 400
    assert handwriting_status_for(HandwritingErrorCode.invalid_model) == 400


def test_handwriting_error_body() -> None:
    """Test handwriting_error_body creates correct payload for domain-specific codes."""
    from platform_core.errors import HandwritingErrorCode, handwriting_error_body

    # Test with default message
    result = handwriting_error_body(HandwritingErrorCode.invalid_image, "req-456")
    assert result["code"] == "invalid_image"
    assert result["message"] == "Failed to decode image."
    assert result["request_id"] == "req-456"

    # Test with custom message
    result_custom = handwriting_error_body(
        HandwritingErrorCode.preprocessing_failed, "req-789", message="Custom error"
    )
    assert result_custom["code"] == "preprocessing_failed"
    assert result_custom["message"] == "Custom error"
    assert result_custom["request_id"] == "req-789"


def test_model_trainer_status_for() -> None:
    """Test model_trainer_status_for returns correct HTTP status codes for domain-specific codes."""
    from platform_core.errors import ModelTrainerErrorCode, model_trainer_status_for

    # Training errors
    assert model_trainer_status_for(ModelTrainerErrorCode.TRAINING_CANCELLED) == 499
    assert model_trainer_status_for(ModelTrainerErrorCode.TRAINING_OOM) == 507
    assert model_trainer_status_for(ModelTrainerErrorCode.TRAINING_NAN_LOSS) == 500
    assert model_trainer_status_for(ModelTrainerErrorCode.TRAINING_DIVERGED) == 500

    # Model errors
    assert model_trainer_status_for(ModelTrainerErrorCode.MODEL_NOT_FOUND) == 404
    assert model_trainer_status_for(ModelTrainerErrorCode.MODEL_LOAD_FAILED) == 500
    assert model_trainer_status_for(ModelTrainerErrorCode.MODEL_INCOMPATIBLE) == 400
    assert model_trainer_status_for(ModelTrainerErrorCode.INVALID_MODEL_SIZE) == 400
    assert model_trainer_status_for(ModelTrainerErrorCode.UNSUPPORTED_BACKEND) == 400

    # Tokenizer errors
    assert model_trainer_status_for(ModelTrainerErrorCode.TOKENIZER_NOT_FOUND) == 404
    assert model_trainer_status_for(ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED) == 500
    assert model_trainer_status_for(ModelTrainerErrorCode.TOKENIZER_TRAIN_FAILED) == 500

    # Dataset errors
    assert model_trainer_status_for(ModelTrainerErrorCode.CORPUS_NOT_FOUND) == 404
    assert model_trainer_status_for(ModelTrainerErrorCode.CORPUS_EMPTY) == 400
    assert model_trainer_status_for(ModelTrainerErrorCode.CORPUS_TOO_LARGE) == 413

    # Run/Job errors
    assert model_trainer_status_for(ModelTrainerErrorCode.RUN_NOT_FOUND) == 404
    assert model_trainer_status_for(ModelTrainerErrorCode.EVAL_NOT_FOUND) == 404
    assert model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND) == 404
    assert model_trainer_status_for(ModelTrainerErrorCode.LOGS_READ_FAILED) == 500

    # Infrastructure errors
    assert model_trainer_status_for(ModelTrainerErrorCode.CUDA_NOT_AVAILABLE) == 503
    assert model_trainer_status_for(ModelTrainerErrorCode.CUDA_OOM) == 507
    assert model_trainer_status_for(ModelTrainerErrorCode.ARTIFACT_UPLOAD_FAILED) == 502
    assert model_trainer_status_for(ModelTrainerErrorCode.ARTIFACT_DOWNLOAD_FAILED) == 502


def test_model_trainer_error_code_enum_values() -> None:
    """Test ModelTrainerErrorCode enum contains correct string values."""
    from platform_core.errors import ModelTrainerErrorCode

    # Verify enum values match string values (ErrorCodeBase inherits from str)
    assert ModelTrainerErrorCode.TRAINING_CANCELLED == "TRAINING_CANCELLED"
    assert ModelTrainerErrorCode.MODEL_NOT_FOUND == "MODEL_NOT_FOUND"
    assert ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED == "TOKENIZER_LOAD_FAILED"
    assert ModelTrainerErrorCode.CORPUS_EMPTY == "CORPUS_EMPTY"
    assert ModelTrainerErrorCode.CUDA_OOM == "CUDA_OOM"
