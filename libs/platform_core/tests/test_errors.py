from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Protocol, runtime_checkable

import pytest

from platform_core.errors import (
    AppError,
    ErrorCode,
    ErrorCodeBase,
    _JSONResponseProto,
    _RequestProto,
    install_exception_handlers,
)
from platform_core.json_utils import load_json_bytes
from platform_core.logging import stdlib_logging
from platform_core.request_context import request_id_var


def _parse_response_body(response: _JSONResponseProto) -> dict[str, str]:
    """Parse JSON response body from JSONResponse."""
    body_bytes = response.body if isinstance(response.body, bytes) else bytes(response.body)
    content = load_json_bytes(body_bytes)
    assert isinstance(content, dict)
    # Validate all values are strings
    result: dict[str, str] = {}
    for key, value in content.items():
        assert isinstance(key, str)
        assert isinstance(value, str)
        result[key] = value
    return result


def test_error_code_user_errors_400() -> None:
    """Test ErrorCode enum contains 400-level user error codes."""
    assert ErrorCode.INVALID_INPUT == "INVALID_INPUT"
    assert ErrorCode.INVALID_JSON == "INVALID_JSON"


def test_error_code_user_errors_401_403() -> None:
    """Test ErrorCode enum contains 401/403-level user error codes."""
    assert ErrorCode.UNAUTHORIZED == "UNAUTHORIZED"
    assert ErrorCode.FORBIDDEN == "FORBIDDEN"


def test_error_code_user_errors_404() -> None:
    """Test ErrorCode enum contains 404-level user error codes."""
    assert ErrorCode.NOT_FOUND == "NOT_FOUND"
    assert ErrorCode.JOB_NOT_FOUND == "JOB_NOT_FOUND"


def test_error_code_user_errors_4xx_other() -> None:
    """Test ErrorCode enum contains other 4xx user error codes."""
    assert ErrorCode.CONFLICT == "CONFLICT"
    assert ErrorCode.PAYLOAD_TOO_LARGE == "PAYLOAD_TOO_LARGE"
    assert ErrorCode.UNSUPPORTED_MEDIA_TYPE == "UNSUPPORTED_MEDIA_TYPE"
    assert ErrorCode.RANGE_NOT_SATISFIABLE == "RANGE_NOT_SATISFIABLE"
    assert ErrorCode.JOB_NOT_READY == "JOB_NOT_READY"
    assert ErrorCode.RATE_LIMIT_EXCEEDED == "RATE_LIMIT_EXCEEDED"


def test_error_code_system_errors_500() -> None:
    """Test ErrorCode enum contains 500-level system error codes."""
    assert ErrorCode.INTERNAL_ERROR == "INTERNAL_ERROR"
    assert ErrorCode.DATABASE_ERROR == "DATABASE_ERROR"
    assert ErrorCode.CONFIG_ERROR == "CONFIG_ERROR"
    assert ErrorCode.JOB_FAILED == "JOB_FAILED"


def test_error_code_system_errors_5xx_other() -> None:
    """Test ErrorCode enum contains other 5xx system error codes."""
    assert ErrorCode.EXTERNAL_SERVICE_ERROR == "EXTERNAL_SERVICE_ERROR"
    assert ErrorCode.SERVICE_UNAVAILABLE == "SERVICE_UNAVAILABLE"
    assert ErrorCode.TIMEOUT == "TIMEOUT"
    assert ErrorCode.INSUFFICIENT_STORAGE == "INSUFFICIENT_STORAGE"


def test_app_error_basic() -> None:
    """Test AppError basic initialization."""
    error = AppError(
        code=ErrorCode.NOT_FOUND,
        message="Resource not found",
    )

    assert error.code == ErrorCode.NOT_FOUND
    assert error.message == "Resource not found"
    assert error.http_status == 404
    assert str(error) == "Resource not found"


def test_app_error_with_explicit_status() -> None:
    """Test AppError initialization with explicit HTTP status."""
    error = AppError(
        code=ErrorCode.NOT_FOUND,
        message="Custom not found",
        http_status=410,
    )

    assert error.code == ErrorCode.NOT_FOUND
    assert error.message == "Custom not found"
    assert error.http_status == 410


def test_app_error_default_status_user_errors_4xx() -> None:
    """Test AppError uses correct default status for all 4xx user errors."""
    test_cases: list[tuple[ErrorCode, int]] = [
        # 400 - Bad Request
        (ErrorCode.INVALID_INPUT, 400),
        (ErrorCode.INVALID_JSON, 400),
        # 401 - Unauthorized
        (ErrorCode.UNAUTHORIZED, 401),
        # 403 - Forbidden
        (ErrorCode.FORBIDDEN, 403),
        # 404 - Not Found
        (ErrorCode.NOT_FOUND, 404),
        (ErrorCode.JOB_NOT_FOUND, 404),
        # 409 - Conflict
        (ErrorCode.CONFLICT, 409),
        # 413 - Payload Too Large
        (ErrorCode.PAYLOAD_TOO_LARGE, 413),
        # 415 - Unsupported Media Type
        (ErrorCode.UNSUPPORTED_MEDIA_TYPE, 415),
        # 416 - Range Not Satisfiable
        (ErrorCode.RANGE_NOT_SATISFIABLE, 416),
        # 425 - Too Early (job not ready)
        (ErrorCode.JOB_NOT_READY, 425),
        # 429 - Too Many Requests
        (ErrorCode.RATE_LIMIT_EXCEEDED, 429),
    ]

    for code, expected_status in test_cases:
        error = AppError(code=code, message="test")
        assert error.http_status == expected_status, f"Expected {expected_status} for {code}"


def test_app_error_default_status_system_errors_5xx() -> None:
    """Test AppError uses correct default status for all 5xx system errors."""
    test_cases: list[tuple[ErrorCode, int]] = [
        # 500 - Internal Server Error
        (ErrorCode.INTERNAL_ERROR, 500),
        (ErrorCode.DATABASE_ERROR, 500),
        (ErrorCode.CONFIG_ERROR, 500),
        (ErrorCode.JOB_FAILED, 500),
        # 502 - Bad Gateway
        (ErrorCode.EXTERNAL_SERVICE_ERROR, 502),
        # 503 - Service Unavailable
        (ErrorCode.SERVICE_UNAVAILABLE, 503),
        # 504 - Gateway Timeout
        (ErrorCode.TIMEOUT, 504),
        # 507 - Insufficient Storage
        (ErrorCode.INSUFFICIENT_STORAGE, 507),
    ]

    for code, expected_status in test_cases:
        error = AppError(code=code, message="test")
        assert error.http_status == expected_status, f"Expected {expected_status} for {code}"


@runtime_checkable
class _MockURL(Protocol):
    @property
    def path(self) -> str: ...


@runtime_checkable
class _MockRequest(Protocol):
    @property
    def url(self) -> _MockURL: ...

    @property
    def method(self) -> str: ...


class MockURL:
    def __init__(self, path: str) -> None:
        self._path = path

    @property
    def path(self) -> str:
        return self._path


class MockRequest:
    def __init__(self, path: str, method: str) -> None:
        self._url = MockURL(path)
        self._method = method

    @property
    def url(self) -> MockURL:
        return self._url

    @property
    def method(self) -> str:
        return self._method


class MockJSONResponse:
    def __init__(self, content: dict[str, str], status_code: int) -> None:
        self.content = content
        self.status_code = status_code


@runtime_checkable
class _ExceptionHandlerProto(Protocol):
    """Protocol for exception handler callable."""

    async def __call__(self, request: _MockRequest, exc: Exception) -> MockJSONResponse: ...


class MockFastAPIApp:
    def __init__(self) -> None:
        self.handlers: dict[
            type[Exception], Callable[[_RequestProto, Exception], Awaitable[_JSONResponseProto]]
        ] = {}

    def add_exception_handler(
        self,
        exc_class_or_status_code: int | type[Exception],
        handler: Callable[[_RequestProto, Exception], Awaitable[_JSONResponseProto]],
    ) -> None:
        if not isinstance(exc_class_or_status_code, int):
            self.handlers[exc_class_or_status_code] = handler


def test_install_exception_handlers_registers_handlers() -> None:
    """Test install_exception_handlers registers AppError and Exception handlers."""
    app = MockFastAPIApp()

    install_exception_handlers(app)

    assert AppError in app.handlers
    assert Exception in app.handlers


def test_install_exception_handlers_app_error_user_error_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test install_exception_handlers logs user errors at INFO level."""
    app = MockFastAPIApp()
    request_id_var: ContextVar[str] = ContextVar("test_request_id", default="")
    request_id_var.set("req-123")

    caplog.set_level(stdlib_logging.INFO)

    install_exception_handlers(
        app,
        request_id_var=request_id_var,
        logger_name="test_errors",
        log_user_errors=True,
    )

    handler = app.handlers[AppError]
    assert callable(handler)

    request = MockRequest(path="/api/test", method="GET")
    error = AppError(
        code=ErrorCode.NOT_FOUND,
        message="Resource not found",
    )

    # Manually invoke the handler (it's async)
    import asyncio

    async def _run_handler() -> _JSONResponseProto:
        return await handler(request, error)

    response: _JSONResponseProto = asyncio.run(_run_handler())

    assert response.status_code == 404
    content = _parse_response_body(response)
    assert content["code"] == "NOT_FOUND"
    assert content["message"] == "Resource not found"
    assert content["request_id"] == "req-123"

    # Check logging
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "INFO"
    assert record.getMessage() == "user_error"


def test_install_exception_handlers_app_error_system_error_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test install_exception_handlers logs system errors at ERROR level with exc_info."""
    app = MockFastAPIApp()
    request_id_var: ContextVar[str] = ContextVar("test_request_id", default="")
    request_id_var.set("req-456")

    caplog.set_level(stdlib_logging.ERROR)

    install_exception_handlers(
        app,
        request_id_var=request_id_var,
        logger_name="test_errors",
    )

    handler = app.handlers[AppError]
    assert callable(handler)

    request = MockRequest(path="/api/system", method="POST")
    error = AppError(
        code=ErrorCode.INTERNAL_ERROR,
        message="System failure",
    )

    import asyncio

    async def _run_handler() -> _JSONResponseProto:
        return await handler(request, error)

    response: _JSONResponseProto = asyncio.run(_run_handler())

    assert response.status_code == 500
    content = _parse_response_body(response)
    assert content["code"] == "INTERNAL_ERROR"
    assert content["message"] == "System failure"
    assert content["request_id"] == "req-456"

    # Check logging - verify ERROR level and message, and that exc_info was requested
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "ERROR"
    assert record.getMessage() == "system_error"
    # Verify exc_info was requested (it's a tuple even if exception context is empty)
    assert record.exc_info is not None or record.exc_text is not None


def test_install_exception_handlers_app_error_no_request_id() -> None:
    """Test install_exception_handlers works without request_id_var."""
    app = MockFastAPIApp()

    install_exception_handlers(app, request_id_var=None, logger_name="test_errors")

    handler = app.handlers[AppError]
    assert callable(handler)

    request = MockRequest(path="/api/test", method="GET")
    error = AppError(code=ErrorCode.FORBIDDEN, message="Access denied")

    import asyncio

    async def _run_handler() -> _JSONResponseProto:
        return await handler(request, error)

    response: _JSONResponseProto = asyncio.run(_run_handler())

    assert response.status_code == 403
    content = _parse_response_body(response)
    assert content["request_id"] == ""


def test_install_exception_handlers_app_error_no_user_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test install_exception_handlers skips user error logging when disabled."""
    app = MockFastAPIApp()

    caplog.set_level(stdlib_logging.INFO)

    install_exception_handlers(
        app,
        request_id_var=None,
        logger_name="test_errors",
        log_user_errors=False,
    )

    handler = app.handlers[AppError]
    assert callable(handler)

    request = MockRequest(path="/api/test", method="GET")
    error = AppError(code=ErrorCode.INVALID_INPUT, message="Bad input")

    import asyncio

    async def _run_handler() -> _JSONResponseProto:
        return await handler(request, error)

    asyncio.run(_run_handler())

    # Should not log user errors
    assert len(caplog.records) == 0


def test_install_exception_handlers_unhandled_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test install_exception_handlers logs unhandled exceptions at ERROR level."""
    app = MockFastAPIApp()
    request_id_var: ContextVar[str] = ContextVar("test_request_id", default="")
    request_id_var.set("req-789")

    caplog.set_level(stdlib_logging.ERROR)

    install_exception_handlers(
        app,
        request_id_var=request_id_var,
        logger_name="test_errors",
    )

    handler = app.handlers[Exception]
    assert callable(handler)

    request = MockRequest(path="/api/crash", method="DELETE")
    error = ValueError("Unexpected error")

    import asyncio

    async def _run_handler() -> _JSONResponseProto:
        return await handler(request, error)

    response: _JSONResponseProto = asyncio.run(_run_handler())

    assert response.status_code == 500
    content = _parse_response_body(response)
    assert content["code"] == "INTERNAL_ERROR"
    assert content["message"] == "Internal server error"
    assert content["request_id"] == "req-789"

    # Check logging - verify ERROR level and message, and that exc_info was requested
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "ERROR"
    assert record.getMessage() == "unhandled_exception"
    # Verify exc_info was requested (it's a tuple even if exception context is empty)
    assert record.exc_info is not None or record.exc_text is not None


def test_install_exception_handlers_unhandled_exception_no_request_id() -> None:
    """Test install_exception_handlers handles unhandled exceptions without request_id_var."""
    app = MockFastAPIApp()

    install_exception_handlers(app, request_id_var=None, logger_name="test_errors")

    handler = app.handlers[Exception]
    assert callable(handler)

    request = MockRequest(path="/api/error", method="PUT")
    error = RuntimeError("Runtime error")

    import asyncio

    async def _run_handler() -> _JSONResponseProto:
        return await handler(request, error)

    response: _JSONResponseProto = asyncio.run(_run_handler())

    assert response.status_code == 500
    content = _parse_response_body(response)
    assert content["request_id"] == ""


def test_install_exception_handlers_app_error_handler_delegates_non_app_error() -> None:
    """Test AppError handler delegates non-AppError exceptions to unhandled handler."""
    app = MockFastAPIApp()

    install_exception_handlers(app, request_id_var=None, logger_name="test_errors")

    app_error_handler = app.handlers[AppError]
    assert callable(app_error_handler)

    request = MockRequest(path="/api/test", method="GET")
    # Pass a non-AppError exception to AppError handler
    error = ValueError("Not an AppError")

    import asyncio

    async def _run_handler() -> _JSONResponseProto:
        return await app_error_handler(request, error)

    response: _JSONResponseProto = asyncio.run(_run_handler())

    # Should delegate to unhandled handler
    assert response.status_code == 500
    content = _parse_response_body(response)
    assert content["code"] == "INTERNAL_ERROR"


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
        app = MockFastAPIApp()

        install_exception_handlers(app, logger_name="test_errors_global_default")

        handler = app.handlers[Exception]
        request = MockRequest(path="/api/crash", method="DELETE")

        import asyncio

        async def _run_handler() -> _JSONResponseProto:
            return await handler(request, RuntimeError("boom"))

        response: _JSONResponseProto = asyncio.run(_run_handler())

        content = _parse_response_body(response)
        assert content["request_id"] == "global-req-123"
    finally:
        request_id_var.reset(token)


def test_install_exception_handlers_custom_internal_error_code() -> None:
    """Unhandled exceptions use provided internal_error_code."""

    class CustomInternal(ErrorCodeBase):
        INTERNAL = "CUSTOM_INTERNAL_ERROR"

    app = MockFastAPIApp()
    install_exception_handlers(
        app,
        request_id_var=None,
        logger_name="test_errors_custom_internal",
        internal_error_code=CustomInternal.INTERNAL,
    )

    handler = app.handlers[Exception]
    request = MockRequest(path="/api/custom", method="GET")

    import asyncio

    async def _run_handler() -> _JSONResponseProto:
        return await handler(request, RuntimeError("unexpected"))

    response: _JSONResponseProto = asyncio.run(_run_handler())

    assert response.status_code == 500
    content = _parse_response_body(response)
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
