from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from enum import Enum
from typing import Generic, Protocol, TypeVar, runtime_checkable

from fastapi.responses import JSONResponse as _FastAPIJSONResponse

from platform_core.logging import get_logger
from platform_core.request_context import request_id_var as _global_request_id_var


class ErrorCodeBase(str, Enum):
    """Base class for service error codes.

    This is a string enum where each member is both an Enum and a str.
    To get the string value, use: code if isinstance(code, str) else str(code)
    """

    value: str


class ErrorCode(ErrorCodeBase):
    """Standard platform error codes for application errors.

    Convention:
    - User errors (4xx): UPPERCASE_WITH_UNDERSCORES
    - System errors (5xx): UPPERCASE_WITH_UNDERSCORES

    All codes are precise and identify a specific issue. No generic/vague codes.
    """

    # User/Client Errors (4xx) - sorted by status code
    INVALID_INPUT = "INVALID_INPUT"  # 400 - validation failed
    INVALID_JSON = "INVALID_JSON"  # 400 - JSON parse error
    UNAUTHORIZED = "UNAUTHORIZED"  # 401 - missing/invalid auth
    FORBIDDEN = "FORBIDDEN"  # 403 - insufficient permissions
    NOT_FOUND = "NOT_FOUND"  # 404 - resource not found
    JOB_NOT_FOUND = "JOB_NOT_FOUND"  # 404 - job ID doesn't exist
    CONFLICT = "CONFLICT"  # 409 - resource conflict
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"  # 413 - request body exceeds limit
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"  # 415 - wrong content type
    RANGE_NOT_SATISFIABLE = "RANGE_NOT_SATISFIABLE"  # 416 - invalid byte range
    JOB_NOT_READY = "JOB_NOT_READY"  # 425 - job still processing
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"  # 429 - too many requests

    # System Errors (5xx) - sorted by status code
    INTERNAL_ERROR = "INTERNAL_ERROR"  # 500 - unexpected server error
    DATABASE_ERROR = "DATABASE_ERROR"  # 500 - database operation failed
    CONFIG_ERROR = "CONFIG_ERROR"  # 500 - configuration missing/invalid
    JOB_FAILED = "JOB_FAILED"  # 500 - job execution failed
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"  # 502 - upstream service failed
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"  # 503 - service not ready
    TIMEOUT = "TIMEOUT"  # 504 - operation timed out
    INSUFFICIENT_STORAGE = "INSUFFICIENT_STORAGE"  # 507 - storage full


class TranscriptErrorCode(ErrorCodeBase):
    """Precise transcript service error codes (no generics)."""

    YOUTUBE_URL_REQUIRED = "YOUTUBE_URL_REQUIRED"
    YOUTUBE_URL_INVALID = "YOUTUBE_URL_INVALID"
    YOUTUBE_URL_UNSUPPORTED = "YOUTUBE_URL_UNSUPPORTED"
    YOUTUBE_VIDEO_ID_INVALID = "YOUTUBE_VIDEO_ID_INVALID"

    TRANSCRIPT_UNAVAILABLE = "TRANSCRIPT_UNAVAILABLE"
    TRANSCRIPT_LANGUAGE_UNAVAILABLE = "TRANSCRIPT_LANGUAGE_UNAVAILABLE"
    TRANSCRIPT_TRANSLATE_UNAVAILABLE = "TRANSCRIPT_TRANSLATE_UNAVAILABLE"
    TRANSCRIPT_LISTING_FAILED = "TRANSCRIPT_LISTING_FAILED"
    TRANSCRIPT_PAYLOAD_INVALID = "TRANSCRIPT_PAYLOAD_INVALID"

    STT_DURATION_UNKNOWN = "STT_DURATION_UNKNOWN"
    STT_TOO_LONG = "STT_TOO_LONG"
    STT_DOWNLOAD_FAILED = "STT_DOWNLOAD_FAILED"
    STT_CHUNKING_DISABLED = "STT_CHUNKING_DISABLED"
    STT_CHUNK_FAILED = "STT_CHUNK_FAILED"
    STT_FFMPEG_MISSING = "STT_FFMPEG_MISSING"


class HandwritingErrorCode(ErrorCodeBase):
    """Domain-specific handwriting service error codes.

    Generic errors (unauthorized, timeout, too_large, etc.) use centralized ErrorCode.
    """

    invalid_image = "invalid_image"
    bad_dimensions = "bad_dimensions"
    preprocessing_failed = "preprocessing_failed"
    malformed_multipart = "malformed_multipart"
    invalid_model = "invalid_model"


class ModelTrainerErrorCode(ErrorCodeBase):
    """Domain-specific model trainer error codes.

    Generic errors (unauthorized, timeout, etc.) use centralized ErrorCode.
    """

    # Training errors
    TRAINING_CANCELLED = "TRAINING_CANCELLED"
    TRAINING_OOM = "TRAINING_OOM"
    TRAINING_NAN_LOSS = "TRAINING_NAN_LOSS"
    TRAINING_DIVERGED = "TRAINING_DIVERGED"

    # Model errors
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    MODEL_INCOMPATIBLE = "MODEL_INCOMPATIBLE"
    INVALID_MODEL_SIZE = "INVALID_MODEL_SIZE"
    UNSUPPORTED_BACKEND = "UNSUPPORTED_BACKEND"

    # Tokenizer errors
    TOKENIZER_NOT_FOUND = "TOKENIZER_NOT_FOUND"
    TOKENIZER_LOAD_FAILED = "TOKENIZER_LOAD_FAILED"
    TOKENIZER_TRAIN_FAILED = "TOKENIZER_TRAIN_FAILED"

    # Dataset errors
    CORPUS_NOT_FOUND = "CORPUS_NOT_FOUND"
    CORPUS_EMPTY = "CORPUS_EMPTY"
    CORPUS_TOO_LARGE = "CORPUS_TOO_LARGE"

    # Run/Job errors
    RUN_NOT_FOUND = "RUN_NOT_FOUND"
    EVAL_NOT_FOUND = "EVAL_NOT_FOUND"
    DATA_NOT_FOUND = "DATA_NOT_FOUND"
    LOGS_READ_FAILED = "LOGS_READ_FAILED"

    # Infrastructure errors
    CUDA_NOT_AVAILABLE = "CUDA_NOT_AVAILABLE"
    CUDA_OOM = "CUDA_OOM"
    ARTIFACT_UPLOAD_FAILED = "ARTIFACT_UPLOAD_FAILED"
    ARTIFACT_DOWNLOAD_FAILED = "ARTIFACT_DOWNLOAD_FAILED"


ErrorCodeType = TypeVar("ErrorCodeType", bound=ErrorCodeBase)


class AppError(Exception, Generic[ErrorCodeType]):
    """Base application error with structured error code and HTTP status.

    Attributes:
        code: Machine-readable error code
        message: Human-readable error message
        http_status: HTTP status code to return

    Example:
        >>> raise AppError(
        ...     code=ErrorCode.NOT_FOUND,
        ...     message="User not found",
        ...     http_status=404
        ... )
    """

    def __init__(self, code: ErrorCodeType, message: str, http_status: int | None = None) -> None:
        """Initialize AppError.

        Args:
            code: Error code enum value
            message: Human-readable error message
            http_status: Optional HTTP status code (defaults based on error code category)
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status if http_status is not None else _default_status_for(code)


_ERROR_CODE_STATUS: dict[ErrorCode, int] = {
    # User/Client Errors (4xx) - sorted by status code
    ErrorCode.INVALID_INPUT: 400,
    ErrorCode.INVALID_JSON: 400,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.JOB_NOT_FOUND: 404,
    ErrorCode.CONFLICT: 409,
    ErrorCode.PAYLOAD_TOO_LARGE: 413,
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: 415,
    ErrorCode.RANGE_NOT_SATISFIABLE: 416,
    ErrorCode.JOB_NOT_READY: 425,
    ErrorCode.RATE_LIMIT_EXCEEDED: 429,
    # System Errors (5xx) - sorted by status code
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.DATABASE_ERROR: 500,
    ErrorCode.CONFIG_ERROR: 500,
    ErrorCode.JOB_FAILED: 500,
    ErrorCode.EXTERNAL_SERVICE_ERROR: 502,
    ErrorCode.SERVICE_UNAVAILABLE: 503,
    ErrorCode.TIMEOUT: 504,
    ErrorCode.INSUFFICIENT_STORAGE: 507,
}


def _default_status_for(code: ErrorCodeBase) -> int:
    """Map error codes to default HTTP status codes."""
    if isinstance(code, ErrorCode):
        return _ERROR_CODE_STATUS.get(code, 500)
    return 500


def _code_value(code: ErrorCodeBase) -> str:
    """Return the string value for any error code enum without exposing Enum repr.

    Since ErrorCodeBase inherits from str, each enum member IS a string.
    We access it directly as a string to get the value, not the Enum repr.
    """
    # ErrorCodeBase(str, Enum) members are strings, so we can return directly
    # This gives us "INVALID_INPUT" not "ErrorCode.INVALID_INPUT"
    result: str = code
    return result


@runtime_checkable
class _RequestProto(Protocol):
    """Minimal protocol for FastAPI Request."""

    @property
    def url(self) -> _URLProto: ...

    @property
    def method(self) -> str: ...


@runtime_checkable
class _URLProto(Protocol):
    """Minimal protocol for Request.url."""

    @property
    def path(self) -> str: ...


@runtime_checkable
class _JSONResponseProto(Protocol):
    """Minimal protocol for FastAPI JSONResponse."""

    def __init__(self, content: dict[str, str], status_code: int) -> None: ...

    @property
    def body(self) -> bytes | memoryview[int]: ...

    @property
    def status_code(self) -> int: ...


_ExceptionHandlerProto = Callable[[_RequestProto, Exception], Awaitable[_JSONResponseProto]]


@runtime_checkable
class _FastAPIAppProto(Protocol):
    """Minimal protocol for FastAPI application adapter.

    Services should create an adapter that wraps FastAPI and converts response types.
    See qr-api/src/qr_api/app.py for reference implementation.
    """

    def add_exception_handler(
        self,
        exc_class_or_status_code: int | type[Exception],
        handler: _ExceptionHandlerProto,
    ) -> None: ...


def install_exception_handlers(
    app: _FastAPIAppProto,
    *,
    request_id_var: ContextVar[str] | None = _global_request_id_var,
    logger_name: str = "app",
    log_user_errors: bool = True,
    internal_error_code: ErrorCodeBase = ErrorCode.INTERNAL_ERROR,
) -> None:
    """Install centralized exception handlers with platform_core logging integration.

    Registers handlers for:
    - AppError: Structured application errors
    - Exception: All unhandled exceptions

    Logging behavior:
    - User errors (4xx): Logged at INFO level without traceback
    - System errors (5xx): Logged at ERROR level with full traceback (exc_info=True)
    - Unhandled exceptions: Logged at ERROR level with full traceback

    Args:
        app: FastAPI application instance
        request_id_var: Optional ContextVar for request ID tracking
        logger_name: Logger name for error logging (default: "app")
        log_user_errors: Whether to log user errors at INFO level (default: True)

    Example:
        >>> from fastapi import FastAPI
        >>> from platform_core.errors import install_exception_handlers
        >>> from platform_core.request_context import request_id_var
        >>>
        >>> app = FastAPI()
        >>> install_exception_handlers(
        ...     app,
        ...     request_id_var=request_id_var,
        ...     logger_name="my-api"
        ... )
    """
    logger = get_logger(logger_name)

    def _json_response(*, content: dict[str, str], status_code: int) -> _JSONResponseProto:
        return _FastAPIJSONResponse(content=content, status_code=status_code)

    async def _app_error_handler(request: _RequestProto, exc: Exception) -> _JSONResponseProto:
        """Handle AppError exceptions with structured logging and response."""
        if not isinstance(exc, AppError):
            # Should not happen, but handle gracefully by delegating
            return await _unhandled_handler(request, exc)

        # Extract request ID from context if available
        rid = request_id_var.get() if request_id_var is not None else ""

        # Determine if this is a user error (4xx) or system error (5xx)
        is_user_error = exc.http_status < 500

        code_value = _code_value(exc.code)
        if is_user_error and log_user_errors:
            # Log user errors at INFO level without traceback
            logger.info(
                "user_error",
                extra={
                    "error_code": code_value,
                    "request_id": rid,
                    "error_message": exc.message,
                    "path": request.url.path,
                    "method": request.method,
                },
            )
        elif not is_user_error:
            # Log system errors at ERROR level with full traceback
            logger.error(
                "system_error",
                extra={
                    "error_code": code_value,
                    "request_id": rid,
                    "error_message": exc.message,
                    "path": request.url.path,
                    "method": request.method,
                },
                exc_info=True,
            )

        # Return structured JSON error response
        response_body: dict[str, str] = {
            "code": code_value,
            "message": exc.message,
            "request_id": rid,
        }
        return _json_response(content=response_body, status_code=exc.http_status)

    async def _unhandled_handler(request: _RequestProto, exc: Exception) -> _JSONResponseProto:
        """Handle all unhandled exceptions with full logging."""
        # Extract request ID from context if available
        rid = request_id_var.get() if request_id_var is not None else ""

        # Log all unhandled exceptions at ERROR level with full traceback
        logger.error(
            "unhandled_exception",
            extra={
                "error_type": type(exc).__name__,
                "request_id": rid,
                "path": request.url.path,
                "method": request.method,
                "error_message": str(exc),
            },
            exc_info=True,
        )

        # Return generic error response (don't expose internal details)
        code_value_internal = _code_value(internal_error_code)
        response_body: dict[str, str] = {
            "code": code_value_internal,
            "message": "Internal server error",
            "request_id": rid,
        }
        return _json_response(content=response_body, status_code=500)

    # Register handlers with FastAPI
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(Exception, _unhandled_handler)


def error_body(code: str, message: str, request_id: str | None) -> dict[str, str | None]:
    """Standard error payload for platform services."""
    return {"code": code, "message": message, "request_id": request_id}


_HANDWRITING_STATUS: dict[HandwritingErrorCode, int] = {
    HandwritingErrorCode.invalid_image: 400,
    HandwritingErrorCode.bad_dimensions: 400,
    HandwritingErrorCode.preprocessing_failed: 400,
    HandwritingErrorCode.malformed_multipart: 400,
    HandwritingErrorCode.invalid_model: 400,
}

_HANDWRITING_MESSAGE: dict[HandwritingErrorCode, str] = {
    HandwritingErrorCode.invalid_image: "Failed to decode image.",
    HandwritingErrorCode.bad_dimensions: "Image dimensions exceed allowed limits.",
    HandwritingErrorCode.preprocessing_failed: "Image preprocessing failed.",
    HandwritingErrorCode.malformed_multipart: "Malformed multipart body.",
    HandwritingErrorCode.invalid_model: "Invalid model file.",
}


def handwriting_status_for(code: HandwritingErrorCode) -> int:
    """HTTP status mapping for handwriting codes."""
    return _HANDWRITING_STATUS.get(code, 500)


def handwriting_error_body(
    code: HandwritingErrorCode, request_id: str, message: str | None = None
) -> dict[str, str]:
    msg = message if message is not None else _HANDWRITING_MESSAGE.get(code, "")
    return {
        "code": code.value,
        "message": msg,
        "request_id": request_id,
    }


_MODEL_TRAINER_STATUS: dict[ModelTrainerErrorCode, int] = {
    # Training errors
    ModelTrainerErrorCode.TRAINING_CANCELLED: 499,
    ModelTrainerErrorCode.TRAINING_OOM: 507,
    ModelTrainerErrorCode.TRAINING_NAN_LOSS: 500,
    ModelTrainerErrorCode.TRAINING_DIVERGED: 500,
    # Model errors
    ModelTrainerErrorCode.MODEL_NOT_FOUND: 404,
    ModelTrainerErrorCode.MODEL_LOAD_FAILED: 500,
    ModelTrainerErrorCode.MODEL_INCOMPATIBLE: 400,
    ModelTrainerErrorCode.INVALID_MODEL_SIZE: 400,
    ModelTrainerErrorCode.UNSUPPORTED_BACKEND: 400,
    # Tokenizer errors
    ModelTrainerErrorCode.TOKENIZER_NOT_FOUND: 404,
    ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED: 500,
    ModelTrainerErrorCode.TOKENIZER_TRAIN_FAILED: 500,
    # Dataset errors
    ModelTrainerErrorCode.CORPUS_NOT_FOUND: 404,
    ModelTrainerErrorCode.CORPUS_EMPTY: 400,
    ModelTrainerErrorCode.CORPUS_TOO_LARGE: 413,
    # Run/Job errors
    ModelTrainerErrorCode.RUN_NOT_FOUND: 404,
    ModelTrainerErrorCode.EVAL_NOT_FOUND: 404,
    ModelTrainerErrorCode.DATA_NOT_FOUND: 404,
    ModelTrainerErrorCode.LOGS_READ_FAILED: 500,
    # Infrastructure errors
    ModelTrainerErrorCode.CUDA_NOT_AVAILABLE: 503,
    ModelTrainerErrorCode.CUDA_OOM: 507,
    ModelTrainerErrorCode.ARTIFACT_UPLOAD_FAILED: 502,
    ModelTrainerErrorCode.ARTIFACT_DOWNLOAD_FAILED: 502,
}


def model_trainer_status_for(code: ModelTrainerErrorCode) -> int:
    """HTTP status mapping for model trainer codes."""
    return _MODEL_TRAINER_STATUS.get(code, 500)


__all__ = [
    "AppError",
    "ErrorCode",
    "ErrorCodeBase",
    "HandwritingErrorCode",
    "ModelTrainerErrorCode",
    "TranscriptErrorCode",
    "_ExceptionHandlerProto",
    "_JSONResponseProto",
    "_RequestProto",
    "error_body",
    "handwriting_error_body",
    "handwriting_status_for",
    "install_exception_handlers",
    "model_trainer_status_for",
]
