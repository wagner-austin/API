"""Error-code vocabulary and the exception that carries one.

Deliberately free of any web framework. The ASGI exception handlers that turn
an :class:`AppError` into a response live in :mod:`platform_core.fastapi`,
which is the module that already owns the FastAPI boundary -- this one is
imported by 304 files across the monorepo, most of which never serve HTTP,
and it has no business pulling in ``fastapi.responses`` to define an enum.
"""

from __future__ import annotations

from enum import Enum
from typing import Generic, TypeVar


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

    # Generic video URL errors
    VIDEO_URL_REQUIRED = "VIDEO_URL_REQUIRED"
    VIDEO_URL_UNSUPPORTED = "VIDEO_URL_UNSUPPORTED"

    # YouTube-specific errors
    YOUTUBE_URL_REQUIRED = "YOUTUBE_URL_REQUIRED"
    YOUTUBE_URL_INVALID = "YOUTUBE_URL_INVALID"
    YOUTUBE_URL_UNSUPPORTED = "YOUTUBE_URL_UNSUPPORTED"
    YOUTUBE_VIDEO_ID_INVALID = "YOUTUBE_VIDEO_ID_INVALID"

    # Vimeo-specific errors
    VIMEO_URL_INVALID = "VIMEO_URL_INVALID"
    VIMEO_VIDEO_ID_INVALID = "VIMEO_VIDEO_ID_INVALID"

    # Direct URL errors
    DIRECT_URL_INVALID = "DIRECT_URL_INVALID"
    DIRECT_URL_EXTENSION_INVALID = "DIRECT_URL_EXTENSION_INVALID"

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
    CORPUS_HOLDOUT_UNSATISFIABLE = "CORPUS_HOLDOUT_UNSATISFIABLE"
    CORPUS_NOT_DECODABLE = "CORPUS_NOT_DECODABLE"

    # Run/Job errors
    RUN_NOT_FOUND = "RUN_NOT_FOUND"
    EVAL_NOT_FOUND = "EVAL_NOT_FOUND"
    DATA_NOT_FOUND = "DATA_NOT_FOUND"
    LOGS_READ_FAILED = "LOGS_READ_FAILED"

    # Cloze evaluation errors
    CLOZE_ITEMS_EMPTY = "CLOZE_ITEMS_EMPTY"
    CLOZE_ITEM_UNSCOREABLE = "CLOZE_ITEM_UNSCOREABLE"
    CLOZE_FINGERPRINT_MISSING = "CLOZE_FINGERPRINT_MISSING"

    # Checkpoint / resume errors
    CHECKPOINT_NOT_FOUND = "CHECKPOINT_NOT_FOUND"
    CHECKPOINT_CORRUPT = "CHECKPOINT_CORRUPT"
    CHECKPOINT_CONFIG_MISMATCH = "CHECKPOINT_CONFIG_MISMATCH"
    CHECKPOINT_SCHEMA_UNSUPPORTED = "CHECKPOINT_SCHEMA_UNSUPPORTED"
    RUN_NOT_RESUMABLE = "RUN_NOT_RESUMABLE"
    RUN_WORKER_DIED = "RUN_WORKER_DIED"

    # Infrastructure errors
    CUDA_NOT_AVAILABLE = "CUDA_NOT_AVAILABLE"
    CUDA_OOM = "CUDA_OOM"
    ARTIFACT_UPLOAD_FAILED = "ARTIFACT_UPLOAD_FAILED"
    ARTIFACT_DOWNLOAD_FAILED = "ARTIFACT_DOWNLOAD_FAILED"


class Hpc3ErrorCode(ErrorCodeBase):
    """Slurm cluster submission and staging error codes.

    Each names one invariant of submitting work to HPC3. There is
    deliberately no generic member: a code that covers everything identifies
    nothing, and the first thing a caller does with one is re-parse the
    message string it was supposed to replace.

    These carry no meaningful HTTP status. They surface through a CLI, and
    ``_default_status_for`` returning 500 for them is correct in the sense
    that nothing consults it.
    """

    # Submission rules -- each maps to one refusal in decode_job_spec.
    GPU_TYPE_UNPINNED = "GPU_TYPE_UNPINNED"
    PARTITION_BILLS = "PARTITION_BILLS"
    PARTITION_GPU_MISMATCH = "PARTITION_GPU_MISMATCH"
    PREEMPTIBLE_RUN_UNPROTECTED = "PREEMPTIBLE_RUN_UNPROTECTED"
    TIME_LIMIT_EXCEEDS_PARTITION = "TIME_LIMIT_EXCEEDS_PARTITION"

    # Sweeps -- many jobs from one template.
    SWEEP_EXCEEDS_GPU_CEILING = "SWEEP_EXCEEDS_GPU_CEILING"
    SWEEP_EXCEEDS_CPU_CEILING = "SWEEP_EXCEEDS_CPU_CEILING"
    SWEEP_EXCEEDS_JOB_CEILING = "SWEEP_EXCEEDS_JOB_CEILING"

    # Staging -- the bytes a run is entitled to read.
    DIGEST_MISMATCH = "DIGEST_MISMATCH"
    MANIFEST_FILE_MISSING = "MANIFEST_FILE_MISSING"
    STAGED_DIGEST_UNEXPECTED = "STAGED_DIGEST_UNEXPECTED"

    # Budget -- our own share of a shared machine, capped before and during.
    BUDGET_PROJECTION_EXCEEDED = "BUDGET_PROJECTION_EXCEEDED"
    BUDGET_CONSUMPTION_EXCEEDED = "BUDGET_CONSUMPTION_EXCEEDED"

    # Preflight -- validating a job against the live scheduler before running it.
    PREFLIGHT_REJECTED = "PREFLIGHT_REJECTED"
    PREFLIGHT_UNPARSABLE = "PREFLIGHT_UNPARSABLE"
    ENV_PATH_MISSING = "ENV_PATH_MISSING"
    ENV_PACKAGE_MISMATCH = "ENV_PACKAGE_MISMATCH"
    ENV_PROBE_UNREADABLE = "ENV_PROBE_UNREADABLE"

    # Workspace configuration -- the one document every command reads.
    WORKSPACE_PROJECT_UNKNOWN = "WORKSPACE_PROJECT_UNKNOWN"
    RUN_FIELD_UNKNOWN = "RUN_FIELD_UNKNOWN"

    # Cluster selection -- which measured machine the rules come from.
    CLUSTER_UNKNOWN = "CLUSTER_UNKNOWN"
    PARTITION_UNKNOWN = "PARTITION_UNKNOWN"

    # Cluster interaction.
    REMOTE_COMMAND_FAILED = "REMOTE_COMMAND_FAILED"
    SACCT_FIELD_UNPARSABLE = "SACCT_FIELD_UNPARSABLE"


class OAuthErrorCode(ErrorCodeBase):
    """OAuth 2.0 error codes for authentication flows.

    Used by platform_core.oauth for generic OAuth operations.
    Services can catch AppError[OAuthErrorCode] and re-raise with
    their own domain-specific error codes if needed.
    """

    # Authorization errors
    AUTH_FAILED = "AUTH_FAILED"
    INVALID_GRANT = "INVALID_GRANT"
    INVALID_STATE = "INVALID_STATE"

    # Token errors
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    TOKEN_EXCHANGE_FAILED = "TOKEN_EXCHANGE_FAILED"
    TOKEN_REFRESH_FAILED = "TOKEN_REFRESH_FAILED"
    MISSING_REFRESH_TOKEN = "MISSING_REFRESH_TOKEN"

    # Network errors
    TOKEN_ENDPOINT_ERROR = "TOKEN_ENDPOINT_ERROR"


class CalendarErrorCode(ErrorCodeBase):
    """Domain-specific calendar service error codes.

    Generic errors (unauthorized, timeout, etc.) use centralized ErrorCode.
    """

    # Authentication errors
    CREDENTIALS_NOT_FOUND = "CREDENTIALS_NOT_FOUND"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    AUTH_FAILED = "AUTH_FAILED"

    # API errors
    CALENDAR_API_ERROR = "CALENDAR_API_ERROR"
    EVENT_NOT_FOUND = "EVENT_NOT_FOUND"
    CALENDAR_NOT_FOUND = "CALENDAR_NOT_FOUND"

    # Competition tracking errors
    COMPETITION_NOT_FOUND = "COMPETITION_NOT_FOUND"
    COMPETITION_ALREADY_EXISTS = "COMPETITION_ALREADY_EXISTS"
    COMPETITIONS_FILE_ERROR = "COMPETITIONS_FILE_ERROR"


class EmailErrorCode(ErrorCodeBase):
    """Domain-specific email service error codes.

    Generic errors (unauthorized, timeout, etc.) use centralized ErrorCode.
    """

    # Authentication errors
    CREDENTIALS_NOT_FOUND = "CREDENTIALS_NOT_FOUND"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    AUTH_FAILED = "AUTH_FAILED"

    # API errors
    EMAIL_API_ERROR = "EMAIL_API_ERROR"
    EMAIL_NOT_FOUND = "EMAIL_NOT_FOUND"
    FOLDER_NOT_FOUND = "FOLDER_NOT_FOUND"
    DRAFT_NOT_FOUND = "DRAFT_NOT_FOUND"

    # Send errors
    SEND_FAILED = "SEND_FAILED"
    INVALID_RECIPIENT = "INVALID_RECIPIENT"
    ATTACHMENT_TOO_LARGE = "ATTACHMENT_TOO_LARGE"

    # Provider errors
    PROVIDER_NOT_CONFIGURED = "PROVIDER_NOT_CONFIGURED"
    PROVIDER_ERROR = "PROVIDER_ERROR"


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
    ModelTrainerErrorCode.CORPUS_HOLDOUT_UNSATISFIABLE: 400,
    ModelTrainerErrorCode.CORPUS_NOT_DECODABLE: 400,
    # Run/Job errors
    ModelTrainerErrorCode.RUN_NOT_FOUND: 404,
    ModelTrainerErrorCode.EVAL_NOT_FOUND: 404,
    ModelTrainerErrorCode.DATA_NOT_FOUND: 404,
    ModelTrainerErrorCode.LOGS_READ_FAILED: 500,
    # Cloze evaluation errors
    ModelTrainerErrorCode.CLOZE_ITEMS_EMPTY: 400,
    ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE: 400,
    # 500, not 404: the record exists and claims to be completed, so the
    # server stored a number it cannot say the configuration of. That is a
    # fault in what was written, not a thing the caller asked for wrongly.
    ModelTrainerErrorCode.CLOZE_FINGERPRINT_MISSING: 500,
    # Checkpoint / resume errors
    ModelTrainerErrorCode.CHECKPOINT_NOT_FOUND: 404,
    ModelTrainerErrorCode.CHECKPOINT_CORRUPT: 500,
    ModelTrainerErrorCode.CHECKPOINT_CONFIG_MISMATCH: 409,
    ModelTrainerErrorCode.CHECKPOINT_SCHEMA_UNSUPPORTED: 409,
    ModelTrainerErrorCode.RUN_NOT_RESUMABLE: 409,
    ModelTrainerErrorCode.RUN_WORKER_DIED: 500,
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
    "CalendarErrorCode",
    "EmailErrorCode",
    "ErrorCode",
    "ErrorCodeBase",
    "HandwritingErrorCode",
    "Hpc3ErrorCode",
    "ModelTrainerErrorCode",
    "OAuthErrorCode",
    "TranscriptErrorCode",
    "error_body",
    "handwriting_error_body",
    "handwriting_status_for",
    "model_trainer_status_for",
]
