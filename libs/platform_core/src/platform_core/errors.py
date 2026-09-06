"""The exception that carries an error code, and how one renders to a caller.

Deliberately free of any web framework. The ASGI exception handlers that turn
an :class:`AppError` into a response live in :mod:`platform_core.fastapi`,
which is the module that already owns the FastAPI boundary -- this one is
imported by 304 files across the monorepo, most of which never serve HTTP,
and it has no business pulling in ``fastapi.responses`` to define an enum.

THE VOCABULARY LIVES NEXT DOOR, in :mod:`platform_core.error_codes`, and is
re-exported here. It was split out on 2026-09-04 when this file reached the
600-line ceiling: eight service enums were growing here beside machinery that
does not grow, which is two roles in one file. The re-export is why that split
cost no importer anything -- ``from platform_core.errors import Hpc3ErrorCode``
still resolves, and every one of the 304 files was left alone.

Names re-exported rather than moved-and-rewritten because the alternative was
a mechanical edit to hundreds of files in a tree several sessions write at
once, to buy a shorter import line and nothing else.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from platform_core.error_codes import (
    CalendarErrorCode,
    EmailErrorCode,
    ErrorCode,
    ErrorCodeBase,
    HandwritingErrorCode,
    ModelTrainerErrorCode,
    OAuthErrorCode,
    TranscriptErrorCode,
)
from platform_core.error_codes_tooling import FleetErrorCode, Hpc3ErrorCode

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
    ModelTrainerErrorCode.CORPUS_MALFORMED_RECORD: 400,
    ModelTrainerErrorCode.ADAPTER_RELOAD_MISMATCH: 500,
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
    # The caller chose a name no strategy answers to, so the request is the
    # thing that is wrong.
    ModelTrainerErrorCode.STRATEGY_NAME_UNKNOWN: 400,
    # The caller asked for the cartridge strategy without the config it needs,
    # or with counts that describe nothing trainable.
    ModelTrainerErrorCode.CARTRIDGE_CONFIG_MISSING: 400,
    ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_INVALID: 400,
    # Two well-formed objects that do not belong together: a saved cartridge
    # and a base model of a different shape. 409 rather than 400 because
    # neither input is wrong on its own.
    ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_MISMATCH: 409,
    # A model that returns no key-value cache cannot host a prefix at all.
    # Nothing the caller passed is malformed, so this is not their error.
    ModelTrainerErrorCode.CARTRIDGE_MODEL_REPORTS_NO_CACHE: 500,
    # The file on disk is missing tensors its own manifest declares.
    ModelTrainerErrorCode.CARTRIDGE_STATE_INCOMPLETE: 500,
    # Two settings that are each fine and cannot both hold. 409, like the
    # geometry mismatch, because neither is wrong on its own.
    ModelTrainerErrorCode.CARTRIDGE_GRADIENT_CHECKPOINTING_UNSUPPORTED: 409,
    # A gain claimed from too few seeds to know whether it is a gain. 400
    # because the seed count is the caller's, and the fix is theirs: run it
    # again with more.
    ModelTrainerErrorCode.CARTRIDGE_MEASUREMENT_UNREPLICATED: 400,
    # A corpus that yields nothing to train or nothing to hold out. 400 for
    # the same reason: the window and the split are the caller's choices.
    ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE: 400,
    # Knowledge-editing errors. The split is 400 for a request the caller
    # composed wrongly, 409 for a request that is well formed and cannot be
    # satisfied at the named site, and 500 for a fault in what the edit did.
    ModelTrainerErrorCode.EDIT_MODULE_NOT_FOUND: 404,
    ModelTrainerErrorCode.EDIT_WEIGHT_NOT_MATRIX: 409,
    ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH: 409,
    ModelTrainerErrorCode.EDIT_KEY_ORTHOGONAL_TO_INPUT: 409,
    ModelTrainerErrorCode.EDIT_PROMPT_PLACEHOLDER_INVALID: 400,
    ModelTrainerErrorCode.EDIT_SUBJECT_NOT_IN_PROMPT: 400,
    ModelTrainerErrorCode.EDIT_ACTIVATION_NOT_CAPTURED: 500,
    ModelTrainerErrorCode.EDIT_RESTORE_MISMATCH: 500,
    ModelTrainerErrorCode.EDIT_VERIFICATION_FAILED: 500,
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
    "FleetErrorCode",
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
