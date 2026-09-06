"""What the SERVICES return: one enum per service, and the base they share.

SPLIT OUT OF :mod:`platform_core.errors` ON 2026-09-04, and the reason is a
guard rather than taste. That module reached the monorepo's 600-line ceiling,
and it had reached it by accumulating VOCABULARY -- enums naming what can go
wrong in several services -- alongside the MACHINERY that carries a code to a
caller. Those are two roles, and only one of them grows every time a service
learns a new failure.

SPLIT AGAIN ON 2026-09-06, when this module hit the same ceiling two days
later. The half that left is :mod:`platform_core.error_codes_tooling`, and the
boundary is observable rather than a matter of taste: everything there belongs
to a ``tools/*`` package and NONE of it is ever rendered as an HTTP status,
while everything here is a service's or the platform's own vocabulary. The
asymmetry was already visible next door -- ``errors.py`` has status maps for
:class:`ErrorCode`, :class:`HandwritingErrorCode` and
:class:`ModelTrainerErrorCode` and none for the tooling enums, because a
command line's refusal to a person is not a response to a request.

So the rule for a new code, and it needs no judgement: does a ``tools/*``
package raise it, and does it reach a human through a terminal rather than a
response body? Then it belongs in ``error_codes_tooling``. Otherwise here.

WHY NOT ONE MODULE PER SERVICE. Because the ban that makes this vocabulary
worth having is ``monorepo_guards.error_rules.ErrorsRule``, which forbids a
local error module anywhere outside ``platform_core``. Scattering these back
into per-service files would satisfy the line ceiling by reintroducing exactly
what that rule exists to prevent, and a reader looking for "what can this
service return" would be back to grepping. Two modules inside the package are
not that; forty scattered across the monorepo would be.

IMPORTERS DO NOT MOVE, still. ``platform_core.errors`` re-exports every name
here and pulls the two tooling enums it already re-exported from their new
home, so the files importing through it keep working untouched. No re-export
was added to THIS module for the names that left: the nineteen lines that
imported them from here directly now name the real module. A re-export that
exists only so stale imports keep resolving is not an interface, and this
workspace does not keep one.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCodeBase(StrEnum):
    """Base class for service error codes.

    A :class:`enum.StrEnum`, so a member IS its own string value: ``str(code)``
    and any f-string render the code itself rather than
    ``ClassName.MEMBER``. Under the older ``(str, Enum)`` spelling those two
    disagreed -- concatenation gave the value while ``str()`` gave the
    qualified name -- which is why the previous docstring had to tell callers
    to write ``code if isinstance(code, str) else str(code)``. They can now
    just use it.
    """


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
    CORPUS_MALFORMED_RECORD = "CORPUS_MALFORMED_RECORD"
    ADAPTER_RELOAD_MISMATCH = "ADAPTER_RELOAD_MISMATCH"

    # Run/Job errors
    RUN_NOT_FOUND = "RUN_NOT_FOUND"
    EVAL_NOT_FOUND = "EVAL_NOT_FOUND"
    DATA_NOT_FOUND = "DATA_NOT_FOUND"
    LOGS_READ_FAILED = "LOGS_READ_FAILED"

    # Cloze evaluation errors
    CLOZE_ITEMS_EMPTY = "CLOZE_ITEMS_EMPTY"
    CLOZE_ITEM_UNSCOREABLE = "CLOZE_ITEM_UNSCOREABLE"
    CLOZE_FINGERPRINT_MISSING = "CLOZE_FINGERPRINT_MISSING"

    # Fine-tuning strategy selection
    STRATEGY_NAME_UNKNOWN = "STRATEGY_NAME_UNKNOWN"

    # Cartridge (trained key-value prefix) errors.
    #
    # Split by who can fix it. An invalid geometry is a malformed artifact or
    # a nonsensical request; a geometry that does not match the model is two
    # valid objects that do not belong together, which is the mistake a
    # caller makes by pointing a saved cartridge at a different base; and a
    # model that reports no cache at all cannot host a prefix, which is a
    # property of the model rather than of anything the caller chose.
    CARTRIDGE_CONFIG_MISSING = "CARTRIDGE_CONFIG_MISSING"
    CARTRIDGE_GEOMETRY_INVALID = "CARTRIDGE_GEOMETRY_INVALID"
    CARTRIDGE_GEOMETRY_MISMATCH = "CARTRIDGE_GEOMETRY_MISMATCH"
    CARTRIDGE_MODEL_REPORTS_NO_CACHE = "CARTRIDGE_MODEL_REPORTS_NO_CACHE"
    CARTRIDGE_STATE_INCOMPLETE = "CARTRIDGE_STATE_INCOMPLETE"
    CARTRIDGE_GRADIENT_CHECKPOINTING_UNSUPPORTED = "CARTRIDGE_GRADIENT_CHECKPOINTING_UNSUPPORTED"
    CARTRIDGE_MEASUREMENT_UNREPLICATED = "CARTRIDGE_MEASUREMENT_UNREPLICATED"
    CARTRIDGE_CORPUS_UNUSABLE = "CARTRIDGE_CORPUS_UNUSABLE"

    # Knowledge-editing errors
    #
    # One code per way a weight edit can be wrong, because they are not one
    # event. A module that is absent is a configuration mistake the caller can
    # fix; a key orthogonal to the input is a property of the model at that
    # position and the same request may succeed elsewhere; a restore that
    # disagrees with its snapshot means the model in memory is no longer the
    # one the measurement was taken on, and nothing after it can be believed.
    EDIT_MODULE_NOT_FOUND = "EDIT_MODULE_NOT_FOUND"
    EDIT_WEIGHT_NOT_MATRIX = "EDIT_WEIGHT_NOT_MATRIX"
    EDIT_UPDATE_SHAPE_MISMATCH = "EDIT_UPDATE_SHAPE_MISMATCH"
    EDIT_KEY_ORTHOGONAL_TO_INPUT = "EDIT_KEY_ORTHOGONAL_TO_INPUT"
    EDIT_PROMPT_PLACEHOLDER_INVALID = "EDIT_PROMPT_PLACEHOLDER_INVALID"
    EDIT_SUBJECT_NOT_IN_PROMPT = "EDIT_SUBJECT_NOT_IN_PROMPT"
    EDIT_ACTIVATION_NOT_CAPTURED = "EDIT_ACTIVATION_NOT_CAPTURED"
    EDIT_RESTORE_MISMATCH = "EDIT_RESTORE_MISMATCH"
    EDIT_VERIFICATION_FAILED = "EDIT_VERIFICATION_FAILED"

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


__all__ = [
    "CalendarErrorCode",
    "EmailErrorCode",
    "ErrorCode",
    "ErrorCodeBase",
    "HandwritingErrorCode",
    "ModelTrainerErrorCode",
    "OAuthErrorCode",
    "TranscriptErrorCode",
]
