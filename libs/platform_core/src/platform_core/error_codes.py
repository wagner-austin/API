"""The error-code vocabulary: one enum per service, and the base they share.

SPLIT OUT OF :mod:`platform_core.errors` ON 2026-09-04, and the reason is a
guard rather than taste. That module reached the monorepo's 600-line ceiling,
and it had reached it by accumulating VOCABULARY -- eight enums naming what can
go wrong in eight different services -- alongside the MACHINERY that carries a
code to a caller. Those are two roles, and only one of them grows every time a
service learns a new failure.

So the split is by role and not by size: codes here, and
:class:`~platform_core.errors.AppError` plus the HTTP rendering next door. The
next service to add an enum adds it here and moves the other file's line count
not at all.

WHY NOT ONE MODULE PER SERVICE. Because the ban that makes this vocabulary
worth having is ``monorepo_guards.error_rules.ErrorsRule``, which forbids a
local error module anywhere outside ``platform_core``. Scattering these back
into per-service files would satisfy the line ceiling by reintroducing exactly
what that rule exists to prevent, and a reader looking for "what can this
service return" would be back to grepping.

IMPORTERS DO NOT MOVE. ``platform_core.errors`` re-exports every name here, so
the 304 files that already import from it keep working and nothing had to be
rewritten to land this.
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
    PROJECT_UNIMAGED = "PROJECT_UNIMAGED"
    WHEEL_TAG_UNKNOWN = "WHEEL_TAG_UNKNOWN"
    GPU_TYPE_UNPINNED = "GPU_TYPE_UNPINNED"
    IMAGED_COMMAND_NEEDS_A_SHELL = "IMAGED_COMMAND_NEEDS_A_SHELL"
    RUN_REMOVES_IMAGE = "RUN_REMOVES_IMAGE"
    PARTITION_BILLS = "PARTITION_BILLS"
    PARTITION_GPU_MISMATCH = "PARTITION_GPU_MISMATCH"
    GPU_MODEL_EXHAUSTED = "GPU_MODEL_EXHAUSTED"
    PREEMPTIBLE_RUN_UNPROTECTED = "PREEMPTIBLE_RUN_UNPROTECTED"
    TIME_LIMIT_EXCEEDS_PARTITION = "TIME_LIMIT_EXCEEDS_PARTITION"

    # Sweeps -- many jobs from one template.
    SWEEP_EXCEEDS_GPU_CEILING = "SWEEP_EXCEEDS_GPU_CEILING"
    SWEEP_EXCEEDS_CPU_CEILING = "SWEEP_EXCEEDS_CPU_CEILING"
    SWEEP_EXCEEDS_JOB_CEILING = "SWEEP_EXCEEDS_JOB_CEILING"

    # Job arrays -- one sbatch call carrying a whole sweep. There is no
    # members-diverge code, deliberately: the array renderer takes the sweep
    # document itself, whose members share the template by construction, so
    # divergence is unrepresentable rather than checked.
    ARRAY_ID_UNPARSABLE = "ARRAY_ID_UNPARSABLE"
    ARRAY_INDICES_EMPTY = "ARRAY_INDICES_EMPTY"

    # Concurrency -- two jobs that would write one file.
    ARTIFACT_ALREADY_IN_FLIGHT = "ARTIFACT_ALREADY_IN_FLIGHT"

    # Campaigns -- a set of runs converging on a declared end state.
    CAMPAIGN_MEMBER_HAS_NO_ARTIFACT = "CAMPAIGN_MEMBER_HAS_NO_ARTIFACT"

    # Image builds -- the one job that is submitted from an already-rendered
    # script rather than from a run document.
    IMAGE_BUILD_SCRIPT_UNREADABLE = "IMAGE_BUILD_SCRIPT_UNREADABLE"
    IMAGE_BUILD_NAME_MISMATCH = "IMAGE_BUILD_NAME_MISMATCH"

    # Staging -- the bytes a run is entitled to read.
    DIGEST_MISMATCH = "DIGEST_MISMATCH"
    MANIFEST_FILE_MISSING = "MANIFEST_FILE_MISSING"
    STAGED_DIGEST_UNEXPECTED = "STAGED_DIGEST_UNEXPECTED"

    # Budget -- our own share of a shared machine, capped before and during.
    BUDGET_PROJECTION_EXCEEDED = "BUDGET_PROJECTION_EXCEEDED"
    BUDGET_CONSUMPTION_EXCEEDED = "BUDGET_CONSUMPTION_EXCEEDED"

    # Bootstrap -- creating the FIRST environment, which capture then probes.
    #
    # These refuse on the creating path, about what the command itself just
    # built, rather than gating a project that already runs. That distinction
    # is the point: every other code here can only ever fire at somebody who
    # has finished, which is how a system accumulates refusals and no
    # on-ramps.
    BOOTSTRAP_ENV_EXISTS = "BOOTSTRAP_ENV_EXISTS"
    BOOTSTRAP_PYTHON_MISMATCH = "BOOTSTRAP_PYTHON_MISMATCH"
    BOOTSTRAP_ENV_NOT_SELF_CONTAINED = "BOOTSTRAP_ENV_NOT_SELF_CONTAINED"

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


class FleetErrorCode(ErrorCodeBase):
    """Dispatching work to the machines on the tailnet.

    A sibling of :class:`Hpc3ErrorCode` rather than a section of it, because
    the two answer to different schedulers. HPC3 has Slurm: partitions, QOS
    ceilings, preemption, service units. A workstation on the tailnet has none
    of those and is bounded instead by the thing Slurm never has to think
    about -- somebody is sitting at it. Folding these into the Slurm enum
    would put codes there that can never fire on a cluster, and the first
    reader to meet ``NODE_OWNER_RESERVED`` in a Slurm traceback would have to
    work out that it cannot happen.

    Same discipline as its sibling: no generic member. A code that covers
    everything identifies nothing.
    """

    # Capacity -- the local analogue of GPU_MODEL_EXHAUSTED, and it exists for
    # the same measured reason. Admissible and going-to-finish are different
    # questions: on 2026-09-04 two overlapping suites on one box held 66
    # processes and 77.9 GB of commit while doing no work at all.
    NODE_MEMORY_EXHAUSTED = "NODE_MEMORY_EXHAUSTED"
    NODE_DISK_EXHAUSTED = "NODE_DISK_EXHAUSTED"
    NODE_OWNER_RESERVED = "NODE_OWNER_RESERVED"
    NODE_UNREACHABLE = "NODE_UNREACHABLE"

    # The lease -- one project's environment, mutated by one dispatch at a
    # time. This is the code that answers the incident the package was written
    # for: `poetry sync` reinstalling a package under a live interpreter.
    LEASE_HELD = "LEASE_HELD"
    LEASE_NOT_HELD = "LEASE_NOT_HELD"
    LEASE_EXPIRED = "LEASE_EXPIRED"

    # Toolchain -- what a node must already have to be dispatched to.
    # Measured 2026-09-04: `make` was present on one node of three.
    NODE_TOOL_MISSING = "NODE_TOOL_MISSING"
    NODE_PYTHON_MISMATCH = "NODE_PYTHON_MISMATCH"

    # Staging -- the bytes the node is entitled to run.
    STAGE_DIGEST_MISMATCH = "STAGE_DIGEST_MISMATCH"
    STAGE_ARCHIVE_UNREADABLE = "STAGE_ARCHIVE_UNREADABLE"

    # Dispatch and its record.
    DISPATCH_FAILED = "DISPATCH_FAILED"
    RUN_UNKNOWN = "RUN_UNKNOWN"
    LEDGER_ROW_UNPARSABLE = "LEDGER_ROW_UNPARSABLE"
    FEED_EVENT_UNPARSABLE = "FEED_EVENT_UNPARSABLE"

    # Workspace configuration -- the one document every command reads.
    WORKSPACE_NODE_UNKNOWN = "WORKSPACE_NODE_UNKNOWN"
    WORKSPACE_PROJECT_UNKNOWN = "WORKSPACE_PROJECT_UNKNOWN"


__all__ = [
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
]
