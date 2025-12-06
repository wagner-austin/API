from platform_core.data_bank_client import (
    AuthorizationError,
    BadRequestError,
    ConflictError,
    DataBankClient,
    DataBankClientError,
    FileInfoDict,
    ForbiddenError,
    HeadInfo,
    InsufficientStorageClientError,
    NotFoundError,
    RangeNotSatisfiableError,
    _decode_upload_response,
)

# Recursive JSON type used in tests for upload decoding coverage.
UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None

__all__ = [
    "AuthorizationError",
    "BadRequestError",
    "ConflictError",
    "DataBankClient",
    "DataBankClientError",
    "FileInfoDict",
    "ForbiddenError",
    "HeadInfo",
    "InsufficientStorageClientError",
    "NotFoundError",
    "RangeNotSatisfiableError",
    "UnknownJson",
    "_decode_upload_response",
]
