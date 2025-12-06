from __future__ import annotations

from platform_core.errors import ErrorCodeBase


class MusicWrappedErrorCode(ErrorCodeBase):
    """Domain-specific error codes for Music Wrapped."""

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NO_LISTENING_HISTORY = "NO_LISTENING_HISTORY"
    UNSUPPORTED_SERVICE = "UNSUPPORTED_SERVICE"
    INVALID_SERVICE = "INVALID_SERVICE"
    INVALID_DATE_RANGE = "INVALID_DATE_RANGE"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    INVALID_TAKEOUT = "INVALID_TAKEOUT"


__all__ = ["MusicWrappedErrorCode"]
