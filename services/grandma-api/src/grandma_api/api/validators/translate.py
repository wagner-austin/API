"""Validation functions for the translate endpoint.

Provides strict validation with clear error propagation.
"""

from __future__ import annotations

from platform_core.errors import AppError, ErrorCode


def validate_token(provided_token: str, expected_token: str) -> None:
    """Validate authentication token matches expected value.

    Args:
        provided_token: The token provided in the request.
        expected_token: The expected token from configuration.

    Raises:
        AppError: With UNAUTHORIZED code if token does not match.
    """
    if provided_token != expected_token:
        raise AppError(
            code=ErrorCode.UNAUTHORIZED,
            message="Invalid token",
            http_status=401,
        )


def validate_audio_bytes(audio_bytes: bytes) -> None:
    """Validate audio bytes are not empty.

    Args:
        audio_bytes: The audio file bytes.

    Raises:
        AppError: With INVALID_INPUT code if audio is empty.
    """
    if len(audio_bytes) == 0:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="No audio file provided",
            http_status=400,
        )


__all__ = [
    "validate_audio_bytes",
    "validate_token",
]
