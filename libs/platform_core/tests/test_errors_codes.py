"""Tests for errors: test_error_code_user_errors_400."""

from __future__ import annotations

from platform_core.errors import (
    AppError,
    ErrorCode,
)


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
