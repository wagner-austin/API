from __future__ import annotations

import pytest

from platform_core.errors import AppError, ErrorCode
from platform_core.security import (
    ApiKeyCheckFn,
    RouteDependencyProto,
    create_api_key_dependency,
    route_dependency,
)


def test_create_api_key_dependency_noop_when_blank() -> None:
    dep = create_api_key_dependency(required_key="  ", error_code=ErrorCode.UNAUTHORIZED)
    # Should not raise when header missing
    dep(None)


def test_create_api_key_dependency_checks_value() -> None:
    dep = create_api_key_dependency(required_key="secret", error_code=ErrorCode.UNAUTHORIZED)
    dep("secret")
    with pytest.raises(AppError) as excinfo:
        dep(None)
    assert excinfo.value.code is ErrorCode.UNAUTHORIZED and excinfo.value.http_status == 401


def test_create_api_key_dependency_wrong_value() -> None:
    """Test that wrong API key raises AppError."""
    dep = create_api_key_dependency(required_key="secret", error_code=ErrorCode.UNAUTHORIZED)
    with pytest.raises(AppError) as excinfo:
        dep("wrong-key")
    assert excinfo.value.code is ErrorCode.UNAUTHORIZED


def test_route_dependency_wraps_function() -> None:
    """Test route_dependency wraps a callable with Depends."""

    def my_check(x_api_key: str | None = None) -> None:
        pass

    result: RouteDependencyProto = route_dependency(my_check)
    # Verify it returns a dependency object - check it's callable
    assert callable(result.dependency)


def test_api_key_check_fn_protocol() -> None:
    """Test that created dependencies match ApiKeyCheckFn protocol."""
    dep: ApiKeyCheckFn = create_api_key_dependency(
        required_key="test", error_code=ErrorCode.UNAUTHORIZED
    )
    # Should be callable with str | None - correct key passes
    dep("test")
    # Incorrect/missing key raises but signature accepts str | None
    with pytest.raises(AppError):
        dep(None)


def test_create_api_key_dependency_with_custom_status() -> None:
    """Test custom HTTP status in dependency."""
    dep = create_api_key_dependency(
        required_key="key",
        error_code=ErrorCode.FORBIDDEN,
        http_status=403,
        message="Custom forbidden",
    )
    with pytest.raises(AppError) as excinfo:
        dep(None)
    assert excinfo.value.http_status == 403
    assert excinfo.value.message == "Custom forbidden"
