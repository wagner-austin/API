from __future__ import annotations

from typing import Protocol

from platform_core.errors import AppError, ErrorCodeBase

# ---------------------------------------------------------------------------
# Protocols for FastAPI dependency injection (avoids importing FastAPI types)
# ---------------------------------------------------------------------------


class ApiKeyCheckFn(Protocol):
    """Protocol for API key check function signature."""

    def __call__(self, x_api_key: str | None = ...) -> None: ...


class RouteDependencyProto(Protocol):
    """Protocol for FastAPI route dependency (Depends instance)."""

    @property
    def dependency(self) -> ApiKeyCheckFn | None: ...


class _DependsCtor(Protocol):
    """Constructor protocol for FastAPI Depends."""

    def __call__(
        self,
        dependency: ApiKeyCheckFn | None = ...,
    ) -> RouteDependencyProto: ...


class _HeaderCtor(Protocol):
    """Constructor protocol for FastAPI Header."""

    def __call__(
        self,
        default: str | None = ...,
        *,
        alias: str | None = ...,
    ) -> str | None: ...


def _get_depends() -> _DependsCtor:
    """Get FastAPI Depends constructor with typed interface."""
    fastapi_mod = __import__("fastapi")
    depends_cls: _DependsCtor = fastapi_mod.Depends
    return depends_cls


def _get_header() -> _HeaderCtor:
    """Get FastAPI Header constructor with typed interface."""
    fastapi_mod = __import__("fastapi")
    header_cls: _HeaderCtor = fastapi_mod.Header
    return header_cls


def route_dependency(dep: ApiKeyCheckFn) -> RouteDependencyProto:
    """Create a typed route dependency for use in FastAPI's dependencies=[...] list.

    This wraps FastAPI's Depends() with proper typing to satisfy strict mypy.

    Args:
        dep: A callable dependency function.

    Returns:
        A dependency instance that can be used in route dependencies.
    """
    depends_ctor = _get_depends()
    result: RouteDependencyProto = depends_ctor(dep)
    return result


def create_api_key_dependency(
    *,
    required_key: str,
    error_code: ErrorCodeBase,
    http_status: int | None = None,
    header_name: str = "X-API-Key",
    message: str = "Unauthorized",
) -> ApiKeyCheckFn:
    """Create a FastAPI dependency that enforces a static API key.

    Args:
        required_key: The expected API key value (whitespace trimmed).
        error_code: Error code to raise when auth fails.
        http_status: Optional HTTP status to use; falls back to default per AppError.
        header_name: Request header to inspect (default: X-API-Key).
        message: Error message to include in the AppError.

    Returns:
        A callable dependency suitable for use in FastAPI route dependencies.
    """
    header_ctor = _get_header()
    key = required_key.strip()
    if key == "":
        # No-op dependency when key is not configured.
        def _pass(x_api_key: str | None = header_ctor(default=None, alias=header_name)) -> None:
            return None

        return _pass

    def _check(x_api_key: str | None = header_ctor(default=None, alias=header_name)) -> None:
        if x_api_key is None or x_api_key != key:
            raise AppError(error_code, message=message, http_status=http_status)

    return _check


__all__ = [
    "ApiKeyCheckFn",
    "RouteDependencyProto",
    "create_api_key_dependency",
    "route_dependency",
]
