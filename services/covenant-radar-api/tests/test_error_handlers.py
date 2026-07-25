"""Tests for the service's domain-to-HTTP error mappings.

Drives the handlers through a real FastAPI app and a real TestClient, so the
mapping is exercised end to end rather than by calling handler functions
directly.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONTypeError

from .conftest import make_route_test_client


def _build_raising_router() -> APIRouter:
    """Build a router whose routes each raise one kind of error.

    Returns:
        Router exposing /json-type, /not-found, /key-error and /other.
    """
    router = APIRouter()

    def _raise_json_type() -> Response:
        raise JSONTypeError("Field 'x' must be an integer")

    def _raise_not_found() -> Response:
        raise AppError(code=ErrorCode.NOT_FOUND, message="Deal not found: d1")

    def _raise_key_error() -> Response:
        raise KeyError("ebitda")

    def _raise_other() -> Response:
        raise RuntimeError("boom")

    router.add_api_route("/json-type", _raise_json_type, methods=["GET"])
    router.add_api_route("/not-found", _raise_not_found, methods=["GET"])
    router.add_api_route("/key-error", _raise_key_error, methods=["GET"])
    router.add_api_route("/other", _raise_other, methods=["GET"])
    return router


class TestErrorMapping:
    """Installed handlers map domain errors onto status codes."""

    def test_json_type_error_becomes_400(self) -> None:
        """A decode type error surfaces as 400 with its message preserved."""
        client = make_route_test_client(_build_raising_router())

        response = client.get("/json-type")

        assert response.status_code == 400
        assert "must be an integer" in response.text

    def test_record_not_found_becomes_404(self) -> None:
        """An absent row surfaces as 404 with its message preserved."""
        client = make_route_test_client(_build_raising_router())

        response = client.get("/not-found")

        assert response.status_code == 404
        assert "Deal not found: d1" in response.text

    def test_bare_key_error_still_becomes_500(self) -> None:
        """A bare KeyError is a defect and is not softened into a 404.

        This is why repositories raise AppError(NOT_FOUND) rather than KeyError:
        mapping KeyError to 404 would hide real bugs behind a client error.
        """
        client = make_route_test_client(_build_raising_router())

        response = client.get("/key-error")

        assert response.status_code == 500

    def test_unmapped_exception_still_becomes_500(self) -> None:
        """An error with no mapping is not softened; it stays a 500."""
        client = make_route_test_client(_build_raising_router())

        response = client.get("/other")

        assert response.status_code == 500
