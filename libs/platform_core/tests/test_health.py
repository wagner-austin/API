"""Tests for platform_core.health module."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from platform_core.health import (
    SERVICE_UNAVAILABLE,
    HealthResponse,
    ReadyResponse,
    build_health_router,
    healthz,
)


def test_healthz_returns_ok() -> None:
    """Test healthz always returns status ok."""
    result: HealthResponse = healthz()
    assert result == {"status": "ok"}


def test_health_response_type_structure() -> None:
    """Test HealthResponse TypedDict structure."""
    response: HealthResponse = {"status": "ok"}
    assert response["status"] == "ok"


def test_ready_response_type_structure_ready() -> None:
    """Test ReadyResponse TypedDict structure for ready state."""
    response: ReadyResponse = {"status": "ready", "reason": None}
    assert response["status"] == "ready"
    assert response["reason"] is None


def test_ready_response_type_structure_degraded() -> None:
    """Test ReadyResponse TypedDict structure for degraded state."""
    response: ReadyResponse = {"status": "degraded", "reason": "test reason"}
    assert response["status"] == "degraded"
    assert response["reason"] == "test reason"


class TestBuildHealthRouter:
    """The router five services each had their own copy of.

    Driven through a real ASGI app rather than by calling the closures, because
    the thing worth proving is what an orchestrator's probe actually receives:
    the status CODE, not just the body. Every one of the five copies set that
    code itself, three via `status.HTTP_503_SERVICE_UNAVAILABLE` and two as a
    bare 503, and a readiness probe answering 200 while its body says
    "degraded" is a service that keeps being sent traffic.
    """

    def _client(self, *, ready: ReadyResponse) -> TestClient:
        """Mount the shared router on a real app.

        Args:
            ready: What this service's readiness probe reports.

        Returns:
            A client bound to an app serving the two health routes.
        """
        app = FastAPI()
        app.include_router(
            build_health_router(
                healthz_route=lambda: {"status": "ok"},
                readyz_route=lambda: ready,
            )
        )
        return TestClient(app)

    def test_healthz_answers_200_and_ok(self) -> None:
        response = self._client(ready={"status": "ready", "reason": None}).get("/healthz")

        body: HealthResponse = response.json()

        assert response.status_code == 200
        assert body == {"status": "ok"}

    def test_a_ready_service_answers_200(self) -> None:
        response = self._client(ready={"status": "ready", "reason": None}).get("/readyz")

        body: ReadyResponse = response.json()

        assert response.status_code == 200
        assert body == {"status": "ready", "reason": None}

    def test_a_degraded_service_answers_503(self) -> None:
        """The mapping this function exists to hold. Each of the five copies
        wrote it separately, so each could have got it wrong separately."""
        response = self._client(ready={"status": "degraded", "reason": "redis down"}).get("/readyz")

        assert response.status_code == SERVICE_UNAVAILABLE

    def test_a_degraded_service_still_returns_its_reason(self) -> None:
        """The 503 does not replace the body: an operator needs the reason,
        and a probe that reports only a code makes them go and look."""
        response = self._client(ready={"status": "degraded", "reason": "redis down"}).get("/readyz")

        body: ReadyResponse = response.json()

        assert body == {"status": "degraded", "reason": "redis down"}

    def test_it_serves_both_probe_paths_and_nothing_else(self) -> None:
        """Asserted by asking the app rather than by reading its route table:
        what an orchestrator can reach is the property, and a route object
        that exists but is not mounted would satisfy the structural check."""
        client = self._client(ready={"status": "ready", "reason": None})

        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        assert client.get("/livez").status_code == 404
