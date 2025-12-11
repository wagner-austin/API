"""Tests for health check endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str

from covenant_radar_api.api.main import create_app

from .conftest import ContainerAndStore


def test_healthz_ok(container_with_store: ContainerAndStore) -> None:
    """Test liveness probe returns ok."""
    client: TestClient = TestClient(create_app(container_with_store.container.settings))

    r = client.get("/healthz")

    assert r.status_code == 200
    body_raw = load_json_str(r.text)
    if type(body_raw) is not dict:
        raise AssertionError("expected dict")
    body = body_raw
    assert body["status"] == "ok"


def test_readyz_ready(container_with_store: ContainerAndStore) -> None:
    """Test readiness probe returns ready when workers present."""
    # container_with_store fixture creates 1 worker by default
    client: TestClient = TestClient(create_app(container_with_store.container.settings))

    r = client.get("/readyz")

    assert r.status_code == 200
    body_raw = load_json_str(r.text)
    if type(body_raw) is not dict:
        raise AssertionError("expected dict")
    body = body_raw
    assert body["status"] == "ready"


def test_readyz_degraded_no_worker(container_with_store: ContainerAndStore) -> None:
    """Test readiness probe returns degraded when no workers."""
    # Remove all workers from fake redis
    container_with_store.container.redis.delete("rq:workers")
    client: TestClient = TestClient(create_app(container_with_store.container.settings))

    r = client.get("/readyz")

    assert r.status_code == 503
    body_raw = load_json_str(r.text)
    if type(body_raw) is not dict:
        raise AssertionError("expected dict")
    body = body_raw
    assert body["status"] == "degraded"
    assert body["reason"] == "no-worker"
