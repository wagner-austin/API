from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis

from model_trainer.api.main import create_app
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.container import ServiceContainer


def _client_and_container() -> tuple[TestClient, ServiceContainer, FakeRedis]:
    app = create_app(load_settings())
    fake = FakeRedis()
    # replace container redis with fake
    container: ServiceContainer = app.state.container
    container.redis = fake
    return TestClient(app), container, fake


def test_readyz_degraded_without_worker(monkeypatch: MonkeyPatch) -> None:
    client, _, fake = _client_and_container()
    resp = client.get("/readyz")
    assert resp.status_code == 503
    obj_raw = load_json_str(resp.text)
    assert isinstance(obj_raw, dict) and "status" in obj_raw
    obj: dict[str, JSONValue] = obj_raw
    status = obj.get("status")
    reason = obj.get("reason")
    assert isinstance(status, str) and status == "degraded"
    assert isinstance(reason, str) and reason in ("no-worker", "redis no-pong", "redis error")
    fake.assert_only_called({"ping", "scard"})


def test_readyz_ready_with_worker(monkeypatch: MonkeyPatch) -> None:
    client, container, fake = _client_and_container()
    # Simulate a worker registered in RQ registry set
    container.redis.sadd("rq:workers", "worker:1")
    resp = client.get("/readyz")
    assert resp.status_code == 200
    obj2_raw = load_json_str(resp.text)
    assert isinstance(obj2_raw, dict) and "status" in obj2_raw
    obj2: dict[str, JSONValue] = obj2_raw
    st2 = obj2.get("status")
    assert isinstance(st2, str) and st2 == "ready"
    fake.assert_only_called({"ping", "scard", "sadd"})
