from __future__ import annotations

from fastapi import FastAPI
from platform_core.json_utils import load_json_str
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis, FakeRedisError, FakeRedisNoPong
from starlette.testclient import TestClient

from model_trainer.api.routes import health
from model_trainer.core.config.settings import Settings, load_settings
from model_trainer.core.services.container import ServiceContainer
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.core.services.registries import ModelRegistry, TokenizerRegistry
from model_trainer.orchestrators.conversation_orchestrator import ConversationOrchestrator
from model_trainer.orchestrators.inference_orchestrator import InferenceOrchestrator
from model_trainer.orchestrators.tokenizer_orchestrator import TokenizerOrchestrator
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator


def _make_container(settings: Settings, r: RedisStrProto) -> ServiceContainer:
    enq = RQEnqueuer("redis://ignored", RQSettings(1, 1, 1, 0, []))
    ds = LocalTextDatasetBuilder()
    model_reg = ModelRegistry(registrations={}, dataset_builder=ds)
    tok_reg = TokenizerRegistry(backends={})
    training = TrainingOrchestrator(
        settings=settings,
        redis_client=r,
        enqueuer=enq,
        model_registry=model_reg,
    )
    inference = InferenceOrchestrator(settings=settings, redis_client=r, enqueuer=enq)
    conversation = ConversationOrchestrator(settings=settings, redis_client=r, enqueuer=enq)
    tokenizer = TokenizerOrchestrator(settings=settings, redis_client=r, enqueuer=enq)
    return ServiceContainer(
        settings=settings,
        redis=r,
        rq_enqueuer=enq,
        training_orchestrator=training,
        inference_orchestrator=inference,
        conversation_orchestrator=conversation,
        tokenizer_orchestrator=tokenizer,
        model_registry=model_reg,
        tokenizer_registry=tok_reg,
        dataset_builder=LocalTextDatasetBuilder(),
    )


def _build_app(container: ServiceContainer) -> TestClient:
    app = FastAPI()
    app.include_router(health.build_router(container))
    return TestClient(app)


def test_readyz_redis_no_pong() -> None:
    s = load_settings()
    r = FakeRedisNoPong()

    client = _build_app(_make_container(s, r))
    res = client.get("/readyz")
    assert res.status_code == 503
    obj_raw = load_json_str(res.text)
    if not isinstance(obj_raw, dict):
        raise AssertionError("Response must be a dict")
    assert obj_raw["status"] == "degraded"
    r.assert_only_called({"ping"})


def test_readyz_no_worker() -> None:
    s = load_settings()
    r = FakeRedis()

    client = _build_app(_make_container(s, r))
    res = client.get("/readyz")
    assert res.status_code == 503
    obj2_raw = load_json_str(res.text)
    if not isinstance(obj2_raw, dict):
        raise AssertionError("Response must be a dict")
    assert obj2_raw["status"] == "degraded"
    r.assert_only_called({"ping", "scard"})


def test_readyz_redis_error() -> None:
    s = load_settings()
    r = FakeRedisError()

    client = _build_app(_make_container(s, r))
    res = client.get("/readyz")
    assert res.status_code == 503
    obj3_raw = load_json_str(res.text)
    if not isinstance(obj3_raw, dict):
        raise AssertionError("Response must be a dict")
    assert obj3_raw["status"] == "degraded"
    r.assert_only_called({"ping"})


def test_readyz_ready() -> None:
    s = load_settings()
    r = FakeRedis()

    # Simulate presence of one worker in RQ registry set
    r.sadd("rq:workers", "worker:1")
    client = _build_app(_make_container(s, r))
    res = client.get("/readyz")
    assert res.status_code == 200
    obj4_raw = load_json_str(res.text)
    if not isinstance(obj4_raw, dict):
        raise AssertionError("Response must be a dict")
    assert obj4_raw["status"] == "ready"
    r.assert_only_called({"ping", "sadd", "scard"})
