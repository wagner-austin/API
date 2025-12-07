from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from fastapi import FastAPI
from fastapi.testclient import TestClient
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis
from pytest import MonkeyPatch

from model_trainer.api.main import create_app
from model_trainer.api.routes import runs as runs_routes
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.core.services.container import ServiceContainer
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.core.services.registries import ModelRegistry, TokenizerRegistry
from model_trainer.orchestrators.conversation_orchestrator import ConversationOrchestrator
from model_trainer.orchestrators.inference_orchestrator import InferenceOrchestrator
from model_trainer.orchestrators.tokenizer_orchestrator import TokenizerOrchestrator
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator


class _SettingsFactory(Protocol):
    def __call__(
        self,
        *,
        artifacts_root: str | None = ...,
        runs_root: str | None = ...,
        logs_root: str | None = ...,
        data_root: str | None = ...,
        data_bank_api_url: str | None = ...,
        data_bank_api_key: str | None = ...,
        threads: int | None = ...,
        redis_url: str | None = ...,
        app_env: Literal["dev", "prod"] | None = ...,
        security_api_key: str | None = ...,
    ) -> Settings: ...


def _mk_app(
    tmp: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> tuple[TestClient, Settings, FakeRedis]:
    s = settings_factory(
        artifacts_root=str(tmp / "artifacts"),
        runs_root=str(tmp / "runs"),
        logs_root=str(tmp / "logs"),
    )

    fake = FakeRedis()

    def _fake_redis_for_kv(url: str) -> RedisStrProto:
        return fake

    monkeypatch.setattr("model_trainer.core.services.container.redis_for_kv", _fake_redis_for_kv)

    # Short-circuit RQ enqueues
    def _enq_train(self: RQEnqueuer, payload: dict[str, str | int | float | bool | None]) -> str:
        return "job-train-1"

    def _enq_eval(self: RQEnqueuer, payload: dict[str, str | int | float | bool | None]) -> str:
        return "job-eval-1"

    monkeypatch.setattr(RQEnqueuer, "enqueue_train", _enq_train)
    monkeypatch.setattr(RQEnqueuer, "enqueue_eval", _enq_eval)
    app = create_app(s)
    return TestClient(app), s, fake


def test_runs_logs_not_found(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    client, _, fake = _mk_app(tmp_path, monkeypatch, settings_factory)
    res = client.get("/runs/unknown/logs")
    assert res.status_code == 404
    fake.assert_only_called(set())


def test_runs_logs_read_failure(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    client, s, fake = _mk_app(tmp_path, monkeypatch, settings_factory)
    log_dir = Path(s["app"]["artifacts_root"]) / "models" / "r1"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "logs.jsonl").mkdir(parents=True, exist_ok=True)
    res = client.get("/runs/r1/logs")
    assert res.status_code == 500
    fake.assert_only_called(set())


def test_runs_logs_tail_content(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    client, s, fake = _mk_app(tmp_path, monkeypatch, settings_factory)
    run_dir = Path(s["app"]["artifacts_root"]) / "models" / "r2"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs.jsonl"
    log_path.write_text("a\n" * 10 + "b\n" + "c\n", encoding="utf-8")
    res = client.get("/runs/r2/logs?tail=2")
    assert res.status_code == 200
    assert res.text.strip().splitlines() == ["b", "c"]
    fake.assert_only_called(set())


def test_runs_logs_stream_not_found(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    client, _, fake = _mk_app(tmp_path, monkeypatch, settings_factory)
    res = client.get("/runs/nope/logs/stream")
    assert res.status_code == 404
    fake.assert_only_called(set())


def test_runs_logs_stream_follow_false(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    client, s, fake = _mk_app(tmp_path, monkeypatch, settings_factory)
    run_dir = Path(s["app"]["artifacts_root"]) / "models" / "r3"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs.jsonl").write_text("x\n" * 3, encoding="utf-8")
    with client.stream("GET", "/runs/r3/logs/stream?tail=2&follow=false") as resp:
        assert resp.status_code == 200
        body = b"".join(list(resp.iter_bytes()))
    # Expect two SSE data lines
    lines = [ln for ln in body.split(b"\n\n") if ln]
    assert len(lines) == 2
    assert lines[0].startswith(b"data: ")
    fake.assert_only_called(set())


def test_runs_eval_result_not_found(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    client, _, fake = _mk_app(tmp_path, monkeypatch, settings_factory)
    res = client.get("/runs/unknown/eval")
    # Exception handler maps to 404
    assert res.status_code == 404
    fake.assert_only_called({"get"})


def test_runs_eval_result_cache_corrupt(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    # Build app and inject fake redis
    app = create_app(settings_factory(artifacts_root=str(tmp_path / "artifacts")))
    container: ServiceContainer = app.state.container
    fake = FakeRedis()
    container.redis = fake
    container.training_orchestrator._redis = fake

    # Pre-populate eval key with non-dict JSON to exercise corrupt branch
    run_id = "run-corrupt"
    from platform_core.trainer_keys import eval_key

    fake.set(eval_key(run_id), "[]")

    client = TestClient(app)
    res = client.get(f"/runs/{run_id}/eval")
    # AppError(DATA_NOT_FOUND) is mapped to 404 by handlers
    assert res.status_code == 404
    fake.assert_only_called({"set", "get"})


def test_runs_train_unsupported_backend_maps_400(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    s = settings_factory(
        artifacts_root=str(tmp_path / "artifacts"),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
    )

    # Enqueue methods
    def _enq_train(self: RQEnqueuer, payload: TrainJobPayload) -> str:
        return "job-x"

    monkeypatch.setattr(RQEnqueuer, "enqueue_train", _enq_train)
    enq = RQEnqueuer("redis://ignored", RQSettings(1, 1, 1, 0, []))
    r = FakeRedis()
    model_reg = ModelRegistry(registrations={}, dataset_builder=LocalTextDatasetBuilder())
    tokenizer_reg = TokenizerRegistry(backends={})
    training = TrainingOrchestrator(
        settings=s,
        redis_client=r,
        enqueuer=enq,
        model_registry=model_reg,
    )
    inference = InferenceOrchestrator(settings=s, redis_client=r, enqueuer=enq)
    conversation = ConversationOrchestrator(settings=s, redis_client=r, enqueuer=enq)
    tokenizer = TokenizerOrchestrator(settings=s, redis_client=r, enqueuer=enq)
    container = ServiceContainer(
        settings=s,
        redis=r,
        rq_enqueuer=enq,
        training_orchestrator=training,
        inference_orchestrator=inference,
        conversation_orchestrator=conversation,
        tokenizer_orchestrator=tokenizer,
        model_registry=model_reg,
        tokenizer_registry=tokenizer_reg,
        dataset_builder=LocalTextDatasetBuilder(),
    )
    app = FastAPI()
    app.include_router(runs_routes.build_router(container), prefix="/runs")
    install_exception_handlers_fastapi(app, logger_name="test", request_id_var=None)
    client = TestClient(app)
    payload = {
        "model_family": "llama",
        "model_size": "s",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "deadbeef",
        "tokenizer_id": "tok",
        "user_id": 1,
    }
    res = client.post("/runs/train", json=payload)
    assert res.status_code == 400
    r.assert_only_called(set())
