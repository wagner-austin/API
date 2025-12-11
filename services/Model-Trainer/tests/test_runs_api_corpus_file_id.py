from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from fastapi.testclient import TestClient
from platform_workers.redis import _RedisBytesClient
from platform_workers.rq_harness import RQClientQueue, RQRetryLike
from platform_workers.testing import (
    FakeQueue,
    FakeRedis,
    FakeRedisBytesClient,
    FakeRetry,
)

from model_trainer.api.main import create_app
from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


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


def test_runs_train_with_corpus_file_id(tmp_path: Path, settings_factory: _SettingsFactory) -> None:
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(artifacts_root=str(artifacts))
    app = create_app(settings)

    container: ServiceContainer = app.state.container  # type narrowing

    # Swap redis to _FakeRedis
    from model_trainer.worker.trainer_job_store import TrainerJobStore

    fake = FakeRedis()
    container.redis = fake
    container.training_orchestrator._redis = fake
    container.training_orchestrator._job_store = TrainerJobStore(fake)

    # Set up fake RQ infrastructure via hooks to capture payload
    capturing_queue = FakeQueue("job-cfid")

    def _fake_rq_connection(url: str) -> _RedisBytesClient:
        return FakeRedisBytesClient()

    def _fake_rq_queue(name: str, connection: _RedisBytesClient) -> RQClientQueue:
        return capturing_queue

    def _fake_rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
        return FakeRetry(max=max_retries, interval=intervals)

    _test_hooks.rq_connection_factory = _fake_rq_connection
    _test_hooks.rq_queue_factory = _fake_rq_queue
    _test_hooks.rq_retry_factory = _fake_rq_retry

    client = TestClient(app)

    body = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.0005,
        "corpus_file_id": "deadbeef",
        "tokenizer_id": "tok-abc",
        "user_id": 1,
    }

    r = client.post("/runs/train", json=body)
    assert r.status_code == 200
    assert capturing_queue.jobs, "payload should be captured"

    # Extract the request payload from the enqueued job
    # EnqueuedJob.args is tuple[_JsonValue, ...] where _JsonValue is recursive dict/list/scalar
    enqueued = capturing_queue.jobs[0]
    assert enqueued.args and len(enqueued.args) > 0

    # The FakeQueue stores the raw dict structure matching TrainJobPayload
    # Get corpus_file_id from nested structure - use str() for safe extraction
    # The payload structure is: {"run_id": ..., "user_id": ..., "request": {"corpus_file_id": ...}}
    payload_str = str(enqueued.args[0])
    assert "deadbeef" in payload_str, f"corpus_file_id not found: {payload_str}"
    fake.assert_only_called({"hset"})
