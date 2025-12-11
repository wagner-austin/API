from __future__ import annotations

from platform_core.json_utils import JSONValue
from platform_workers.redis import _RedisBytesClient
from platform_workers.testing import FakeQueue, FakeRedis, FakeRedisBytesClient, FakeRetry

from model_trainer.api.schemas.runs import TrainRequest
from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator


def _get_payload_user_id(raw: JSONValue) -> int:
    """Extract user_id from a raw JSON payload dict."""
    if not isinstance(raw, dict):
        raise AssertionError("payload must be dict")
    val = raw.get("user_id")
    if not isinstance(val, int):
        raise AssertionError("user_id must be int")
    return val


def test_orchestrator_threads_user_id() -> None:
    s = load_settings()
    r = FakeRedis()

    # Set up fake RQ infrastructure via hooks
    fake_queue = FakeQueue(job_id="job-1")

    def _fake_rq_connection(url: str) -> _RedisBytesClient:
        return FakeRedisBytesClient()

    def _fake_rq_queue(name: str, connection: _RedisBytesClient) -> FakeQueue:
        return fake_queue

    def _fake_rq_retry(*, max_retries: int, intervals: list[int]) -> FakeRetry:
        return FakeRetry(max=max_retries, interval=intervals)

    _test_hooks.rq_connection_factory = _fake_rq_connection
    _test_hooks.rq_queue_factory = _fake_rq_queue
    _test_hooks.rq_retry_factory = _fake_rq_retry

    enq = RQEnqueuer(
        redis_url="redis://localhost:6379/0",
        settings=RQSettings(
            job_timeout_sec=60,
            result_ttl_sec=60,
            failure_ttl_sec=60,
            retry_max=0,
            retry_intervals=[],
        ),
    )

    orch = TrainingOrchestrator(settings=s, redis_client=r, enqueuer=enq, model_registry=None)
    req = TrainRequest(
        model_family="gpt2",
        model_size="small",
        max_seq_len=128,
        num_epochs=1,
        batch_size=2,
        learning_rate=5e-4,
        corpus_file_id="deadbeef",
        tokenizer_id="tok1",
        user_id=42,
        holdout_fraction=0.01,
        seed=42,
        pretrained_run_id=None,
        freeze_embed=False,
        gradient_clipping=1.0,
        optimizer="adamw",
        device="cpu",
        early_stopping_patience=5,
        test_split_ratio=0.15,
        finetune_lr_cap=5e-5,
        precision="auto",
    )
    out = orch.enqueue_training(req)
    assert out["run_id"]

    # Verify user_id was passed correctly in the payload
    assert len(fake_queue.jobs) == 1
    job = fake_queue.jobs[0]
    user_id = _get_payload_user_id(job.args[0])
    assert user_id == 42

    r.assert_only_called({"hset"})
