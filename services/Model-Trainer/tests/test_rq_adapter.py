from __future__ import annotations

from platform_workers.redis import RedisBytesProto, _RedisBytesClient
from platform_workers.rq_harness import RQClientQueue, RQJobLike, RQRetryLike
from platform_workers.testing import FakeJob, FakeRedisBytesClient, FakeRetry

from model_trainer.core import _test_hooks
from model_trainer.core.contracts.queue import (
    EvalJobPayload,
    TokenizerTrainPayload,
    TrainJobPayload,
)
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings

# Recursive JSON type matching rq_harness
_JsonValue = dict[str, "_JsonValue"] | list["_JsonValue"] | str | int | float | bool | None

# Kwargs dict for tracking enqueue calls
_KwargsDict = dict[str, int | str | None]


class _TrackingQueue(RQClientQueue):
    """Queue that tracks enqueue calls for test assertions."""

    def __init__(self) -> None:
        self.last: tuple[str, _JsonValue, _KwargsDict] | None = None

    def enqueue(
        self,
        func_ref: str,
        *args: _JsonValue,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> RQJobLike:
        # args[0] is the payload dict
        payload: _JsonValue = args[0] if args else {}
        kwargs: _KwargsDict = {
            "job_timeout": job_timeout,
            "result_ttl": result_ttl,
            "failure_ttl": failure_ttl,
            "description": description,
        }
        self.last = (func_ref, payload, kwargs)
        desc_str = description if description is not None else "job"
        return FakeJob(f"id:{desc_str}")


def test_rq_enqueuer_methods() -> None:
    # Set up tracking queue
    fake_queue = _TrackingQueue()

    def _fake_rq_queue(name: str, connection: _RedisBytesClient) -> RQClientQueue:
        return fake_queue

    def _fake_rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
        return FakeRetry(max=max_retries, interval=intervals)

    def _fake_redis_raw_for_rq(url: str) -> RedisBytesProto:
        return FakeRedisBytesClient()

    # Inject fakes via _test_hooks
    orig_queue = _test_hooks.rq_queue_factory
    orig_retry = _test_hooks.rq_retry_factory
    orig_conn = _test_hooks.rq_connection_factory

    _test_hooks.rq_queue_factory = _fake_rq_queue
    _test_hooks.rq_retry_factory = _fake_rq_retry
    _test_hooks.rq_connection_factory = _fake_redis_raw_for_rq

    try:
        settings = RQSettings(
            job_timeout_sec=60,
            result_ttl_sec=120,
            failure_ttl_sec=180,
            retry_max=3,
            retry_intervals=[1, 2, 3],
        )
        enq = RQEnqueuer(redis_url="redis://localhost/0", settings=settings)

        # Train job
        train_payload: TrainJobPayload = {
            "run_id": "run-1",
            "request": {
                "model_family": "gpt2",
                "model_size": "s",
                "max_seq_len": 16,
                "num_epochs": 1,
                "batch_size": 1,
                "learning_rate": 1e-3,
                "corpus_file_id": "deadbeef",
                "tokenizer_id": "tok",
                "holdout_fraction": 0.01,
                "seed": 42,
                "pretrained_run_id": None,
                "freeze_embed": False,
                "gradient_clipping": 1.0,
                "optimizer": "adamw",
                "device": "cpu",
                "precision": "auto",
                "data_num_workers": None,
                "data_pin_memory": None,
                "early_stopping_patience": 0,
                "test_split_ratio": 0.0,
                "finetune_lr_cap": 0.0,
            },
            "user_id": 1,
        }
        jid = enq.enqueue_train(train_payload)
        assert jid.startswith("id:train:run-1")
        last = fake_queue.last
        assert last is not None and len(last) == 3
        path, payload, kwargs = last
        assert path == "model_trainer.worker.train_job.process_train_job"
        # payload is typed as _JsonValue but we know it's a dict from the implementation
        if not isinstance(payload, dict):
            raise AssertionError("payload must be dict")
        assert payload["run_id"] == "run-1"
        assert payload["user_id"] == 1
        # Verify all request fields are passed through
        req = payload["request"]
        if not isinstance(req, dict):
            raise AssertionError("request must be dict")
        assert req["model_family"] == "gpt2"
        assert req["model_size"] == "s"
        assert req["max_seq_len"] == 16
        assert req["num_epochs"] == 1
        assert req["batch_size"] == 1
        assert req["learning_rate"] == 1e-3
        assert req["corpus_file_id"] == "deadbeef"
        assert req["tokenizer_id"] == "tok"
        assert req["holdout_fraction"] == 0.01
        assert req["seed"] == 42
        assert req["pretrained_run_id"] is None
        assert req["freeze_embed"] is False
        assert req["gradient_clipping"] == 1.0
        assert req["optimizer"] == "adamw"
        assert req["device"] == "cpu"
        assert req["precision"] == "auto"
        assert req["data_num_workers"] is None
        assert req["data_pin_memory"] is None
        assert req["early_stopping_patience"] == 0
        assert req["test_split_ratio"] == 0.0
        assert req["finetune_lr_cap"] == 0.0
        assert kwargs["job_timeout"] == 60

        # Eval job
        fake_queue.last = None
        eval_payload: EvalJobPayload = {
            "run_id": "run-1",
            "split": "validation",
            "path_override": None,
        }
        jid2 = enq.enqueue_eval(eval_payload)
        assert jid2.startswith("id:eval:run-1:validation")
        last2 = fake_queue.last
        assert last2 is not None and len(last2) == 3
        path2, payload2, _kwargs2 = last2
        assert path2 == "model_trainer.worker.eval_job.process_eval_job"
        if not isinstance(payload2, dict):
            raise AssertionError("payload2 must be dict")
        assert payload2["run_id"] == "run-1"
        assert payload2["split"] == "validation"
        assert payload2["path_override"] is None

        # Tokenizer job
        fake_queue.last = None
        tok_payload: TokenizerTrainPayload = {
            "tokenizer_id": "tok-1",
            "method": "bpe",
            "vocab_size": 128,
            "min_frequency": 1,
            "corpus_file_id": "deadbeef",
            "holdout_fraction": 0.1,
            "seed": 42,
        }
        jid3 = enq.enqueue_tokenizer(tok_payload)
        assert jid3.startswith("id:tokenizer:tok-1")
        last3 = fake_queue.last
        assert last3 is not None and len(last3) == 3
        path3, payload3, _kwargs3 = last3
        assert path3 == "model_trainer.worker.tokenizer_worker.process_tokenizer_train_job"
        if not isinstance(payload3, dict):
            raise AssertionError("payload3 must be dict")
        assert payload3["tokenizer_id"] == "tok-1"
        assert payload3["method"] == "bpe"
        assert payload3["vocab_size"] == 128
        assert payload3["min_frequency"] == 1
        assert payload3["corpus_file_id"] == "deadbeef"
        assert payload3["holdout_fraction"] == 0.1
        assert payload3["seed"] == 42
    finally:
        # Restore original hooks
        _test_hooks.rq_queue_factory = orig_queue
        _test_hooks.rq_retry_factory = orig_retry
        _test_hooks.rq_connection_factory = orig_conn
