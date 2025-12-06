from __future__ import annotations

from platform_workers.redis import RedisBytesProto, _RedisBytesClient
from platform_workers.rq_harness import RQClientQueue, RQJobLike, RQRetryLike
from pytest import MonkeyPatch

from model_trainer.core.contracts.queue import (
    EvalJobPayload,
    TokenizerTrainPayload,
    TrainJobPayload,
)
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings


class _FakeJob(RQJobLike):
    def __init__(self, job_id: str) -> None:
        self._id = job_id

    def get_id(self) -> str:
        return self._id


# Recursive JSON type matching rq_harness
_JsonValue = dict[str, "_JsonValue"] | list["_JsonValue"] | str | int | float | bool | None

# Kwargs dict for tracking enqueue calls
_KwargsDict = dict[str, int | str | None]


class _FakeQueue(RQClientQueue):
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
        return _FakeJob(f"id:{desc_str}")


class _FakeRetry(RQRetryLike):
    """Fake Retry class matching RQ's interface."""

    def __init__(self, max_retries: int, intervals: list[int]) -> None:
        self.max_retries = max_retries
        self.intervals = intervals


class _FakeRedisBytesClient(RedisBytesProto):
    """Fake Redis client implementing RedisBytesProto."""

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def close(self) -> None:
        pass


def _install_fakes(monkeypatch: MonkeyPatch, queue_holder: dict[str, _FakeQueue]) -> None:
    fake_queue = _FakeQueue()
    queue_holder["q"] = fake_queue

    def _fake_rq_queue(name: str, connection: _RedisBytesClient) -> RQClientQueue:
        return fake_queue

    def _fake_rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
        return _FakeRetry(max_retries=max_retries, intervals=intervals)

    def _fake_redis_raw_for_rq(url: str) -> RedisBytesProto:
        return _FakeRedisBytesClient()

    # Patch at the import location in rq_adapter module
    monkeypatch.setattr("model_trainer.core.services.queue.rq_adapter.rq_queue", _fake_rq_queue)
    monkeypatch.setattr("model_trainer.core.services.queue.rq_adapter.rq_retry", _fake_rq_retry)
    monkeypatch.setattr(
        "model_trainer.core.services.queue.rq_adapter.redis_raw_for_rq", _fake_redis_raw_for_rq
    )


def test_rq_enqueuer_methods(monkeypatch: MonkeyPatch) -> None:
    holder: dict[str, _FakeQueue] = {}
    _install_fakes(monkeypatch, holder)

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
    last = holder["q"].last
    assert last is not None and len(last) == 3
    path, payload, kwargs = last
    assert path == "model_trainer.worker.train_job.process_train_job"
    assert isinstance(payload, dict) and "run_id" in payload
    assert "request" in payload and "user_id" in payload
    assert payload.get("run_id") == "run-1"
    assert kwargs["job_timeout"] == 60

    # Eval job
    holder["q"].last = None
    eval_payload: EvalJobPayload = {
        "run_id": "run-1",
        "split": "validation",
        "path_override": None,
    }
    jid2 = enq.enqueue_eval(eval_payload)
    assert jid2.startswith("id:eval:run-1:validation")
    last2 = holder["q"].last
    assert last2 is not None and len(last2) == 3
    path2, payload2, _kwargs2 = last2
    assert path2 == "model_trainer.worker.eval_job.process_eval_job"
    assert isinstance(payload2, dict) and "run_id" in payload2
    assert "split" in payload2 and "path_override" in payload2
    assert payload2.get("split") == "validation"

    # Tokenizer job
    holder["q"].last = None
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
    last3 = holder["q"].last
    assert last3 is not None and len(last3) == 3
    path3, payload3, _kwargs3 = last3
    assert path3 == "model_trainer.worker.tokenizer_worker.process_tokenizer_train_job"
    assert isinstance(payload3, dict) and "tokenizer_id" in payload3
    assert "method" in payload3 and "vocab_size" in payload3
    assert payload3.get("method") == "bpe"
