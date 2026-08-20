from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_workers.redis import RedisBytesProto, _RedisBytesClient
from platform_workers.rq_harness import RQClientQueue, RQJobLike, RQRetryLike
from platform_workers.testing import FakeJob, FakeRedisBytesClient, FakeRetry

from model_trainer.core import _test_hooks
from model_trainer.core.contracts.queue import (
    ClozeJobPayload,
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
    """Queue that tracks enqueue calls for test assertions.

    ``last_retry`` is held separately from ``last`` because the retry policy
    is not a JSON scalar. It is recorded at all because it was previously the
    one enqueue argument no test could see, and a train job silently carrying
    a retry is what let a failed run requeue itself behind an operator's back.
    """

    def __init__(self) -> None:
        self.last: tuple[str, _JsonValue, _KwargsDict] | None = None
        self.last_retry: RQRetryLike | None = None
        self.removed: list[str] = []
        self.remove_result = 1

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
        self.last_retry = retry
        desc_str = description if description is not None else "job"
        return FakeJob(f"id:{desc_str}")

    def remove(self, job_or_id: str) -> int:
        """Record a removal and report the configured outcome.

        Args:
            job_or_id: The job id the adapter asked to remove.

        Returns:
            ``remove_result``, so a test can present both the job-was-pending
            and the job-already-taken cases.
        """
        self.removed.append(job_or_id)
        return self.remove_result


class _Fakes:
    """The queue a test enqueues into, and every retry the adapter built.

    Attributes:
        queue: Tracking queue receiving each enqueue call.
        retries: Retry objects the adapter asked the factory for, in order.
            An empty list after an enqueue means that job type never
            requested a retry policy at all.
    """

    def __init__(self, queue: _TrackingQueue, retries: list[RQRetryLike]) -> None:
        self.queue = queue
        self.retries = retries


def _make_rq_fakes() -> Generator[_Fakes, None, None]:
    """Bind the RQ seams to fakes for one test and restore them afterwards.

    Yields:
        The tracking queue and the list of retries the adapter constructed.
    """
    fake_queue = _TrackingQueue()
    retries: list[RQRetryLike] = []

    def _fake_rq_queue(name: str, connection: _RedisBytesClient) -> RQClientQueue:
        return fake_queue

    def _fake_rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
        retry = FakeRetry(max=max_retries, interval=intervals)
        retries.append(retry)
        return retry

    def _fake_redis_raw_for_rq(url: str) -> RedisBytesProto:
        return FakeRedisBytesClient()

    orig_queue = _test_hooks.rq_queue_factory
    orig_retry = _test_hooks.rq_retry_factory
    orig_conn = _test_hooks.rq_connection_factory

    _test_hooks.rq_queue_factory = _fake_rq_queue
    _test_hooks.rq_retry_factory = _fake_rq_retry
    _test_hooks.rq_connection_factory = _fake_redis_raw_for_rq
    try:
        yield _Fakes(fake_queue, retries)
    finally:
        _test_hooks.rq_queue_factory = orig_queue
        _test_hooks.rq_retry_factory = orig_retry
        _test_hooks.rq_connection_factory = orig_conn


rq_fakes = pytest.fixture(_make_rq_fakes)


def _settings() -> RQSettings:
    """Build settings whose retry policy is distinguishable from no policy.

    Returns:
        Settings with a non-zero retry budget, so a job that carries the
        policy and a job that carries none cannot be confused.
    """
    return RQSettings(
        job_timeout_sec=60,
        result_ttl_sec=120,
        failure_ttl_sec=180,
        retry_max=3,
        retry_intervals=[1, 2, 3],
    )


def _train_payload() -> TrainJobPayload:
    """Build a minimal but complete training payload.

    Returns:
        A payload the adapter can encode without any field missing.
    """
    return {
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
            "loss_mask_prefix_separator": None,
            "hub_model_id": None,
            "finetuning_strategy": "full",
            "lora": None,
            "quantization": None,
            "gguf_export": None,
        },
        "user_id": 1,
        "resume": False,
    }


def test_a_train_job_is_enqueued_with_no_retry_policy(rq_fakes: _Fakes) -> None:
    """A failed training run must stay failed rather than requeue itself.

    The adapter must not even ask for a retry object, so the assertion is on
    the factory having gone uncalled as well as on the argument being None.
    """
    enq = RQEnqueuer(redis_url="redis://localhost/0", settings=_settings())

    enq.enqueue_train(_train_payload())

    assert rq_fakes.queue.last_retry is None
    assert rq_fakes.retries == []


def test_removing_a_pending_job_reports_it_was_removed(rq_fakes: _Fakes) -> None:
    """The adapter must pass the id through and read RQ's own count.

    The count is what separates "cancelled it before a worker took it" from
    "too late", which is the entire decision the cancel path makes.
    """
    enq = RQEnqueuer(redis_url="redis://localhost/0", settings=_settings())
    rq_fakes.queue.remove_result = 1

    assert enq.remove_queued_job("job-7") is True
    assert rq_fakes.queue.removed == ["job-7"]


def test_removing_a_job_a_worker_already_took_reports_false(rq_fakes: _Fakes) -> None:
    """RQ removes nothing once a worker has popped the job off the list."""
    enq = RQEnqueuer(redis_url="redis://localhost/0", settings=_settings())
    rq_fakes.queue.remove_result = 0

    assert enq.remove_queued_job("job-7") is False
    assert rq_fakes.queue.removed == ["job-7"]


def test_an_inference_job_keeps_the_retry_policy(rq_fakes: _Fakes) -> None:
    """Retries stay where they are cheap and the failures are transient.

    The retry handed to the queue must be the one the adapter built from
    settings, so the identity is asserted rather than its mere presence.
    """
    enq = RQEnqueuer(redis_url="redis://localhost/0", settings=_settings())

    enq.enqueue_cloze(
        ClozeJobPayload(
            run_id="run-1", request_id="req-1", items_file_id="deadbeef", max_seq_len=16
        )
    )

    assert len(rq_fakes.retries) == 1
    assert rq_fakes.queue.last_retry is rq_fakes.retries[0]


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
                "loss_mask_prefix_separator": None,
                "hub_model_id": None,
                "finetuning_strategy": "full",
                "lora": None,
                "quantization": None,
                "gguf_export": None,
            },
            "user_id": 1,
            "resume": False,
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
        assert req["hub_model_id"] is None
        assert req["finetuning_strategy"] == "full"
        assert req["lora"] is None
        assert req["quantization"] is None
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


def test_a_baseline_cloze_job_is_enqueued_against_the_model_not_a_run(
    rq_fakes: _Fakes,
) -> None:
    """The description must name the model and item set, because there is no run.

    That description is the only human-readable handle on the job in the queue,
    and a baseline carries no run id to fall back on.
    """
    enq = RQEnqueuer(redis_url="redis://localhost/0", settings=_settings())

    enq.enqueue_baseline_cloze(
        {
            "hub_model_id": "gpt2",
            "items_file_id": "file-1",
            "max_seq_len": 128,
            "device": "cpu",
        }
    )

    last = rq_fakes.queue.last
    if last is None:
        raise AssertionError("expected an enqueue call")
    func_ref, payload, kwargs = last
    assert func_ref == "model_trainer.worker.baseline_cloze_job.process_baseline_cloze_job"
    assert kwargs["description"] == "baseline-cloze:gpt2:file-1"
    if not isinstance(payload, dict):
        raise AssertionError(f"expected payload dict, got {type(payload)}")
    assert payload["hub_model_id"] == "gpt2"
