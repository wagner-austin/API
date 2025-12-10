from __future__ import annotations

from typing import Final

from platform_core.json_utils import JSONValue
from platform_core.queues import TRAINER_QUEUE
from platform_workers.redis import _RedisBytesClient, redis_raw_for_rq
from platform_workers.rq_harness import RQClientQueue, RQJobLike, rq_queue, rq_retry

from ...contracts.conversation import ChatJobPayload
from ...contracts.queue import (
    EvalJobPayload,
    GenerateJobPayload,
    ScoreJobPayload,
    TokenizerTrainPayload,
    TrainJobPayload,
)


class RQSettings:
    job_timeout_sec: int
    result_ttl_sec: int
    failure_ttl_sec: int
    retry_max: int
    retry_intervals: list[int]

    def __init__(
        self: RQSettings,
        job_timeout_sec: int,
        result_ttl_sec: int,
        failure_ttl_sec: int,
        retry_max: int,
        retry_intervals: list[int],
    ) -> None:
        self.job_timeout_sec = job_timeout_sec
        self.result_ttl_sec = result_ttl_sec
        self.failure_ttl_sec = failure_ttl_sec
        self.retry_max = retry_max
        self.retry_intervals = retry_intervals


class RQEnqueuer:
    redis_url: str
    settings: RQSettings
    queue_name: Final[str] = TRAINER_QUEUE

    def __init__(self: RQEnqueuer, redis_url: str, settings: RQSettings) -> None:
        self.redis_url = redis_url
        self.settings = settings

    def _connection(self: RQEnqueuer) -> _RedisBytesClient:
        return redis_raw_for_rq(self.redis_url)

    def enqueue_train(self: RQEnqueuer, payload: TrainJobPayload) -> str:
        conn = self._connection()
        q: RQClientQueue = rq_queue(self.queue_name, connection=conn)
        retry = rq_retry(
            max_retries=self.settings.retry_max, intervals=self.settings.retry_intervals
        )
        req = payload["request"]
        payload_dict: dict[str, JSONValue] = {
            "run_id": payload["run_id"],
            "user_id": payload["user_id"],
            "request": {
                "model_family": req["model_family"],
                "model_size": req["model_size"],
                "max_seq_len": req["max_seq_len"],
                "num_epochs": req["num_epochs"],
                "batch_size": req["batch_size"],
                "learning_rate": req["learning_rate"],
                "corpus_file_id": req["corpus_file_id"],
                "tokenizer_id": req["tokenizer_id"],
                "holdout_fraction": req["holdout_fraction"],
                "seed": req["seed"],
                "pretrained_run_id": req["pretrained_run_id"],
                "freeze_embed": req["freeze_embed"],
                "gradient_clipping": req["gradient_clipping"],
                "optimizer": req["optimizer"],
                "device": req["device"],
                "precision": req["precision"],
                "data_num_workers": req["data_num_workers"],
                "data_pin_memory": req["data_pin_memory"],
                "early_stopping_patience": req["early_stopping_patience"],
                "test_split_ratio": req["test_split_ratio"],
                "finetune_lr_cap": req["finetune_lr_cap"],
            },
        }
        job: RQJobLike = q.enqueue(
            "model_trainer.worker.train_job.process_train_job",
            payload_dict,
            job_timeout=self.settings.job_timeout_sec,
            result_ttl=self.settings.result_ttl_sec,
            failure_ttl=self.settings.failure_ttl_sec,
            retry=retry,
            description=f"train:{payload['run_id']}",
        )
        return job.get_id()

    def enqueue_eval(self: RQEnqueuer, payload: EvalJobPayload) -> str:
        conn = self._connection()
        q: RQClientQueue = rq_queue(self.queue_name, connection=conn)
        retry = rq_retry(
            max_retries=self.settings.retry_max, intervals=self.settings.retry_intervals
        )
        payload_dict: dict[str, JSONValue] = {
            "run_id": payload["run_id"],
            "split": payload["split"],
            "path_override": payload["path_override"],
        }
        job: RQJobLike = q.enqueue(
            "model_trainer.worker.eval_job.process_eval_job",
            payload_dict,
            job_timeout=self.settings.job_timeout_sec,
            result_ttl=self.settings.result_ttl_sec,
            failure_ttl=self.settings.failure_ttl_sec,
            retry=retry,
            description=f"eval:{payload['run_id']}:{payload['split']}",
        )
        return job.get_id()

    def enqueue_tokenizer(self: RQEnqueuer, payload: TokenizerTrainPayload) -> str:
        conn = self._connection()
        q: RQClientQueue = rq_queue(self.queue_name, connection=conn)
        retry = rq_retry(
            max_retries=self.settings.retry_max, intervals=self.settings.retry_intervals
        )
        payload_dict: dict[str, JSONValue] = {
            "tokenizer_id": payload["tokenizer_id"],
            "method": payload["method"],
            "vocab_size": payload["vocab_size"],
            "min_frequency": payload["min_frequency"],
            "corpus_file_id": payload["corpus_file_id"],
            "holdout_fraction": payload["holdout_fraction"],
            "seed": payload["seed"],
        }
        job: RQJobLike = q.enqueue(
            "model_trainer.worker.tokenizer_worker.process_tokenizer_train_job",
            payload_dict,
            job_timeout=self.settings.job_timeout_sec,
            result_ttl=self.settings.result_ttl_sec,
            failure_ttl=self.settings.failure_ttl_sec,
            retry=retry,
            description=f"tokenizer:{payload['tokenizer_id']}",
        )
        return job.get_id()

    def enqueue_score(self: RQEnqueuer, payload: ScoreJobPayload) -> str:
        conn = self._connection()
        q: RQClientQueue = rq_queue(self.queue_name, connection=conn)
        retry = rq_retry(
            max_retries=self.settings.retry_max, intervals=self.settings.retry_intervals
        )
        payload_dict: dict[str, JSONValue] = {
            "run_id": payload["run_id"],
            "request_id": payload["request_id"],
            "text": payload["text"],
            "path": payload["path"],
            "detail_level": payload["detail_level"],
            "top_k": payload["top_k"],
            "seed": payload["seed"],
        }
        job: RQJobLike = q.enqueue(
            "model_trainer.worker.score_job.process_score_job",
            payload_dict,
            job_timeout=self.settings.job_timeout_sec,
            result_ttl=self.settings.result_ttl_sec,
            failure_ttl=self.settings.failure_ttl_sec,
            retry=retry,
            description=f"score:{payload['run_id']}:{payload['request_id']}",
        )
        return job.get_id()

    def enqueue_generate(self: RQEnqueuer, payload: GenerateJobPayload) -> str:
        conn = self._connection()
        q: RQClientQueue = rq_queue(self.queue_name, connection=conn)
        retry = rq_retry(
            max_retries=self.settings.retry_max, intervals=self.settings.retry_intervals
        )
        stop_seq_json: list[JSONValue] = list(payload["stop_sequences"])
        payload_dict: dict[str, JSONValue] = {
            "run_id": payload["run_id"],
            "request_id": payload["request_id"],
            "prompt_text": payload["prompt_text"],
            "prompt_path": payload["prompt_path"],
            "max_new_tokens": payload["max_new_tokens"],
            "temperature": payload["temperature"],
            "top_k": payload["top_k"],
            "top_p": payload["top_p"],
            "stop_on_eos": payload["stop_on_eos"],
            "stop_sequences": stop_seq_json,
            "seed": payload["seed"],
            "num_return_sequences": payload["num_return_sequences"],
        }
        job: RQJobLike = q.enqueue(
            "model_trainer.worker.generate_job.process_generate_job",
            payload_dict,
            job_timeout=self.settings.job_timeout_sec,
            result_ttl=self.settings.result_ttl_sec,
            failure_ttl=self.settings.failure_ttl_sec,
            retry=retry,
            description=f"generate:{payload['run_id']}:{payload['request_id']}",
        )
        return job.get_id()

    def enqueue_chat(self: RQEnqueuer, payload: ChatJobPayload) -> str:
        conn = self._connection()
        q: RQClientQueue = rq_queue(self.queue_name, connection=conn)
        retry = rq_retry(
            max_retries=self.settings.retry_max, intervals=self.settings.retry_intervals
        )
        payload_dict: dict[str, JSONValue] = {
            "run_id": payload["run_id"],
            "session_id": payload["session_id"],
            "request_id": payload["request_id"],
            "prompt": payload["prompt"],
            "max_new_tokens": payload["max_new_tokens"],
            "temperature": payload["temperature"],
            "top_k": payload["top_k"],
            "top_p": payload["top_p"],
        }
        job: RQJobLike = q.enqueue(
            "model_trainer.worker.chat_job.process_chat_job",
            payload_dict,
            job_timeout=self.settings.job_timeout_sec,
            result_ttl=self.settings.result_ttl_sec,
            failure_ttl=self.settings.failure_ttl_sec,
            retry=retry,
            description=f"chat:{payload['run_id']}:{payload['session_id']}:{payload['request_id']}",
        )
        return job.get_id()
