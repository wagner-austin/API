"""Tests for TrainingOrchestrator.enqueue_resume and the resume route."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.job_types import JobStatusLiteral
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str, narrow_json_to_dict
from platform_workers.redis import _RedisBytesClient
from platform_workers.testing import FakeQueue, FakeRedis, FakeRedisBytesClient, FakeRetry

from model_trainer.api.main import create_app
from model_trainer.api.schemas.runs import TrainRequest
from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings, load_settings
from model_trainer.core.infra.paths import checkpoints_dir
from model_trainer.core.services.container import ServiceContainer
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator
from model_trainer.worker.trainer_job_store import TrainerJobStore

RUN_ID = "run-resumable"


def _make_request() -> TrainRequest:
    return {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 128,
        "num_epochs": 1,
        "batch_size": 2,
        "learning_rate": 5e-4,
        "corpus_file_id": "deadbeef",
        "tokenizer_id": "tok1",
        "user_id": 42,
        "holdout_fraction": 0.01,
        "seed": 42,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cpu",
        "early_stopping_patience": 5,
        "test_split_ratio": 0.15,
        "finetune_lr_cap": 5e-5,
        "loss_mask_prefix_separator": None,
        "precision": "auto",
        "data_num_workers": None,
        "data_pin_memory": None,
        "hub_model_id": None,
        "finetuning_strategy": "full",
        "lora": None,
        "quantization": None,
        "gguf_export": None,
    }


def _install_fake_rq(fake_queue: FakeQueue) -> None:
    def _fake_rq_connection(url: str) -> _RedisBytesClient:
        return FakeRedisBytesClient()

    def _fake_rq_queue(name: str, connection: _RedisBytesClient) -> FakeQueue:
        return fake_queue

    def _fake_rq_retry(*, max_retries: int, intervals: list[int]) -> FakeRetry:
        return FakeRetry(max=max_retries, interval=intervals)

    _test_hooks.rq_connection_factory = _fake_rq_connection
    _test_hooks.rq_queue_factory = _fake_rq_queue
    _test_hooks.rq_retry_factory = _fake_rq_retry


def _enqueuer() -> RQEnqueuer:
    return RQEnqueuer(
        redis_url="redis://localhost:6379/0",
        settings=RQSettings(
            job_timeout_sec=60,
            result_ttl_sec=60,
            failure_ttl_sec=60,
            retry_max=0,
            retry_intervals=[],
        ),
    )


def _make_orchestrator(tmp_path: Path, redis: FakeRedis) -> tuple[TrainingOrchestrator, Settings]:
    settings = load_settings()
    settings["app"]["artifacts_root"] = str(tmp_path / "artifacts")
    orch = TrainingOrchestrator(
        settings=settings, redis_client=redis, enqueuer=_enqueuer(), model_registry=None
    )
    return orch, settings


def _seed_status(redis: FakeRedis, run_id: str, status: JobStatusLiteral) -> None:
    now = datetime.utcnow()
    TrainerJobStore(redis).save(
        {
            "job_id": run_id,
            "user_id": 42,
            "status": status,
            "progress": 100,
            "message": "seeded",
            "created_at": now,
            "updated_at": now,
            "error": None,
            "artifact_file_id": None,
        },
    )


def _touch_checkpoint(settings: Settings, run_id: str) -> None:
    checkpoints_dir(settings).mkdir(parents=True, exist_ok=True)
    (checkpoints_dir(settings) / f"{run_id}.pt").write_bytes(b"\x00")


def _payload_field(raw: JSONValue, key: str) -> JSONValue:
    assert isinstance(raw, dict)
    return raw[key]


class TestEnqueueResume:
    """Status and checkpoint gates on the resume enqueue."""

    def test_unknown_run_is_refused(self, tmp_path: Path) -> None:
        redis = FakeRedis()
        orch, _ = _make_orchestrator(tmp_path, redis)
        with pytest.raises(AppError) as excinfo:
            _ = orch.enqueue_resume(RUN_ID, _make_request())
        exc: AppError[ModelTrainerErrorCode] = excinfo.value
        assert exc.code == ModelTrainerErrorCode.RUN_NOT_FOUND
        redis.assert_only_called({"hgetall"})

    @pytest.mark.parametrize("status", ["queued", "processing", "completed"])
    def test_non_failed_run_is_refused(self, tmp_path: Path, status: JobStatusLiteral) -> None:
        redis = FakeRedis()
        orch, _ = _make_orchestrator(tmp_path, redis)
        _seed_status(redis, RUN_ID, status)
        with pytest.raises(AppError) as excinfo:
            _ = orch.enqueue_resume(RUN_ID, _make_request())
        exc: AppError[ModelTrainerErrorCode] = excinfo.value
        assert exc.code == ModelTrainerErrorCode.RUN_NOT_RESUMABLE
        assert status in str(exc)
        redis.assert_only_called({"hset", "hgetall"})

    def test_failed_run_without_checkpoint_is_refused(self, tmp_path: Path) -> None:
        redis = FakeRedis()
        orch, _ = _make_orchestrator(tmp_path, redis)
        _seed_status(redis, RUN_ID, "failed")
        with pytest.raises(AppError) as excinfo:
            _ = orch.enqueue_resume(RUN_ID, _make_request())
        exc: AppError[ModelTrainerErrorCode] = excinfo.value
        assert exc.code == ModelTrainerErrorCode.CHECKPOINT_NOT_FOUND
        redis.assert_only_called({"hset", "hgetall"})

    def test_failed_run_with_checkpoint_enqueues_resume_execution(self, tmp_path: Path) -> None:
        redis = FakeRedis()
        fake_queue = FakeQueue(job_id="job-resume-1")
        _install_fake_rq(fake_queue)
        orch, settings = _make_orchestrator(tmp_path, redis)
        _seed_status(redis, RUN_ID, "failed")
        _touch_checkpoint(settings, RUN_ID)

        out = orch.enqueue_resume(RUN_ID, _make_request())
        assert out["run_id"] == RUN_ID
        assert out["job_id"] == "job-resume-1"

        assert len(fake_queue.jobs) == 1
        raw = fake_queue.jobs[0].args[0]
        assert _payload_field(raw, "run_id") == RUN_ID
        assert _payload_field(raw, "resume") is True

        status = TrainerJobStore(redis).load(RUN_ID)
        assert status is not None and status["status"] == "queued"
        assert status["message"] == "resume queued"
        redis.assert_only_called({"hset", "hgetall"})

    def test_fresh_enqueue_carries_resume_false(self, tmp_path: Path) -> None:
        redis = FakeRedis()
        fake_queue = FakeQueue(job_id="job-fresh-1")
        _install_fake_rq(fake_queue)
        orch, _ = _make_orchestrator(tmp_path, redis)

        out = orch.enqueue_training(_make_request())
        assert out["job_id"] == "job-fresh-1"
        raw = fake_queue.jobs[0].args[0]
        assert _payload_field(raw, "resume") is False
        redis.assert_only_called({"hset"})


class TestResumeRoute:
    """The HTTP surface maps orchestration errors to status codes."""

    def test_resume_unknown_run_returns_404(self) -> None:
        app = create_app(load_settings())
        client = TestClient(app)
        response = client.post(
            "/runs/run-never-seen/resume",
            content=dump_json_str(dict(_make_request())),
        )
        assert response.status_code == 404
        error_body = narrow_json_to_dict(load_json_str(response.text))
        assert error_body["code"] == "RUN_NOT_FOUND"

    def test_resume_success_returns_run_and_job(self, tmp_path: Path) -> None:
        settings = load_settings()
        settings["app"]["artifacts_root"] = str(tmp_path / "artifacts")
        app = create_app(settings)
        container: ServiceContainer = app.state.container
        fake_redis = FakeRedis()
        container.redis = fake_redis
        # The container builds its orchestrator eagerly against the real
        # redis client; rebuild it against the fake for this test.
        container.training_orchestrator = TrainingOrchestrator(
            settings=settings,
            redis_client=fake_redis,
            enqueuer=container.rq_enqueuer,
            model_registry=container.model_registry,
        )
        fake_queue = FakeQueue(job_id="job-route-1")
        _install_fake_rq(fake_queue)

        _seed_status(fake_redis, RUN_ID, "failed")
        _touch_checkpoint(settings, RUN_ID)

        client = TestClient(app)
        response = client.post(
            f"/runs/{RUN_ID}/resume",
            content=dump_json_str(dict(_make_request())),
        )
        assert response.status_code == 200
        body = narrow_json_to_dict(load_json_str(response.text))
        assert body["run_id"] == RUN_ID
        assert body["job_id"] == "job-route-1"
        assert len(fake_queue.jobs) == 1
        fake_redis.assert_only_called({"hset", "hgetall"})
