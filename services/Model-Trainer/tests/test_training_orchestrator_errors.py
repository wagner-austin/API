from __future__ import annotations

from datetime import datetime

import pytest
from platform_core.errors import AppError
from platform_core.job_types import JobStatusLiteral
from platform_core.json_utils import dump_json_str, load_json_str
from platform_core.trainer_keys import eval_key
from platform_workers.testing import FakeRedis

from model_trainer.api.schemas.runs import EvaluateRequest, TrainRequest
from model_trainer.core.config.settings import load_settings
from model_trainer.core.contracts.queue import EvalJobPayload, TrainJobPayload
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.core.services.registries import ModelRegistry
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator
from model_trainer.worker.trainer_job_store import TrainerJobStore


def test_trainer_job_store_initial_status() -> None:
    """Test TrainerJobStore.initial_status helper method."""
    fake = FakeRedis()
    store = TrainerJobStore(fake)
    status = store.initial_status(
        job_id="run-init", user_id=42, message="initializing", status="queued"
    )
    assert status["job_id"] == "run-init"
    assert status["user_id"] == 42
    assert status["status"] == "queued"
    assert status["message"] == "initializing"
    assert status["progress"] == 0
    assert status["error"] is None
    assert status["artifact_file_id"] is None
    assert status["created_at"].year >= 2020 and status["created_at"] <= datetime.utcnow()
    assert status["updated_at"].year >= 2020 and status["updated_at"] <= datetime.utcnow()
    fake.assert_only_called(set())


def _save_status(
    fake: FakeRedis, run_id: str, status: JobStatusLiteral, message: str | None = None
) -> None:
    now = datetime.utcnow()
    TrainerJobStore(fake).save(
        {
            "job_id": run_id,
            "user_id": 0,
            "status": status,
            "progress": 0,
            "message": message,
            "created_at": now,
            "updated_at": now,
            "error": None,
            "artifact_file_id": None,
        }
    )


class _FakeEnq(RQEnqueuer):
    def __init__(self: _FakeEnq) -> None:
        super().__init__(redis_url="redis://x", settings=RQSettings(1, 1, 1, 0, []))
        self.last: tuple[str, TrainJobPayload | EvalJobPayload] | None = None

    def enqueue_train(self: _FakeEnq, payload: TrainJobPayload) -> str:
        self.last = ("train", payload)
        return "job-train"

    def enqueue_eval(self: _FakeEnq, payload: EvalJobPayload) -> str:
        self.last = ("eval", payload)
        return "job-eval"


def test_orchestrator_unsupported_model_raises() -> None:
    fake = FakeRedis()
    reg = ModelRegistry(registrations={}, dataset_builder=LocalTextDatasetBuilder())
    orch = TrainingOrchestrator(
        settings=load_settings(),
        redis_client=fake,
        enqueuer=_FakeEnq(),
        model_registry=reg,
    )
    req = TrainRequest(
        model_family="llama",
        model_size="s",
        max_seq_len=16,
        num_epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        corpus_file_id="deadbeef",
        tokenizer_id="tok",
        user_id=0,
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
    with pytest.raises(AppError):
        _ = orch.enqueue_training(req)
    fake.assert_only_called(set())


def test_orchestrator_status_missing() -> None:
    fake = FakeRedis()
    orch = TrainingOrchestrator(
        settings=load_settings(),
        redis_client=fake,
        enqueuer=_FakeEnq(),
        model_registry=None,
    )
    with pytest.raises(AppError):
        _ = orch.get_status("no-run")
    fake.assert_only_called({"hgetall"})


def test_orchestrator_eval_missing_run_returns_failed() -> None:
    fake = FakeRedis()
    orch = TrainingOrchestrator(
        settings=load_settings(),
        redis_client=fake,
        enqueuer=_FakeEnq(),
        model_registry=None,
    )
    out = orch.enqueue_evaluation("no-run", EvaluateRequest(split="validation"))
    assert out["status"] == "failed"
    fake.assert_only_called({"hgetall"})


def test_orchestrator_status_queued() -> None:
    fake = FakeRedis()
    _save_status(fake, "run-q", "queued")
    orch = TrainingOrchestrator(
        settings=load_settings(),
        redis_client=fake,
        enqueuer=_FakeEnq(),
        model_registry=None,
    )
    res = orch.get_status("run-q")
    assert res["status"] == "queued"
    fake.assert_only_called({"hset", "hgetall", "get"})


def test_orchestrator_status_running() -> None:
    fake = FakeRedis()
    _save_status(fake, "run-r", "processing")
    orch = TrainingOrchestrator(
        settings=load_settings(),
        redis_client=fake,
        enqueuer=_FakeEnq(),
        model_registry=None,
    )
    res = orch.get_status("run-r")
    assert res["status"] == "running"
    fake.assert_only_called({"hset", "hgetall", "get"})


def test_orchestrator_status_completed() -> None:
    fake = FakeRedis()
    _save_status(fake, "run-c", "completed")
    orch = TrainingOrchestrator(
        settings=load_settings(),
        redis_client=fake,
        enqueuer=_FakeEnq(),
        model_registry=None,
    )
    res = orch.get_status("run-c")
    assert res["status"] == "completed"
    fake.assert_only_called({"hset", "hgetall", "get"})


def test_orchestrator_status_failed() -> None:
    fake = FakeRedis()
    _save_status(fake, "run-f", "failed")
    orch = TrainingOrchestrator(
        settings=load_settings(),
        redis_client=fake,
        enqueuer=_FakeEnq(),
        model_registry=None,
    )
    res = orch.get_status("run-f")
    assert res["status"] == "failed"
    fake.assert_only_called({"hset", "hgetall", "get"})


def test_orchestrator_eval_enqueues_and_sets_cache() -> None:
    fake = FakeRedis()
    # Mark run as present
    _save_status(fake, "run-ok", "processing")
    orch = TrainingOrchestrator(
        settings=load_settings(),
        redis_client=fake,
        enqueuer=_FakeEnq(),
        model_registry=None,
    )
    eval_request: EvaluateRequest = {"split": "validation"}
    out = orch.enqueue_evaluation("run-ok", eval_request)
    assert out["status"] == "queued"
    raw_obj: str | int | bool | None = fake.get(eval_key("run-ok"))
    if not isinstance(raw_obj, str):
        raise AssertionError(f"Expected raw_obj to be str, got {type(raw_obj)}")
    cached_eval = load_json_str(raw_obj)
    if not isinstance(cached_eval, dict):
        raise AssertionError(f"Expected cached_eval to be dict, got {type(cached_eval)}")
    assert cached_eval["status"] == "queued"
    assert cached_eval["split"] == "validation"
    fake.assert_only_called({"hset", "hgetall", "set", "get"})


@pytest.mark.parametrize(
    ("status_value", "expected"),
    [
        ("queued", "queued"),
        ("running", "running"),
        ("completed", "completed"),
        ("failed", "failed"),
        ("unknown", "failed"),
    ],
)
def test_orchestrator_get_evaluation_status_variants(status_value: str, expected: str) -> None:
    fake = FakeRedis()
    cache = {
        "status": status_value,
        "split": "validation",
        "loss": 1.0,
        "ppl": 2.0,
        "artifact": "/tmp/a",
    }
    fake.set(eval_key("run-status"), dump_json_str(cache))
    orch = TrainingOrchestrator(
        settings=load_settings(),
        redis_client=fake,
        enqueuer=_FakeEnq(),
        model_registry=None,
    )
    out = orch.get_evaluation("run-status")
    assert out["status"] == expected
    fake.assert_only_called({"set", "get"})
