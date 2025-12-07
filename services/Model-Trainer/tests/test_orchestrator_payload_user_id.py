from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch
from platform_workers.testing import FakeRedis

from model_trainer.api.schemas.runs import TrainRequest
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator


def test_orchestrator_threads_user_id(monkeypatch: MonkeyPatch) -> None:
    s = load_settings()
    r = FakeRedis()
    # Real enqueuer instance for type compatibility
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
    seen: dict[str, int] = {}

    def _fake_enqueue_train(payload: dict[str, str | int | float | bool | None]) -> str:
        val = payload.get("user_id")
        assert val == 42, f"Expected user_id to be 42, got {val}"
        if not isinstance(val, int):
            raise AssertionError(f"Expected user_id to be int, got {type(val)}")
        seen["user_id"] = val
        return "job-1"

    monkeypatch.setattr(enq, "enqueue_train", _fake_enqueue_train)
    orch = TrainingOrchestrator(settings=s, redis_client=r, enqueuer=enq, model_registry=None)
    # Stub CorpusFetcher to avoid network and provide a local path
    import tempfile
    from pathlib import Path

    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
            pass

        def fetch(self: _CF, file_id: str) -> Path:
            return Path(tempfile.gettempdir())

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)
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
    assert out["run_id"] and seen.get("user_id") == 42
    r.assert_only_called({"hset"})
