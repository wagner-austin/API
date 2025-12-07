from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_workers.testing import FakeRedis

from model_trainer.api.schemas.runs import TrainRequest
from model_trainer.api.schemas.tokenizers import TokenizerTrainRequest
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.orchestrators.tokenizer_orchestrator import TokenizerOrchestrator
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator


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


def test_tokenizer_orchestrator_rejects_empty_corpus_file_id(tmp_path: Path) -> None:
    s = load_settings()
    # point artifacts root to tmp to avoid unrelated writes
    s["app"]["artifacts_root"] = str(tmp_path / "artifacts")
    r = FakeRedis()
    orch = TokenizerOrchestrator(settings=s, redis_client=r, enqueuer=_enqueuer())
    # Provide whitespace corpus_file_id which passes schema min_length but fails after strip
    req = TokenizerTrainRequest(
        method="bpe",
        vocab_size=128,
        min_frequency=1,
        corpus_file_id=" ",
        holdout_fraction=0.1,
        seed=1,
    )
    with pytest.raises(AppError) as ei:
        _ = orch.enqueue_training(req)
    exc: AppError[ModelTrainerErrorCode] = ei.value
    assert exc.code == ModelTrainerErrorCode.CORPUS_NOT_FOUND
    r.assert_only_called(set())


def test_training_orchestrator_rejects_empty_corpus_file_id(tmp_path: Path) -> None:
    s = load_settings()
    s["app"]["artifacts_root"] = str(tmp_path / "artifacts")
    r = FakeRedis()
    orch = TrainingOrchestrator(
        settings=s, redis_client=r, enqueuer=_enqueuer(), model_registry=None
    )
    req = TrainRequest(
        model_family="gpt2",
        model_size="s",
        max_seq_len=16,
        num_epochs=1,
        batch_size=1,
        learning_rate=5e-4,
        corpus_file_id=" ",
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
    with pytest.raises(AppError) as ei2:
        _ = orch.enqueue_training(req)
    exc2: AppError[ModelTrainerErrorCode] = ei2.value
    assert exc2.code == ModelTrainerErrorCode.CORPUS_NOT_FOUND
    r.assert_only_called(set())
