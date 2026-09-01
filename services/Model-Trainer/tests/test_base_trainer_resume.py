"""Integration tests for epoch-boundary checkpointing and explicit resume.

The gold assertion: a run interrupted after two epochs and resumed produces
bit-identical final weights and loss to the same run trained start to finish,
because the checkpoint restores model, optimizer and every RNG stream at the
epoch boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from platform_core.determinism_record import UNPINNED_STACK, determinism_record
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import JSONValue, load_json_str, narrow_json_to_dict

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    TrainingCheckpointMeta,
)
from model_trainer.core.contracts.model import (
    ModelBackend,
    ModelTrainConfig,
    PreparedLMModel,
    TrainOutcome,
)
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.infra.paths import model_dir
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.backend_factory import create_char_lstm_backend
from model_trainer.core.services.tokenizer.char_backend import CharBackend
from model_trainer.core.services.training.base_trainer import BaseTrainer
from model_trainer.core.services.training.base_trainer_core import AdamW
from model_trainer.core.services.training.checkpoint import (
    TrainingCheckpoint,
    capture_rng_states,
    checkpoint_exists,
    load_training_checkpoint,
    save_training_checkpoint,
)

TOTAL_EPOCHS = 3


def _write_tiny_corpus(root: Path) -> str:
    out_dir = root / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_lines = ["aba", "abbaba", "abaaba", "babbab", "ababab", "bababa"]
    (out_dir / "tiny.txt").write_text("\n".join(corpus_lines * 10) + "\n", encoding="utf-8")
    return str(out_dir)


def _train_char_tokenizer(root: Path, corpus_path: str) -> str:
    tok_out = root / "artifacts" / "tokenizers" / "tok1"
    cfg = TokenizerTrainConfig(
        method="char",
        vocab_size=0,
        min_frequency=1,
        corpus_path=corpus_path,
        holdout_fraction=0.05,
        seed=42,
        out_dir=str(tok_out),
    )
    _ = CharBackend().train(cfg)
    return "tok1"


def _make_cfg(corpus_path: str, tokenizer_id: str) -> ModelTrainConfig:
    return {
        "model_family": "char_lstm",
        "model_size": "tiny",
        "max_seq_len": 16,
        "num_epochs": TOTAL_EPOCHS,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "tokenizer_id": tokenizer_id,
        "corpus_path": corpus_path,
        "corpus_format": "lines",
        "holdout_fraction": 0.0,
        "seed": 42,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cpu",
        "data_num_workers": 0,
        "data_pin_memory": False,
        "early_stopping_patience": 0,
        "test_split_ratio": 0.0,
        "finetune_lr_cap": 0.0,
        "loss_mask_prefix_separator": None,
        "precision": "fp32",
        "finetuning_strategy": "full",
        "hub_model_id": None,
        "lora": None,
        "quantization": None,
        "gguf_export": None,
    }


UNPINNED = determinism_record(UNPINNED_STACK, {})
"""What this test ran under: nothing pinned, recorded as that rather than null."""


def _noop(_: float) -> None:
    return None


def _never() -> bool:
    return False


class _CancelAfterCalls:
    """Cancellation callback that fires after a fixed number of checks."""

    def __init__(self, calls_before_cancel: int) -> None:
        self._remaining = calls_before_cancel

    def __call__(self) -> bool:
        if self._remaining <= 0:
            return True
        self._remaining -= 1
        return False


_ProgressCallback = Callable[
    [int, int, float, float, float, float, float | None, float | None], None
]


def _train(
    backend: ModelBackend,
    cfg: ModelTrainConfig,
    settings: Settings,
    *,
    run_id: str,
    cancelled: Callable[[], bool],
    resume: bool,
    prepared: PreparedLMModel,
    progress: _ProgressCallback | None = None,
) -> TrainOutcome:
    """Run one training execution through the backend under test."""
    return backend.train(
        cfg,
        settings,
        run_id=run_id,
        heartbeat=_noop,
        cancelled=cancelled,
        resume=resume,
        prepared=prepared,
        progress=progress,
        determinism=UNPINNED,
    )


def _prepare(settings: Settings, cfg: ModelTrainConfig) -> tuple[PreparedLMModel, ModelBackend]:
    backend = create_char_lstm_backend(LocalTextDatasetBuilder())
    tokenizer_id = cfg["tokenizer_id"]
    assert tokenizer_id is not None
    tok_dir = Path(settings["app"]["artifacts_root"]) / "tokenizers" / tokenizer_id
    handle = CharBackend().load(str(tok_dir / "tokenizer.json"))
    # prepare() draws initial weights from the global torch RNG, and train()
    # seeds only after preparation; seed here so every prepared model in a
    # test starts from identical weights.
    torch.manual_seed(cfg["seed"])
    prepared = backend.prepare(cfg, settings, tokenizer=handle)
    return prepared, backend


def _read_manifest(settings: Settings, run_id: str) -> dict[str, JSONValue]:
    text = (model_dir(settings, run_id) / "manifest.json").read_text(encoding="utf-8")
    decoded = narrow_json_to_dict(load_json_str(text))
    return dict(decoded)


def _count_batch_checks_per_epoch(steps_total: int) -> int:
    """Batches per epoch, derived from a full run's step count."""
    assert steps_total % TOTAL_EPOCHS == 0
    return steps_total // TOTAL_EPOCHS


def test_interrupted_run_resumes_bit_identical(
    settings_with_paths: Settings, tmp_path: Path
) -> None:
    corpus_path = _write_tiny_corpus(tmp_path)
    tokenizer_id = _train_char_tokenizer(tmp_path, corpus_path)
    cfg = _make_cfg(corpus_path, tokenizer_id)

    # Reference: the same run trained start to finish in one execution,
    # with the loss trajectory tracked so learning itself is asserted.
    losses: list[float] = []

    def _track(
        step: int,
        epoch: int,
        loss: float,
        train_ppl: float,
        grad_norm: float,
        samples_per_sec: float,
        val_loss: float | None,
        val_ppl: float | None,
    ) -> None:
        losses.append(loss)

    prepared_full, backend = _prepare(settings_with_paths, cfg)
    full = _train(
        backend,
        cfg,
        settings_with_paths,
        run_id="run-full",
        cancelled=_never,
        resume=False,
        prepared=prepared_full,
        progress=_track,
    )
    assert full["cancelled"] is False
    loss_before = losses[0]
    loss_after = losses[-1]
    assert loss_after < loss_before
    assert checkpoint_exists(settings_with_paths, "run-full") is False
    full_manifest = _read_manifest(settings_with_paths, "run-full")
    assert full_manifest["resumed_from_epoch"] is None

    # Interrupted: cancel at the first batch of epoch 3, after two completed
    # epochs. The cancellation callback is checked once per batch.
    batches_per_epoch = _count_batch_checks_per_epoch(full["steps"])
    prepared_interrupted, _ = _prepare(settings_with_paths, cfg)
    interrupted = _train(
        backend,
        cfg,
        settings_with_paths,
        run_id="run-resumed",
        cancelled=_CancelAfterCalls(2 * batches_per_epoch),
        resume=False,
        prepared=prepared_interrupted,
        progress=None,
    )
    assert interrupted["cancelled"] is True
    assert interrupted["steps"] == 2 * batches_per_epoch
    assert checkpoint_exists(settings_with_paths, "run-resumed") is True
    persisted = load_training_checkpoint(settings_with_paths, "run-resumed")
    assert persisted.meta["epochs_completed"] == 2
    assert persisted.meta["global_step"] == 2 * batches_per_epoch

    # Resume: a fresh execution continues from the checkpoint and must land
    # exactly where the uninterrupted run landed.
    prepared_resumed, _ = _prepare(settings_with_paths, cfg)
    resumed = _train(
        backend,
        cfg,
        settings_with_paths,
        run_id="run-resumed",
        cancelled=_never,
        resume=True,
        prepared=prepared_resumed,
        progress=None,
    )
    assert resumed["cancelled"] is False
    assert resumed["steps"] == full["steps"]
    assert resumed["loss"] == full["loss"]
    assert resumed["perplexity"] == full["perplexity"]

    full_state = prepared_full.model.state_dict()
    resumed_state = prepared_resumed.model.state_dict()
    assert sorted(full_state) == sorted(resumed_state)
    for name, tensor in full_state.items():
        assert torch.equal(tensor, resumed_state[name]), f"weights differ at {name}"

    assert checkpoint_exists(settings_with_paths, "run-resumed") is False
    manifest = _read_manifest(settings_with_paths, "run-resumed")
    assert manifest["resumed_from_epoch"] == 2
    assert manifest["steps"] == full["steps"]


def test_resume_without_checkpoint_is_refused(
    settings_with_paths: Settings, tmp_path: Path
) -> None:
    corpus_path = _write_tiny_corpus(tmp_path)
    tokenizer_id = _train_char_tokenizer(tmp_path, corpus_path)
    cfg = _make_cfg(corpus_path, tokenizer_id)
    prepared, backend = _prepare(settings_with_paths, cfg)

    with pytest.raises(AppError) as excinfo:
        _ = _train(
            backend,
            cfg,
            settings_with_paths,
            run_id="run-nothing-persisted",
            cancelled=_never,
            resume=True,
            prepared=prepared,
            progress=None,
        )
    exc: AppError[ModelTrainerErrorCode] = excinfo.value
    assert exc.code == ModelTrainerErrorCode.CHECKPOINT_NOT_FOUND


def test_resume_with_changed_config_is_refused(
    settings_with_paths: Settings, tmp_path: Path
) -> None:
    corpus_path = _write_tiny_corpus(tmp_path)
    tokenizer_id = _train_char_tokenizer(tmp_path, corpus_path)
    cfg = _make_cfg(corpus_path, tokenizer_id)

    prepared_probe, backend = _prepare(settings_with_paths, cfg)
    probe = _train(
        backend,
        cfg,
        settings_with_paths,
        run_id="run-probe-mismatch",
        cancelled=_never,
        resume=False,
        prepared=prepared_probe,
        progress=None,
    )
    batches_per_epoch = _count_batch_checks_per_epoch(probe["steps"])

    prepared, _ = _prepare(settings_with_paths, cfg)
    interrupted = _train(
        backend,
        cfg,
        settings_with_paths,
        run_id="run-mismatch",
        cancelled=_CancelAfterCalls(batches_per_epoch),
        resume=False,
        prepared=prepared,
        progress=None,
    )
    assert interrupted["cancelled"] is True
    assert checkpoint_exists(settings_with_paths, "run-mismatch") is True

    changed = _make_cfg(corpus_path, tokenizer_id)
    changed["seed"] = 43
    changed["learning_rate"] = 5e-4
    prepared_again, _ = _prepare(settings_with_paths, changed)
    with pytest.raises(AppError) as excinfo:
        _ = _train(
            backend,
            changed,
            settings_with_paths,
            run_id="run-mismatch",
            cancelled=_never,
            resume=True,
            prepared=prepared_again,
            progress=None,
        )
    exc2: AppError[ModelTrainerErrorCode] = excinfo.value
    assert exc2.code == ModelTrainerErrorCode.CHECKPOINT_CONFIG_MISMATCH
    assert "learning_rate, seed" in str(exc2)
    # The refused resume must leave the checkpoint untouched.
    assert checkpoint_exists(settings_with_paths, "run-mismatch") is True


def test_resume_of_fully_trained_checkpoint_completes_without_stepping(
    settings_with_paths: Settings, tmp_path: Path
) -> None:
    corpus_path = _write_tiny_corpus(tmp_path)
    tokenizer_id = _train_char_tokenizer(tmp_path, corpus_path)
    cfg = _make_cfg(corpus_path, tokenizer_id)
    prepared, backend = _prepare(settings_with_paths, cfg)

    # Interrupt during the FINAL epoch, so the checkpoint records every epoch
    # but the run still ended cancelled.
    full_steps_probe = _train(
        backend,
        cfg,
        settings_with_paths,
        run_id="run-probe",
        cancelled=_never,
        resume=False,
        prepared=prepared,
        progress=None,
    )
    batches_per_epoch = _count_batch_checks_per_epoch(full_steps_probe["steps"])

    prepared_b, _ = _prepare(settings_with_paths, cfg)
    interrupted = _train(
        backend,
        cfg,
        settings_with_paths,
        run_id="run-tail",
        cancelled=_CancelAfterCalls(TOTAL_EPOCHS * batches_per_epoch - 1),
        resume=False,
        prepared=prepared_b,
        progress=None,
    )
    assert interrupted["cancelled"] is True
    persisted = load_training_checkpoint(settings_with_paths, "run-tail")
    assert persisted.meta["epochs_completed"] == TOTAL_EPOCHS - 1

    prepared_c, _ = _prepare(settings_with_paths, cfg)
    resumed = _train(
        backend,
        cfg,
        settings_with_paths,
        run_id="run-tail",
        cancelled=_never,
        resume=True,
        prepared=prepared_c,
        progress=None,
    )
    assert resumed["cancelled"] is False
    assert resumed["steps"] == full_steps_probe["steps"]
    assert checkpoint_exists(settings_with_paths, "run-tail") is False


def test_apply_checkpoint_restores_best_and_summaries(
    settings_with_paths: Settings, tmp_path: Path
) -> None:
    """The restore path rehydrates early-stopping state, best-model marker,
    counters, summaries and accumulated timing."""
    corpus_path = _write_tiny_corpus(tmp_path)
    tokenizer_id = _train_char_tokenizer(tmp_path, corpus_path)
    cfg = _make_cfg(corpus_path, tokenizer_id)
    prepared, _ = _prepare(settings_with_paths, cfg)

    trainer = BaseTrainer(
        prepared,
        cfg,
        settings_with_paths,
        run_id="run-apply",
        redis_hb=_noop,
        cancelled=_never,
        resume=True,
        determinism=UNPINNED,
    )
    meta: TrainingCheckpointMeta = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": "run-apply",
        "epochs_completed": 2,
        "global_step": 20,
        "last_loss": 0.75,
        "best_val_loss": 0.9,
        "epochs_no_improve": 1,
        "best_saved": True,
        "total_samples_processed": 40,
        "total_tokens_processed": 640,
        "elapsed_seconds": 33.5,
        "started_at_iso": "2026-08-18T01:02:03",
        "epoch_summaries": [
            {"epoch": 0, "train_loss": 1.5, "train_ppl": 4.5, "val_loss": 1.2, "val_ppl": 3.3},
            {"epoch": 1, "train_loss": 1.0, "train_ppl": 2.7, "val_loss": 0.9, "val_ppl": 2.5},
        ],
        "config": cfg,
    }
    restored = TrainingCheckpoint(
        meta=meta,
        model_state=prepared.model.state_dict(),
        optimizer_state={"state": {}, "param_groups": []},
        rng=capture_rng_states(),
    )
    trainer._require_matching_config(meta)
    trainer._apply_checkpoint(restored)

    assert trainer._es_state == {"best_val_loss": 0.9, "epochs_no_improve": 1}
    assert trainer._best_checkpoint_path == Path(str(model_dir(settings_with_paths, "run-apply")))
    assert trainer._total_samples_processed == 40
    assert trainer._total_tokens_processed == 640
    assert trainer._epoch_summaries == [
        (0, 1.5, 4.5, 1.2, 3.3),
        (1, 1.0, 2.7, 0.9, 2.5),
    ]
    assert trainer._elapsed_before == 33.5
    assert trainer._training_start_iso == "2026-08-18T01:02:03"
    assert trainer._resumed_from_epoch == 2


def test_resume_with_all_epochs_complete_skips_the_loop(
    settings_with_paths: Settings, tmp_path: Path
) -> None:
    """A checkpoint recording every epoch resumes straight to completion,
    reporting the checkpoint's own step count and loss."""
    corpus_path = _write_tiny_corpus(tmp_path)
    tokenizer_id = _train_char_tokenizer(tmp_path, corpus_path)
    cfg = _make_cfg(corpus_path, tokenizer_id)
    prepared, backend = _prepare(settings_with_paths, cfg)

    meta: TrainingCheckpointMeta = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": "run-complete",
        "epochs_completed": TOTAL_EPOCHS,
        "global_step": 39,
        "last_loss": 0.25,
        "best_val_loss": None,
        "epochs_no_improve": 0,
        "best_saved": False,
        "total_samples_processed": 78,
        "total_tokens_processed": 1248,
        "elapsed_seconds": 5.0,
        "started_at_iso": "2026-08-18T02:00:00",
        "epoch_summaries": [],
        "config": cfg,
    }
    # The optimizer state must match a real AdamW over this model's
    # parameters; torch refuses a state dict with foreign param groups.
    optimizer_state = AdamW(prepared.model.parameters(), lr=cfg["learning_rate"]).state_dict()
    _ = save_training_checkpoint(
        settings_with_paths,
        TrainingCheckpoint(
            meta=meta,
            model_state=prepared.model.state_dict(),
            optimizer_state=optimizer_state,
            rng=capture_rng_states(),
        ),
    )

    resumed = _train(
        backend,
        cfg,
        settings_with_paths,
        run_id="run-complete",
        cancelled=_never,
        resume=True,
        prepared=prepared,
        progress=None,
    )
    assert resumed["cancelled"] is False
    assert resumed["steps"] == 39
    assert resumed["loss"] == 0.25
    assert checkpoint_exists(settings_with_paths, "run-complete") is False
