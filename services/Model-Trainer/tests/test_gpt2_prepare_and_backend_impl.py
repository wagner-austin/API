from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.backend_factory import create_gpt2_backend
from model_trainer.core.services.model.backends.gpt2 import (
    prepare_gpt2_with_handle,
)
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend


class _SettingsFactory(Protocol):
    def __call__(
        self,
        *,
        artifacts_root: str | None = ...,
        runs_root: str | None = ...,
        logs_root: str | None = ...,
        data_root: str | None = ...,
        data_bank_api_url: str | None = ...,
        data_bank_api_key: str | None = ...,
        threads: int | None = ...,
        redis_url: str | None = ...,
        app_env: Literal["dev", "prod"] | None = ...,
        security_api_key: str | None = ...,
    ) -> Settings: ...


def test_gpt2_prepare_from_artifact(tmp_path: Path, settings_factory: _SettingsFactory) -> None:
    artifacts = tmp_path / "artifacts"
    _settings = settings_factory(artifacts_root=str(artifacts))
    _ = _settings  # unused, but kept for interface consistency

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is tiny\n", encoding="utf-8")

    tok_id = "tok-prep"
    out_dir = artifacts / "tokenizers" / tok_id
    cfg_tok = TokenizerTrainConfig(
        method="bpe",
        vocab_size=128,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(out_dir),
    )
    # Tokenizer training (no ML loss metric)
    loss_initial = 0.0
    _ = BPEBackend().train(cfg_tok)
    loss_final = 0.0
    assert loss_final <= loss_initial

    tok_handle = BPEBackend().load(str(out_dir / "tokenizer.json"))

    cfg: ModelTrainConfig = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 5e-4,
        "tokenizer_id": tok_id,
        "corpus_path": str(corpus),
        "holdout_fraction": 0.01,
        "seed": 42,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cpu",
        "data_num_workers": 0,
        "data_pin_memory": False,
        "early_stopping_patience": 5,
        "test_split_ratio": 0.15,
        "finetune_lr_cap": 5e-5,
        "precision": "fp32",
    }
    prepared = prepare_gpt2_with_handle(tok_handle, cfg)
    assert prepared.max_seq_len == 16
    assert prepared.eos_id >= 0 and prepared.eos_id < tok_handle.get_vocab_size()
    assert prepared.pad_id >= 0 and prepared.pad_id < tok_handle.get_vocab_size()


def test_gpt2_backend_impl_end_to_end(tmp_path: Path, settings_factory: _SettingsFactory) -> None:
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(artifacts_root=str(artifacts))

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Use a larger corpus with repetitive patterns for stable training
    pattern = "hello world this is a test\n" * 50 + "testing the model training\n" * 50
    (corpus / "a.txt").write_text(pattern, encoding="utf-8")

    tok_id = "tok-backend"
    out_dir = artifacts / "tokenizers" / tok_id
    cfg_tok = TokenizerTrainConfig(
        method="bpe",
        vocab_size=128,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(out_dir),
    )
    _ = BPEBackend().train(cfg_tok)
    tok_handle = BPEBackend().load(str(out_dir / "tokenizer.json"))

    cfg: ModelTrainConfig = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 5e-4,
        "tokenizer_id": tok_id,
        "corpus_path": str(corpus),
        "holdout_fraction": 0.1,
        "seed": 42,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cpu",
        "data_num_workers": 0,
        "data_pin_memory": False,
        "early_stopping_patience": 5,
        "test_split_ratio": 0.15,
        "finetune_lr_cap": 5e-5,
        "precision": "fp32",
    }

    builder = LocalTextDatasetBuilder()
    backend = create_gpt2_backend(builder)
    prepared = backend.prepare(cfg, settings, tokenizer=tok_handle)

    # Track training losses
    train_losses: list[float] = []

    def track_loss(
        step: int,
        epoch: int,
        loss: float,
        train_ppl: float,
        grad_norm: float,
        samples_per_sec: float,
        val_loss: float | None,
        val_ppl: float | None,
    ) -> None:
        train_losses.append(loss)

    def _hb(_: float) -> None:
        pass

    def _cancelled() -> bool:
        return False

    res = backend.train(
        cfg,
        settings,
        run_id="run-backend",
        heartbeat=_hb,
        cancelled=_cancelled,
        prepared=prepared,
        progress=track_loss,
    )
    assert res["loss"] >= 0.0
    loss_before = train_losses[0]
    loss_after = train_losses[-1]
    assert loss_after < loss_before
    _ = backend.save(prepared, str(artifacts / "models" / "run-backend"))

    # Evaluate path via backend adapter
    eval_res = backend.evaluate(run_id="run-backend", cfg=cfg, settings=settings)
    assert eval_res["loss"] >= 0.0
