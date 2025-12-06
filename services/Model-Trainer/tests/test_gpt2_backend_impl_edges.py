from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.backend_factory import create_gpt2_backend
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


def test_gpt2_backend_impl_name_and_type_errors(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(artifacts_root=str(artifacts))

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\n", encoding="utf-8")

    tok_id = "tok-bpe"
    out_dir = artifacts / "tokenizers" / tok_id
    # Train real tokenizer
    from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig

    cfg_tok = TokenizerTrainConfig(
        method="bpe",
        vocab_size=128,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(out_dir),
    )
    loss_initial = 0.0
    _ = BPEBackend().train(cfg_tok)
    tok_handle = BPEBackend().load(str(out_dir / "tokenizer.json"))
    loss_final = 0.0
    assert loss_final <= loss_initial

    cfg: ModelTrainConfig = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 8,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
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
    backend = create_gpt2_backend(LocalTextDatasetBuilder())
    assert backend.name() == "gpt2"
    # Prepare and save works for proper type
    prepared = backend.prepare(cfg, settings, tokenizer=tok_handle)
    _ = backend.save(prepared, str(artifacts / "models" / "run-x"))
