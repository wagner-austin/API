from __future__ import annotations

from pathlib import Path

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.model.backends.char_lstm.io import (
    get_model_max_seq_len,
)
from model_trainer.core.services.model.backends.char_lstm.model import (
    CharLSTM,
    CharLSTMModel,
)
from model_trainer.core.services.model.backends.char_lstm.prepare import (
    prepare_char_lstm_with_handle,
)
from model_trainer.core.services.tokenizer.char_backend import CharBackend


def _make_char_tokenizer(tmp: Path, corpus_dir: Path) -> tuple[str, str]:
    out_dir = tmp / "artifacts" / "tokenizers" / "tokC"
    cfg = TokenizerTrainConfig(
        method="char",
        vocab_size=0,
        min_frequency=1,
        corpus_path=str(corpus_dir),
        holdout_fraction=0.05,
        seed=42,
        out_dir=str(out_dir),
    )
    CharBackend().train(cfg)
    return "tokC", str(out_dir)


def test_get_model_max_seq_len_default_branch(tmp_path: Path) -> None:
    # Build a minimal model then force a non-int n_positions to hit default branch
    m = CharLSTM(vocab_size=8, embed_dim=4, hidden_dim=6, num_layers=1, dropout=0.0, max_seq_len=32)
    delattr(m.config, "n_positions")
    assert get_model_max_seq_len(m) == 256


def test_prepare_small_and_medium_and_wrapper_from_pretrained(
    settings_with_paths: Settings, tmp_path: Path
) -> None:
    # Tiny corpus to create a real tokenizer handle
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "a.txt").write_text("aba\nabba\n", encoding="utf-8")
    tok_id, _ = _make_char_tokenizer(tmp_path, corpus)
    tok_dir = Path(settings_with_paths["app"]["artifacts_root"]) / "tokenizers" / tok_id
    handle = CharBackend().load(str(tok_dir / "tokenizer.json"))

    # Prepare two sizes to cover size branches
    prep_small = prepare_char_lstm_with_handle(
        handle,
        {
            "model_family": "char_lstm",
            "model_size": "small",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 2,
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
        },
    )
    prep_med = prepare_char_lstm_with_handle(
        handle,
        {
            "model_family": "char_lstm",
            "model_size": "medium",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 2,
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
        },
    )
    # Save one and load via wrapper to cover wrapper.from_pretrained and .config
    out_dir = tmp_path / "save"
    out_dir.mkdir(parents=True, exist_ok=True)
    prep_small.model.save_pretrained(str(out_dir))
    wrapper = CharLSTMModel.from_pretrained(str(out_dir))
    # Access config property to cover line
    _ = wrapper.config
    # Access named_parameters to cover line
    named_params = wrapper.named_parameters()
    assert isinstance(named_params, list) and len(named_params) > 0
    # Sanity check for med prep object existence
    assert prep_med.max_seq_len == 16
