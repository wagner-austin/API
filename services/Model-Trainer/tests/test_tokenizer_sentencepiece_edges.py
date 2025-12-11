from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError

from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.tokenizer.spm_backend import (
    SentencePieceBackend,
    train_spm_tokenizer,
)


def test_sentencepiece_backend_empty_corpus_error(tmp_path: Path) -> None:
    """Test that empty corpus directory raises AppError (no tokenizer creation)."""
    cfg = TokenizerTrainConfig(
        method="sentencepiece",
        vocab_size=128,
        min_frequency=1,
        corpus_path=str(tmp_path),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(tmp_path / "tok"),
    )
    # Empty corpus should raise before any model is built
    # Call underlying function directly to avoid guard false positive
    with pytest.raises(AppError, match="No text files found"):
        train_spm_tokenizer(cfg.corpus_path, cfg.out_dir, cfg)


def test_sentencepiece_inspect_missing_manifest(tmp_path: Path) -> None:
    """Test that inspecting missing tokenizer raises AppError."""
    backend = SentencePieceBackend()
    missing_dir = tmp_path / "nope"
    with pytest.raises(AppError, match="manifest not found"):
        backend.inspect(str(missing_dir))
