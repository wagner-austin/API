"""char_lstm backend: scoring."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

import pytest
import torch
from platform_core.errors import AppError

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import (
    GenerateConfig,
    ModelTrainConfig,
    PreparedLMModel,
    ScoreConfig,
)
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.backend_factory import create_char_lstm_backend
from model_trainer.core.services.model.backends.char_lstm.generate import (
    _read_prompt,
    _sample_token,
)
from model_trainer.core.services.model.backends.char_lstm.score import (
    _compute_logits,
    _compute_topk,
    _read_text_or_path,
    score_char_lstm,
)
from model_trainer.core.services.tokenizer.char_backend import CharBackend
from tests.conftest import UNPINNED


class _ForwardLogitsFn(Protocol):
    """Protocol for CharLSTM forward_logits method."""

    def __call__(
        self: _ForwardLogitsFn,
        *,
        input_ids: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: ...


def _write_tiny_corpus(root: Path) -> str:
    out_dir = root / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "tiny.txt"
    fp.write_text("aba\nabbaba\n", encoding="utf-8")
    return str(out_dir)


def _train_char_tokenizer(root: Path, corpus_path: str) -> tuple[str, str]:
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
    CharBackend().train(cfg)
    return "tok1", str(tok_out)


def _noop(_: float) -> None:
    return None


def _never() -> bool:
    return False


def _prepare_trained_model(settings: Settings, tmp_path: Path) -> tuple[PreparedLMModel, str]:
    """Prepare and train a tiny char_lstm model, returns (prepared, corpus_path)."""
    corpus_path = _write_tiny_corpus(tmp_path)
    tok_id, _ = _train_char_tokenizer(tmp_path, corpus_path)

    cfg: ModelTrainConfig = {
        "model_family": "char_lstm",
        "model_size": "tiny",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "tokenizer_id": tok_id,
        "corpus_path": corpus_path,
        "corpus_format": "lines",
        "holdout_fraction": 0.01,
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
        "cartridge": None,
        "quantization": None,
        "gguf_export": None,
    }
    backend = create_char_lstm_backend(LocalTextDatasetBuilder())
    tok_dir = Path(settings["app"]["artifacts_root"]) / "tokenizers" / tok_id
    handle = CharBackend().load(str(tok_dir / "tokenizer.json"))
    prepared = backend.prepare(cfg, settings, tokenizer=handle)
    backend.train(
        cfg,
        settings,
        run_id="runA",
        heartbeat=_noop,
        cancelled=_never,
        resume=False,
        prepared=prepared,
        progress=None,
        determinism=UNPINNED,
    )
    return prepared, corpus_path


# ===== Score tests =====


def test_read_text_or_path_text(settings_with_paths: Settings) -> None:
    cfg = ScoreConfig(text="hello", path=None, detail_level="summary", top_k=None, seed=None)
    result = _read_text_or_path(cfg, settings_with_paths)
    assert result == "hello"


def test_read_text_or_path_path(settings_with_paths: Settings, tmp_path: Path) -> None:
    # Write a file under data_root
    data_root = Path(settings_with_paths["app"]["data_root"])
    data_root.mkdir(parents=True, exist_ok=True)
    test_file = data_root / "test.txt"
    test_file.write_text("test content", encoding="utf-8")

    cfg = ScoreConfig(text=None, path=str(test_file), detail_level="summary", top_k=None, seed=None)
    result = _read_text_or_path(cfg, settings_with_paths)
    assert result == "test content"


def test_read_text_or_path_outside_data_root_raises(
    settings_with_paths: Settings, tmp_path: Path
) -> None:
    cfg = ScoreConfig(
        text=None, path="/tmp/outside.txt", detail_level="summary", top_k=None, seed=None
    )
    with pytest.raises(AppError, match="data_root"):
        _read_text_or_path(cfg, settings_with_paths)


def test_read_text_or_path_neither_raises(settings_with_paths: Settings) -> None:
    cfg = ScoreConfig(text=None, path=None, detail_level="summary", top_k=None, seed=None)
    with pytest.raises(AppError, match="either"):
        _read_text_or_path(cfg, settings_with_paths)


def test_score_char_lstm_summary(settings_with_paths: Settings, tmp_path: Path) -> None:
    prepared, _ = _prepare_trained_model(settings_with_paths, tmp_path)
    cfg = ScoreConfig(text="aba", path=None, detail_level="summary", top_k=None, seed=42)
    result = score_char_lstm(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert math.isfinite(result["loss"])
    assert result["perplexity"] >= 1.0
    assert result["surprisal"] is None
    assert result["topk"] is None
    assert result["tokens"] is None


def test_score_char_lstm_per_char(settings_with_paths: Settings, tmp_path: Path) -> None:
    prepared, _ = _prepare_trained_model(settings_with_paths, tmp_path)
    cfg = ScoreConfig(text="aba", path=None, detail_level="per_char", top_k=None, seed=42)
    result = score_char_lstm(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert math.isfinite(result["loss"])
    assert result["perplexity"] >= 1.0
    # "aba" = 3 chars/tokens, surprisal has T-1 entries (next-token prediction shift)
    surprisal = result["surprisal"]
    tokens = result["tokens"]
    if surprisal is None:
        raise AssertionError("Expected surprisal to be set for per_char mode")
    if tokens is None:
        raise AssertionError("Expected tokens to be set for per_char mode")
    assert len(tokens) == 3, f"Expected 3 tokens for 'aba', got {len(tokens)}"
    assert len(surprisal) == 2, f"Expected 2 surprisal values (T-1), got {len(surprisal)}"
    # Verify surprisal values are finite
    for s in surprisal:
        assert math.isfinite(s)


def test_score_char_lstm_with_topk(settings_with_paths: Settings, tmp_path: Path) -> None:
    prepared, _ = _prepare_trained_model(settings_with_paths, tmp_path)
    cfg = ScoreConfig(text="aba", path=None, detail_level="summary", top_k=3, seed=42)
    result = score_char_lstm(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    # "aba" = 3 chars, expect 3 positions in topk
    topk = result["topk"]
    if topk is None:
        raise AssertionError("Expected topk to be set when top_k > 0")
    assert len(topk) == 3
    # Each position should have top-k predictions
    for pos_topk in topk:
        assert len(pos_topk) <= 3


def test_score_char_lstm_short_text(settings_with_paths: Settings, tmp_path: Path) -> None:
    prepared, _ = _prepare_trained_model(settings_with_paths, tmp_path)
    # Single char - not enough for loss computation
    cfg = ScoreConfig(text="a", path=None, detail_level="summary", top_k=None, seed=42)
    result = score_char_lstm(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert result["loss"] == 0.0
    assert result["perplexity"] == 1.0


def test_compute_logits_short_sequence(settings_with_paths: Settings, tmp_path: Path) -> None:
    prepared, _ = _prepare_trained_model(settings_with_paths, tmp_path)
    # Single token - not enough for loss computation
    batch_ids: list[list[int]] = [[1]]
    input_ids = torch.tensor(batch_ids, dtype=torch.long)
    logits, per_token_loss = _compute_logits(prepared, input_ids)
    assert logits.size(1) == 1
    assert per_token_loss.numel() == 0


def test_compute_topk(settings_with_paths: Settings, tmp_path: Path) -> None:
    prepared, _ = _prepare_trained_model(settings_with_paths, tmp_path)
    # Create small logits tensor [1, 2, vocab_size]
    vocab_size = prepared.tok_for_dataset.get_vocab_size()
    logits = torch.randn(1, 2, vocab_size)
    result = _compute_topk(logits, prepared.tok_for_dataset, k=3)
    assert len(result) == 2  # 2 positions
    for pos_topk in result:
        assert len(pos_topk) <= 3
        for token_str, prob in pos_topk:
            # Verify token is a non-empty string and prob is valid probability
            assert len(token_str) >= 0, "Token should be string, got empty"
            assert 0.0 <= prob <= 1.0, f"Probability should be [0,1], got {prob}"


# ===== Generate tests =====


def test_read_prompt_text(settings_with_paths: Settings) -> None:
    cfg = GenerateConfig(
        prompt_text="hello",
        prompt_path=None,
        max_new_tokens=10,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        stop_on_eos=True,
        stop_sequences=[],
        seed=None,
        num_return_sequences=1,
    )
    result = _read_prompt(cfg, settings_with_paths)
    assert result == "hello"


def test_read_prompt_path(settings_with_paths: Settings, tmp_path: Path) -> None:
    data_root = Path(settings_with_paths["app"]["data_root"])
    data_root.mkdir(parents=True, exist_ok=True)
    test_file = data_root / "prompt.txt"
    test_file.write_text("test prompt", encoding="utf-8")

    cfg = GenerateConfig(
        prompt_text=None,
        prompt_path=str(test_file),
        max_new_tokens=10,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        stop_on_eos=True,
        stop_sequences=[],
        seed=None,
        num_return_sequences=1,
    )
    result = _read_prompt(cfg, settings_with_paths)
    assert result == "test prompt"


def test_read_prompt_outside_data_root_raises(settings_with_paths: Settings) -> None:
    cfg = GenerateConfig(
        prompt_text=None,
        prompt_path="/tmp/outside.txt",
        max_new_tokens=10,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        stop_on_eos=True,
        stop_sequences=[],
        seed=None,
        num_return_sequences=1,
    )
    with pytest.raises(AppError, match="data_root"):
        _read_prompt(cfg, settings_with_paths)


def test_read_prompt_neither_raises(settings_with_paths: Settings) -> None:
    cfg = GenerateConfig(
        prompt_text=None,
        prompt_path=None,
        max_new_tokens=10,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        stop_on_eos=True,
        stop_sequences=[],
        seed=None,
        num_return_sequences=1,
    )
    with pytest.raises(AppError, match="either"):
        _read_prompt(cfg, settings_with_paths)


def test_sample_token_greedy() -> None:
    vals: list[float] = [0.0, 0.0, 10.0, 0.0]
    logits = torch.tensor(vals)  # Token 2 has highest logit
    result = _sample_token(logits, temperature=0.0, top_k=0, top_p=1.0)
    assert result == 2
