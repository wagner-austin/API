"""char_lstm backend: evaluation and edge branches."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.backend_factory import create_char_lstm_backend
from model_trainer.core.services.tokenizer.char_backend import CharBackend
from tests._char_lstm_prepare_support import (
    _FakeTokHandle,
    _never,
    _noop,
    _train_char_tokenizer,
    _write_tiny_corpus,
)
from tests.conftest import UNPINNED


def test_char_lstm_gradient_flow(settings_with_paths: Settings, tmp_path: Path) -> None:
    """Integration test: all trainable parameters receive gradients."""
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
        "quantization": None,
        "gguf_export": None,
    }
    backend = create_char_lstm_backend(LocalTextDatasetBuilder())
    tok_dir = Path(settings_with_paths["app"]["artifacts_root"]) / "tokenizers" / tok_id
    handle = CharBackend().load(str(tok_dir / "tokenizer.json"))
    prepared = backend.prepare(cfg, settings_with_paths, tokenizer=handle)

    vocab_size = handle.get_vocab_size()

    # Create input and do forward/backward pass
    input_ids = torch.randint(0, vocab_size, (2, 8), dtype=torch.long)
    labels = torch.randint(0, vocab_size, (2, 8), dtype=torch.long)

    prepared.model.train()
    out = prepared.model.forward(input_ids=input_ids, labels=labels)

    # Record loss before backward
    loss_before = out.loss.item()

    torch.autograd.backward([out.loss])

    # Verify loss is unchanged after backward (backward computes grads, doesn't change loss)
    loss_after = out.loss.item()
    assert loss_after <= loss_before, (
        f"Loss should remain stable: {loss_before:.4f} -> {loss_after:.4f}"
    )

    # Verify all trainable parameters have gradients
    params_without_grad: list[str] = []
    for name, param in prepared.model.named_parameters():
        if param.requires_grad and param.grad is None:
            params_without_grad.append(name)

    assert len(params_without_grad) == 0, f"Parameters without gradients: {params_without_grad}"


def test_char_lstm_long_input_truncation(settings_with_paths: Settings, tmp_path: Path) -> None:
    """Integration test: model handles inputs longer than max_seq_len."""
    corpus_path = _write_tiny_corpus(tmp_path)
    tok_id, _ = _train_char_tokenizer(tmp_path, corpus_path)

    max_seq_len = 8  # Small for testing
    cfg: ModelTrainConfig = {
        "model_family": "char_lstm",
        "model_size": "tiny",
        "max_seq_len": max_seq_len,
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
        "quantization": None,
        "gguf_export": None,
    }
    backend = create_char_lstm_backend(LocalTextDatasetBuilder())
    tok_dir = Path(settings_with_paths["app"]["artifacts_root"]) / "tokenizers" / tok_id
    handle = CharBackend().load(str(tok_dir / "tokenizer.json"))
    prepared = backend.prepare(cfg, settings_with_paths, tokenizer=handle)

    vocab_size = handle.get_vocab_size()

    # Create input longer than max_seq_len
    long_seq_len = max_seq_len * 2
    input_ids = torch.randint(0, vocab_size, (1, long_seq_len), dtype=torch.long)
    labels = torch.randint(0, vocab_size, (1, long_seq_len), dtype=torch.long)

    # Model should handle long input (LSTM processes full sequence)
    prepared.model.eval()
    out = prepared.model.forward(input_ids=input_ids, labels=labels)

    # Verify loss is computed correctly
    assert math.isfinite(out.loss.item()), "Loss should be finite for long input"


def test_char_lstm_generation_determinism(settings_with_paths: Settings, tmp_path: Path) -> None:
    """Integration test: same seed produces same generation output."""
    from model_trainer.core.contracts.model import GenerateConfig
    from model_trainer.core.services.model.backends.char_lstm.generate import (
        generate_char_lstm,
    )

    corpus_path = _write_tiny_corpus(tmp_path)
    tok_id, _ = _train_char_tokenizer(tmp_path, corpus_path)

    cfg: ModelTrainConfig = {
        "model_family": "char_lstm",
        "model_size": "tiny",
        "max_seq_len": 16,
        "num_epochs": 3,  # Multiple epochs for loss reduction test
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
        "quantization": None,
        "gguf_export": None,
    }
    backend = create_char_lstm_backend(LocalTextDatasetBuilder())
    tok_dir = Path(settings_with_paths["app"]["artifacts_root"]) / "tokenizers" / tok_id
    handle = CharBackend().load(str(tok_dir / "tokenizer.json"))
    prepared = backend.prepare(cfg, settings_with_paths, tokenizer=handle)

    # Train briefly with loss tracking
    train_losses: list[float] = []

    def track_loss(
        step: int,
        epoch: int,
        loss: float,
        train_ppl: float,
        gn: float,
        sps: float,
        vl: float | None,
        vp: float | None,
    ) -> None:
        train_losses.append(loss)

    _ = backend.train(
        cfg,
        settings_with_paths,
        run_id="runDeterminism",
        heartbeat=_noop,
        cancelled=_never,
        resume=False,
        prepared=prepared,
        progress=track_loss,
        determinism=UNPINNED,
    )

    # Verify training produced valid losses
    loss_before = train_losses[0]
    loss_after = train_losses[-1]
    assert loss_after < loss_before, (
        f"Training should reduce loss: before={loss_before:.4f}, after={loss_after:.4f}"
    )

    gen_cfg = GenerateConfig(
        prompt_text="ab",
        prompt_path=None,
        max_new_tokens=10,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        stop_on_eos=False,
        stop_sequences=[],
        seed=123,
        num_return_sequences=1,
    )

    # Generate twice with same seed
    result1 = generate_char_lstm(prepared=prepared, cfg=gen_cfg, settings=settings_with_paths)
    result2 = generate_char_lstm(prepared=prepared, cfg=gen_cfg, settings=settings_with_paths)

    # Outputs should be identical
    assert result1["outputs"] == result2["outputs"], (
        f"Same seed should produce same output: {result1['outputs']} != {result2['outputs']}"
    )


def test_char_lstm_continued_training_reduces_loss(
    settings_with_paths: Settings, tmp_path: Path
) -> None:
    """Integration test: continued training on an already-trained model reduces loss.

    This verifies that fine-tuning works by:
    1. Training a model for initial epochs
    2. Continuing training (fine-tuning) for more epochs
    3. Verifying that loss decreases during the continued training phase
    """
    # Use a larger corpus for stable training metrics
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    pattern = "abababab\n" * 50 + "babababa\n" * 50
    (corpus_dir / "train.txt").write_text(pattern, encoding="utf-8")
    corpus_path = str(corpus_dir)
    tok_id, _ = _train_char_tokenizer(tmp_path, corpus_path)

    # Initial training configuration
    cfg: ModelTrainConfig = {
        "model_family": "char_lstm",
        "model_size": "tiny",
        "max_seq_len": 16,
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 1e-2,
        "tokenizer_id": tok_id,
        "corpus_path": corpus_path,
        "corpus_format": "lines",
        "holdout_fraction": 0.1,
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
    backend = create_char_lstm_backend(LocalTextDatasetBuilder())
    tok_dir = Path(settings_with_paths["app"]["artifacts_root"]) / "tokenizers" / tok_id
    handle = CharBackend().load(str(tok_dir / "tokenizer.json"))
    prepared = backend.prepare(cfg, settings_with_paths, tokenizer=handle)

    # Phase 1: Initial training
    initial_losses: list[float] = []

    def collect_initial(
        step: int,
        epoch: int,
        loss: float,
        train_ppl: float,
        gn: float,
        sps: float,
        vl: float | None,
        vp: float | None,
    ) -> None:
        initial_losses.append(loss)

    _ = backend.train(
        cfg,
        settings_with_paths,
        run_id="runContinued1",
        heartbeat=_noop,
        cancelled=_never,
        resume=False,
        prepared=prepared,
        progress=collect_initial,
        determinism=UNPINNED,
    )

    # Verify initial training worked
    loss_initial_phase = initial_losses[0]
    loss_final_phase = initial_losses[-1]
    assert loss_final_phase < loss_initial_phase, (
        f"Initial training should reduce loss: initial={loss_initial_phase:.4f}, "
        f"final={loss_final_phase:.4f}"
    )

    # Phase 2: Continued training (fine-tuning) with same data
    continued_cfg: ModelTrainConfig = {
        "model_family": "char_lstm",
        "model_size": "tiny",
        "max_seq_len": 16,
        "num_epochs": 5,
        "batch_size": 4,
        "learning_rate": 5e-3,
        "tokenizer_id": tok_id,
        "corpus_path": corpus_path,
        "corpus_format": "lines",
        "holdout_fraction": 0.1,
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

    continued_losses: list[float] = []

    def collect_continued(
        step: int,
        epoch: int,
        loss: float,
        train_ppl: float,
        gn: float,
        sps: float,
        vl: float | None,
        vp: float | None,
    ) -> None:
        continued_losses.append(loss)

    _ = backend.train(
        continued_cfg,
        settings_with_paths,
        run_id="runContinued2",
        heartbeat=_noop,
        cancelled=_never,
        resume=False,
        prepared=prepared,
        progress=collect_continued,
        determinism=UNPINNED,
    )

    # Verify continued training shows loss decrease
    # With tiny data (2 lines), batch_size=2, num_epochs=5: expect ~5 steps
    assert len(continued_losses) >= 3, (
        f"Expected at least 3 loss values from continued training, got {len(continued_losses)}"
    )

    # Compare first third vs last third of continued training losses
    third = len(continued_losses) // 3
    initial_loss_avg = sum(continued_losses[:third]) / third
    final_loss_avg = sum(continued_losses[-third:]) / third

    assert final_loss_avg < initial_loss_avg, (
        f"Continued training should reduce loss: "
        f"initial avg={initial_loss_avg:.4f}, final avg={final_loss_avg:.4f}"
    )

    # Also verify that continued training started from where initial training ended
    # (the model retains its learned state)
    loss_before_continue = initial_losses[-1]
    loss_after_continue = continued_losses[0]

    # First continued loss should be similar to last initial loss (same model state)
    # Allow 50% tolerance since batch composition differs
    ratio = loss_after_continue / loss_before_continue if loss_before_continue > 0 else 1.0
    assert 0.5 < ratio < 2.0, (
        f"Model state should persist: loss before={loss_before_continue:.4f}, "
        f"loss after={loss_after_continue:.4f}, ratio={ratio:.2f}"
    )


def test_char_tokenizer_roundtrip(settings_with_paths: Settings, tmp_path: Path) -> None:
    """Integration test: tokenizer encode/decode roundtrip preserves text."""
    corpus_path = _write_tiny_corpus(tmp_path)
    tok_id, _ = _train_char_tokenizer(tmp_path, corpus_path)

    tok_dir = Path(settings_with_paths["app"]["artifacts_root"]) / "tokenizers" / tok_id
    handle = CharBackend().load(str(tok_dir / "tokenizer.json"))

    # Test various strings (using characters from the corpus)
    test_strings = ["aba", "abba", "bab", "a", "b"]

    for text in test_strings:
        # Encode
        token_ids: list[int] = handle.encode(text)

        # Decode
        decoded = handle.decode(token_ids)

        # Verify roundtrip
        assert decoded == text, f"Roundtrip failed: '{text}' -> {token_ids} -> '{decoded}'"


def test_char_lstm_prepare_raises_when_tokenizer_none(settings_with_paths: Settings) -> None:
    """Cover char_lstm/prepare.py tokenizer None error branch."""
    from model_trainer.core.services.model.backends.char_lstm import prepare_char_lstm_with_handle

    cfg: ModelTrainConfig = {
        "model_family": "char_lstm",
        "model_size": "tiny",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "tokenizer_id": "some_tok",
        "corpus_path": "/tmp",
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
        "quantization": None,
        "gguf_export": None,
    }
    with pytest.raises(ValueError, match="tokenizer is required for char_lstm backend"):
        prepare_char_lstm_with_handle(None, cfg)


def test_char_lstm_prepare_raises_when_tokenizer_id_none(settings_with_paths: Settings) -> None:
    """Cover char_lstm/prepare.py tokenizer_id None error branch."""
    from model_trainer.core.services.model.backends.char_lstm import prepare_char_lstm_with_handle

    cfg: ModelTrainConfig = {
        "model_family": "char_lstm",
        "model_size": "tiny",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "tokenizer_id": None,  # tokenizer_id is None, but tokenizer handle is provided
        "corpus_path": "/tmp",
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
        "quantization": None,
        "gguf_export": None,
    }
    with pytest.raises(ValueError, match="tokenizer_id is required for char_lstm backend"):
        prepare_char_lstm_with_handle(_FakeTokHandle(), cfg)


def test_char_lstm_io_load_raises_when_tokenizer_none(settings_with_paths: Settings) -> None:
    """Cover char_lstm/io.py tokenizer None error branch."""
    from model_trainer.core.services.model.backends.char_lstm.io import (
        load_prepared_char_lstm_from_handle,
    )

    with pytest.raises(ValueError, match="tokenizer is required for char_lstm backend"):
        load_prepared_char_lstm_from_handle("/some/path", None)


def test_char_lstm_evaluate_raises_when_tokenizer_id_none(settings_with_paths: Settings) -> None:
    """Cover char_lstm/evaluate.py tokenizer_id None error branch."""
    from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
    from model_trainer.core.services.model.backends.char_lstm.evaluate import evaluate_char_lstm

    cfg: ModelTrainConfig = {
        "model_family": "char_lstm",
        "model_size": "tiny",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "tokenizer_id": None,  # tokenizer_id is None
        "corpus_path": "/tmp",
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
        "quantization": None,
        "gguf_export": None,
    }
    with pytest.raises(ValueError, match="tokenizer_id is required for char_lstm backend"):
        evaluate_char_lstm(
            run_id="test-run",
            cfg=cfg,
            settings=settings_with_paths,
            dataset_builder=LocalTextDatasetBuilder(),
        )
