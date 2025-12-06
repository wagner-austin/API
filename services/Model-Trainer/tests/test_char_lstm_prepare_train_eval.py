from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.backend_factory import create_char_lstm_backend
from model_trainer.core.services.tokenizer.char_backend import CharBackend


def _write_tiny_corpus(root: Path) -> str:
    out_dir = root / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "tiny.txt"
    # Expanded corpus for meaningful training - original 10 bytes was too small
    # Need multiple batches to show loss reduction across training steps
    corpus_lines = ["aba", "abbaba", "abaaba", "babbab", "ababab", "bababa"]
    corpus_text = "\n".join(corpus_lines * 10) + "\n"  # ~300 bytes
    fp.write_text(corpus_text, encoding="utf-8")
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
    stats = CharBackend().train(cfg)
    assert stats.token_count >= 4
    return "tok1", str(tok_out)


def _noop(_: float) -> None:
    return None


def _never() -> bool:
    return False


def test_char_lstm_end_to_end_small(settings_with_paths: Settings, tmp_path: Path) -> None:
    corpus_path = _write_tiny_corpus(tmp_path)
    tok_id, _ = _train_char_tokenizer(tmp_path, corpus_path)

    cfg: ModelTrainConfig = {
        "model_family": "char_lstm",
        "model_size": "tiny",
        "max_seq_len": 16,
        "num_epochs": 3,  # Multiple epochs to get multiple training steps
        "batch_size": 2,
        "learning_rate": 1e-3,
        "tokenizer_id": tok_id,
        "corpus_path": corpus_path,
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
        "precision": "fp32",
    }
    backend = create_char_lstm_backend(LocalTextDatasetBuilder())

    # Prepare with tokenizer handle via backend API
    tok_dir = Path(settings_with_paths["app"]["artifacts_root"]) / "tokenizers" / tok_id
    handle = CharBackend().load(str(tok_dir / "tokenizer.json"))
    prepared = backend.prepare(cfg, settings_with_paths, tokenizer=handle)

    # Track losses during training
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

    # Train for 1 epoch on tiny data
    out = backend.train(
        cfg,
        settings_with_paths,
        run_id="runA",
        heartbeat=_noop,
        cancelled=_never,
        prepared=prepared,
        progress=track_loss,
    )
    assert out["steps"] >= 1
    assert math.isfinite(out["loss"])
    assert out["perplexity"] >= 1.0

    # Verify training made progress
    loss_before = train_losses[0]
    loss_after = train_losses[-1]
    assert loss_after < loss_before, (
        f"Training should reduce loss: before={loss_before:.4f}, after={loss_after:.4f}"
    )

    # Save and evaluate path
    models_dir = Path(settings_with_paths["app"]["artifacts_root"]) / "models"
    art = backend.save(prepared, out_dir=str(models_dir / "runA"))
    assert Path(art["out_dir"]).exists()

    # Load path to cover backend.load branch as well
    loaded = backend.load(art["out_dir"], settings_with_paths, tokenizer=handle)
    # Verify loaded model config matches original - access via getattr for ConfigLike
    loaded_vocab: int | None = getattr(loaded.model.config, "vocab_size", None)
    prepared_vocab: int | None = getattr(prepared.model.config, "vocab_size", None)
    if loaded_vocab is None or prepared_vocab is None:
        raise AssertionError("Expected vocab_size on both configs")
    assert loaded_vocab == prepared_vocab

    ev = backend.evaluate(run_id="runA", cfg=cfg, settings=settings_with_paths)
    assert math.isfinite(ev["loss"])
    assert ev["perplexity"] >= 1.0


def test_char_lstm_invalid_size_raises(settings_with_paths: Settings, tmp_path: Path) -> None:
    corpus_path = _write_tiny_corpus(tmp_path)
    tok_id, _ = _train_char_tokenizer(tmp_path, corpus_path)
    cfg: ModelTrainConfig = {
        "model_family": "char_lstm",
        "model_size": "unknown",
        "max_seq_len": 8,
        "num_epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "tokenizer_id": tok_id,
        "corpus_path": corpus_path,
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
        "precision": "fp32",
    }
    backend = create_char_lstm_backend(LocalTextDatasetBuilder())
    tok_dir = Path(settings_with_paths["app"]["artifacts_root"]) / "tokenizers" / tok_id
    handle = CharBackend().load(str(tok_dir / "tokenizer.json"))
    with pytest.raises(AppError, match="invalid model_size") as exc_info:
        _ = backend.prepare(cfg, settings_with_paths, tokenizer=handle)
    err: AppError[ModelTrainerErrorCode] = exc_info.value
    assert err.code == ModelTrainerErrorCode.INVALID_MODEL_SIZE


def test_char_lstm_freeze_embed_preserves_embedding_weights(
    settings_with_paths: Settings, tmp_path: Path
) -> None:
    """Integration test: verify freeze_embed=True keeps embedding weights unchanged."""
    corpus_path = _write_tiny_corpus(tmp_path)
    tok_id, _ = _train_char_tokenizer(tmp_path, corpus_path)

    cfg: ModelTrainConfig = {
        "model_family": "char_lstm",
        "model_size": "tiny",
        "max_seq_len": 16,
        "num_epochs": 2,
        "batch_size": 2,
        "learning_rate": 1e-2,
        "tokenizer_id": tok_id,
        "corpus_path": corpus_path,
        "holdout_fraction": 0.01,
        "seed": 42,
        "pretrained_run_id": None,
        "freeze_embed": True,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cpu",
        "data_num_workers": 0,
        "data_pin_memory": False,
        "early_stopping_patience": 0,
        "test_split_ratio": 0.0,
        "finetune_lr_cap": 0.0,
        "precision": "fp32",
    }
    backend = create_char_lstm_backend(LocalTextDatasetBuilder())
    tok_dir = Path(settings_with_paths["app"]["artifacts_root"]) / "tokenizers" / tok_id
    handle = CharBackend().load(str(tok_dir / "tokenizer.json"))
    prepared = backend.prepare(cfg, settings_with_paths, tokenizer=handle)

    # Capture embedding weights before training
    embedding_weights_before: dict[str, torch.Tensor] = {}
    other_weights_before: dict[str, torch.Tensor] = {}
    for name, param in prepared.model.named_parameters():
        if "embedding" in name.lower():
            embedding_weights_before[name] = param.detach().clone()
        else:
            other_weights_before[name] = param.detach().clone()

    # CharLSTM has exactly 1 embedding parameter (embedding.weight)
    assert len(embedding_weights_before) == 1, (
        f"Expected 1 embedding param, found {len(embedding_weights_before)}: "
        f"{list(embedding_weights_before.keys())}"
    )
    # CharLSTM has LSTM weights, biases, and projection layer params
    assert len(other_weights_before) >= 5, (
        f"Expected at least 5 non-embedding params, found {len(other_weights_before)}"
    )

    # Track losses during training
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

    # Train with freeze_embed=True
    out = backend.train(
        cfg,
        settings_with_paths,
        run_id="runFreeze",
        heartbeat=_noop,
        cancelled=_never,
        prepared=prepared,
        progress=track_loss,
    )
    assert out["steps"] >= 1, "Training should complete at least 1 step"

    # Verify training reduced loss (even with frozen embeddings)
    loss_before = train_losses[0]
    loss_after = train_losses[-1]
    assert loss_after < loss_before, (
        f"Training should reduce loss: before={loss_before:.4f}, after={loss_after:.4f}"
    )

    # Verify embedding weights are unchanged
    for name, param in prepared.model.named_parameters():
        if "embedding" in name.lower():
            before = embedding_weights_before[name]
            current = param.detach()
            assert torch.equal(current, before), f"Embedding param {name} should be unchanged"

    # Verify at least some non-embedding weights changed (proves training occurred)
    any_changed = False
    for name, param in prepared.model.named_parameters():
        if "embedding" not in name.lower() and name in other_weights_before:
            before = other_weights_before[name]
            current = param.detach()
            if not torch.equal(current, before):
                any_changed = True
                break
    assert any_changed, "Non-embedding weights should change during training"


def test_char_lstm_training_reduces_loss(settings_with_paths: Settings, tmp_path: Path) -> None:
    """Integration test: verify training actually reduces loss over time."""
    # Create a larger corpus for more meaningful training
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    # Repetitive pattern that the model should learn
    pattern = "abababab\n" * 50 + "babababa\n" * 50
    (corpus_dir / "train.txt").write_text(pattern, encoding="utf-8")
    corpus_path = str(corpus_dir)

    tok_id, _ = _train_char_tokenizer(tmp_path, corpus_path)

    cfg: ModelTrainConfig = {
        "model_family": "char_lstm",
        "model_size": "tiny",
        "max_seq_len": 16,
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 1e-2,
        "tokenizer_id": tok_id,
        "corpus_path": corpus_path,
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
        "precision": "fp32",
    }
    backend = create_char_lstm_backend(LocalTextDatasetBuilder())
    tok_dir = Path(settings_with_paths["app"]["artifacts_root"]) / "tokenizers" / tok_id
    handle = CharBackend().load(str(tok_dir / "tokenizer.json"))
    prepared = backend.prepare(cfg, settings_with_paths, tokenizer=handle)

    # Capture losses during training via progress callback
    losses: list[float] = []

    def capture_loss(
        step: int,
        epoch: int,
        loss: float,
        train_ppl: float,
        gn: float,
        sps: float,
        vl: float | None,
        vp: float | None,
    ) -> None:
        losses.append(loss)

    out = backend.train(
        cfg,
        settings_with_paths,
        run_id="runLossTest",
        heartbeat=_noop,
        cancelled=_never,
        prepared=prepared,
        progress=capture_loss,
    )

    # Verify we captured multiple losses
    assert len(losses) >= 3, f"Expected at least 3 loss samples, got {len(losses)}"

    # Verify training completed successfully
    assert out["steps"] >= 1, "Training should complete at least 1 step"
    assert math.isfinite(out["loss"]), "Final loss should be finite"
    assert out["perplexity"] >= 1.0, "Perplexity should be >= 1.0"

    # Verify loss trend is downward: compare first third average to last third average
    n = len(losses)
    third = n // 3
    if third > 0:
        initial_loss_avg = sum(losses[:third]) / third
        final_loss_avg = sum(losses[-third:]) / third
        assert final_loss_avg < initial_loss_avg, (
            f"Loss should decrease: initial avg={initial_loss_avg:.4f}, "
            f"final avg={final_loss_avg:.4f}"
        )


def test_char_lstm_save_load_consistency(settings_with_paths: Settings, tmp_path: Path) -> None:
    """Integration test: saved model produces same outputs as original."""
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
        "precision": "fp32",
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
        run_id="runSaveLoad",
        heartbeat=_noop,
        cancelled=_never,
        prepared=prepared,
        progress=track_loss,
    )

    # Verify training produced valid losses
    loss_before = train_losses[0]
    loss_after = train_losses[-1]
    assert loss_after < loss_before, (
        f"Training should reduce loss: before={loss_before:.4f}, after={loss_after:.4f}"
    )

    # Create test input
    test_input = torch.randint(1, 5, (1, 4), dtype=torch.long)
    labels = torch.randint(2, 6, (1, 4), dtype=torch.long)

    # Get output from original model
    prepared.model.eval()
    original_out = prepared.model.forward(input_ids=test_input, labels=labels)
    original_loss = original_out.loss.item()

    # Save the model
    models_dir = tmp_path / "models" / "runSaveLoad"
    art = backend.save(prepared, out_dir=str(models_dir))

    # Load into new model
    loaded = backend.load(art["out_dir"], settings_with_paths, tokenizer=handle)

    # Get output from loaded model
    loaded.model.eval()
    loaded_out = loaded.model.forward(input_ids=test_input, labels=labels)
    loaded_loss = loaded_out.loss.item()

    # Verify outputs match
    assert abs(original_loss - loaded_loss) < 1e-5, (
        f"Loaded model output differs: original={original_loss}, loaded={loaded_loss}"
    )


def test_char_lstm_forward_pass_shapes(settings_with_paths: Settings, tmp_path: Path) -> None:
    """Integration test: forward pass produces correct output tensor shapes."""
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
        "precision": "fp32",
    }
    backend = create_char_lstm_backend(LocalTextDatasetBuilder())
    tok_dir = Path(settings_with_paths["app"]["artifacts_root"]) / "tokenizers" / tok_id
    handle = CharBackend().load(str(tok_dir / "tokenizer.json"))
    prepared = backend.prepare(cfg, settings_with_paths, tokenizer=handle)

    # Test various batch sizes and sequence lengths
    test_cases: list[tuple[int, int]] = [(1, 4), (2, 8), (4, 16)]
    vocab_size = handle.get_vocab_size()

    for batch_size, seq_len in test_cases:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

        prepared.model.eval()
        out = prepared.model.forward(input_ids=input_ids, labels=labels)

        # Verify loss is a scalar
        assert out.loss.dim() == 0, f"Loss should be scalar, got shape {out.loss.shape}"
        assert math.isfinite(out.loss.item()), "Loss should be finite"


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
        "precision": "fp32",
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
        "precision": "fp32",
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
        "precision": "fp32",
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
        prepared=prepared,
        progress=track_loss,
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
        "precision": "fp32",
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
        prepared=prepared,
        progress=collect_initial,
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
        "precision": "fp32",
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
        prepared=prepared,
        progress=collect_continued,
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
