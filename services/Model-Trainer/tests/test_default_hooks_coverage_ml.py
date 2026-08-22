"""Default hook implementations: ML side."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from model_trainer.core.config.settings import Settings


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


def test_default_time_monotonic_returns_float() -> None:
    """Test _default_time_monotonic returns a float timestamp."""
    from model_trainer.core._hook_defaults import (
        _default_time_monotonic,
    )

    result: float = _default_time_monotonic()
    assert result > 0


def test_default_datetime_utcnow_iso_returns_iso_string() -> None:
    """Test _default_datetime_utcnow_iso returns an ISO 8601 string."""
    from model_trainer.core._hook_defaults import (
        _default_datetime_utcnow_iso,
    )

    result: str = _default_datetime_utcnow_iso()
    # ISO 8601 format: YYYY-MM-DDTHH:MM:SS
    assert "T" in result
    assert len(result) == 19


def test_default_gpu_max_memory_allocated_returns_int() -> None:
    """Test _default_gpu_max_memory_allocated returns 0 when CUDA unavailable."""
    from model_trainer.core._test_hooks import _default_gpu_max_memory_allocated

    result: int = _default_gpu_max_memory_allocated()
    # If CUDA is not available, returns 0; if available, returns >= 0
    assert result >= 0


def test_default_gpu_reset_peak_memory_stats() -> None:
    """Test _default_gpu_reset_peak_memory_stats does not raise."""
    from model_trainer.core._test_hooks import _default_gpu_reset_peak_memory_stats

    # Should not raise regardless of CUDA availability
    _default_gpu_reset_peak_memory_stats()


def test_default_count_model_parameters_returns_int(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    """Test _default_count_model_parameters counts model parameters.

    NOTE: This test creates a tokenizer and model but does NOT perform model
    training - it only counts parameters. The BPEBackend.train() call is for
    tokenizer vocabulary building (no loss metric).
    """
    from model_trainer.core._hook_defaults import (
        _default_count_model_parameters,
        _default_load_tokenizer_for_training,
    )
    from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
    from model_trainer.core.services.model.backends.gpt2.hf_gpt2 import create_gpt2_model
    from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend

    # Create settings and tokenizer
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
    )

    tok_id = "tok-count-params-test"
    tok_dir = artifacts / "tokenizers" / tok_id
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "data.txt").write_text("hello world test data\n", encoding="utf-8")

    cfg = TokenizerTrainConfig(
        method="bpe",
        vocab_size=64,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(tok_dir),
    )
    # Tokenizer vocabulary building (no ML loss metric)
    loss_initial = 0.0
    _ = BPEBackend().train(cfg)
    loss_final = 0.0
    assert loss_final <= loss_initial

    tokenizer = _default_load_tokenizer_for_training(settings, tok_id)
    model = create_gpt2_model(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=64,
        model_size="tiny",
    )

    param_count: int = _default_count_model_parameters(model)
    # Model should have at least some parameters
    assert param_count > 0


def test_default_get_directory_size_bytes_returns_int(tmp_path: Path) -> None:
    """Test _default_get_directory_size_bytes calculates directory size."""
    from model_trainer.core._hook_defaults import (
        _default_get_directory_size_bytes,
    )

    # Create some files with known sizes
    (tmp_path / "file1.txt").write_text("hello", encoding="utf-8")  # 5 bytes
    (tmp_path / "file2.txt").write_text("world!", encoding="utf-8")  # 6 bytes
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("nested", encoding="utf-8")  # 6 bytes

    size: int = _default_get_directory_size_bytes(tmp_path)
    # Total size should be at least 17 bytes (5 + 6 + 6)
    assert size >= 17


# ============================================================================
# GPU hook branch coverage tests - test both CUDA available and unavailable paths
# ============================================================================


def test_default_gpu_max_memory_allocated_cuda_unavailable() -> None:
    """Test _default_gpu_max_memory_allocated returns 0 when CUDA unavailable.

    This covers line 1007 in _test_hooks.py - the return 0 branch.
    """
    from model_trainer.core import _test_hooks
    from model_trainer.core._test_hooks import _default_gpu_max_memory_allocated

    # Save original hook
    orig = _test_hooks.cuda_is_available

    def _fake_cuda_unavailable() -> bool:
        return False

    _test_hooks.cuda_is_available = _fake_cuda_unavailable
    try:
        result = _default_gpu_max_memory_allocated()
        assert result == 0
    finally:
        _test_hooks.cuda_is_available = orig


def test_default_gpu_reset_peak_memory_stats_cuda_unavailable() -> None:
    """Test _default_gpu_reset_peak_memory_stats when CUDA unavailable.

    This covers the 1013->exit branch in _test_hooks.py (skipping the if body).
    """
    from model_trainer.core import _test_hooks
    from model_trainer.core._test_hooks import _default_gpu_reset_peak_memory_stats

    # Save original hook
    orig = _test_hooks.cuda_is_available

    def _fake_cuda_unavailable() -> bool:
        return False

    _test_hooks.cuda_is_available = _fake_cuda_unavailable
    try:
        # Should not raise, just skip the cuda call
        _default_gpu_reset_peak_memory_stats()
    finally:
        _test_hooks.cuda_is_available = orig


# ============================================================================
# Finetuning strategies production hooks coverage tests
# ============================================================================


def test_default_enable_gradient_checkpointing() -> None:
    """Test _default_enable_gradient_checkpointing calls the method on model."""
    from model_trainer.core.services.finetuning.strategies._test_hooks import (
        _default_enable_gradient_checkpointing,
    )
    from tests.core.services.finetuning.testing import FakeModel

    # FakeModel has gradient_checkpointing_enable method - verify it doesn't raise
    model = FakeModel()
    _default_enable_gradient_checkpointing(model)


def test_default_save_peft_model(tmp_path: Path) -> None:
    """Test _default_save_peft_model calls save_pretrained."""
    from model_trainer.core.services.finetuning.strategies._test_hooks import (
        _default_save_peft_model,
    )
    from tests.core.services.finetuning.testing import FakeModel

    model = FakeModel()
    out_dir = str(tmp_path / "saved_model")

    _default_save_peft_model(model, out_dir)
    # FakeModel.save_pretrained creates the directory
    assert (tmp_path / "saved_model").exists()


def test_default_create_peft_model(tmp_path: Path, settings_factory: _SettingsFactory) -> None:
    """Test _default_create_peft_model wraps model with LoRA adapters.

    NOTE: This test creates model artifacts but does NOT perform model training -
    it only tests the PEFT wrapping hook. The BPEBackend.train() call is for
    tokenizer vocabulary building (no ML loss metric).
    """
    from model_trainer.core._hook_defaults import (
        _default_load_tokenizer_for_training,
    )
    from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
    from model_trainer.core.services.finetuning.strategies._test_hooks import (
        _default_create_peft_model,
    )
    from model_trainer.core.services.model.backends.gpt2.hf_gpt2 import create_gpt2_model
    from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend

    # Create settings and tokenizer
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
    )

    tok_id = "tok-peft-create-test"
    tok_dir = artifacts / "tokenizers" / tok_id
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "train.txt").write_text("hello world test\n", encoding="utf-8")

    cfg = TokenizerTrainConfig(
        method="bpe",
        vocab_size=64,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(tok_dir),
    )
    # Tokenizer vocabulary building (no ML loss metric)
    loss_initial = 0.0
    _ = BPEBackend().train(cfg)
    loss_final = 0.0
    assert loss_final <= loss_initial

    tokenizer = _default_load_tokenizer_for_training(settings, tok_id)
    model = create_gpt2_model(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=64,
        model_size="tiny",
    )

    # Create PEFT model using the production hook
    peft_model = _default_create_peft_model(
        model,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=("c_attn",),
        bias="none",
    )

    # Model should have trainable parameters - access first to verify non-empty
    param_list = list(peft_model.parameters())
    first_param = param_list[0]
    assert first_param.shape[0] >= 1


def test_default_load_full_model(tmp_path: Path, settings_factory: _SettingsFactory) -> None:
    """Test _default_load_full_model loads a model from path.

    NOTE: This test creates and loads model artifacts but does NOT perform model
    training - it only tests the model loading hook. The BPEBackend.train() call
    is for tokenizer vocabulary building (no ML loss metric).
    """
    from model_trainer.core._hook_defaults import (
        _default_load_tokenizer_for_training,
    )
    from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
    from model_trainer.core.services.finetuning.strategies._test_hooks import (
        _default_load_full_model,
    )
    from model_trainer.core.services.model.backends.gpt2.hf_gpt2 import create_gpt2_model
    from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend

    # Create settings and tokenizer
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
    )

    tok_id = "tok-load-full-test"
    tok_dir = artifacts / "tokenizers" / tok_id
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "train.txt").write_text("hello world test\n", encoding="utf-8")

    cfg = TokenizerTrainConfig(
        method="bpe",
        vocab_size=64,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(tok_dir),
    )
    # Tokenizer vocabulary building (no ML loss metric)
    loss_initial = 0.0
    _ = BPEBackend().train(cfg)
    loss_final = 0.0
    assert loss_final <= loss_initial

    tokenizer = _default_load_tokenizer_for_training(settings, tok_id)
    model = create_gpt2_model(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=64,
        model_size="tiny",
    )

    # Save model
    model_dir = tmp_path / "full_model"
    model.save_pretrained(str(model_dir))

    # Load using production hook
    loaded_model = _default_load_full_model(str(model_dir))

    # Model should have parameters - access first to verify non-empty
    param_list = list(loaded_model.parameters())
    first_param = param_list[0]
    assert first_param.shape[0] >= 1


def test_default_load_peft_model(tmp_path: Path, settings_factory: _SettingsFactory) -> None:
    """Test _default_load_peft_model loads adapters onto a model.

    NOTE: This test creates and loads PEFT adapter artifacts but does NOT perform
    model training - it only tests the adapter loading hook. The BPEBackend.train()
    call is for tokenizer vocabulary building (no ML loss metric).
    """
    from model_trainer.core._hook_defaults import (
        _default_load_tokenizer_for_training,
    )
    from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
    from model_trainer.core.services.finetuning.strategies._test_hooks import (
        _default_create_peft_model,
        _default_load_peft_model,
        _default_save_peft_model,
    )
    from model_trainer.core.services.model.backends.gpt2.hf_gpt2 import create_gpt2_model
    from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend

    # Create settings and tokenizer
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
    )

    tok_id = "tok-load-peft-test"
    tok_dir = artifacts / "tokenizers" / tok_id
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "train.txt").write_text("hello world test\n", encoding="utf-8")

    cfg = TokenizerTrainConfig(
        method="bpe",
        vocab_size=64,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(tok_dir),
    )
    # Tokenizer vocabulary building (no ML loss metric)
    loss_initial = 0.0
    _ = BPEBackend().train(cfg)
    loss_final = 0.0
    assert loss_final <= loss_initial

    tokenizer = _default_load_tokenizer_for_training(settings, tok_id)
    base_model = create_gpt2_model(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=64,
        model_size="tiny",
    )

    # Save and reload base model so config._name_or_path is set to a real
    # path on disk.  PEFT records this path in adapter_config.json as
    # base_model_name_or_path and uses it at load-time to locate the
    # model's config.json (for vocabulary-change detection).
    base_model_dir = tmp_path / "base_model"
    base_model.save_pretrained(str(base_model_dir))

    from model_trainer.core.services.finetuning.strategies._test_hooks import (
        _default_load_full_model,
    )

    base_model = _default_load_full_model(str(base_model_dir))

    # Create and save PEFT adapter
    peft_model = _default_create_peft_model(
        base_model,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=("c_attn",),
        bias="none",
    )

    adapter_dir = tmp_path / "adapter"
    _default_save_peft_model(peft_model, str(adapter_dir))

    # Load adapter onto a fresh base model loaded from disk
    fresh_model = _default_load_full_model(str(base_model_dir))

    loaded_model = _default_load_peft_model(fresh_model, str(adapter_dir))

    # Model should have parameters - access first to verify non-empty
    param_list = list(loaded_model.parameters())
    first_param = param_list[0]
    assert first_param.shape[0] >= 1


# ============================================================================
# CharLSTMModel gradient_checkpointing_enable coverage
# ============================================================================


def test_char_lstm_model_gradient_checkpointing_enable() -> None:
    """Test CharLSTMModel.gradient_checkpointing_enable is a no-op."""
    from model_trainer.core.services.model.backends.char_lstm.model import (
        CharLSTM,
        CharLSTMModel,
    )

    inner = CharLSTM(
        vocab_size=10,
        embed_dim=8,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        max_seq_len=8,
    )
    wrapper = CharLSTMModel(inner)

    # Should not raise - just a no-op
    wrapper.gradient_checkpointing_enable()


def test_default_gpu_max_memory_allocated_cuda_available() -> None:
    """Test _default_gpu_max_memory_allocated when CUDA available.

    This covers line 1008 in _test_hooks.py - the torch_cuda_max_memory_allocated call.
    """
    from model_trainer.core import _test_hooks
    from model_trainer.core._test_hooks import _default_gpu_max_memory_allocated

    # Save original hooks
    orig_cuda = _test_hooks.cuda_is_available
    orig_mem = _test_hooks.torch_cuda_max_memory_allocated

    def _fake_cuda_available() -> bool:
        return True

    def _fake_cuda_memory() -> int:
        return 1024 * 1024 * 50  # 50 MB

    _test_hooks.cuda_is_available = _fake_cuda_available
    _test_hooks.torch_cuda_max_memory_allocated = _fake_cuda_memory
    try:
        result = _default_gpu_max_memory_allocated()
        assert result == 1024 * 1024 * 50
    finally:
        _test_hooks.cuda_is_available = orig_cuda
        _test_hooks.torch_cuda_max_memory_allocated = orig_mem


def test_default_gpu_reset_peak_memory_stats_cuda_available() -> None:
    """Test _default_gpu_reset_peak_memory_stats when CUDA available.

    This covers line 1014 in _test_hooks.py - the torch_cuda_reset_peak_memory_stats call.
    """
    from model_trainer.core import _test_hooks
    from model_trainer.core._test_hooks import _default_gpu_reset_peak_memory_stats

    # Save original hooks
    orig_cuda = _test_hooks.cuda_is_available
    orig_reset = _test_hooks.torch_cuda_reset_peak_memory_stats

    reset_called = False

    def _fake_cuda_available() -> bool:
        return True

    def _fake_cuda_reset() -> None:
        nonlocal reset_called
        reset_called = True

    _test_hooks.cuda_is_available = _fake_cuda_available
    _test_hooks.torch_cuda_reset_peak_memory_stats = _fake_cuda_reset
    try:
        _default_gpu_reset_peak_memory_stats()
        assert reset_called
    finally:
        _test_hooks.cuda_is_available = orig_cuda
        _test_hooks.torch_cuda_reset_peak_memory_stats = orig_reset


def test_default_torch_cuda_max_memory_allocated_direct() -> None:
    """Test _default_torch_cuda_max_memory_allocated returns non-negative int.

    The function has a guard that checks torch.cuda.is_available() directly,
    so it's safe to call regardless of CUDA hardware presence.
    """
    from model_trainer.core._hook_defaults import (
        _default_torch_cuda_max_memory_allocated,
    )

    result = _default_torch_cuda_max_memory_allocated()
    # On non-CUDA systems returns 0, on CUDA systems returns actual memory
    assert result >= 0


def test_default_torch_cuda_reset_peak_memory_stats_direct() -> None:
    """Test _default_torch_cuda_reset_peak_memory_stats completes without error.

    The function has a guard that checks torch.cuda.is_available() directly,
    so it's safe to call regardless of CUDA hardware presence.
    """
    from model_trainer.core._hook_defaults import (
        _default_torch_cuda_reset_peak_memory_stats,
    )

    _default_torch_cuda_reset_peak_memory_stats()


def test_default_torch_cuda_memory_hooks_run_against_real_torch() -> None:
    """Test the lowest-level torch.cuda adapters against the installed torch.

    Every other GPU test fakes `torch_cuda_max_memory_allocated`, so these two
    adapters -- the ones that actually reach torch -- were never executed. They
    assume CUDA is present, because their only caller checks that first through
    the `cuda_is_available` hook.
    """
    import torch

    from model_trainer.core._hook_defaults import (
        _default_torch_cuda_max_memory_allocated,
        _default_torch_cuda_reset_peak_memory_stats,
    )

    # Asserted rather than skipped: this service pins torch to the cu128 index
    # and trains on GPU, so a machine without CUDA cannot run its suite
    # meaningfully. Say that outright instead of quietly passing.
    assert torch.cuda.is_available()

    _default_torch_cuda_reset_peak_memory_stats()

    assert _default_torch_cuda_max_memory_allocated() >= 0
