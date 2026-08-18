"""Tests for GPU memory calculation path in base_trainer.py.

This test specifically covers line 326 in base_trainer.py - the peak_gpu_memory_mb
calculation when device is 'cuda' and GPU memory is reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import torch

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.backend_factory import create_char_lstm_backend
from model_trainer.core.services.tokenizer.char_backend import CharBackend


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


def _write_tiny_corpus(root: Path) -> str:
    out_dir = root / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "tiny.txt"
    corpus_lines = ["aba", "abbaba", "abaaba", "babbab", "ababab", "bababa"]
    corpus_text = "\n".join(corpus_lines * 10) + "\n"
    fp.write_text(corpus_text, encoding="utf-8")
    return str(out_dir)


def _train_char_tokenizer(root: Path, corpus_path: str, tok_id: str) -> str:
    tok_out = root / "artifacts" / "tokenizers" / tok_id
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
    return str(tok_out)


def _noop(_: float) -> None:
    return None


def _never() -> bool:
    return False


def test_gpu_memory_mb_calculation_path(tmp_path: Path, settings_factory: _SettingsFactory) -> None:
    """Test base_trainer.py line 326 - GPU memory calculation when device='cuda'.

    This test overrides hooks to simulate a 'cuda' device scenario with GPU memory
    reported, without requiring actual CUDA hardware. The device='cuda' in config
    is what triggers the calculation path, and we override gpu_max_memory_allocated
    to return a non-zero value.

    NOTE: We use device='cpu' for actual tensor operations to avoid CUDA dependency,
    but the cfg["device"] string is set to 'cuda' to exercise the conditional path.
    The base_trainer checks cfg["device"] as a string, not actual tensor placement.
    """
    # Set up paths
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
    )

    # Create corpus and tokenizer
    corpus_path = _write_tiny_corpus(tmp_path)
    tok_id = "tok-gpu-test"
    tok_dir = _train_char_tokenizer(tmp_path, corpus_path, tok_id)

    # Create config with device='cuda' string to trigger the conditional
    # NOTE: Actual training uses CPU tensors, but cfg["device"] is checked as string
    cfg: ModelTrainConfig = {
        "model_family": "char_lstm",
        "model_size": "tiny",
        "max_seq_len": 16,
        "num_epochs": 3,  # Multiple epochs to verify loss reduction
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
        "device": "cpu",  # Use CPU for actual tensors
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
    handle = CharBackend().load(str(Path(tok_dir) / "tokenizer.json"))
    prepared = backend.prepare(cfg, settings, tokenizer=handle)

    # Save original hooks
    orig_gpu_mem = _test_hooks.gpu_max_memory_allocated
    orig_cuda_is_available = _test_hooks.cuda_is_available
    orig_torch_device = _test_hooks.torch_device
    orig_gpu_reset = _test_hooks.gpu_reset_peak_memory_stats

    # Override hooks to simulate CUDA scenario
    def _fake_gpu_memory() -> int:
        return 1024 * 1024 * 100  # 100 MB

    def _fake_cuda_available() -> bool:
        return True

    def _fake_torch_device(device_str: str) -> torch.device:
        # Always return CPU device for actual tensor operations
        return torch.device("cpu")

    def _fake_gpu_reset() -> None:
        # No-op since we're not actually using CUDA
        pass

    _test_hooks.gpu_max_memory_allocated = _fake_gpu_memory
    _test_hooks.cuda_is_available = _fake_cuda_available
    _test_hooks.torch_device = _fake_torch_device
    _test_hooks.gpu_reset_peak_memory_stats = _fake_gpu_reset

    # Temporarily modify cfg to have device='cuda' to trigger the conditional
    # The actual tensor operations happen on CPU, but the string check passes
    cfg["device"] = "cuda"

    # Track losses during training for loss decrease verification
    train_losses: list[float] = []

    def track_progress(
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

    try:
        # Run training - this exercises line 326 calculation path
        out = backend.train(
            cfg,
            settings,
            run_id="run-gpu-test",
            heartbeat=_noop,
            cancelled=_never,
            resume=False,
            prepared=prepared,
            progress=track_progress,
        )
        # Verify training completed
        assert out["steps"] >= 1

        # Verify loss decreases (guard rule requirement)
        loss_before = train_losses[0]
        loss_after = train_losses[-1]
        assert loss_after < loss_before, (
            f"Training should reduce loss: before={loss_before:.4f}, after={loss_after:.4f}"
        )
    finally:
        # Restore hooks
        _test_hooks.gpu_max_memory_allocated = orig_gpu_mem
        _test_hooks.cuda_is_available = orig_cuda_is_available
        _test_hooks.torch_device = orig_torch_device
        _test_hooks.gpu_reset_peak_memory_stats = orig_gpu_reset
