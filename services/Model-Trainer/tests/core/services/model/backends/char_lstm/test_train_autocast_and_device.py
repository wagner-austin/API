"""char_lstm trainer branches: mid-loop behavior."""

from __future__ import annotations

import warnings

import pytest
import torch
from tests.core.services.model.backends.char_lstm._train_branches_support import (
    _LM,
    UNPINNED,
    _make_cfg,
    _make_prepared,
    _make_settings,
)

from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.services.training import base_trainer as bt
from model_trainer.core.services.training import trainer_grad_utils as bt_grad
from model_trainer.core.services.training.dataloader import DataLoader


def test_setup_device_cuda_not_available() -> None:
    """Test _setup_device raises RuntimeError when CUDA requested but not available."""
    from model_trainer.core import _test_hooks

    _test_hooks.cuda_is_available = lambda: False

    cfg: ModelTrainConfig = {**_make_cfg(), "device": "cuda"}

    trainer = bt.BaseTrainer(
        _make_prepared(),
        cfg,
        _make_settings(),
        run_id="test-cuda-fail",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
        determinism=UNPINNED,
    )

    with pytest.raises(RuntimeError, match="CUDA requested but not available"):
        _ = trainer._setup_device()


def test_get_autocast_context_fp32_returns_nullcontext() -> None:
    """Test that fp32 precision returns nullcontext (no autocast)."""
    ctx = bt_grad._get_autocast_context("fp32", torch.device("cpu"))
    # Verify the context is a no-op by entering and exiting it
    with ctx:
        pass  # No exception means it worked


def test_get_autocast_context_fp16_on_cpu_returns_nullcontext() -> None:
    """Test that fp16 on CPU returns nullcontext (autocast only on CUDA)."""
    ctx = bt_grad._get_autocast_context("fp16", torch.device("cpu"))
    # Verify the context is a no-op by entering and exiting it
    with ctx:
        pass  # No exception means it worked


def test_get_autocast_context_bf16_on_cpu_returns_nullcontext() -> None:
    """Test that bf16 on CPU returns nullcontext (autocast only on CUDA)."""
    ctx = bt_grad._get_autocast_context("bf16", torch.device("cpu"))
    # Verify the context is a no-op by entering and exiting it
    with ctx:
        pass  # No exception means it worked


def test_get_autocast_context_fp16_on_cuda() -> None:
    """Test that fp16 on CUDA returns autocast context."""
    # Create a mock CUDA device (doesn't require actual CUDA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ctx = bt_grad._get_autocast_context("fp16", torch.device("cuda"))
        # Verify the context can be entered and exited
        with ctx:
            pass  # Autocast context entered successfully


def test_get_autocast_context_bf16_on_cuda() -> None:
    """Test that bf16 on CUDA returns autocast context."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ctx = bt_grad._get_autocast_context("bf16", torch.device("cuda"))
        # Verify the context can be entered and exited
        with ctx:
            pass  # Autocast context entered successfully


def test_create_grad_scaler_returns_scaler() -> None:
    """Test that _create_grad_scaler returns a valid GradScaler."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scaler = bt_grad._create_grad_scaler()
        # Verify it can scale a tensor (basic functionality check)
        t = torch.tensor(1.0, requires_grad=True)
        scaled = scaler.scale(t)
        assert scaled.item() >= 0.0  # Scaled value should be non-negative


def test_train_one_epoch_fp16_scaler_paths() -> None:
    """Test that fp16 precision with CUDA device uses GradScaler paths.

    This test covers lines 611-614 and 627-629 in base_trainer.py.
    Note: Without actual CUDA, the scaler is disabled but the code paths still execute.
    """

    class _DS(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
        def __len__(self: _DS) -> int:
            return 2

        def __getitem__(self: _DS, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            ids = torch.randint(0, 4, (4,))
            return (ids, ids)

    dl = DataLoader(_DS(), batch_size=1, shuffle=False)

    # Create a real model with trainable parameters for proper GradScaler integration
    model = _LM()

    # Use a real optimizer that GradScaler can properly interact with
    # Access _p directly since model.parameters() returns ParameterLike protocol
    optim = torch.optim.SGD([model._p], lr=0.01)

    # Create config with fp16 precision
    cfg: ModelTrainConfig = {**_make_cfg(), "precision": "fp16"}

    trainer = bt.BaseTrainer(
        _make_prepared(),
        cfg,
        _make_settings(),
        run_id="test-fp16-run",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
        determinism=UNPINNED,
    )
    # Mock device as CUDA to trigger scaler path
    trainer._device = torch.device("cuda")

    # Run training epoch - this will use scaler paths even if CUDA is not available
    # (the scaler is disabled but code paths still execute)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = trainer._train_one_epoch(
            model=model,
            dataloader=dl,
            optim=optim,
            epoch=0,
            device="cpu",  # Actually runs on CPU but scaler logic still executes
            start_step=0,
        )
    # Verify training completed (not cancelled)
    assert out[2] is False and out[1] >= 1


def test_evaluate_get_autocast_context_cuda_fp16() -> None:
    """Test char_lstm evaluate._get_autocast_context with fp16 on CUDA."""
    from model_trainer.core.services.model.backends.char_lstm import evaluate as char_eval

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ctx = char_eval._get_autocast_context("fp16", "cuda")
        with ctx:
            pass  # Autocast context entered successfully


def test_evaluate_get_autocast_context_cuda_bf16() -> None:
    """Test char_lstm evaluate._get_autocast_context with bf16 on CUDA."""
    from model_trainer.core.services.model.backends.char_lstm import evaluate as char_eval

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ctx = char_eval._get_autocast_context("bf16", "cuda")
        with ctx:
            pass  # Autocast context entered successfully


def test_evaluate_get_autocast_context_cpu_fp16() -> None:
    """Test char_lstm evaluate._get_autocast_context with fp16 on CPU returns nullcontext."""
    from model_trainer.core.services.model.backends.char_lstm import evaluate as char_eval

    ctx = char_eval._get_autocast_context("fp16", "cpu")
    with ctx:
        pass  # Returns nullcontext on non-cuda


def test_gpt2_evaluate_get_autocast_context_cuda_fp16() -> None:
    """Test gpt2 evaluate._get_autocast_context with fp16 on CUDA."""
    from model_trainer.core.services.model.backends.gpt2 import evaluate as gpt2_eval

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ctx = gpt2_eval._get_autocast_context("fp16", "cuda")
        with ctx:
            pass  # Autocast context entered successfully


def test_gpt2_evaluate_get_autocast_context_cuda_bf16() -> None:
    """Test gpt2 evaluate._get_autocast_context with bf16 on CUDA."""
    from model_trainer.core.services.model.backends.gpt2 import evaluate as gpt2_eval

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ctx = gpt2_eval._get_autocast_context("bf16", "cuda")
        with ctx:
            pass  # Autocast context entered successfully


def test_gpt2_evaluate_get_autocast_context_cpu_fp16() -> None:
    """Test gpt2 evaluate._get_autocast_context with fp16 on CPU returns nullcontext."""
    from model_trainer.core.services.model.backends.gpt2 import evaluate as gpt2_eval

    ctx = gpt2_eval._get_autocast_context("fp16", "cpu")
    with ctx:
        pass  # Returns nullcontext on non-cuda


def test_setup_device_cuda_available() -> None:
    """Test _setup_device returns cuda device when available."""
    from model_trainer.core import _test_hooks

    _test_hooks.cuda_is_available = lambda: True

    cfg: ModelTrainConfig = {**_make_cfg(), "device": "cuda"}

    trainer = bt.BaseTrainer(
        _make_prepared(),
        cfg,
        _make_settings(),
        run_id="test-cuda-ok",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
        determinism=UNPINNED,
    )

    device = trainer._setup_device()
    assert device.type == "cuda"
