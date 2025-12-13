"""Tests for platform_ml.device_selector module.

Achieves 100% statement and branch coverage by testing all device resolution paths,
precision resolution, and batch size recommendations.
Uses FakeTorchModule from testing.py to test actual code without GPU hardware.
"""

from __future__ import annotations

import pytest

from platform_ml import torch_types
from platform_ml.device_selector import (
    recommended_batch_size,
    resolve_device,
    resolve_precision,
)
from platform_ml.testing import FakeTorchModule
from platform_ml.torch_types import _TorchModuleProtocol


def test_resolve_device_cpu_passthrough() -> None:
    """Test that explicitly requested 'cpu' device is returned as-is."""
    result = resolve_device("cpu")
    assert result == "cpu"


def test_resolve_device_cuda_passthrough() -> None:
    """Test that explicitly requested 'cuda' device is returned as-is."""
    result = resolve_device("cuda")
    assert result == "cuda"


def test_resolve_device_auto_with_cuda_available() -> None:
    """Test that 'auto' resolves to 'cuda' when CUDA is available."""
    fake_torch = FakeTorchModule(cuda_available=True)

    def _fake_import() -> _TorchModuleProtocol:
        return fake_torch

    torch_types._import_torch = _fake_import
    result = resolve_device("auto")
    assert result == "cuda"


def test_resolve_device_auto_with_cuda_unavailable() -> None:
    """Test that 'auto' resolves to 'cpu' when CUDA is unavailable."""
    fake_torch = FakeTorchModule(cuda_available=False)

    def _fake_import() -> _TorchModuleProtocol:
        return fake_torch

    torch_types._import_torch = _fake_import
    result = resolve_device("auto")
    assert result == "cpu"


def test_resolve_device_auto_uses_hook() -> None:
    """Test that 'auto' resolution uses the hook, not torch directly.

    This verifies the hook is actually called, enabling test isolation.
    """
    from platform_ml.testing import FakeCudaModule

    fake_cuda = FakeCudaModule(cuda_available=False)
    fake_torch = FakeTorchModule(cuda_module=fake_cuda)

    def _fake_import() -> _TorchModuleProtocol:
        return fake_torch

    torch_types._import_torch = _fake_import
    resolve_device("auto")
    assert fake_cuda.is_available_call_count == 1, "Hook should have been called exactly once"


# =============================================================================
# Precision resolution tests
# =============================================================================


def test_resolve_precision_fp32_on_cuda() -> None:
    """fp32 is allowed on any device."""
    assert resolve_precision("fp32", "cuda") == "fp32"


def test_resolve_precision_fp32_on_cpu() -> None:
    """fp32 is allowed on any device."""
    assert resolve_precision("fp32", "cpu") == "fp32"


def test_resolve_precision_fp16_on_cuda() -> None:
    """fp16 is allowed on cuda."""
    assert resolve_precision("fp16", "cuda") == "fp16"


def test_resolve_precision_fp16_on_cpu_raises() -> None:
    """fp16 is NOT allowed on cpu - should raise RuntimeError."""
    with pytest.raises(RuntimeError, match=r"fp16.*not supported on CPU"):
        resolve_precision("fp16", "cpu")


def test_resolve_precision_bf16_on_cuda() -> None:
    """bf16 is allowed on cuda."""
    assert resolve_precision("bf16", "cuda") == "bf16"


def test_resolve_precision_bf16_on_cpu_raises() -> None:
    """bf16 is NOT allowed on cpu - should raise RuntimeError."""
    with pytest.raises(RuntimeError, match=r"bf16.*not supported on CPU"):
        resolve_precision("bf16", "cpu")


def test_resolve_precision_auto_on_cuda() -> None:
    """auto resolves to fp16 on cuda."""
    assert resolve_precision("auto", "cuda") == "fp16"


def test_resolve_precision_auto_on_cpu() -> None:
    """auto resolves to fp32 on cpu."""
    assert resolve_precision("auto", "cpu") == "fp32"


# =============================================================================
# Batch size recommendation tests
# =============================================================================


def test_recommended_batch_size_bumps_on_cuda_small_batch() -> None:
    """Small batch sizes (<= 4) get bumped to 8 on CUDA."""
    assert recommended_batch_size(4, "cuda") == 8
    assert recommended_batch_size(2, "cuda") == 8
    assert recommended_batch_size(1, "cuda") == 8


def test_recommended_batch_size_preserves_on_cuda_large_batch() -> None:
    """Larger batch sizes (> 4) are preserved on CUDA."""
    assert recommended_batch_size(8, "cuda") == 8
    assert recommended_batch_size(16, "cuda") == 16
    assert recommended_batch_size(32, "cuda") == 32


def test_recommended_batch_size_preserves_on_cpu() -> None:
    """All batch sizes are preserved on CPU."""
    assert recommended_batch_size(4, "cpu") == 4
    assert recommended_batch_size(8, "cpu") == 8
    assert recommended_batch_size(1, "cpu") == 1
