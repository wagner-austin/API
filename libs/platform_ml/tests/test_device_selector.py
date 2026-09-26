"""Tests for platform_ml.device_selector module.

Achieves 100% statement and branch coverage by testing all device resolution paths,
precision resolution, and batch size recommendations.
Uses FakeTorchModule from testing.py to test actual code without GPU hardware.
"""

from __future__ import annotations

import pytest

from platform_ml import torch_types
from platform_ml.device_selector import (
    RequestedDevice,
    RequestedPrecision,
    ResolvedDevice,
    ResolvedPrecision,
    recommended_batch_size,
    resolve_device,
    resolve_precision,
)
from platform_ml.testing import FakeTorchModule
from platform_ml.torch_types import _TorchModuleProtocol


def test_requested_device_words_are_the_resolved_words_plus_auto() -> None:
    """A request names every concrete device by the same word, plus 'auto'."""
    resolved = [device.value for device in ResolvedDevice]
    assert [device.value for device in RequestedDevice] == [*resolved, "auto"]


def test_requested_precision_words_are_the_resolved_words_plus_auto() -> None:
    """A request names every concrete precision by the same word, plus 'auto'."""
    resolved = [precision.value for precision in ResolvedPrecision]
    assert [precision.value for precision in RequestedPrecision] == [*resolved, "auto"]


def test_resolve_device_cpu_passthrough() -> None:
    """Test that explicitly requested CPU is returned as-is."""
    assert resolve_device(RequestedDevice.CPU) is ResolvedDevice.CPU


def test_resolve_device_cuda_passthrough() -> None:
    """Test that explicitly requested CUDA is returned as-is."""
    assert resolve_device(RequestedDevice.CUDA) is ResolvedDevice.CUDA


def test_resolve_device_auto_with_cuda_available() -> None:
    """Test that AUTO resolves to CUDA when CUDA is available."""
    fake_torch = FakeTorchModule(cuda_available=True)

    def _fake_import() -> _TorchModuleProtocol:
        return fake_torch

    torch_types._import_torch = _fake_import
    assert resolve_device(RequestedDevice.AUTO) is ResolvedDevice.CUDA


def test_resolve_device_auto_with_cuda_unavailable() -> None:
    """Test that AUTO resolves to CPU when CUDA is unavailable."""
    fake_torch = FakeTorchModule(cuda_available=False)

    def _fake_import() -> _TorchModuleProtocol:
        return fake_torch

    torch_types._import_torch = _fake_import
    assert resolve_device(RequestedDevice.AUTO) is ResolvedDevice.CPU


def test_resolve_device_auto_uses_hook() -> None:
    """Test that AUTO resolution uses the hook, not torch directly.

    This verifies the hook is actually called, enabling test isolation.
    """
    from platform_ml.testing import FakeCudaModule

    fake_cuda = FakeCudaModule(cuda_available=False)
    fake_torch = FakeTorchModule(cuda_module=fake_cuda)

    def _fake_import() -> _TorchModuleProtocol:
        return fake_torch

    torch_types._import_torch = _fake_import
    resolve_device(RequestedDevice.AUTO)
    assert fake_cuda.is_available_call_count == 1, "Hook should have been called exactly once"


# =============================================================================
# Precision resolution tests
# =============================================================================


def test_resolve_precision_fp32_on_any_device() -> None:
    """FP32 is allowed on every device."""
    for device in ResolvedDevice:
        assert resolve_precision(RequestedPrecision.FP32, device) is ResolvedPrecision.FP32


def test_resolve_precision_fp16_on_cuda() -> None:
    """FP16 is allowed on CUDA."""
    resolved = resolve_precision(RequestedPrecision.FP16, ResolvedDevice.CUDA)
    assert resolved is ResolvedPrecision.FP16


def test_resolve_precision_fp16_on_cpu_raises() -> None:
    """FP16 is NOT allowed on CPU - should raise RuntimeError."""
    with pytest.raises(RuntimeError, match=r"^fp16 precision is not supported on CPU$"):
        resolve_precision(RequestedPrecision.FP16, ResolvedDevice.CPU)


def test_resolve_precision_bf16_on_cuda() -> None:
    """BF16 is allowed on CUDA."""
    resolved = resolve_precision(RequestedPrecision.BF16, ResolvedDevice.CUDA)
    assert resolved is ResolvedPrecision.BF16


def test_resolve_precision_bf16_on_cpu_raises() -> None:
    """BF16 is NOT allowed on CPU - should raise RuntimeError."""
    with pytest.raises(RuntimeError, match=r"^bf16 precision is not supported on CPU$"):
        resolve_precision(RequestedPrecision.BF16, ResolvedDevice.CPU)


def test_resolve_precision_auto_on_cuda() -> None:
    """AUTO resolves to FP16 on CUDA."""
    resolved = resolve_precision(RequestedPrecision.AUTO, ResolvedDevice.CUDA)
    assert resolved is ResolvedPrecision.FP16


def test_resolve_precision_auto_on_cpu() -> None:
    """AUTO resolves to FP32 on CPU."""
    resolved = resolve_precision(RequestedPrecision.AUTO, ResolvedDevice.CPU)
    assert resolved is ResolvedPrecision.FP32


# =============================================================================
# Batch size recommendation tests
# =============================================================================


def test_recommended_batch_size_bumps_on_cuda_small_batch() -> None:
    """Small batch sizes (<= 4) get bumped to 8 on CUDA."""
    assert recommended_batch_size(4, ResolvedDevice.CUDA) == 8
    assert recommended_batch_size(2, ResolvedDevice.CUDA) == 8
    assert recommended_batch_size(1, ResolvedDevice.CUDA) == 8


def test_recommended_batch_size_preserves_on_cuda_large_batch() -> None:
    """Larger batch sizes (> 4) are preserved on CUDA."""
    assert recommended_batch_size(8, ResolvedDevice.CUDA) == 8
    assert recommended_batch_size(16, ResolvedDevice.CUDA) == 16
    assert recommended_batch_size(32, ResolvedDevice.CUDA) == 32


def test_recommended_batch_size_preserves_on_cpu() -> None:
    """All batch sizes are preserved on CPU."""
    assert recommended_batch_size(4, ResolvedDevice.CPU) == 4
    assert recommended_batch_size(8, ResolvedDevice.CPU) == 8
    assert recommended_batch_size(1, ResolvedDevice.CPU) == 1
