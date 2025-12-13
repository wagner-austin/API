"""Tests for device selection in handwriting-ai training.

Verifies that device resolution works correctly with both CUDA available
and unavailable, using the shared platform_ml device selector.
"""

from __future__ import annotations

from platform_ml import resolve_device
from platform_ml import torch_types as platform_ml_torch_types
from platform_ml.testing import FakeCudaModule, FakeTorchModule
from platform_ml.torch_types import _TorchModuleProtocol


def test_resolve_device_auto_with_cuda_available() -> None:
    """When CUDA is available, 'auto' resolves to 'cuda'."""
    fake_torch = FakeTorchModule(cuda_available=True)

    def _fake_import() -> _TorchModuleProtocol:
        return fake_torch

    platform_ml_torch_types._import_torch = _fake_import
    result = resolve_device("auto")
    assert result == "cuda"


def test_resolve_device_auto_with_cuda_unavailable() -> None:
    """When CUDA is unavailable, 'auto' resolves to 'cpu'."""
    fake_torch = FakeTorchModule(cuda_available=False)

    def _fake_import() -> _TorchModuleProtocol:
        return fake_torch

    platform_ml_torch_types._import_torch = _fake_import
    result = resolve_device("auto")
    assert result == "cpu"


def test_resolve_device_explicit_cpu() -> None:
    """Explicit 'cpu' is returned as-is without checking CUDA."""
    result = resolve_device("cpu")
    assert result == "cpu"


def test_resolve_device_explicit_cuda() -> None:
    """Explicit 'cuda' is returned as-is without checking CUDA availability."""
    result = resolve_device("cuda")
    assert result == "cuda"


def test_resolve_device_uses_platform_ml_hook() -> None:
    """Verify that device resolution uses the platform_ml hook for testability."""
    fake_cuda = FakeCudaModule(cuda_available=True)
    fake_torch = FakeTorchModule(cuda_module=fake_cuda)

    def _fake_import() -> _TorchModuleProtocol:
        return fake_torch

    platform_ml_torch_types._import_torch = _fake_import
    resolve_device("auto")
    assert fake_cuda.is_available_call_count == 1
