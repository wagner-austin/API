from __future__ import annotations

import torch
from torch import nn

from handwriting_ai import _test_hooks
from handwriting_ai.training.train_utils import (
    bytes_of_model_and_grads,
    torch_allocator_stats,
)


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Create two parameters with specific sizes
        # p1: 100 elements of float32 (4 bytes each) = 400 bytes
        # p2: 50 elements of float64 (8 bytes each) = 400 bytes with grad
        self.p1 = nn.Parameter(torch.zeros(100, dtype=torch.float32))
        self.p2 = nn.Parameter(torch.zeros(50, dtype=torch.float64))


def test_bytes_of_model_and_grads() -> None:
    # Create model and set up gradients
    m = _FakeModel()
    # Manually set grad on p2 only
    m.p2.grad = torch.zeros(50, dtype=torch.float64)

    param_b, grad_b = bytes_of_model_and_grads(m)
    # p1: 100 x 4 bytes = 400, p2: 50 x 8 bytes = 400
    assert param_b == (100 * 4) + (50 * 8)
    # Only p2 has grad: 50 x 8 = 400
    assert grad_b == (50 * 8)


def test_torch_allocator_stats_sane() -> None:
    available, allocated, reserved, max_alloc = torch_allocator_stats()
    if not available:
        assert allocated == 0 and reserved == 0 and max_alloc == 0
    else:
        assert allocated >= 0 and reserved >= 0 and max_alloc >= 0


def test_torch_allocator_stats_cuda_branch() -> None:
    # Simulate CUDA available and provide deterministic values
    _test_hooks.torch_cuda_is_available = lambda: True
    _test_hooks.torch_cuda_current_device = lambda: 0
    _test_hooks.torch_cuda_memory_allocated = lambda dev: 123456
    _test_hooks.torch_cuda_memory_reserved = lambda dev: 234567
    _test_hooks.torch_cuda_max_memory_allocated = lambda dev: 345678

    available, allocated, reserved, max_alloc = torch_allocator_stats()
    assert available is True
    assert allocated == 123456
    assert reserved == 234567
    assert max_alloc == 345678
