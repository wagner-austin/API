"""Tests for platform_ml.testing fake classes.

Ensures complete coverage of FakeTensor, FakeDevice, FakeDType, FakeCudaModule,
and FakeTorchModule.
"""

from __future__ import annotations

from platform_ml.testing import (
    FakeCudaModule,
    FakeDevice,
    FakeDType,
    FakeTensor,
    FakeTorchModule,
)


class TestFakeDevice:
    """Tests for FakeDevice."""

    def test_default_values(self) -> None:
        """Test default device is cpu with no index."""
        device = FakeDevice()
        assert device.type == "cpu"
        assert device.index is None

    def test_custom_values(self) -> None:
        """Test custom device type and index."""
        device = FakeDevice(device_type="cuda", index=0)
        assert device.type == "cuda"
        assert device.index == 0


class TestFakeDType:
    """Tests for FakeDType."""

    def test_instantiation(self) -> None:
        """Test FakeDType satisfies DTypeProtocol (empty protocol)."""
        dtype = FakeDType()
        # DTypeProtocol is empty - just verify construction succeeds
        assert type(dtype).__name__ == "FakeDType"


class TestFakeTensor:
    """Tests for FakeTensor."""

    def test_default_values(self) -> None:
        """Test default tensor has empty shape and cpu device."""
        tensor = FakeTensor()
        assert tensor.shape == ()
        assert type(tensor.dtype).__name__ == "FakeDType"
        assert tensor.device.type == "cpu"
        assert tensor.grad is None

    def test_custom_shape(self) -> None:
        """Test tensor with custom shape."""
        tensor = FakeTensor(shape=(2, 3, 4))
        assert tensor.shape == (2, 3, 4)

    def test_numel_empty(self) -> None:
        """Test numel returns 1 for empty shape."""
        tensor = FakeTensor()
        assert tensor.numel() == 1

    def test_numel_with_shape(self) -> None:
        """Test numel returns product of dimensions."""
        tensor = FakeTensor(shape=(2, 3, 4))
        assert tensor.numel() == 24

    def test_element_size(self) -> None:
        """Test element_size returns 4."""
        tensor = FakeTensor()
        assert tensor.element_size() == 4

    def test_item(self) -> None:
        """Test item returns 0.0."""
        tensor = FakeTensor()
        assert tensor.item() == 0.0

    def test_tolist(self) -> None:
        """Test tolist returns empty list."""
        tensor = FakeTensor()
        assert tensor.tolist() == []

    def test_detach(self) -> None:
        """Test detach returns self."""
        tensor = FakeTensor()
        assert tensor.detach() is tensor

    def test_cpu(self) -> None:
        """Test cpu returns tensor on cpu device."""
        tensor = FakeTensor(shape=(2, 3), device_type="cuda")
        cpu_tensor = tensor.cpu()
        assert cpu_tensor.device.type == "cpu"
        assert cpu_tensor.shape == (2, 3)

    def test_clone(self) -> None:
        """Test clone returns new tensor with same shape."""
        tensor = FakeTensor(shape=(2, 3))
        cloned = tensor.clone()
        assert cloned is not tensor
        assert cloned.shape == (2, 3)

    def test_cuda(self) -> None:
        """Test cuda returns tensor on cuda device."""
        tensor = FakeTensor(shape=(2, 3))
        cuda_tensor = tensor.cuda()
        assert cuda_tensor.device.type == "cuda"
        assert cuda_tensor.shape == (2, 3)

    def test_cuda_with_device(self) -> None:
        """Test cuda with device index."""
        tensor = FakeTensor()
        cuda_tensor = tensor.cuda(device=1)
        assert cuda_tensor.device.type == "cuda"

    def test_to_with_string(self) -> None:
        """Test to with string device."""
        tensor = FakeTensor(shape=(2, 3))
        moved = tensor.to("cuda")
        assert moved.device.type == "cuda"
        assert moved.shape == (2, 3)

    def test_to_with_device(self) -> None:
        """Test to with DeviceProtocol."""
        tensor = FakeTensor()
        target_device = FakeDevice(device_type="cuda", index=0)
        moved = tensor.to(target_device)
        assert moved.device.type == "cuda"

    def test_add(self) -> None:
        """Test __add__ returns self."""
        tensor = FakeTensor()
        result = tensor + 1.0
        assert result is tensor

    def test_mul(self) -> None:
        """Test __mul__ returns self."""
        tensor = FakeTensor()
        result = tensor * 2.0
        assert result is tensor

    def test_truediv(self) -> None:
        """Test __truediv__ returns self."""
        tensor = FakeTensor()
        result = tensor / 2.0
        assert result is tensor


class TestFakeCudaModule:
    """Tests for FakeCudaModule."""

    def test_default_unavailable(self) -> None:
        """Test default cuda is unavailable."""
        cuda = FakeCudaModule()
        assert cuda.is_available() is False
        assert cuda.is_available_call_count == 1

    def test_available(self) -> None:
        """Test cuda available when configured."""
        cuda = FakeCudaModule(cuda_available=True)
        assert cuda.is_available() is True

    def test_call_count_increments(self) -> None:
        """Test call count increments on each call."""
        cuda = FakeCudaModule()
        cuda.is_available()
        cuda.is_available()
        cuda.is_available()
        assert cuda.is_available_call_count == 3


class TestFakeTorchModule:
    """Tests for FakeTorchModule."""

    def test_default_values(self) -> None:
        """Test default torch module."""
        torch = FakeTorchModule()
        assert torch.cuda.is_available() is False
        assert torch.get_num_threads() == 1
        assert torch.set_num_threads_calls == []
        assert torch.manual_seed_calls == []

    def test_cuda_available(self) -> None:
        """Test cuda available when configured."""
        torch = FakeTorchModule(cuda_available=True)
        assert torch.cuda.is_available() is True

    def test_custom_num_threads(self) -> None:
        """Test custom num_threads."""
        torch = FakeTorchModule(num_threads=8)
        assert torch.get_num_threads() == 8

    def test_set_num_threads_records_calls(self) -> None:
        """Test set_num_threads records calls."""
        torch = FakeTorchModule()
        torch.set_num_threads(4)
        torch.set_num_threads(8)
        assert torch.set_num_threads_calls == [4, 8]

    def test_manual_seed_records_calls(self) -> None:
        """Test manual_seed records calls and returns tensor."""
        torch = FakeTorchModule()
        result = torch.manual_seed(42)
        torch.manual_seed(123)
        assert torch.manual_seed_calls == [42, 123]
        assert result.shape == ()

    def test_custom_cuda_module(self) -> None:
        """Test custom cuda module injection."""
        cuda = FakeCudaModule(cuda_available=True)
        torch = FakeTorchModule(cuda_module=cuda)
        torch.cuda.is_available()
        assert cuda.is_available_call_count == 1
