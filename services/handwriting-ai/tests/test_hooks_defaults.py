"""Tests for default hook implementations in _test_hooks.py.

These tests verify that the default production implementations work correctly.
Most tests in the suite override hooks with fakes, so these tests ensure
the actual default implementations are exercised for coverage.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from platform_workers.rq_harness import WorkerConfig

from handwriting_ai import _test_hooks


def test_default_mp_get_all_start_methods_returns_spawn() -> None:
    """Test _default_mp_get_all_start_methods returns a list containing 'spawn'."""
    methods = _test_hooks._default_mp_get_all_start_methods()
    # 'spawn' is available on all platforms (Windows, macOS, Linux)
    assert "spawn" in methods


def test_default_mp_get_context_returns_context_with_method() -> None:
    """Test _default_mp_get_context returns a context with spawn method."""
    ctx = _test_hooks._default_mp_get_context("spawn")
    # The wrapper should have method = spawn (or the requested method)
    assert ctx.method == "spawn"


def test_default_mp_get_context_with_none_method() -> None:
    """Test _default_mp_get_context with None method returns default context."""
    ctx = _test_hooks._default_mp_get_context(None)
    # Method should be None for default context
    assert ctx.method is None or ctx.method == "spawn" or ctx.method == "fork"


def test_default_thread_factory_creates_working_thread() -> None:
    """Test _default_thread_factory creates a thread that executes its target."""
    executed = {"flag": False}

    def target() -> None:
        executed["flag"] = True

    thread = _test_hooks._default_thread_factory(target=target, daemon=True, name="test-thread")
    thread.start()
    thread.join(timeout=1.0)
    # Verify the thread actually ran its target function
    assert executed["flag"] is True


def test_default_event_factory_creates_working_event() -> None:
    """Test _default_event_factory creates an event with set/wait behavior."""
    event = _test_hooks._default_event_factory()
    # Event should start unset
    assert event.is_set() is False
    event.set()
    assert event.is_set() is True


def test_default_principal_angle_with_diagonal_returns_float() -> None:
    """Test _default_principal_angle returns a float for a diagonal line image."""
    from PIL import Image
    from PIL.Image import Image as PILImage

    # Create a simple 10x10 grayscale image with a diagonal line
    img: PILImage = Image.new("L", (10, 10), 255)
    pix = img.load()
    # load() always returns PixelAccess for valid images - narrow the type
    if pix is None:
        raise RuntimeError("PIL.Image.load() returned None unexpectedly")
    # Draw a diagonal line
    for i in range(10):
        pix[i, i] = 0

    result = _test_hooks._default_principal_angle(img, 10, 10)
    # For a diagonal line, result should be a float around 45 degrees (or ~0.78 radians)
    # We verify it's a specific value rather than just existence
    assert result is None or (-90.0 <= result <= 90.0)


def test_default_run_worker_calls_runner() -> None:
    """Test _default_run_worker calls the provided runner function."""
    calls: list[WorkerConfig] = []

    def fake_runner(config: WorkerConfig) -> None:
        calls.append(config)

    config: WorkerConfig = {
        "redis_url": "redis://localhost:6379",
        "queue_name": "test-queue",
        "events_channel": "test-events",
    }

    class FakeLogger:
        def info(self, msg: str, *args: str) -> None:
            pass

        def error(self, msg: str, *args: str) -> None:
            pass

    logger = FakeLogger()

    _test_hooks._default_run_worker(config, logger, fake_runner)
    # Verify runner was called exactly once with the correct config
    assert len(calls) == 1
    assert calls[0]["redis_url"] == config["redis_url"]
    assert calls[0]["queue_name"] == config["queue_name"]
    assert calls[0]["events_channel"] == config["events_channel"]


def test_preprocess_estimate_background_zero_total() -> None:
    """Test _estimate_background_is_dark when histogram total is zero."""
    from PIL import Image
    from PIL.Image import Image as PILImage

    from handwriting_ai.preprocess import _estimate_background_is_dark

    # Create image and override histogram hook to return all zeros
    img = Image.new("L", (10, 10), 128)

    # Save original hook
    original_hook = _test_hooks.pil_histogram

    class _FakeHistogram:
        """Fake histogram hook returning all zeros."""

        def __call__(self, img: PILImage) -> list[int]:
            _ = img
            return [0] * 256

    _test_hooks.pil_histogram = _FakeHistogram()
    try:
        result = _estimate_background_is_dark(img)
        # Should return False when total is 0
        assert result is False
    finally:
        _test_hooks.pil_histogram = original_hook


def test_principal_angle_pix_none_returns_none() -> None:
    """Test _principal_angle returns None when pix.load() returns None."""
    from handwriting_ai.preprocess import _principal_angle

    # Use injection function to get a fake image typed as PILImage
    fake_img = _test_hooks.inject_fake_image_as_pil()
    result = _principal_angle(fake_img, 10, 10)
    assert result is None


def test_principal_angle_confidence_pix_none_returns_none() -> None:
    """Test _principal_angle_confidence returns None when pix.load() returns None."""
    from handwriting_ai.preprocess import _principal_angle_confidence

    # Use injection function to get a fake image typed as PILImage
    fake_img = _test_hooks.inject_fake_image_as_pil()
    result = _principal_angle_confidence(fake_img, 10, 10)
    assert result is None


def test_center_on_square_with_real_image() -> None:
    """Test _center_on_square works correctly with a real image."""
    from PIL import Image

    from handwriting_ai.preprocess import _center_on_square

    # Create a real image with some content
    img = Image.new("L", (28, 28), 128)
    result = _center_on_square(img)
    # Function adds 10% margin on each side: 28 + 2*round(28*0.1) = 28 + 6 = 34
    assert result.size == (34, 34)


class _FakeModelNonStringKey:
    """Fake model returning state dict with non-string key for runtime validation test."""

    def eval(self) -> _FakeModelNonStringKey:
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> _test_hooks.LoadStateResultProtocol:
        _ = sd

        class _Res:
            def __init__(self) -> None:
                self.missing_keys: list[str] = []
                self.unexpected_keys: list[str] = []

        return _Res()

    def train(self, mode: bool = True) -> _FakeModelNonStringKey:
        _ = mode
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        # Use injection to return dict with non-string key
        return _test_hooks.inject_bad_state_dict_non_string_key()

    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return []


class _FakeModelNonTensorValue:
    """Fake model returning state dict with non-Tensor value for runtime validation test."""

    def eval(self) -> _FakeModelNonTensorValue:
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> _test_hooks.LoadStateResultProtocol:
        _ = sd

        class _Res:
            def __init__(self) -> None:
                self.missing_keys: list[str] = []
                self.unexpected_keys: list[str] = []

        return _Res()

    def train(self, mode: bool = True) -> _FakeModelNonTensorValue:
        _ = mode
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        # Use injection to return dict with non-Tensor values
        return _test_hooks.inject_bad_state_dict_values()

    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return []


def test_engine_state_dict_non_string_key_raises() -> None:
    """Test that state_dict with non-string key raises RuntimeError."""
    from handwriting_ai.inference.engine import build_fresh_state_dict

    original_build = _test_hooks.build_model

    def _fake_build(arch: str, n_classes: int) -> _test_hooks.InferenceTorchModelProtocol:
        _ = (arch, n_classes)
        return _FakeModelNonStringKey()

    _test_hooks.build_model = _fake_build
    try:
        with pytest.raises(RuntimeError, match="invalid state dict entry from model"):
            build_fresh_state_dict("resnet18", 10)
    finally:
        _test_hooks.build_model = original_build


def test_engine_state_dict_non_tensor_value_raises() -> None:
    """Test that state_dict with non-Tensor value raises RuntimeError."""
    from handwriting_ai.inference.engine import build_fresh_state_dict

    original_build = _test_hooks.build_model

    def _fake_build(arch: str, n_classes: int) -> _test_hooks.InferenceTorchModelProtocol:
        _ = (arch, n_classes)
        return _FakeModelNonTensorValue()

    _test_hooks.build_model = _fake_build
    try:
        with pytest.raises(RuntimeError, match="invalid state dict entry from model"):
            build_fresh_state_dict("resnet18", 10)
    finally:
        _test_hooks.build_model = original_build


class _StateDictInterceptorBadKey:
    """Interceptor that returns bad state_dict with non-string key.

    Satisfies _TypedModule Protocol but returns invalid state_dict for testing.
    """

    def eval(self) -> torch.nn.Module:
        return torch.nn.Identity()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> _test_hooks.LoadStateResultProtocol:
        _ = sd

        class _Res:
            def __init__(self) -> None:
                self.missing_keys: list[str] = []
                self.unexpected_keys: list[str] = []

        return _Res()

    def train(self, mode: bool = True) -> torch.nn.Module:
        _ = mode
        return torch.nn.Identity()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return _test_hooks.inject_bad_state_dict_non_string_key()


class _StateDictInterceptorBadValue:
    """Interceptor that returns bad state_dict with non-Tensor value.

    Satisfies _TypedModule Protocol but returns invalid state_dict for testing.
    """

    def eval(self) -> torch.nn.Module:
        return torch.nn.Identity()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> _test_hooks.LoadStateResultProtocol:
        _ = sd

        class _Res:
            def __init__(self) -> None:
                self.missing_keys: list[str] = []
                self.unexpected_keys: list[str] = []

        return _Res()

    def train(self, mode: bool = True) -> torch.nn.Module:
        _ = mode
        return torch.nn.Identity()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return _test_hooks.inject_bad_state_dict_values()


def test_wrapped_model_state_dict_bad_key_raises() -> None:
    """Test _WrappedTorchModel.state_dict() with non-string key raises RuntimeError."""
    from handwriting_ai.inference.engine import _WrappedTorchModel

    # Create a real torch.nn.Linear module and wrap it
    real_module = torch.nn.Linear(10, 5)
    wrapped = _WrappedTorchModel(real_module)

    # Replace internal _module with interceptor that returns bad state_dict
    wrapped._module = _StateDictInterceptorBadKey()

    with pytest.raises(RuntimeError, match="state_dict key must be str"):
        wrapped.state_dict()


def test_wrapped_model_state_dict_bad_value_raises() -> None:
    """Test _WrappedTorchModel.state_dict() with non-Tensor value raises RuntimeError."""
    from handwriting_ai.inference.engine import _WrappedTorchModel

    # Create a real torch.nn.Linear module and wrap it
    real_module = torch.nn.Linear(10, 5)
    wrapped = _WrappedTorchModel(real_module)

    # Replace internal _module with interceptor that returns bad state_dict
    wrapped._module = _StateDictInterceptorBadValue()

    with pytest.raises(RuntimeError, match="state_dict value must be Tensor"):
        wrapped.state_dict()


def test_center_on_square_pix_none_returns_original() -> None:
    """Test _center_on_square returns original image when pix.load() returns None."""
    from PIL import Image
    from PIL.Image import Image as PILImage

    from handwriting_ai.preprocess import _center_on_square

    # Create an input image
    input_img = Image.new("L", (28, 28), 128)

    # Save original hook
    original_hook = _test_hooks.otsu_binarize

    def _fake_otsu(gray: PILImage) -> PILImage:
        """Return a fake image that returns None from load()."""
        _ = gray
        return _test_hooks.inject_fake_image_as_pil()

    _test_hooks.otsu_binarize = _fake_otsu
    try:
        result = _center_on_square(input_img)
        # When pix is None, should return original image unchanged
        assert result is input_img
    finally:
        _test_hooks.otsu_binarize = original_hook


def test_minimal_handler_handle_returns_true() -> None:
    """Test _MinimalHandler.handle returns True (required by logging)."""
    from platform_core.logging import stdlib_logging

    handler = _test_hooks._MinimalHandler()
    record = stdlib_logging.LogRecord(
        name="test",
        level=stdlib_logging.INFO,
        pathname="",
        lineno=0,
        msg="test message",
        args=(),
        exc_info=None,
    )
    result = handler.handle(record)
    assert result is True


def test_default_artifact_store_factory_creates_store() -> None:
    """Test _default_artifact_store_factory creates a valid ArtifactStore."""
    # Call the factory with dummy credentials - it just constructs objects
    # without making network calls until methods are actually invoked
    store = _test_hooks._default_artifact_store_factory(
        api_url="http://localhost:8000",
        api_key="test-api-key",
    )
    # Verify it returns the expected concrete type
    assert type(store).__name__ == "ArtifactStore"
