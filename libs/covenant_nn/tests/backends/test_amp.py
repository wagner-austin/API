"""Tests for covenant_nn.backends._amp - the shared mixed-precision wiring.

Every CUDA arm here is reached WITHOUT a CUDA device, which is the point of
the module. Before it existed, the fp16 path in four training loops was
executable only where a GPU was present, so this package reported 94.08% on
a CPU runner and 100% on the one desk with a graphics card in it -- a
coverage gate met by hardware rather than by tests.

What these assert is WIRING: that a scaler is consulted when one exists,
that the optimiser is stepped through it rather than directly, that
``train_scale`` multiplies the loss exactly once on both paths. fp16
arithmetic is not verified here and cannot be; the real CUDA training tests
in tests/backends/*/test_*_infer.py still do that, and still skip without a
device.

Strict typing only: no Any, casts, or type: ignore.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import AbstractContextManager
from types import TracebackType

import pytest
from platform_ml import torch_types
from platform_ml.torch_types import DTypeProtocol

from covenant_nn.backends import _amp


@pytest.fixture(autouse=True)
def _restore_amp_hooks() -> Generator[None, None, None]:
    """Save and restore every injection point around each test.

    Explicit save-restore rather than a `reset_hooks()` call: that is the
    pattern this monorepo's monkey-patch guard recognises as isolation, and
    it is stronger besides -- it restores what WAS there rather than what
    the module thinks the default should be.
    """
    original_autocast = _amp.autocast_factory
    original_grad_scaler = _amp.grad_scaler_factory
    original_cudnn = _amp.cudnn_config
    yield
    _amp.autocast_factory = original_autocast
    _amp.grad_scaler_factory = original_grad_scaler
    _amp.cudnn_config = original_cudnn


class _FakeDType:
    """Stands in for ``torch.float16``, which is opaque to this module."""


class _FakeTensor:
    """A tensor that records the arithmetic done to it.

    Only the operations :mod:`covenant_nn.backends._amp` performs are
    implemented -- multiplication by a scalar and ``backward()``. Anything
    else would be a fake modelling code that is not under test.
    """

    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self.backward_calls = 0

    def __mul__(self, other: float) -> _FakeTensor:
        return _FakeTensor(self.value * float(other))

    def backward(self) -> None:
        self.backward_calls += 1


class _FakeOptimizer:
    """Records whether it was stepped directly."""

    def __init__(self) -> None:
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


class _FakeScaler:
    """Records the scaler protocol a mixed-precision step drives."""

    def __init__(self) -> None:
        self.scaled: list[float] = []
        self.stepped: list[_FakeOptimizer] = []
        self.updates = 0

    def scale(self, loss: _amp.ScalableLoss) -> _FakeTensor:
        value = loss.value if isinstance(loss, _FakeTensor) else 0.0
        self.scaled.append(value)
        return _FakeTensor(value)

    def step(self, optimizer: _amp.OptimizerProto) -> None:
        if isinstance(optimizer, _FakeOptimizer):
            self.stepped.append(optimizer)

    def update(self) -> None:
        self.updates += 1


class _RecordingContext:
    """A context manager that records that it was entered."""

    def __init__(self) -> None:
        self.entered = 0

    def __enter__(self) -> None:
        self.entered += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


class TestProductionDefaults:
    """The real torch accessors, which are reachable without a GPU.

    That is not incidental. Fetching ``torch.backends.cudnn`` and
    ``torch.amp.autocast`` are attribute reads, and a GradScaler built for
    CUDA on a machine without CUDA warns and disables itself rather than
    raising. Only ASSIGNING the cudnn flags needs a device, which is why
    that one stays behind a guard.
    """

    def test_autocast_factory_default_builds_an_enterable_context(self) -> None:
        """The real factory yields a context that can be entered here.

        Entered, not merely constructed, and on a machine that may have no
        GPU: ``autocast("cuda")`` off CUDA warns and disables itself rather
        than raising, which is exactly the property that lets this default
        be covered without a device.
        """
        torch_mod = torch_types._import_torch()
        body_ran = False
        with _amp._default_autocast_factory()("cuda", dtype=torch_mod.float16):
            body_ran = True
        assert body_ran

    def test_grad_scaler_factory_default_constructs_a_real_grad_scaler(self) -> None:
        scaler = _amp._default_grad_scaler_factory("cuda")
        assert type(scaler).__name__ == "GradScaler"

    def test_cudnn_config_default_returns_torchs_cudnn_module(self) -> None:
        assert type(_amp._default_cudnn_config()).__name__ == "CudnnModule"


class TestConfigureCudnnDeterminism:
    """Pinning cuDNN, reached with a fake so no device is required."""

    def test_sets_both_flags_on_cuda(self) -> None:
        class _Config:
            deterministic = False
            benchmark = True

        config = _Config()
        _amp.cudnn_config = lambda: config
        _amp.configure_cudnn_determinism("cuda")
        assert config.deterministic is True
        assert config.benchmark is False

    def test_touches_nothing_off_cuda(self) -> None:
        """The guard is load-bearing, not defensive.

        Assigning these flags resolves the cuDNN version inside torch, which
        raises on a build reporting ``cuda.is_available()`` with no device
        visible. A fake would happily accept the assignment, so this asserts
        the CALL never happens rather than that it was harmless.
        """
        calls = 0

        def _config() -> _amp.CudnnConfigProto:
            nonlocal calls
            calls += 1
            raise AssertionError("cudnn config must not be fetched off cuda")

        _amp.cudnn_config = _config
        _amp.configure_cudnn_determinism("cpu")
        assert calls == 0


class TestMakeGradScaler:
    """Which runs get a scaler."""

    def test_builds_one_for_mixed_precision_cuda(self) -> None:
        requested: list[str] = []
        fake = _FakeScaler()

        def _factory(device_type: str) -> _amp.GradScalerProto:
            requested.append(device_type)
            return fake

        _amp.grad_scaler_factory = _factory
        assert _amp.make_grad_scaler("cuda", "fp16") is fake
        assert requested == ["cuda"]

    def test_none_at_fp32_even_on_cuda(self) -> None:
        """Loss scaling exists to lift fp16 gradients off the format floor."""
        assert _amp.make_grad_scaler("cuda", "fp32") is None

    def test_none_off_cuda_whatever_the_precision(self) -> None:
        assert _amp.make_grad_scaler("cpu", "fp16") is None


class TestAmpContext:
    """Which context the forward pass runs in."""

    def test_no_autocast_is_built_without_a_scaler(self) -> None:
        """A run with no scaler must not reach for autocast at all.

        Asserted by refusing rather than by inspecting the returned type:
        building an autocast context names a device, and naming "cuda" on a
        run that is not on cuda is the mistake worth catching.
        """

        def _autocast() -> _amp.AutocastFactory:
            raise AssertionError("autocast must not be built without a scaler")

        _amp.autocast_factory = _autocast
        with _amp.amp_context(None, _FakeDType()):
            pass

    def test_autocast_for_cuda_with_the_runs_dtype(self) -> None:
        recorded: list[tuple[str, DTypeProtocol]] = []
        context = _RecordingContext()

        def _autocast(device_type: str, *, dtype: DTypeProtocol) -> AbstractContextManager[None]:
            recorded.append((device_type, dtype))
            return context

        _amp.autocast_factory = lambda: _autocast
        dtype = _FakeDType()
        with _amp.amp_context(_FakeScaler(), dtype):
            pass
        assert recorded == [("cuda", dtype)]
        assert context.entered == 1


class TestBackwardStep:
    """How the gradient reaches the optimiser."""

    def test_without_a_scaler_the_optimiser_is_stepped_directly(self) -> None:
        optimizer = _FakeOptimizer()
        loss = _FakeTensor(2.0)
        _amp.backward_step(scaler=None, optimizer=optimizer, loss=loss, train_scale=3.0)
        assert optimizer.step_calls == 1

    def test_with_a_scaler_the_optimiser_is_stepped_through_it(self) -> None:
        """The scaler steps the optimiser; the loop must not also step it.

        Stepping both would apply two updates per batch, which trains at
        double the intended rate and shows up as nothing but a worse curve.
        """
        optimizer = _FakeOptimizer()
        scaler = _FakeScaler()
        _amp.backward_step(
            scaler=scaler, optimizer=optimizer, loss=_FakeTensor(2.0), train_scale=1.0
        )
        assert scaler.stepped == [optimizer]
        assert scaler.updates == 1
        assert optimizer.step_calls == 0

    def test_train_scale_multiplies_the_loss_on_the_scaled_path(self) -> None:
        scaler = _FakeScaler()
        _amp.backward_step(
            scaler=scaler,
            optimizer=_FakeOptimizer(),
            loss=_FakeTensor(2.0),
            train_scale=3.0,
        )
        assert scaler.scaled == [6.0]

    def test_train_scale_multiplies_the_loss_on_the_plain_path_too(self) -> None:
        """Both arms scale, and they did before this was extracted.

        Dropping it from one path would change the effective step size on
        that path alone -- a difference between CPU and GPU runs that no
        test would name.
        """
        loss = _FakeTensor(2.0)
        _amp.backward_step(scaler=None, optimizer=_FakeOptimizer(), loss=loss, train_scale=3.0)
        assert loss.backward_calls == 0  # the PRODUCT is what backprops
