"""The mixed-precision wiring every backend's training loop shares.

WHY THIS MODULE EXISTS. Four training loops -- LSTM classifier, LSTM
regressor, MLP classifier, MLP regressor -- carried the same AMP block,
written out four times, and each wrote its FORWARD PASS TWICE: once under
``autocast`` and once under ``nullcontext``, differing in nothing but the
context. Every one of those lines was reachable only with a CUDA device, so
on a CPU runner the package sat at 94.08% and the gate that says "100%" was
being met by the graphics card in one particular desk.

Splitting the two decisions apart fixes both problems at once:

  * :func:`amp_context` decides WHAT CONTEXT the forward runs in. The
    forward itself is then written once, unconditionally, so a CPU test
    covers it.
  * :func:`backward_step` decides HOW THE GRADIENT IS APPLIED -- through
    the scaler or straight through the optimiser. It is four lines either
    way, and both arms take fakes, so both are reachable without a GPU.

WHAT IS DELIBERATELY NOT CLAIMED. Driving these with fakes verifies the
WIRING -- that a scaler is consulted when one exists, that the optimiser is
stepped through it rather than directly, that the loss is multiplied by
``train_scale`` exactly once on both paths. It does not verify fp16
numerics, and nothing here replaces the real CUDA training tests. Those
still run, and they are what says the arithmetic is right.

Protocols are structural, so the per-backend ``_GradScalerProto`` and
``_OptimizerProto`` definitions already in each module satisfy these
parameters without importing from here.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

from platform_ml.torch_types import DTypeProtocol


class ScalableLoss(Protocol):
    """The two things a backward step does to a loss.

    Narrower than ``TensorProtocol`` on purpose. A backward step multiplies
    the loss by the training scale and calls ``backward()`` on the product;
    it never asks for a shape, a device or a dtype. Declaring the twenty-odd
    members it does not use would oblige every test double to implement them
    -- which is how a fake ends up modelling code that is not under test.
    A real tensor satisfies this structurally.
    """

    def __mul__(self, other: float) -> ScalableLoss: ...
    def backward(self) -> None: ...


class GradScalerProto(Protocol):
    """The part of ``torch.amp.GradScaler`` a training loop uses."""

    def scale(self, loss: ScalableLoss) -> ScalableLoss: ...
    def step(self, optimizer: OptimizerProto) -> None: ...
    def update(self) -> None: ...


class OptimizerProto(Protocol):
    """The part of a torch optimiser a training loop uses."""

    def zero_grad(self) -> None: ...
    def step(self) -> None: ...


class AutocastFactory(Protocol):
    """``torch.amp.autocast``, as a callable."""

    def __call__(
        self, device_type: str, *, dtype: DTypeProtocol
    ) -> AbstractContextManager[None]: ...


class CudnnConfigProto(Protocol):
    """The two ``torch.backends.cudnn`` flags this package pins."""

    deterministic: bool
    benchmark: bool


def _default_autocast_factory() -> AutocastFactory:
    """Fetch ``torch.amp.autocast``.

    Returns:
        The real factory. Fetching it is a plain attribute read and needs no
        CUDA device, which is why this default is reachable in a CPU test
        rather than being a line only a GPU could execute.
    """
    amp_mod = __import__("torch.amp", fromlist=["autocast"])
    factory: AutocastFactory = amp_mod.autocast
    return factory


def _default_grad_scaler_factory(device_type: str) -> GradScalerProto:
    """Construct a ``torch.amp.GradScaler`` for ``device_type``.

    Args:
        device_type: The device the scaler is for, e.g. ``"cuda"``.

    Returns:
        The scaler. Constructing one for CUDA on a machine without CUDA
        warns and yields a disabled scaler rather than raising, so this
        default is also reachable without a device.
    """
    amp_mod = __import__("torch.amp", fromlist=["GradScaler"])
    scaler: GradScalerProto = amp_mod.GradScaler(device_type)
    return scaler


def _default_cudnn_config() -> CudnnConfigProto:
    """Fetch ``torch.backends.cudnn``.

    Returns:
        The module object. FETCHING it is safe anywhere; ASSIGNING to its
        flags is not, which is why :func:`configure_cudnn_determinism`
        stays guarded on the device. See that function.
    """
    backends_mod = __import__("torch.backends", fromlist=["cudnn"])
    config: CudnnConfigProto = backends_mod.cudnn
    return config


#: Injection points. Rebound by tests to reach the CUDA arms without a GPU,
#: and left alone in production.
autocast_factory: Callable[[], AutocastFactory] = _default_autocast_factory
grad_scaler_factory: Callable[[str], GradScalerProto] = _default_grad_scaler_factory
cudnn_config: Callable[[], CudnnConfigProto] = _default_cudnn_config


def configure_cudnn_determinism(device: str) -> None:
    """Pin cuDNN to deterministic algorithms when training on CUDA.

    GUARDED ON THE DEVICE, and it has to be. These look like flags that
    would be harmless to set anywhere -- no CPU kernel reads them -- but
    torch resolves the cuDNN version on assignment, and on a build that
    reports ``cuda.is_available()`` while exposing no device (the state
    ``CUDA_VISIBLE_DEVICES=""`` produces) that raises
    ``ValueError: min() arg is an empty sequence`` out of
    ``torch/backends/cudnn/__init__.py``. Measured, not assumed: setting
    these unconditionally turned two passing tests red.

    Args:
        device: The resolved device string.
    """
    if device != "cuda":
        return
    config = cudnn_config()
    config.deterministic = True
    config.benchmark = False


def make_grad_scaler(device: str, precision: str) -> GradScalerProto | None:
    """Build the gradient scaler this run needs, if it needs one.

    Args:
        device: The resolved device string.
        precision: The resolved precision, e.g. ``"fp16"`` or ``"fp32"``.

    Returns:
        A scaler for mixed-precision CUDA training, or None. Loss scaling
        exists to keep fp16 gradients off the floor of the format; at fp32,
        or off CUDA, there is nothing to rescue and a scaler would only add
        a multiply.
    """
    if device != "cuda" or precision == "fp32":
        return None
    return grad_scaler_factory("cuda")


def amp_context(
    scaler: GradScalerProto | None,
    float16: DTypeProtocol,
) -> AbstractContextManager[None]:
    """Name the context the forward pass runs in.

    Args:
        scaler: The run's gradient scaler, or None when it has none.
        float16: The ``torch.float16`` dtype object.

    Returns:
        An autocast context when a scaler is in play, else a null context.
        Returning a context rather than branching at the call site is the
        point: the forward pass is then written ONCE instead of once per
        arm, which is how it used to be duplicated in all four loops.
    """
    if scaler is None:
        return nullcontext()
    return autocast_factory()("cuda", dtype=float16)


def backward_step(
    *,
    scaler: GradScalerProto | None,
    optimizer: OptimizerProto,
    loss: ScalableLoss,
    train_scale: float,
) -> None:
    """Apply the gradient, through the scaler when there is one.

    ``train_scale`` multiplies the loss in BOTH arms, and did before this
    was extracted. It is the learning-rate scale expressed on the loss, so
    dropping it from either arm would silently change the effective step
    size on that path only.

    Args:
        scaler: The run's gradient scaler, or None.
        optimizer: The optimiser to step.
        loss: The batch loss, unscaled.
        train_scale: Multiplier applied to the loss before backward.
    """
    scaled_loss = loss * float(train_scale)
    if scaler is None:
        scaled_loss.backward()
        optimizer.step()
        return
    scaler.scale(scaled_loss).backward()
    scaler.step(optimizer)
    scaler.update()


__all__ = [
    "AutocastFactory",
    "CudnnConfigProto",
    "GradScalerProto",
    "OptimizerProto",
    "ScalableLoss",
    "amp_context",
    "autocast_factory",
    "backward_step",
    "configure_cudnn_determinism",
    "cudnn_config",
    "grad_scaler_factory",
    "make_grad_scaler",
]
