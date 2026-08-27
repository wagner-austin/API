"""Run one attention call under each backend and record what each produced.

:mod:`sdpa_shapes` says which calls exist and why; this issues them.

WHY THE OPERANDS ARE BUILT THE WAY THE MODEL BUILDS THEM. GPT-2's
``_split_heads`` reshapes ``[batch, seq, heads*head_dim]`` and PERMUTES to
``[batch, heads, seq, head_dim]``, leaving a non-contiguous tensor -- and
transformers only forces contiguity when an attention mask is present, which
it is not on this path. Backend eligibility depends on strides, so a probe
that handed the dispatcher neat contiguous tensors could measure a different
selection than the model gets. The permute is reproduced here for that
reason, not for tidiness.

WHY OPERANDS ARE GENERATED ON THE CPU AND MOVED. The CUDA RNG is per-device,
so generating on the device would hand two cards different inputs and produce
a difference that says nothing about how they attend. Same discipline as
:mod:`gemm_probe`.

WHAT "UNAVAILABLE" MEANS HERE. Forcing a backend the build or the card cannot
provide makes torch raise ``RuntimeError: No available kernel``. That refusal
IS the measurement, so it is caught and recorded -- but only that refusal:
any other ``RuntimeError`` propagates, because an out-of-memory or a shape
error recorded as "this backend is unavailable" would be a false fact about
the hardware.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Final

import torch
import torch.backends.cuda as backends_cuda
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing_extensions import TypedDict

from model_trainer.core.services.model.sdpa_shapes import (
    BACKEND_KEYS,
    SDPA_SEED,
    SdpaShape,
)
from model_trainer.core.services.model.tensor_digest import (
    describe_tensor,
    require_reproduced,
)

#: The forced backends, by the key a record carries. Checked against
#: :data:`~sdpa_shapes.BACKEND_KEYS` by the suite, so a key declared there
#: with no backend here -- or the reverse -- fails rather than silently
#: dropping a column from every record.
BACKENDS: Final[dict[str, SDPBackend]] = {
    "math": SDPBackend.MATH,
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}

#: What torch says when the only permitted backend cannot run this call.
#: TWO wordings, because it depends on the device and both are real: cuda
#: raises the first, cpu the second, measured 2026-08-27 on torch
#: 2.6.0+cu124. Matched rather than treating every ``RuntimeError`` as
#: unavailability, so a genuine failure -- an out-of-memory, a dtype
#: mismatch -- stays loud instead of being recorded as a fact about the
#: hardware. A torch version that rewords these will raise here rather than
#: quietly reporting every backend as unavailable, which is the failure
#: direction to prefer.
NO_KERNEL_MESSAGES: Final[tuple[str, ...]] = (
    "No available kernel",
    "No viable backend for scaled_dot_product_attention",
)


def is_no_kernel_refusal(message: str) -> bool:
    """Say whether a failure message is torch declining for want of a kernel.

    Args:
        message: The ``RuntimeError``'s text.

    Returns:
        True when it is the refusal this probe records as unavailability.
    """
    return any(marker in message for marker in NO_KERNEL_MESSAGES)


class SdpaMeasurement(TypedDict):
    """One attention call, measured under every backend.

    Attributes:
        default_digest: Digest of the unforced call's output.
        digests: Digest per backend key, for the backends that ran.
        available: Whether forcing each backend produced a result.
        eligible: What ``can_use_*`` said, per key in
            :data:`~sdpa_shapes.ELIGIBLE_KEYS`.
    """

    default_digest: float
    digests: dict[str, float]
    available: dict[str, bool]
    eligible: dict[str, bool]


def sdpa_operands(shape: SdpaShape, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one call's query, key and value, identically on every device.

    Args:
        shape: The call to build for.
        device: Where the operands must end up.

    Returns:
        ``(query, key, value)``, each ``[1, heads, sequence_len, head_dim]``
        and non-contiguous, matching what ``_split_heads`` hands the
        dispatcher inside the model.
    """
    torch.manual_seed(SDPA_SEED)
    width = shape["heads"] * shape["head_dim"]
    flat = [torch.randn(1, shape["sequence_len"], width, dtype=torch.float32) for _ in range(3)]
    return (
        flat[0]
        .to(device)
        .view(1, shape["sequence_len"], shape["heads"], shape["head_dim"])
        .permute(0, 2, 1, 3),
        flat[1]
        .to(device)
        .view(1, shape["sequence_len"], shape["heads"], shape["head_dim"])
        .permute(0, 2, 1, 3),
        flat[2]
        .to(device)
        .view(1, shape["sequence_len"], shape["heads"], shape["head_dim"])
        .permute(0, 2, 1, 3),
    )


def sdpa_output(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Run the attention call exactly as GPT-2's sdpa path runs it.

    ``is_causal=True`` with no mask and zero dropout is what
    ``GPT2SdpaAttention.forward`` passes in eval with no cache, read at
    source rather than assumed.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.

    Returns:
        The attention output.
    """
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
    )


def forced_sdpa_output(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, backend: SDPBackend
) -> torch.Tensor | None:
    """Run the attention call with one backend forced.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        backend: The only backend to permit.

    Returns:
        The output, or None when torch refused because that backend has no
        kernel for this configuration.

    Raises:
        RuntimeError: Any failure that is NOT the no-kernel refusal. An
            out-of-memory recorded as "this backend is unavailable" would be
            a false fact about the hardware, so only the one message is
            treated as a measurement.
    """
    # Bound to a typed name before use: `sdpa_kernel` returns a context
    # manager whose type parameter is Any, and this package forbids
    # expressions of that type.
    manager: AbstractContextManager[None] = sdpa_kernel([backend])
    try:
        with manager:
            return sdpa_output(query, key, value)
    except RuntimeError as refusal:
        if not is_no_kernel_refusal(str(refusal)):
            raise
        return None


def sdpa_eligibility(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> dict[str, bool]:
    """Ask torch which backends it considers usable for this call.

    Recorded beside the forced runs rather than instead of them: this is an
    opinion about a configuration, and the forced runs are what happened.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.

    Returns:
        One entry per key in :data:`~sdpa_shapes.ELIGIBLE_KEYS`.
    """
    params = backends_cuda.SDPAParams(query, key, value, None, 0.0, True, False)
    return {
        "flash": backends_cuda.can_use_flash_attention(params),
        "efficient": backends_cuda.can_use_efficient_attention(params),
        "cudnn": backends_cuda.can_use_cudnn_attention(params),
    }


def default_digest(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, what: str, device: str
) -> float:
    """Run the unforced call twice and digest it.

    Returns a plain float, not an optional one: the dispatcher always has the
    math fallback available, so this call has no refusal to report. An earlier
    revision shared one function with :func:`forced_digest` and therefore
    carried a "the unforced call produced nothing" arm that no input could
    reach -- an arm nobody has checked says what it means. The types are split
    so it cannot exist.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        what: What is being run, for a self-reproduction failure message.
        device: Where it ran, for that message.

    Returns:
        The folded digest of the output.

    Raises:
        RuntimeError: When the same call on the same device produced two
            different tensors.
    """
    first = sdpa_output(query, key, value)
    second = sdpa_output(query, key, value)
    return describe_tensor(require_reproduced(first.cpu(), second.cpu(), what, device))[0]


def forced_digest(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    backend: SDPBackend,
    what: str,
    device: str,
) -> float | None:
    """Run one forced call twice and digest it.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        backend: The backend to force.
        what: What is being run, for a self-reproduction failure message.
        device: Where it ran, for that message.

    Returns:
        The folded digest, or None when that backend has no kernel here.

    Raises:
        RuntimeError: When the same call on the same device produced two
            different tensors, or propagated from
            :func:`forced_sdpa_output`.
    """
    # One condition rather than two sequential ones. Availability does not
    # vary within a process, so "the first ran and the second refused" is a
    # state no input reaches -- and a separate arm for it would be an arm
    # nobody has checked says what it means.
    first = forced_sdpa_output(query, key, value, backend)
    second = forced_sdpa_output(query, key, value, backend)
    if first is None or second is None:
        return None
    return describe_tensor(require_reproduced(first.cpu(), second.cpu(), what, device))[0]


def probe_sdpa(shape: SdpaShape, device: str) -> SdpaMeasurement:
    """Measure one attention call under the dispatcher and under each backend.

    Args:
        shape: The call to measure.
        device: Device to run it on.

    Returns:
        The measurement.

    Raises:
        RuntimeError: When a call did not reproduce itself on this device.
    """
    query, key, value = sdpa_operands(shape, device)
    label = f"attention h{shape['heads']} s{shape['sequence_len']} d{shape['head_dim']}"

    default = default_digest(query, key, value, f"{label} (default)", device)

    digests: dict[str, float] = {}
    available: dict[str, bool] = {}
    for name in BACKEND_KEYS:
        digest = forced_digest(query, key, value, BACKENDS[name], f"{label} ({name})", device)
        available[name] = digest is not None
        if digest is not None:
            digests[name] = digest

    return SdpaMeasurement(
        default_digest=default,
        digests=digests,
        available=available,
        eligible=sdpa_eligibility(query, key, value),
    )


__all__ = [
    "BACKENDS",
    "NO_KERNEL_MESSAGES",
    "SdpaMeasurement",
    "default_digest",
    "forced_digest",
    "forced_sdpa_output",
    "is_no_kernel_refusal",
    "probe_sdpa",
    "sdpa_eligibility",
    "sdpa_operands",
    "sdpa_output",
]
