"""Reduce a tensor to two numbers that change whenever its bits do.

Shared by every cross-card measurement here, because the alternative is two
spellings of a 48-bit fold that are free to drift apart. :mod:`gemm_probe`
folds one GEMM's output; :mod:`forward_trace` folds several thousand
activations. The fold itself must be the same fold, or a digest recorded by
one cannot be read against a digest recorded by the other.

WHY A FOLDED DIGEST RATHER THAN A SUM. The question is bitwise identity, and
a sum can hide a difference: two elements differing by ``+d`` and ``-d``
cancel exactly. A hash of the bytes cannot cancel. Forty-eight bits of it
fold into a float64 without loss -- every integer below 2**53 is exactly
representable -- so bitwise identity becomes float equality, which is what
:func:`~platform_core.run_record.agree_across_runs` already tests.

WHY THE SUM IS THERE ANYWAY. Once the digest says two tensors differ, the
next question is by how much: a last-bit rounding difference and a different
computation both show up as "digest differs", and only a magnitude separates
them.

WHY THE SUM IS CHUNKED RATHER THAN EXACT. :func:`math.fsum` is correctly
rounded, so its result does not depend on the order it sums in -- but it
needs the whole sequence, and these tensors run to hundreds of millions of
elements. So the sum is ``fsum`` over per-chunk ``fsum`` partials at a FIXED
chunk size. That is not the exactly-rounded sum of the tensor; it is a
deterministic function of the tensor's bytes, which is all this has to be.
Two runs that chunk identically produce identical sums, and the chunk size is
a module constant rather than an argument so that they always do.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Final

import torch

#: How many bytes of the digest fold into an observation. Six bytes is 48
#: bits, comfortably inside float64's exact-integer range (2**53); taking
#: seven would start rounding and two different tensors could then record the
#: same number, which is the one failure this observation must not have.
DIGEST_BYTES = 6

#: Elements rendered at a time. Bounds peak memory -- a chunk becomes a
#: Python list and a bytes object, so a 1.5-billion-element trace would
#: otherwise need tens of gigabytes -- and fixes the sum, per the module
#: docstring. Changing it changes every recorded sum, so it is a constant
#: that must not become a parameter.
CHUNK_ELEMENTS = 1 << 20

#: Floating dtypes this can render exactly, and the ``struct`` code for each.
#: float16 and bfloat16 are deliberately absent: neither has a ``struct``
#: code, and rendering them through float32 would make two different tensors
#: capable of producing one digest.
_FLOAT_FORMATS: Final[dict[torch.dtype, str]] = {
    torch.float32: "f",
    torch.float64: "d",
}

#: Integral dtypes this accepts. All are widened to int64 and rendered with
#: one code: the widening is exact for every one of them, so it cannot merge
#: two distinct tensors, and token ids and position ids -- the only integral
#: tensors a probe forward pass crosses a module boundary with -- are the
#: reason this path exists at all.
_INTEGRAL_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)

_INTEGRAL_FORMAT = "q"


def fold_digest(digest: bytes) -> float:
    """Reduce a hash to a float that changes whenever it does.

    Args:
        digest: The hash to fold, at least :data:`DIGEST_BYTES` long.

    Returns:
        Its first :data:`DIGEST_BYTES` bytes, as an exact float.

    Raises:
        ValueError: If the digest is shorter than :data:`DIGEST_BYTES`. A
            short digest would silently fold to a small number and collide
            with every other short one.
    """
    if len(digest) < DIGEST_BYTES:
        raise ValueError(f"digest must be at least {DIGEST_BYTES} bytes, got {len(digest)}")
    return float(int.from_bytes(digest[:DIGEST_BYTES], "big"))


def _describe_float(flat: torch.Tensor, code: str) -> tuple[float, float]:
    """Fold a flat floating-point tensor, chunk by chunk.

    Args:
        flat: A one-dimensional CPU tensor of a dtype in
            :data:`_FLOAT_FORMATS`.
        code: That dtype's ``struct`` code.

    Returns:
        ``(folded digest, chunked sum)``.
    """
    hasher = hashlib.sha256()
    partials: list[float] = []
    for start in range(0, flat.numel(), CHUNK_ELEMENTS):
        values: list[float] = flat[start : start + CHUNK_ELEMENTS].tolist()
        hasher.update(struct.pack(f"<{len(values)}{code}", *values))
        partials.append(math.fsum(values))
    return fold_digest(hasher.digest()), math.fsum(partials)


def _describe_integral(flat: torch.Tensor) -> tuple[float, float]:
    """Fold a flat integral tensor, chunk by chunk.

    Separate from :func:`_describe_float` only so that each can annotate the
    list ``tolist`` actually returns. One function covering both would have
    to annotate ``list[float]`` for a call that yields ``list[int]``, which
    is a small lie in the one module whose whole job is exactness.

    Args:
        flat: A one-dimensional CPU tensor, already widened to int64.

    Returns:
        ``(folded digest, chunked sum)``.
    """
    hasher = hashlib.sha256()
    partials: list[float] = []
    for start in range(0, flat.numel(), CHUNK_ELEMENTS):
        values: list[int] = flat[start : start + CHUNK_ELEMENTS].tolist()
        hasher.update(struct.pack(f"<{len(values)}{_INTEGRAL_FORMAT}", *values))
        partials.append(math.fsum(values))
    return fold_digest(hasher.digest()), math.fsum(partials)


def require_reproduced(
    first: torch.Tensor, second: torch.Tensor, what: str, device: str
) -> torch.Tensor:
    """Return one of two runs of a computation, refusing if they differ.

    Every cross-card measurement here rests on a device reproducing itself.
    If it does not, nothing measured ACROSS cards means anything, so the run
    stops rather than recording a number whose own device cannot repeat it.

    A separate function rather than a branch inside each caller, because it is
    the one part a test can exercise in BOTH directions: the arithmetic under
    study is deterministic within a device, so the failing arm cannot be
    reached by running it -- and an arm no test can reach is an arm nobody has
    checked says what it means.

    Args:
        first: The first run's output.
        second: The second run's.
        what: What was run, for the message, e.g. ``"a GEMM M64xK128xN64"``.
        device: Where it ran, for the message.

    Returns:
        ``first``, once the two are known to agree.

    Raises:
        RuntimeError: If they differ.
    """
    if not torch.equal(first, second):
        raise RuntimeError(
            f"{what} did not reproduce itself on {device}; "
            "nothing measured across cards would mean anything"
        )
    return first


def describe_tensor(tensor: torch.Tensor) -> tuple[float, float]:
    """Reduce a tensor to its identity and its magnitude.

    Args:
        tensor: The tensor to describe, on any device.

    Returns:
        ``(folded digest, chunked float64 sum)``. Both are computed on the
        CPU from the tensor's own bytes, so the only device-dependent step in
        any measurement using this is the arithmetic that produced the
        tensor.

    Raises:
        ValueError: If the tensor holds a NaN, or if its dtype is one this
            cannot render exactly. NaN is refused because ``struct`` does not
            preserve a NaN's payload bits, so two different NaNs would fold
            to one digest -- and a NaN in a probe is a finding to look at
            rather than a number to record. An unsupported dtype is refused
            rather than converted, because every conversion that would make
            it renderable can map two distinct tensors onto one digest.
    """
    flat = tensor.detach().to("cpu").contiguous().flatten()

    code = _FLOAT_FORMATS.get(flat.dtype)
    if code is not None:
        if bool(torch.isnan(flat).any().item()):
            raise ValueError(
                "refusing to digest a tensor holding NaN: struct drops the payload bits, "
                "so two different NaNs would record one digest"
            )
        return _describe_float(flat, code)

    if flat.dtype in _INTEGRAL_DTYPES:
        return _describe_integral(flat.to(torch.int64))

    raise ValueError(
        f"cannot render dtype {flat.dtype} exactly; "
        f"supported: {', '.join(str(d) for d in (*_FLOAT_FORMATS, *_INTEGRAL_DTYPES))}"
    )


__all__ = [
    "CHUNK_ELEMENTS",
    "DIGEST_BYTES",
    "describe_tensor",
    "fold_digest",
    "require_reproduced",
]
