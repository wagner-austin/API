"""Several ways to compute one GEMM, differing only in who chooses the order.

WHY THIS EXISTS. Every determinism control shipped so far -- ``CUBLAS_WORKSPACE_CONFIG``,
``use_deterministic_algorithms``, TF32 off, ``CUBLASLT_WORKSPACE_SIZE=0``, the
SDPA math pin -- constrains WHICH vendor kernel cuBLAS picks. None of them
removes the vendor's remaining freedom, and that freedom is per-architecture
by design: tile sizes and split counts come from heuristics keyed on compute
capability, SM count and L2 size. A different split is a different summation
order, and float addition is not associative, so the same operands round
differently. That is the whole residual the v24 four-card trace is left with,
and no further flag can reach it -- the trace that found it already had every
flag on.

So the next control is not a flag. It is owning the reduction order, which
means not calling the vendor's GEMM -- or calling it on pieces small enough
that its remaining freedom does not matter. These are the arms:

* :data:`CUBLAS_ARM` is ``addmm``, the baseline the ladder and the trace have
  always run. It is here so the other two are read against it in the same
  record rather than against a number recovered from a different run.

* :data:`FP64_ARM` widens to float64, multiplies, and narrows back. It is
  what a practitioner reaches for first and it is NOT a determinism argument:
  cuBLAS still picks the kernel and still picks the order, so two cards can
  still disagree -- just ~2**-29 further down, where narrowing to float32
  usually but not always rounds the disagreement away. "Usually" is the
  reason to measure it rather than recommend it. Its cost also varies by an
  order of magnitude across the cards under test, since fp64 throughput is
  1:2 on the V100, A100 and A30 and 1:64 on the L40S.

* :data:`RANK1_ARM` is the one with an argument behind it. It accumulates K
  rank-one outer products in ascending k, so the sum is a SEQUENCE OF
  ELEMENTWISE OPERATIONS and there is no reduction for the hardware to
  reorder: every output element is produced by one thread doing one multiply
  and one add, K times, in an order the program fixes rather than the
  heuristic. Vectorisation and occupancy still differ per card and cannot
  change that -- they change which thread runs when, not what it computes.
  Volta and Ampere both implement IEEE-754 binary32 arithmetic, so the same
  operations in the same order give the same bits.

  It is quadratically wasteful of bandwidth -- K passes over an MxN
  accumulator instead of one -- so it is an existence proof and a measurement
  instrument, NOT a proposal for the forward pass. What it establishes, if it
  holds, is that the ceiling is reachable, which is the thing no amount of
  configuration could tell us.

* :data:`OWNED_ARM` extends the rank-one argument through the BACKWARD pass:
  same forward, and the two gradient products plus the bias gradient's
  row-sum are program-ordered too, via the autograd Function in
  :mod:`owned_backward`. It exists because the v29 train-step measurement
  showed the forward proof buys zero agreeing gradients -- see the constant's
  own docstring for what it owns and what it deliberately leaves.

* The ``block<N>`` arms are the middle ground, and the one a real
  implementation would resemble. Each cuts K into fixed pieces of N and adds
  the pieces in ascending k, so the PROGRAM fixes how the reduction is cut
  while the VENDOR still chooses how to reduce within a piece. That is not a
  proof, and it is not meant to be: it measures HOW MUCH of the order has to
  be owned before the cards agree, which is the number someone choosing an
  implementation actually needs. See :data:`BLOCK_SIZES` for why the widths
  are a prediction rather than a sweep.

WHY NOT TRITON, WHICH IS THE OBVIOUS ANSWER AND IS ALREADY IN THE IMAGE. A
kernel with a fixed ``BLOCK_K`` and no split-K would keep tensor cores and
most of the speed, and it remains the right shape for production. Two things
stand in the way, and only one of them is about Triton.

The first is that a Triton kernel would not settle the question by itself.
``tl.dot`` lowers to the MMA instruction of the target architecture, whose
shape differs between sm_70 and sm_80, so such a kernel fixes the block
SCHEDULE and inherits the intra-tile order from the card -- exactly as the
``block<N>`` arms fix the chunking and inherit the intra-chunk order from
cuBLAS. The block arms therefore test the same property one level up, in code
that runs everywhere, and a negative result from them would predict a
negative result from Triton without writing it.

The second is this package's own bar, and it is worth stating plainly rather
than working around. Coverage is ``fail_under = 100`` with ``omit = []``, and
Triton does not install on Windows at all -- so a ``@triton.jit`` body cannot
be executed, and cannot be covered, by the suite that gates every commit
here. Shipping one would mean adding a coverage exemption, which is the
thing this workspace bans by name. It needs a GPU-capable test runner, not a
cleverer import.
"""

from __future__ import annotations

from typing import Final, Protocol

import torch

#: ``addmm``: the vendor picks the kernel and the order.
CUBLAS_ARM = "cublas"

#: float64 throughout, narrowed at the end. The vendor still picks the order.
FP64_ARM = "fp64"

#: K rank-one updates in ascending k. The program fixes the order.
RANK1_ARM = "rank1"

#: The rank-one arm with its BACKWARD owned too. The 2026-08-31 train-step
#: measurement showed why this exists: under every arm above, zero gradients
#: agree across four cards at any rung, because a matmul's backward is two
#: MORE matmuls -- the input gradient reduces over the output width, the
#: weight gradient over the batch -- and autograd hands both to the vendor
#: regardless of how the forward was computed. This arm routes the forward
#: AND both gradient products AND the bias gradient's row-sum through the
#: same ascending-order accumulation, so a training step's every reduction
#: through a projection is program-ordered. What it deliberately does NOT
#: touch: the backward of everything that is not a projection -- layer norm,
#: softmax, GELU, cross-entropy -- which stays autograd's, so a residual
#: under this arm NAMES those, the way the whole-model trace named the
#: projections.
OWNED_ARM = "owned"

#: The K-block widths the ``block<N>`` arms chunk a reduction into.
#:
#: WHY THESE THREE, AND WHY THEY ARE A PREDICTION RATHER THAN A SWEEP. The
#: 2026-08-30 boundary bracket measured where the V100 leaves the sm_80+ cards
#: on an unchunked matmul: at M >= 3072 AND K >= 1152, with K=1024 the
#: largest reduction length that still agreed. A ``block<N>`` arm holds M
#: fixed and cuts K into pieces of N, so if the disagreement really is a
#: property of the reduction LENGTH the vendor is handed, then chunking to
#: 1024 should agree and chunking to 1280 should not -- on the same shapes,
#: in the same run. 256 is the control far below the line.
#:
#: That is falsifiable in the direction that matters. If ``block1280`` agrees
#: the threshold is not about the length handed to one call, and the story
#: the bracket told needs revising rather than extending.
BLOCK_SIZES: Final[tuple[int, ...]] = (256, 1024, 1280)

#: The chunked arms, named for their width so the label carries it.
BLOCK_ARMS: Final[tuple[str, ...]] = tuple(f"block{size}" for size in BLOCK_SIZES)

#: Every arm, in the order a report should read them: baseline, the cheap
#: attempt, the one with a forward proof, the one that extends it through
#: the backward, then the chunked middle ground. Declared as a tuple rather
#: than inferred from the dispatch below, so a report can name the arms
#: without importing torch -- the reason :mod:`gemm_shapes` is separate from
#: :mod:`gemm_probe`.
KERNEL_ARMS: Final[tuple[str, ...]] = (CUBLAS_ARM, FP64_ARM, RANK1_ARM, OWNED_ARM, *BLOCK_ARMS)


def require_block_size(arm: str) -> int:
    """Return the K-block width a ``block<N>`` arm names.

    Args:
        arm: One of :data:`BLOCK_ARMS`.

    Returns:
        The width in elements.

    Raises:
        ValueError: When ``arm`` is not a block arm. Parsed from the declared
            table rather than from the digits in the string: a name is a block
            arm because it is IN :data:`BLOCK_ARMS`, and reading an integer
            out of an arbitrary string would accept ``block7`` -- a width
            nothing declares and no record could be compared against.
    """
    if arm not in BLOCK_ARMS:
        raise ValueError(f"{arm!r} is not one of {', '.join(BLOCK_ARMS)}")
    return BLOCK_SIZES[BLOCK_ARMS.index(arm)]


def cublas_addmm(bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute ``bias + x @ w`` on the cuBLASLt path.

    ``addmm`` rather than ``mm``: the fused bias epilogue is what routes this
    to cuBLASLt, measured under ``CUBLASLT_LOG_LEVEL=4``. See
    :mod:`gemm_shapes` for the operand orientation and why it matters.

    Args:
        bias: ``[M]``.
        x: ``[N, K]``.
        w: ``[K, M]``.

    Returns:
        ``[N, M]``.
    """
    return torch.addmm(bias, x, w)


def fp64_addmm(bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute the same product in float64 and narrow the result.

    Narrowed rather than returned wide, because the question is whether two
    cards produce the same FLOAT32 tensor. Returning float64 would compare a
    quantity the forward pass never holds and would make the arm incomparable
    with the other two.

    Args:
        bias: ``[M]``, float32.
        x: ``[N, K]``, float32.
        w: ``[K, M]``, float32.

    Returns:
        ``[N, M]``, float32.
    """
    wide = torch.addmm(bias.double(), x.double(), w.double())
    return wide.float()


def rank1_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute ``x @ w`` as K rank-one updates in ascending k.

    The product without the bias, because two callers need exactly this and
    one of them has no bias to add. ``lm_head`` is
    ``nn.Linear(n_embd, vocab, bias=False)``, and handing it a zero bias would
    not be free: ``0.0 + -0.0`` is ``+0.0``, so a zeroed bias can flip the
    sign bit of a negative zero and change bytes a digest reads. Consistently,
    on every card -- but a probe should not introduce a difference it then has
    to argue is harmless.

    ``addr_`` rather than ``add_(torch.outer(...))``: one elementwise kernel
    over the accumulator instead of an allocation and two, which at K up to
    8192 is the difference between seconds and tens of seconds. It computes
    the same thing.

    ``w`` may be a transposed view -- ``Linear`` stores ``[out, in]`` and this
    wants ``[in, out]`` -- so ``w[k, :]`` is a strided read. Measured to give
    the same result as the contiguous form to the last bit.

    Args:
        x: ``[N, K]``.
        w: ``[K, M]``.

    Returns:
        ``[N, M]``.
    """
    accumulator = torch.zeros(x.shape[0], w.shape[1], dtype=x.dtype, device=x.device)
    for k in range(w.shape[0]):
        accumulator.addr_(x[:, k], w[k, :])
    return accumulator


def rank1_addmm(bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute the same product as K rank-one updates, then add the bias.

    The bias is added at the END, not folded into the accumulator first. Both
    are fixed orders and they give different bits: a bias added first
    participates in the rounding of all K subsequent adds. Last matches how
    ``addmm`` is written -- ``bias + (x @ w)`` -- so the arms differ in the
    reduction under study and not also in where the bias went.

    Args:
        bias: ``[M]``.
        x: ``[N, K]``.
        w: ``[K, M]``.

    Returns:
        ``[N, M]``.
    """
    return bias + rank1_matmul(x, w)


def blocked_matmul(x: torch.Tensor, w: torch.Tensor, block: int) -> torch.Tensor:
    """Compute ``x @ w`` as ceil(K/block) cuBLASLt calls summed in ascending k.

    The middle ground between :func:`cublas_addmm` and :func:`rank1_matmul`,
    and the one a real implementation would resemble. The PROGRAM fixes how
    the reduction is cut and the order the pieces are added; the VENDOR still
    chooses how to reduce within a piece. So it is not a proof the way the
    rank-one arm is -- it measures how much of the order has to be owned
    before the cards agree, which is the number someone choosing an
    implementation actually needs.

    IT ACCUMULATES WITH ``addmm``, NOT ``matmul``, AND THAT IS THE WHOLE
    CORRECTNESS OF THE ARM. The first version chunked with ``torch.matmul``
    and every card disagreed worse than the unchunked baseline. Measured
    2026-08-30 on an RTX 3090 Ti, one shape, one set of operands, with and
    without ``CUBLASLT_WORKSPACE_SIZE=0``::

        addmm    338779f0ee4467ae -> fa28a1f6d2ae3b64   (control reaches it)
        matmul   c6856afb742b9f0a -> c6856afb742b9f0a   (control does not)

    ``torch.matmul`` takes the legacy ``cublasSgemm`` entry point, which
    ``CUBLASLT_WORKSPACE_SIZE`` structurally cannot touch -- the same fact
    that makes GPT-2's bias-free ``lm_head`` the one matmul the split-K
    control never reached. So a matmul-chunked arm ran with the control that
    buys sm_80+ agreement silently switched off, and measured the entry point
    rather than the chunking. Accumulating into the running sum keeps every
    piece on the path the control governs, which is the only way the
    comparison against the unchunked arm isolates one variable.

    A consequence worth naming: under a block arm ``lm_head`` moves from the
    legacy path onto cuBLASLt, because every piece here is an ``addmm``. That
    is a real difference from the untreated arm and is deliberate -- the arm
    exists to chunk the path the control governs -- but it means an
    ``lm_head`` row under a block arm is not comparable with one under
    ``cublas``.

    The tail is whatever is left when K is not a multiple of ``block``, added
    in its own turn rather than padded: padding with zeros would change the
    number of terms and put a rounding difference into the arm under study.

    Args:
        x: ``[N, K]``.
        w: ``[K, M]``.
        block: K-block width, from :func:`require_block_size`.

    Returns:
        ``[N, M]``.
    """
    accumulator = torch.zeros(x.shape[0], w.shape[1], dtype=x.dtype, device=x.device)
    for start in range(0, w.shape[0], block):
        stop = min(start + block, w.shape[0])
        accumulator = torch.addmm(accumulator, x[:, start:stop], w[start:stop, :])
    return accumulator


def blocked_addmm(bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor, block: int) -> torch.Tensor:
    """Compute the chunked product, then add the bias.

    Bias last, matching :func:`rank1_addmm`, so the arms differ in the
    reduction under study and not also in where the bias went.

    Args:
        bias: ``[M]``.
        x: ``[N, K]``.
        w: ``[K, M]``.
        block: K-block width.

    Returns:
        ``[N, M]``.
    """
    return bias + blocked_matmul(x, w, block)


def accumulate_rows(grad_out: torch.Tensor) -> torch.Tensor:
    """Sum the rows of a matrix in ascending order.

    The owned form of ``grad_out.sum(dim=0)``, which is what autograd
    computes for a broadcast bias's gradient -- a reduction over the batch
    dimension whose order the vendor's kernel chooses. Row-at-a-time
    ``add_`` makes each output element a sequence of elementwise adds in an
    order the program fixes, the same argument :func:`rank1_matmul` makes
    for the product.

    Args:
        grad_out: ``[N, M]``.

    Returns:
        ``[M]``.
    """
    total = torch.zeros(grad_out.shape[1], dtype=grad_out.dtype, device=grad_out.device)
    for row in range(grad_out.shape[0]):
        total.add_(grad_out[row])
    return total


class _ApplyMatmulProto(Protocol):
    """``OwnedMatmul.apply``, with the type its stub loses."""

    def __call__(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor: ...


class _ApplyAddmmProto(Protocol):
    """``OwnedAddmm.apply``, with the type its stub loses."""

    def __call__(self, bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor: ...


class _MatmulFunctionProto(Protocol):
    """The class object carrying the owned matmul's ``apply``."""

    apply: _ApplyMatmulProto


class _AddmmFunctionProto(Protocol):
    """The class object carrying the owned addmm's ``apply``."""

    apply: _ApplyAddmmProto


def _owned_matmul_apply() -> _ApplyMatmulProto:
    """Reach ``OwnedMatmul.apply`` without naming the class in an expression.

    The Functions live in :mod:`owned_backward` and are reached through
    ``__import__`` -- the ``module.Conv1D`` pattern from
    :mod:`kernel_arm_modules` -- because ``torch.autograd.Function`` carries
    ``Any`` in its stub, and naming a subclass in an expression trips this
    package's contains-Any check while a class DEFINITION does not. The
    import is inside the function, so importing this module still costs no
    torch-graph machinery, and the lookup is a ``sys.modules`` hit after the
    first call.

    Returns:
        The apply callable, typed.
    """
    module = __import__(
        "model_trainer.core.services.model.owned_backward", fromlist=["OwnedMatmul"]
    )
    function: _MatmulFunctionProto = module.OwnedMatmul
    return function.apply


def _owned_addmm_apply() -> _ApplyAddmmProto:
    """Reach ``OwnedAddmm.apply``, typed. See :func:`_owned_matmul_apply`.

    Returns:
        The apply callable, typed.
    """
    module = __import__("model_trainer.core.services.model.owned_backward", fromlist=["OwnedAddmm"])
    function: _AddmmFunctionProto = module.OwnedAddmm
    return function.apply


def owned_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute ``x @ w`` with forward and backward orders both owned.

    Bit-identical to :func:`rank1_matmul` in the FORWARD -- it is the same
    accumulation -- so every forward-only record under this arm must equal
    the rank-one arm's, which is a free cross-check the tests assert. What
    differs is what autograd does afterwards.

    Args:
        x: ``[N, K]``.
        w: ``[K, M]``.

    Returns:
        ``[N, M]``.
    """
    return _owned_matmul_apply()(x, w)


def owned_addmm(bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute ``bias + x @ w`` with every reduction owned.

    Args:
        bias: ``[M]``.
        x: ``[N, K]``.
        w: ``[K, M]``.

    Returns:
        ``[N, M]``.
    """
    return _owned_addmm_apply()(bias, x, w)


def matmul_by_arm(arm: str, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute ``x @ w`` by the named arm, with no bias anywhere.

    The bias-free twin of :func:`gemm_by_arm`, for the one caller that has no
    bias: ``lm_head`` is ``nn.Linear(n_embd, vocab, bias=False)``. Passing it
    a zeroed bias instead would not be free -- ``0.0 + -0.0`` is ``+0.0``, so
    a zero bias can flip the sign bit of a negative zero and change bytes a
    digest reads -- and it would also route the cuBLAS arm to ``addmm``,
    whose fused epilogue takes a DIFFERENT library entry point than the
    ``mm`` an untreated ``lm_head`` actually uses. The whole point of the
    cuBLAS arm is to be the untreated path.

    Args:
        arm: One of :data:`KERNEL_ARMS`.
        x: ``[N, K]``.
        w: ``[K, M]``.

    Returns:
        ``[N, M]``, float32.

    Raises:
        ValueError: Propagated from :func:`require_kernel_arm`.
    """
    named = require_kernel_arm(arm)
    if named == CUBLAS_ARM:
        return torch.matmul(x, w)
    if named == FP64_ARM:
        return torch.matmul(x.double(), w.double()).float()
    if named == OWNED_ARM:
        return owned_matmul(x, w)
    if named in BLOCK_ARMS:
        return blocked_matmul(x, w, require_block_size(named))
    return rank1_matmul(x, w)


def require_kernel_arm(raw: str) -> str:
    """Return the arm named, refusing anything else.

    Refused rather than defaulted for the reason
    :func:`~model_trainer.core.services.model.control_arms.require_control_arm`
    refuses: a record whose arm was guessed is a record naming a condition it
    may not have run under, and here the arm decides the arithmetic itself.

    Args:
        raw: The flag's value.

    Returns:
        ``raw``, once it is known to be one of :data:`KERNEL_ARMS`.

    Raises:
        ValueError: When it is not.
    """
    if raw not in KERNEL_ARMS:
        raise ValueError(f"kernel must be one of {', '.join(KERNEL_ARMS)}; got {raw!r}")
    return raw


def gemm_by_arm(arm: str, bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Dispatch one GEMM to the named arm.

    Args:
        arm: One of :data:`KERNEL_ARMS`.
        bias: ``[M]``.
        x: ``[N, K]``.
        w: ``[K, M]``.

    Returns:
        ``[N, M]``, float32.

    Raises:
        ValueError: Propagated from :func:`require_kernel_arm`.
    """
    named = require_kernel_arm(arm)
    if named == CUBLAS_ARM:
        return cublas_addmm(bias, x, w)
    if named == FP64_ARM:
        return fp64_addmm(bias, x, w)
    if named == OWNED_ARM:
        return owned_addmm(bias, x, w)
    if named in BLOCK_ARMS:
        return blocked_addmm(bias, x, w, require_block_size(named))
    return rank1_addmm(bias, x, w)


__all__ = [
    "BLOCK_ARMS",
    "BLOCK_SIZES",
    "CUBLAS_ARM",
    "FP64_ARM",
    "KERNEL_ARMS",
    "OWNED_ARM",
    "RANK1_ARM",
    "accumulate_rows",
    "blocked_addmm",
    "blocked_matmul",
    "cublas_addmm",
    "fp64_addmm",
    "gemm_by_arm",
    "matmul_by_arm",
    "owned_addmm",
    "owned_matmul",
    "rank1_addmm",
    "rank1_matmul",
    "require_block_size",
    "require_kernel_arm",
]
