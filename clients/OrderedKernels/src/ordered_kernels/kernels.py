"""The fixed-order CUDA kernels, and the NVRTC plumbing that launches them.

WHY CUDA-C STRINGS AND NOT TRITON. Two reasons, both structural. Triton's
``tl.dot`` lowers to each architecture's matrix-unit instructions, whose
internal accumulation order differs per chip -- the exact freedom this
package exists to remove -- so the fast path of Triton is the wrong path
here regardless of tooling. And a ``@triton.jit`` body is Python source that
never executes as Python, so no coverage tool can hold it to this
workspace's 100% bar. A CUDA source STRING is data; every Python line in
this package runs for real on a GPU and is covered for real.

THE ORDER CONTRACT, WHICH IS THE WHOLE PACKAGE. Every output element is
produced by ONE thread as a strictly ascending-k sequence of
``acc = acc + x*w`` -- separate multiply, separate add, two roundings --
because that is bit-for-bit the arithmetic of
:func:`~model_trainer.core.services.model.deterministic_gemm.rank1_matmul`,
whose records span seven GPUs. Compiled with ``--fmad=false`` so the
compiler cannot contract the pair into one fused rounding, and with no
fast-math anywhere so it cannot reassociate. Parallelism lives ACROSS
output elements; within one element nothing is parallel, which is why SM
count, tile scheduling and occupancy -- everything that differs between
cards -- cannot touch the bits. The tail of a K tile is bounds-guarded
rather than zero-padded: a padded ``acc + 0.0`` is not a no-op in IEEE
arithmetic (``-0.0 + 0.0`` is ``+0.0``), and a kernel that adds terms the
oracle does not add is a different computation.

WHAT MAKES IT FAST WHERE RANK-ONE IS NOT. The rank-one arm re-reads an
entire M x N accumulator K times from global memory. This kernel is the
classic shared-memory tiled GEMM: each K-tile of both operands is staged
once into shared memory and every product term is read from there, so
global traffic drops by the tile width -- while each element's private
``acc`` chain stays strictly sequential. Owning the order never required
giving up tiling; it only forbids split-K and per-element multi-accumulator
tricks.
"""

from __future__ import annotations

from types import TracebackType
from typing import Final, Protocol

import torch

#: One thread block computes a TILE x TILE output tile: 16x16 threads, each
#: owning a MICRO x MICRO register patch, over K slices of K_SLICE staged in
#: shared memory. Register tiling is the speed lever the naive version left
#: on the table -- each staged value feeds MICRO outputs instead of one, so
#: shared traffic per multiply drops fourfold -- and it is bit-neutral BY
#: CONSTRUCTION: every output element still belongs to exactly one thread,
#: whose k-chain is the same strictly ascending sequence of separate
#: multiply-and-add the naive kernel ran. Measured before the change: the
#: naive tile priced at 1.8x the vendor at small shapes and 12-128x at
#: large ones; the register tile exists for the large end.
TILE: Final = 64

#: Threads per block edge; TILE / THREADS outputs per thread per edge.
THREADS: Final = 16

#: Outputs per thread per edge.
MICRO: Final = 4

#: K-slice width staged per iteration.
K_SLICE: Final = 16

#: Threads per block for the two one-thread-per-output-line kernels:
#: ``ordered_rowsum`` (one thread per column) and ``ordered_lastdim_sum``
#: (one thread per row). A launch width, never an arithmetic parameter.
ROWSUM_BLOCK: Final = 128

#: The CUDA source. A string on purpose -- see the module docstring.
CUDA_SOURCE: Final = r"""
extern "C" __global__ void ordered_gemm_nn(
    const float* __restrict__ x,     // N x K, row-major, contiguous
    const float* __restrict__ w,     // K x M, row-major, contiguous
    const float* __restrict__ bias,  // M, or unused when use_bias == 0
    float* __restrict__ out,         // N x M, row-major, contiguous
    const int n_rows, const int k_dim, const int m_cols, const int use_bias)
{
    // 64x64 output tile per block; 16x16 threads; a 4x4 register patch per
    // thread. Each output element belongs to ONE thread and its k-chain is
    // strictly ascending with separate multiply and add -- the register
    // tiling changes how many elements a thread owns, never the order any
    // one element is summed in.
    //
    // blockIdx.z selects the batch slice: each z computes one independent
    // n x k @ k x m product from densely packed operands. A 2-D launch uses
    // grid.z == 1, where every offset below is zero and the arithmetic is
    // untouched -- the seven-GPU record corpus still pins this kernel.
    const float* xb = x + (size_t)blockIdx.z * n_rows * k_dim;
    const float* wb = w + (size_t)blockIdx.z * k_dim * m_cols;
    float* outb = out + (size_t)blockIdx.z * n_rows * m_cols;
    __shared__ float xs[64][16];
    __shared__ float ws[16][64];
    const int row0 = blockIdx.y * 64;
    const int col0 = blockIdx.x * 64;
    const int tid = threadIdx.y * 16 + threadIdx.x;
    float acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < k_dim; k0 += 16) {
        // 256 threads cooperatively stage 64x16 of x and 16x64 of w.
        // Out-of-range positions stage 0.0f, which is harmless here: the
        // K tail is bounds-guarded below (a padded 0.0f term would flip a
        // -0.0f accumulator), and out-of-range rows/cols are masked at the
        // store, so their padded products are never written anywhere.
        for (int s = tid; s < 64 * 16; s += 256) {
            const int r = s / 16;
            const int k = s % 16;
            xs[r][k] = (row0 + r < n_rows && k0 + k < k_dim)
                ? xb[(row0 + r) * k_dim + (k0 + k)] : 0.0f;
            ws[k][r] = (k0 + k < k_dim && col0 + r < m_cols)
                ? wb[(k0 + k) * m_cols + (col0 + r)] : 0.0f;
        }
        __syncthreads();
        const int k_stop = (k_dim - k0 < 16) ? (k_dim - k0) : 16;
        for (int kk = 0; kk < k_stop; ++kk) {
            float xr[4];
            float wc[4];
            for (int i = 0; i < 4; ++i)
                xr[i] = xs[threadIdx.y * 4 + i][kk];
            for (int j = 0; j < 4; ++j)
                wc[j] = ws[kk][threadIdx.x * 4 + j];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    // Separate multiply and add, two roundings, matching
                    // addr_'s arithmetic; --fmad=false forbids contraction.
                    acc[i][j] = acc[i][j] + xr[i] * wc[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i) {
        const int row = row0 + threadIdx.y * 4 + i;
        if (row < n_rows) {
            for (int j = 0; j < 4; ++j) {
                const int col = col0 + threadIdx.x * 4 + j;
                if (col < m_cols) {
                    outb[row * m_cols + col] = use_bias ? (bias[col] + acc[i][j]) : acc[i][j];
                }
            }
        }
    }
}

extern "C" __global__ void ordered_rowsum(
    const float* __restrict__ grad,  // N x M, row-major, contiguous
    float* __restrict__ out,         // M
    const int n_rows, const int m_cols)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < m_cols) {
        float acc = 0.0f;
        for (int r = 0; r < n_rows; ++r) {
            acc = acc + grad[r * m_cols + col];
        }
        out[col] = acc;
    }
}

extern "C" __global__ void ordered_lastdim_sum(
    const float* __restrict__ rows,  // R x C, row-major, contiguous
    float* __restrict__ out,         // R
    const int n_rows, const int m_cols)
{
    // One thread per ROW, ascending column order -- the softmax
    // denominator's reduction, owned the way ordered_rowsum owns the bias
    // gradient's.
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n_rows) {
        float acc = 0.0f;
        for (int c = 0; c < m_cols; ++c) {
            acc = acc + rows[row * m_cols + c];
        }
        out[row] = acc;
    }
}
"""

#: NVRTC options. ``--fmad=false`` is the bit-compatibility linchpin: with
#: contraction allowed, ``a + x*w`` may compile to one fused rounding and
#: the kernel would compute DIFFERENT bits than the seven-GPU record corpus
#: it exists to match.
NVRTC_OPTIONS: Final[tuple[str, ...]] = ("--fmad=false",)


class CupyArrayProto(Protocol):
    """A cupy ndarray view over a torch tensor, by one honest attribute.

    The kernels receive these as launch arguments and never touch them from
    Python; naming one real cupy-array member keeps the Protocol structural
    without smuggling ``object`` into an annotation.
    """

    @property
    def ndim(self) -> int: ...


class RawKernelProto(Protocol):
    """One compiled kernel, callable with a launch configuration."""

    def __call__(
        self,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: tuple[CupyArrayProto | int, ...],
    ) -> None: ...


class RawModuleProto(Protocol):
    """A compiled NVRTC module, from which kernels are fetched by name."""

    def get_function(self, name: str) -> RawKernelProto: ...


class ExternalStreamProto(Protocol):
    """A cupy stream wrapping an existing CUDA stream pointer."""

    def __enter__(self) -> ExternalStreamProto: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class _CupySurfaceProto(Protocol):
    """The four members of cupy this module touches.

    A Protocol over a dynamic import (the ``module.Conv1D`` pattern), because
    cupy's annotations do not survive this workspace's strict settings and
    the surface actually used is tiny enough to state.
    """

    def raw_module(self, code: str, options: tuple[str, ...]) -> RawModuleProto: ...
    def external_stream(self, ptr: int) -> ExternalStreamProto: ...
    def from_dlpack(self, tensor: torch.Tensor) -> CupyArrayProto: ...


class _RawModuleCtorProto(Protocol):
    def __call__(self, *, code: str, options: tuple[str, ...]) -> RawModuleProto: ...


class _ExternalStreamCtorProto(Protocol):
    def __call__(self, ptr: int) -> ExternalStreamProto: ...


class _FromDlpackProto(Protocol):
    def __call__(self, tensor: torch.Tensor) -> CupyArrayProto: ...


class _Cupy:
    """The cupy surface, resolved once and typed at the boundary."""

    def __init__(self) -> None:
        module = __import__("cupy", fromlist=["RawModule"])
        raw_module_ctor: _RawModuleCtorProto = module.RawModule
        self._raw_module_ctor = raw_module_ctor
        stream_module = __import__("cupy.cuda", fromlist=["ExternalStream"])
        external_stream_ctor: _ExternalStreamCtorProto = stream_module.ExternalStream
        self._external_stream_ctor = external_stream_ctor
        from_dlpack: _FromDlpackProto = module.from_dlpack
        self._from_dlpack = from_dlpack

    def raw_module(self, code: str, options: tuple[str, ...]) -> RawModuleProto:
        return self._raw_module_ctor(code=code, options=options)

    def external_stream(self, ptr: int) -> ExternalStreamProto:
        return self._external_stream_ctor(ptr)

    def from_dlpack(self, tensor: torch.Tensor) -> CupyArrayProto:
        return self._from_dlpack(tensor)


_cupy: _CupySurfaceProto | None = None
_module: RawModuleProto | None = None


def _surface() -> _CupySurfaceProto:
    """Resolve cupy once per process."""
    global _cupy
    if _cupy is None:
        _cupy = _Cupy()
    return _cupy


def _kernels() -> RawModuleProto:
    """Compile the CUDA source once per process and cache the module."""
    global _module
    if _module is None:
        _module = _surface().raw_module(CUDA_SOURCE, NVRTC_OPTIONS)
    return _module


def _require_f32_cuda(tensor: torch.Tensor, name: str, dims: int) -> torch.Tensor:
    """Return a contiguous view of a float32 CUDA tensor of ``dims`` rank.

    Contiguity is imposed HERE, with torch's own deterministic copy, so the
    kernels can assume plain row-major layout. The backward pass hands in
    transposed views and pays one copy each -- bandwidth, not bits.

    Args:
        tensor: The operand.
        name: Its role, for the message.
        dims: The rank the caller's kernel expects.

    Returns:
        ``tensor.contiguous()``.

    Raises:
        ValueError: For a non-CUDA, non-float32 or wrong-rank operand.
            Refused rather than converted: a silently-widened or
            silently-moved operand would be a different computation than the
            record claims.
    """
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor; ordered kernels have no CPU path")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {tensor.dtype}")
    if tensor.dim() != dims:
        raise ValueError(f"{name} must be {dims}-D, got {tensor.dim()}-D")
    # Detached because torch refuses to export a grad-requiring tensor over
    # dlpack, and these kernels only READ memory -- autograd's bookkeeping
    # happens in the Functions above them, never here. Same storage, no copy.
    return tensor.detach().contiguous()


def _launch(
    kernel_name: str,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    args: tuple[CupyArrayProto | int, ...],
) -> None:
    """Launch one kernel on torch's CURRENT stream.

    Through ``ExternalStream`` so the launch is ordered against torch's own
    work without a device-wide synchronize -- cupy would otherwise use its
    own stream and race the producer of the operands.

    Args:
        kernel_name: Which kernel in :data:`CUDA_SOURCE`.
        grid: Blocks.
        block: Threads per block.
        args: Kernel arguments -- cupy array views and ints.
    """
    stream = _surface().external_stream(torch.cuda.current_stream().cuda_stream)
    kernel = _kernels().get_function(kernel_name)
    with stream:
        kernel(grid, block, args)


def gemm(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Compute ``x @ w`` (plus ``bias``, added last) in program-fixed order.

    Bit-identical to ``rank1_matmul`` / ``rank1_addmm`` by construction --
    same ascending-k order, same two-rounding multiply-add, bias after the
    whole product -- and asserted against them in the suite rather than
    trusted. What differs is only speed: shared-memory tiling instead of K
    passes over the accumulator.

    Args:
        x: ``[N, K]``, float32, CUDA.
        w: ``[K, M]``, float32, CUDA; a transposed view is fine and is
            copied contiguous.
        bias: ``[M]`` or None.

    Returns:
        ``[N, M]``, float32, on the same device.

    Raises:
        ValueError: Propagated from the operand checks, or for a shape
            mismatch between the operands.
    """
    x2 = _require_f32_cuda(x, "x", 2)
    w2 = _require_f32_cuda(w, "w", 2)
    if x2.shape[1] != w2.shape[0]:
        raise ValueError(f"inner dimensions differ: x is {tuple(x2.shape)}, w {tuple(w2.shape)}")
    n_rows, k_dim = int(x2.shape[0]), int(x2.shape[1])
    m_cols = int(w2.shape[1])
    use_bias = 0
    if bias is None:
        bias_arg = torch.empty(0, dtype=torch.float32, device=x2.device)
    else:
        if not bias.is_cuda or bias.dtype != torch.float32 or bias.dim() != 1:
            raise ValueError("bias must be a 1-D float32 CUDA tensor")
        if int(bias.shape[0]) != m_cols:
            raise ValueError(f"bias has {int(bias.shape[0])} elements for {m_cols} columns")
        bias_arg = bias.detach().contiguous()
        use_bias = 1
    out = torch.empty(n_rows, m_cols, dtype=torch.float32, device=x2.device)
    surface = _surface()
    grid = ((m_cols + TILE - 1) // TILE, (n_rows + TILE - 1) // TILE, 1)
    _launch(
        "ordered_gemm_nn",
        grid,
        (THREADS, THREADS, 1),
        (
            surface.from_dlpack(x2),
            surface.from_dlpack(w2),
            surface.from_dlpack(bias_arg),
            surface.from_dlpack(out),
            n_rows,
            k_dim,
            m_cols,
            use_bias,
        ),
    )
    return out


def rowsum(grad: torch.Tensor) -> torch.Tensor:
    """Sum the rows of a matrix in ascending order, one thread per column.

    The kernel twin of
    :func:`~model_trainer.core.services.model.deterministic_gemm.accumulate_rows`,
    for the bias gradient; bit-identity is asserted in the suite.

    Args:
        grad: ``[N, M]``, float32, CUDA.

    Returns:
        ``[M]``.

    Raises:
        ValueError: Propagated from the operand checks.
    """
    g2 = _require_f32_cuda(grad, "grad", 2)
    n_rows, m_cols = int(g2.shape[0]), int(g2.shape[1])
    out = torch.empty(m_cols, dtype=torch.float32, device=g2.device)
    surface = _surface()
    grid = ((m_cols + ROWSUM_BLOCK - 1) // ROWSUM_BLOCK, 1, 1)
    _launch(
        "ordered_rowsum",
        grid,
        (ROWSUM_BLOCK, 1, 1),
        (surface.from_dlpack(g2), surface.from_dlpack(out), n_rows, m_cols),
    )
    return out


def gemm_batched(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute ``x[b] @ w[b]`` for every batch slice, each in fixed order.

    The same kernel as :func:`gemm` -- ``blockIdx.z`` selects the slice, and
    every output element is still one thread's strictly ascending k-chain --
    so per slice this is bit-identical to :func:`gemm` on that slice, which
    the suite asserts. Attention's two matmuls are the consumer: 12 heads of
    a real GPT-2 are 12 slices of one launch instead of 12 launches.

    Args:
        x: ``[B, N, K]``, float32, CUDA; a strided view is fine and is
            copied contiguous.
        w: ``[B, K, M]``, float32, CUDA.

    Returns:
        ``[B, N, M]``, float32, on the same device.

    Raises:
        ValueError: Propagated from the operand checks, or for mismatched
            batch or inner dimensions.
    """
    x3 = _require_f32_cuda(x, "x", 3)
    w3 = _require_f32_cuda(w, "w", 3)
    if x3.shape[0] != w3.shape[0]:
        raise ValueError(f"batch sizes differ: x has {int(x3.shape[0])}, w {int(w3.shape[0])}")
    if x3.shape[2] != w3.shape[1]:
        raise ValueError(f"inner dimensions differ: x is {tuple(x3.shape)}, w {tuple(w3.shape)}")
    batch, n_rows, k_dim = int(x3.shape[0]), int(x3.shape[1]), int(x3.shape[2])
    m_cols = int(w3.shape[2])
    bias_arg = torch.empty(0, dtype=torch.float32, device=x3.device)
    out = torch.empty(batch, n_rows, m_cols, dtype=torch.float32, device=x3.device)
    surface = _surface()
    grid = ((m_cols + TILE - 1) // TILE, (n_rows + TILE - 1) // TILE, batch)
    _launch(
        "ordered_gemm_nn",
        grid,
        (THREADS, THREADS, 1),
        (
            surface.from_dlpack(x3),
            surface.from_dlpack(w3),
            surface.from_dlpack(bias_arg),
            surface.from_dlpack(out),
            n_rows,
            k_dim,
            m_cols,
            0,
        ),
    )
    return out


def lastdim_sum(rows: torch.Tensor) -> torch.Tensor:
    """Sum each row left to right, one thread per row.

    The softmax denominator's reduction: where :func:`rowsum` folds N rows
    into one (the bias gradient), this folds each row's C columns into one
    scalar, in ascending column order.

    Args:
        rows: ``[R, C]``, float32, CUDA.

    Returns:
        ``[R]``.

    Raises:
        ValueError: Propagated from the operand checks.
    """
    r2 = _require_f32_cuda(rows, "rows", 2)
    n_rows, m_cols = int(r2.shape[0]), int(r2.shape[1])
    out = torch.empty(n_rows, dtype=torch.float32, device=r2.device)
    surface = _surface()
    grid = ((n_rows + ROWSUM_BLOCK - 1) // ROWSUM_BLOCK, 1, 1)
    _launch(
        "ordered_lastdim_sum",
        grid,
        (ROWSUM_BLOCK, 1, 1),
        (surface.from_dlpack(r2), surface.from_dlpack(out), n_rows, m_cols),
    )
    return out


__all__ = [
    "CUDA_SOURCE",
    "K_SLICE",
    "MICRO",
    "NVRTC_OPTIONS",
    "ROWSUM_BLOCK",
    "THREADS",
    "TILE",
    "CupyArrayProto",
    "ExternalStreamProto",
    "RawKernelProto",
    "RawModuleProto",
    "gemm",
    "gemm_batched",
    "lastdim_sum",
    "rowsum",
]
