"""The individual GEMMs the probe ladder issues, isolated so they can be read.

WHY THIS EXISTS. The ladder measured that cards disagree at some rungs, and
the cuBLASLt trace measured that they choose different kernels for almost
every GEMM shape -- 31 of 32. Those two facts do not join up: `medium` draws
three different kernels, including three different split-K counts, and still
returns a bit-identical loss on all three cards. So kernel divergence does
not imply numeric divergence, and nothing yet says which GEMM, if any,
carries the difference at the rungs that DO disagree.

A whole forward pass cannot answer that. Its loss is one number aggregating
every operation, so a difference anywhere lands in the same place. This runs
ONE GEMM at a time and compares its OUTPUT TENSOR across cards, which turns
"the rung disagreed" into "this matmul disagreed, and that one did not".

HOW A SHAPE MAPS TO A CALL, WHICH WAS MEASURED AND NOT REASONED. cuBLASLt
logs operands in column-major, so a call logged ``A[M x K] B[K x N]`` is
issued from torch as ``addmm(bias[M], x[N, K], w[K, M])``. That mapping was
checked by running the call under ``CUBLASLT_LOG_LEVEL=4`` and comparing the
logged descriptors to the ladder's: ``addmm(b, x[64,4096], w[4096,1024])``
logs ``Adesc=[rows=1024 cols=4096] Bdesc=[rows=4096 cols=64]``, which is the
ladder's medium MLP-projection call exactly.

WHY ``addmm`` AND NOT ``mm``. The bias is not decoration. ``torch.mm`` does
not reach cuBLASLt at all -- measured: it logs nothing under a trace and
takes the legacy ``cublasSgemm`` path. HF's ``Conv1D`` uses ``addmm``, whose
fused bias epilogue (``epilogue=EPILOGUE_BIAS`` in the log) is what routes it
to cuBLASLt. A probe built on ``mm`` would exercise a different library entry
point and answer a question nobody asked.
"""

from __future__ import annotations

from typing import Final

from typing_extensions import TypedDict


class GemmShape(TypedDict):
    """One matmul, in the orientation cuBLASLt reports it.

    Attributes:
        rows: ``M``, the output width -- the weight's output dimension.
        inner: ``K``, the summed dimension. This is the one that matters: it
            is what split-K partitions, so it is where a device-dependent
            reduction order can enter.
        cols: ``N``, the batch-times-sequence dimension.
        origin: Which ladder rung and role this call comes from, so a result
            here can be read against the rung that produced it.
    """

    rows: int
    inner: int
    cols: int
    origin: str


#: The batch-times-sequence dimension every ladder GEMM shares, because the
#: ladder runs one sequence of the gate rung's length.
GEMM_COLS = 64

#: Seven calls spanning the three cases the trace turned up, chosen so the
#: result can discriminate rather than merely accumulate:
#:
#: * ``128x128`` is the ONE shape of 32 where all three cards chose the same
#:   algorithm. If its output differs, something is wrong with this probe
#:   rather than with the cards.
#: * The ``small`` and ``medium`` calls drew DIFFERENT algorithms on every
#:   card -- including split-K counts of 9, 6 and 3 on one of them -- while
#:   their rungs returned bit-identical losses. These are the cases that
#:   decide whether a kernel difference can be numerically invisible.
#: * The ``large`` and ``xl`` calls come from the rungs that DID disagree.
#:
#: Declared as constants rather than derived from a model, for the reason the
#: probe ladder is: a shape assembled at runtime is one nobody can reproduce
#: without also recovering how it was assembled.
GEMM_SHAPES: Final[dict[str, GemmShape]] = {
    "tiny-attn-proj": {
        "rows": 128,
        "inner": 128,
        "cols": GEMM_COLS,
        "origin": "tiny rung, attention output projection; all three cards chose one algorithm",
    },
    "small-attn-proj": {
        "rows": 768,
        "inner": 768,
        "cols": GEMM_COLS,
        "origin": "small rung, attention output projection; rung agreed across cards",
    },
    "small-mlp-proj": {
        "rows": 768,
        "inner": 3072,
        "cols": GEMM_COLS,
        "origin": "small rung, MLP output projection; split-K 9/6/3, rung agreed across cards",
    },
    "medium-attn-proj": {
        "rows": 1024,
        "inner": 1024,
        "cols": GEMM_COLS,
        "origin": "medium rung, attn output projection; A100 split-K 6 vs A30 none, rung agreed",
    },
    "medium-mlp-proj": {
        "rows": 1024,
        "inner": 4096,
        "cols": GEMM_COLS,
        "origin": "medium rung, MLP output projection; A100 split-K 6 vs A30 4, rung agreed",
    },
    "large-attn-proj": {
        "rows": 1280,
        "inner": 1280,
        "cols": GEMM_COLS,
        "origin": "large rung, attn output projection; A100 split-K 3 vs A30 none, rung DISAGREED",
    },
    "large-mlp-proj": {
        "rows": 1280,
        "inner": 5120,
        "cols": GEMM_COLS,
        "origin": "large rung, MLP output projection; A100 and A30 chose alike, rung DISAGREED",
    },
    "xl-mlp-proj": {
        "rows": 1600,
        "inner": 6400,
        "cols": GEMM_COLS,
        "origin": "xl rung, MLP output projection; rung DISAGREED with V100 the outlier",
    },
}

#: Seed for the operands. One seed for the whole table, generated on the CPU
#: and moved to the device, so every card multiplies the SAME bits. Generating
#: on the GPU would use the device RNG and hand different cards different
#: inputs, which would produce a difference that says nothing about GEMMs.
GEMM_SEED = 42

GEMM_EXPERIMENT = "gemm-attribution"

#: Suffix for the observation carrying a shape's output identity.
DIGEST_SUFFIX = "digest48"

#: Suffix for the observation carrying its magnitude.
SUM_SUFFIX = "sum"


def gemm_label(name: str, shape: GemmShape, suffix: str) -> str:
    """Name one measurement of one shape.

    The dimensions are in the label rather than only the table, so a record
    read on its own still says what was multiplied -- and a shape edited
    without renaming cannot quietly reuse an old name.

    Args:
        name: The table key.
        shape: Its dimensions.
        suffix: Which measurement, :data:`DIGEST_SUFFIX` or :data:`SUM_SUFFIX`.

    Returns:
        e.g. ``gemm-medium-mlp-proj-M1024-K4096-N64|digest48``.
    """
    return f"gemm-{name}-M{shape['rows']}-K{shape['inner']}-N{shape['cols']}|{suffix}"


__all__ = [
    "DIGEST_SUFFIX",
    "GEMM_COLS",
    "GEMM_EXPERIMENT",
    "GEMM_SEED",
    "GEMM_SHAPES",
    "SUM_SUFFIX",
    "GemmShape",
    "gemm_label",
]
