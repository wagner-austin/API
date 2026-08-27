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

#: The sweep's output widths, and the reduction lengths crossed against them.
#:
#: WHY A SWEEP EXISTS BESIDE THE TABLE ABOVE. That table answers attribution --
#: which of the LADDER's matmuls carries a rung's difference -- and for that it
#: must be the ladder's shapes. It cannot answer the other half. Of the 32
#: shapes the ladder issues, exactly TWO drew the same algorithm on the V100
#: and the A30, so "does one kernel imply one result" rested on a single
#: instance, which is an anecdote wearing a claim's clothes.
#:
#: A sweep fixes that by not choosing. Every grid point records its kernel and
#: its output, and whichever cell of the 2x2 a point lands in, it lands there
#: on its own. Small ``K`` is over-represented deliberately: the two shapes
#: where the cards already agreed both had ``K=128``, so that is where the
#: same-kernel cases are expected to come from, and a sweep that skipped it
#: would have re-created the shortage it exists to fix.
#:
#: ``K`` is the summed dimension, which is what split-K partitions, so it is
#: the axis a device-dependent reduction order enters through. ``M`` moves
#: because tiling depends on the output width too.
SWEEP_ROWS: Final[tuple[int, ...]] = (128, 256, 512, 1024, 2048)
SWEEP_INNERS: Final[tuple[int, ...]] = (128, 256, 512, 1024, 2048, 4096, 8192)


#: The name every grid point is labelled under. ONE name for the whole grid,
#: not one per point: :func:`gemm_label` already appends the dimensions, so a
#: per-point name encoding them too produced
#: ``gemm-sweep-M1024-K1024-M1024-K1024-N64`` -- correct and unique, and the
#: kind of sloppiness that makes a reader distrust the rest of the artifact.
#: Labels stay unique because the dimensions differ.
SWEEP_NAME = "sweep"


def _sweep() -> tuple[GemmShape, ...]:
    """Build the grid.

    Generated from two declared lists rather than typed out, which keeps the
    probe reproducible from the source alone -- the lists ARE the shapes, and
    nobody has to recover how a hand-written table was assembled.

    Returns:
        One entry per grid point, in a stable order. A tuple rather than a
        mapping because every point shares :data:`SWEEP_NAME`; the label, not
        the key, is what distinguishes them.
    """
    return tuple(
        GemmShape(
            rows=rows,
            inner=inner,
            cols=GEMM_COLS,
            origin=f"sweep grid point, M={rows} K={inner}",
        )
        for rows in SWEEP_ROWS
        for inner in SWEEP_INNERS
    )


#: The grid.
GEMM_SWEEP: Final[tuple[GemmShape, ...]] = _sweep()

#: A realistic training batch: eight sequences of 512 tokens flattened, which
#: is what the batch-times-sequence dimension is in a real step.
#:
#: WHY THE BATCHED TABLE EXISTS. Timing at :data:`GEMM_COLS` could not resolve
#: most shapes. Every matmul call carries a fixed ~100 microseconds of Python
#: dispatch and kernel launch, and at one 64-token sequence the arithmetic is
#: a few microseconds -- so the measurement was overhead with a rounding error
#: of signal on top, and reported the same time for a 128x128 matmul as for a
#: 1600x6400 one.
#:
#: That is not a shape nobody cares about. Split-K is SELECTED on seven of the
#: eight ladder calls -- measured from the traces; only ``tiny-attn-proj`` at
#: K=128 draws none, which is what one would expect when there is nothing to
#: split. So the shapes the first benchmark could not resolve are shapes the
#: intervention really does change; they were unmeasured, not unaffected.
#:
#: Raising the batch fixes the measurement and makes it more honest at the
#: same time: it is the regime real training runs in, where the arithmetic is
#: hundreds of microseconds and the fixed overhead is noise instead of the
#: signal.
BATCH_COLS = 4096


def _batched() -> tuple[tuple[str, GemmShape], ...]:
    """The ladder's calls at a realistic batch size.

    Returns:
        ``(name, shape)`` pairs mirroring :data:`GEMM_SHAPES`, at
        :data:`BATCH_COLS` instead of one short sequence.
    """
    return tuple(
        (
            f"batched-{name}",
            GemmShape(
                rows=shape["rows"],
                inner=shape["inner"],
                cols=BATCH_COLS,
                origin=f"{shape['origin']} -- at a {BATCH_COLS}-row batch",
            ),
        )
        for name, shape in GEMM_SHAPES.items()
    )


#: The ladder's calls at a real batch size.
GEMM_BATCHED: Final[tuple[tuple[str, GemmShape], ...]] = _batched()

#: Batch sizes swept between the two regimes already measured.
#:
#: WHY. Disabling split-K costs +20% to +85% at ``GEMM_COLS`` and 0-13% at
#: :data:`BATCH_COLS`, and those are the only two points anyone has. That is
#: enough to say the cost depends on batch size and not enough to say WHERE it
#: goes away -- which is the number someone sizing a real job actually needs.
#: Doubling from one to the other turns two points into eight.
CROSSOVER_COLS: Final[tuple[int, ...]] = (64, 128, 256, 512, 1024, 2048, 4096)

#: Which calls to sweep. The three MLP projections, because they are the only
#: shapes whose cost at ``GEMM_COLS`` cleared the per-call overhead floor and
#: so the only ones with a cost to lose. Sweeping the attention projections
#: too would double the runtime to watch numbers that started at zero.
CROSSOVER_SOURCES: Final[tuple[str, ...]] = (
    "medium-mlp-proj",
    "large-mlp-proj",
    "xl-mlp-proj",
)


def _crossover() -> tuple[tuple[str, GemmShape], ...]:
    """Build the batch-size sweep.

    Returns:
        ``(name, shape)`` pairs, one per source call per batch size. Each
        source keeps ONE name across the sweep; the labels differ because the
        batch dimension does, which is the same reason the grid shares
        :data:`SWEEP_NAME`.
    """
    return tuple(
        (
            f"crossover-{name}",
            GemmShape(
                rows=GEMM_SHAPES[name]["rows"],
                inner=GEMM_SHAPES[name]["inner"],
                cols=cols,
                origin=f"{name} swept to a {cols}-row batch",
            ),
        )
        for name in CROSSOVER_SOURCES
        for cols in CROSSOVER_COLS
    )


#: The batch-size sweep.
GEMM_CROSSOVER: Final[tuple[tuple[str, GemmShape], ...]] = _crossover()

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


def probed_shapes() -> tuple[tuple[str, GemmShape], ...]:
    """Every shape one run measures: the ladder's calls, then the sweep.

    Pairs rather than a mapping, because every grid point shares
    :data:`SWEEP_NAME` and a mapping would collapse them to one. What has to be
    unique is the LABEL, and that is checked here rather than assumed.

    A shape can legitimately appear in both tables -- the ladder's
    ``tiny-attn-proj`` is M128/K128, which is also a grid point -- and it is
    measured under each name rather than deduplicated. That is deliberate: the
    two labels must carry the same digest, so the overlap is a free check that
    the result depends on the DIMENSIONS and not on which table asked.

    Returns:
        ``(name, shape)`` pairs, ladder first then the grid.

    Raises:
        ValueError: Propagated from :func:`require_unique_labels`.
    """
    pairs = [(name, shape) for name, shape in GEMM_SHAPES.items()]
    pairs += [(SWEEP_NAME, shape) for shape in GEMM_SWEEP]
    return require_unique_labels(tuple(pairs))


def timed_shapes() -> tuple[tuple[str, GemmShape], ...]:
    """Every shape the split-K cost benchmark times.

    The ladder's calls at one short sequence and again at a real batch. The
    sweep grid is deliberately absent: it exists to populate the
    same-kernel/same-output table, which is a question about VALUES, and
    timing thirty-five more shapes to answer it would cost minutes and say
    nothing.

    The batched twins carry a different name rather than a different table
    position, so a record read on its own distinguishes ``large-mlp-proj``
    from ``batched-large-mlp-proj`` without consulting the dimensions.

    Returns:
        ``(name, shape)`` pairs, single-sequence first then batched.

    Raises:
        ValueError: Propagated from :func:`require_unique_labels`.
    """
    pairs = [(name, shape) for name, shape in GEMM_SHAPES.items()]
    pairs += list(GEMM_BATCHED)
    pairs += list(GEMM_CROSSOVER)
    return require_unique_labels(tuple(pairs))


def require_unique_labels(
    pairs: tuple[tuple[str, GemmShape], ...],
) -> tuple[tuple[str, GemmShape], ...]:
    """Return the pairs, refusing if any two would share a label.

    A separate function rather than a check inside :func:`probed_shapes`, for
    the reason :func:`require_reproduced` is one: the declared tables do not
    collide, so the failing arm is unreachable through them, and an arm no
    test can drive is an arm nobody has confirmed says what it means.

    Args:
        pairs: The ``(name, shape)`` pairs to check.

    Returns:
        ``pairs`` unchanged, once the labels are known distinct.

    Raises:
        ValueError: If two entries produce one label. That would silently drop
            an observation -- and ``run_record`` would reject the duplicate
            name anyway, at a point much further from the cause.
    """
    labels = [gemm_label(name, shape, DIGEST_SUFFIX) for name, shape in pairs]
    duplicated = sorted({label for label in labels if labels.count(label) > 1})
    if duplicated:
        raise ValueError(f"two probed shapes share a label: {duplicated}")
    return pairs


__all__ = [
    "BATCH_COLS",
    "CROSSOVER_COLS",
    "CROSSOVER_SOURCES",
    "DIGEST_SUFFIX",
    "GEMM_BATCHED",
    "GEMM_COLS",
    "GEMM_CROSSOVER",
    "GEMM_EXPERIMENT",
    "GEMM_SEED",
    "GEMM_SHAPES",
    "GEMM_SWEEP",
    "SUM_SUFFIX",
    "SWEEP_INNERS",
    "SWEEP_NAME",
    "SWEEP_ROWS",
    "GemmShape",
    "gemm_label",
    "probed_shapes",
    "require_unique_labels",
    "timed_shapes",
]
