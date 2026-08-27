"""Which attention calls to probe, and what each measurement is called.

The names, not the arithmetic. :mod:`sdpa_probe` runs one; this says what
there is to run, for the same reason :mod:`probe_shapes` is separate from
:mod:`known_answer_probe` -- naming a measurement must not require importing
torch, and the report that reads finished records runs on a laptop.

WHAT QUESTION THIS ANSWERS. The forward trace
([[a-loss-agrees-where-the-computation-does-not]], measured 2026-08-27)
localised the `tiny` rung's entire cross-card divergence to ONE operation:
`scaled_dot_product_attention`. Everything up to and including the QKV matmul
is bit-identical on a V100, an A30 and an A100; the attention result is not,
and every later tensor inherits it. What the trace could NOT say is why --
`F.scaled_dot_product_attention` is a dispatcher over several kernels, and
which one each card selects was never recorded.

HOW SELECTION IS MEASURED RATHER THAN INFERRED. ``can_use_flash_attention``
and friends report which backends torch considers ELIGIBLE, which is torch's
opinion about a configuration and not a record of what ran. So this probe
also runs the call once per backend with that backend FORCED, and digests
each output. The backend whose forced output is bit-identical to the
unforced one is the backend the unforced call used. Two failure modes of
that method are real and are left visible rather than resolved: if NO forced
output matches, the default used something none of them reproduced; if
SEVERAL match, the measurement cannot separate them. The record carries the
digests and the report says which case it is.

WHY THE SHAPES ARE DERIVED FROM THE LADDER. Every shape here is the attention
call some ladder rung actually makes, computed from
:data:`~probe_shapes.PROBE_SHAPES` and :data:`~model_sizes.GPT2_MODEL_SIZES`
rather than restated. A rung reshaped there changes what this probes instead
of leaving this measuring a shape nothing runs.
"""

from __future__ import annotations

from typing import Final

from typing_extensions import TypedDict

from model_trainer.core.services.model.cost_labels import (
    DEFAULT_KEY,
    FALSE_VALUE,
    FITTED_SUFFIX,
    PEAK_SUFFIX,
    SECONDS_SUFFIX,
    SPREAD_SUFFIX,
    TRUE_VALUE,
    labelled,
)
from model_trainer.core.services.model.model_sizes import GPT2_MODEL_SIZES
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES

#: Distinct from every other experiment here. "Which kernel did this card
#: select" is not "where does agreement break" and not "do these outputs
#: match", so a record answering one must not be differenced against a record
#: answering another.
SDPA_EXPERIMENT = "sdpa-backend-selection"

#: Seed for the operands. Shared with the probe ladder deliberately: the
#: question is about kernel selection, which the seed does not touch, and a
#: second seed would be a second thing to keep in step for no gain.
SDPA_SEED = 42


class SdpaShape(TypedDict):
    """One attention call, as some ladder rung makes it.

    Attributes:
        rung: The ladder rung this call comes from.
        heads: Attention heads, which is the batch dimension of the
            underlying matmuls once the head axis is folded in.
        head_dim: Width of one head. 64 at every GPT-2 size, which is worth
            recording rather than assuming: it is what makes the rungs differ
            in head COUNT alone.
        sequence_len: Query and key length.
    """

    rung: str
    heads: int
    head_dim: int
    sequence_len: int


def sdpa_shape_for(rung: str) -> SdpaShape:
    """Build the attention call one ladder rung makes.

    Args:
        rung: A key of :data:`~probe_shapes.PROBE_SHAPES`.

    Returns:
        That rung's attention shape.

    Raises:
        KeyError: If the rung is not one the ladder declares, or names a
            model size the shared table lacks.
    """
    shape = PROBE_SHAPES[rung]
    dims = GPT2_MODEL_SIZES[shape["model_size"]]
    return SdpaShape(
        rung=rung,
        heads=dims["n_head"],
        head_dim=dims["hidden_size"] // dims["n_head"],
        sequence_len=shape["sequence_len"],
    )


def sdpa_shapes() -> tuple[SdpaShape, ...]:
    """Every attention call the ladder makes, in ladder order.

    Returns:
        One shape per declared rung.
    """
    return tuple(sdpa_shape_for(rung) for rung in PROBE_SHAPES)


#: Sequence lengths the cost sweep walks.
#:
#: WHY A SWEEP AND NOT THE LADDER'S SHAPES. The correctness result says
#: pinning ``MATH`` makes attention card-invariant. Its price is dominated by
#: ONE axis: the math path materialises the full ``[batch, heads, seq, seq]``
#: score matrix, which is quadratic in sequence length, where the fused
#: kernel is not. Quoting a cost at the ladder's own 64 tokens would repeat
#: the mistake the split-K page had to correct, where a table measured at a
#: single unrepresentative point became "the cost of reproducibility". The
#: sweep runs to 4096 so the wall, if there is one, is inside the measured
#: range rather than past its edge.
COST_LENGTHS: Final[tuple[int, ...]] = (64, 128, 256, 512, 1024, 2048, 4096)

#: Batches the cost sweep walks. One sequence is the regime the ladder
#: probes; eight is a plausible training batch, and the memory the math path
#: needs scales with it.
COST_BATCHES: Final[tuple[int, ...]] = (1, 8)

#: Heads and width for the cost sweep: gpt2-small's attention, which is the
#: smallest size anyone trains rather than the smallest the ladder defines.
COST_HEADS = 12
COST_HEAD_DIM = 64


class SdpaCostShape(TypedDict):
    """One attention call to time.

    Attributes:
        name: What to call it in a record.
        batch: Sequences in the batch.
        heads: Attention heads.
        head_dim: Width of one head.
        sequence_len: Query and key length.
    """

    name: str
    batch: int
    heads: int
    head_dim: int
    sequence_len: int


def cost_shapes() -> tuple[SdpaCostShape, ...]:
    """Every attention call the cost sweep times, in order.

    Returns:
        The batch-by-length grid, then the ladder's own eight calls at batch
        one -- so the configurations the correctness result covers have a
        stated price of their own rather than one interpolated from the grid.
    """
    grid = tuple(
        SdpaCostShape(
            name=f"grid-b{batch}-s{length}",
            batch=batch,
            heads=COST_HEADS,
            head_dim=COST_HEAD_DIM,
            sequence_len=length,
        )
        for batch in COST_BATCHES
        for length in COST_LENGTHS
    )
    ladder = tuple(
        SdpaCostShape(
            name=f"rung-{shape['rung']}",
            batch=1,
            heads=shape["heads"],
            head_dim=shape["head_dim"],
            sequence_len=shape["sequence_len"],
        )
        for shape in sdpa_shapes()
    )
    return grid + ladder


def cost_prefix(shape: SdpaCostShape) -> str:
    """Name one timed call, without saying which measurement.

    Args:
        shape: The call.

    Returns:
        e.g. ``cost-grid-b8-s2048-h12-d64``.
    """
    return f"cost-{shape['name']}-h{shape['heads']}-d{shape['head_dim']}"


def cost_label(shape: SdpaCostShape, backend: str, suffix: str) -> str:
    """Name one timing of one call.

    Args:
        shape: The call.
        backend: :data:`DEFAULT_KEY` or a key of :data:`BACKEND_KEYS`.
        suffix: Which measurement.

    Returns:
        e.g. ``cost-grid-b8-s2048-h12-d64|math|seconds``.
    """
    return labelled(cost_prefix(shape), backend, suffix)


#: Distinct from the selection experiment. "Which kernel ran" and "what it
#: cost" are different questions and their records must not be differenced.
SDPA_COST_EXPERIMENT = "sdpa-backend-cost"

#: Backends to force, one at a time. ``ERROR`` and ``OVERRIDEABLE`` are
#: deliberately absent: neither is a kernel a call can land on.
BACKEND_KEYS: Final[tuple[str, ...]] = ("math", "flash", "efficient", "cudnn")

#: Backends torch will give an eligibility opinion about. ``math`` has no
#: ``can_use_math_attention`` because it is the fallback that is always
#: compiled in -- so for it, "did forcing it work" is the only evidence
#: there is, which is exactly why availability is measured separately from
#: eligibility for all four.
ELIGIBLE_KEYS: Final[tuple[str, ...]] = ("flash", "efficient", "cudnn")

#: Suffix for the observation carrying an output's identity.
DIGEST_SUFFIX = "digest48"

#: Suffix for whether forcing that backend produced a result at all. This is
#: MEASURED -- the call was made and either returned or refused.
AVAILABLE_SUFFIX = "available"

#: Suffix for whether torch considers the backend usable for this shape.
#: This is torch's OPINION, recorded beside the measurement so the two can be
#: checked against each other rather than one standing in for the other.
ELIGIBLE_SUFFIX = "eligible"


def sdpa_label(shape: SdpaShape, backend: str, suffix: str) -> str:
    """Name one measurement of one attention call.

    The dimensions are in the label rather than only the table, so a record
    read on its own still says what was computed -- and a rung reshaped
    without renaming cannot quietly reuse an old name.

    Args:
        shape: The call.
        backend: :data:`DEFAULT_KEY` or one of :data:`BACKEND_KEYS`.
        suffix: Which measurement.

    Returns:
        e.g. ``sdpa-tiny-h2-d64-s64|efficient|digest48``.
    """
    return (
        f"sdpa-{shape['rung']}"
        f"-h{shape['heads']}"
        f"-d{shape['head_dim']}"
        f"-s{shape['sequence_len']}"
        f"|{backend}|{suffix}"
    )


__all__ = [
    "AVAILABLE_SUFFIX",
    "BACKEND_KEYS",
    "COST_BATCHES",
    "COST_HEADS",
    "COST_HEAD_DIM",
    "COST_LENGTHS",
    "DEFAULT_KEY",
    "DIGEST_SUFFIX",
    "ELIGIBLE_KEYS",
    "ELIGIBLE_SUFFIX",
    "FALSE_VALUE",
    "FITTED_SUFFIX",
    "PEAK_SUFFIX",
    "SDPA_COST_EXPERIMENT",
    "SDPA_EXPERIMENT",
    "SDPA_SEED",
    "SECONDS_SUFFIX",
    "SPREAD_SUFFIX",
    "TRUE_VALUE",
    "SdpaCostShape",
    "SdpaShape",
    "cost_label",
    "cost_prefix",
    "cost_shapes",
    "sdpa_label",
    "sdpa_shape_for",
    "sdpa_shapes",
]
