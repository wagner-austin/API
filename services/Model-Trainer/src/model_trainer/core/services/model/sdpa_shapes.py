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


#: The unforced call -- what the dispatcher chose on its own. Named like a
#: backend so it sits in the same column as the forced runs, because the
#: whole reading is "which forced digest equals this one".
DEFAULT_KEY = "default"

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

#: How a boolean is carried in a record, which holds only numbers.
TRUE_VALUE = 1.0
FALSE_VALUE = 0.0


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
    "DEFAULT_KEY",
    "DIGEST_SUFFIX",
    "ELIGIBLE_KEYS",
    "ELIGIBLE_SUFFIX",
    "FALSE_VALUE",
    "SDPA_EXPERIMENT",
    "SDPA_SEED",
    "TRUE_VALUE",
    "SdpaShape",
    "sdpa_label",
    "sdpa_shape_for",
    "sdpa_shapes",
]
