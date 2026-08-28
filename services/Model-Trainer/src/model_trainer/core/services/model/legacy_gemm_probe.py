"""The one matmul the cuBLASLt switch cannot reach, and whether it can be.

THE OPEN QUESTION THIS ANSWERS. With `CUBLASLT_WORKSPACE_SIZE=0` the A100 and
the A30 agree on 1,017 of `xl`'s 1,018 traced tensors. The single survivor is
`lm_head`, and the reason is structural: GPT-2's output projection is
``nn.Linear(n_embd, vocab, bias=False)``, no bias means no fused epilogue, and
without an epilogue torch does not route the call to cuBLASLt at all -- it
takes the legacy ``cublasSgemm`` entry point, which the workspace variable
governs nothing about. One tensor, and it is enough to move a reported loss.

`gemm_shapes` says why its own probe could not answer this: it is built on
``addmm`` deliberately, because "the bias is not decoration ... a probe built
on ``mm`` would exercise a different library entry point and answer a question
nobody asked". Somebody is asking it now.

THE EXPERIMENT, AND WHY IT IS DECISIVE. For each shape this computes the same
product twice on one card:

* ``mm(x, w)`` -- what `lm_head` does today, on the legacy path.
* ``addmm(zeros, x, w)`` -- the same arithmetic, with a bias of zero. Adding
  0.0 is exact in IEEE-754, so the two agree in real arithmetic and any
  difference between them is reduction ORDER; and the zero bias supplies the
  fused epilogue that routes the call to cuBLASLt.

Comparing the ``mm`` digest across cards says whether the legacy path is where
the divergence lives. Comparing the ``addmm`` digest across cards says whether
routing it to cuBLASLt would fix it. If the second agrees where the first does
not, then the residual `lm_head` divergence has a remedy that costs one add of
zero per output element -- and this module is what establishes that rather
than assuming it.

WHY THE OPERANDS COME FROM `gemm_probe`. Same builder, same seed, same
CPU-then-move discipline. A second spelling would be free to drift, and a
difference caused by different inputs would look exactly like the finding.
"""

from __future__ import annotations

from typing import Final

import torch
from typing_extensions import TypedDict

from model_trainer.core.services.model.gemm_probe import (
    describe_output,
    gemm_operands,
)
from model_trainer.core.services.model.gemm_shapes import GEMM_COLS, GemmShape

#: The output projections of the ladder's four rungs, in cuBLASLt's
#: orientation.
#:
#: ``rows`` is the vocabulary, which is what makes these different from every
#: shape the existing probe carries: an output projection is far wider than it
#: is deep, where an attention or MLP projection is roughly square. ``inner``
#: is ``n_embd``, and it is the summed dimension -- the one a split reduction
#: would partition.
#:
#: The gate rung is included and expected NOT to diverge: the trace found
#: `lm_head` agreeing at `tiny`, whose K is 128 against 1024-1600 at the
#: larger rungs, and a short reduction gives a split nothing to do. A rung
#: where the effect is absent is what distinguishes "K is the variable" from
#: "this path always differs".
LM_HEAD_SHAPES: Final[tuple[GemmShape, ...]] = (
    {"rows": 512, "inner": 128, "cols": GEMM_COLS, "origin": "tiny-lm-head"},
    {"rows": 512, "inner": 1024, "cols": GEMM_COLS, "origin": "medium-lm-head"},
    {"rows": 512, "inner": 1280, "cols": GEMM_COLS, "origin": "large-lm-head"},
    {"rows": 512, "inner": 1600, "cols": GEMM_COLS, "origin": "xl-lm-head"},
    # The real vocabulary, which no ladder rung uses and every real model
    # does. If the effect depends on the output width rather than on K, this
    # is the row that shows it.
    {"rows": 50257, "inner": 768, "cols": GEMM_COLS, "origin": "gpt2-real-vocab"},
)

#: What the bias-free call is named in a record.
LEGACY_ARM = "mm"

#: What the zero-bias call is named in a record.
EPILOGUE_ARM = "addmm-zero-bias"

ARMS: Final[tuple[str, ...]] = (LEGACY_ARM, EPILOGUE_ARM)


class ArmOutputs(TypedDict):
    """One shape's two products, computed on one card.

    Attributes:
        legacy: ``mm(x, w)``, the bias-free path `lm_head` takes.
        epilogue: ``addmm(zeros, x, w)``, the same product routed through
            cuBLASLt by a bias that adds nothing.
    """

    legacy: torch.Tensor
    epilogue: torch.Tensor


def arm_outputs(shape: GemmShape, device: str) -> ArmOutputs:
    """Compute one shape's product on both library entry points.

    Args:
        shape: The call to issue.
        device: Where to issue it.

    Returns:
        Both products, from identical operands.
    """
    bias, x, w = gemm_operands(shape, device)
    return ArmOutputs(
        legacy=torch.mm(x, w),
        epilogue=torch.addmm(torch.zeros_like(bias), x, w),
    )


def arms_agree(outputs: ArmOutputs) -> bool:
    """Whether the two entry points produced bit-identical tensors.

    A zero bias is exact, so in real arithmetic these are the same product.
    A difference here is reduction order alone -- which is the whole subject,
    and is why this is reported rather than asserted.

    Args:
        outputs: One shape's two products.

    Returns:
        True when every bit matches.
    """
    return bool(torch.equal(outputs["legacy"], outputs["epilogue"]))


def arm_identity(shape: GemmShape, device: str) -> tuple[tuple[float, float], ...]:
    """Describe both arms of one shape.

    Args:
        shape: The call to issue.
        device: Where to issue it.

    Returns:
        ``((legacy_digest, legacy_sum), (epilogue_digest, epilogue_sum))``.
    """
    outputs = arm_outputs(shape, device)
    return (describe_output(outputs["legacy"]), describe_output(outputs["epilogue"]))


__all__ = [
    "ARMS",
    "EPILOGUE_ARM",
    "LEGACY_ARM",
    "LM_HEAD_SHAPES",
    "ArmOutputs",
    "arm_identity",
    "arm_outputs",
    "arms_agree",
]
