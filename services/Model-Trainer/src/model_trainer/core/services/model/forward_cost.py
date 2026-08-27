"""Time a whole GPT-2 forward pass, with and without the attention pin.

The number the per-call benchmark cannot give. :mod:`sdpa_timing` measured
attention alone and found pinning ``SDPBackend.MATH`` costs 4-7x. A forward
pass is not one attention call: it is the QKV and output projections, the
MLP, layernorm, GELU, the output projection over the vocabulary and the loss,
plus one attention call per layer. So the end-to-end multiplier is the
per-call one weighted by attention's share of the pass, and only measuring it
says what that share is.

WHY THE SHAPES ARE NOT THE LADDER'S. The ladder runs one 64-token sequence at
a 512-token vocabulary, which is a probe and not a workload -- and this
investigation has already had to retract one cost table for being measured in
a regime nobody trains in. These shapes carry the real GPT-2 vocabulary
(50,257) at batches and lengths a training run would use. The gate rung is
kept as ONE row so the probe's own regime has an end-to-end number too, and
it is labelled as the probe it is.

WHY THE VOCABULARY MATTERS ENOUGH TO SAY TWICE. The output projection is
``batch x sequence x hidden x vocab``, so a 512-token vocabulary makes it a
rounding error and a 50,257-token one makes it one of the largest matmuls in
the pass. Attention's SHARE -- which is what the end-to-end multiplier
measures -- moves with that choice. A table that did not say which vocabulary
it used would not be reproducible.

WHY FORWARD ONLY. The correctness result it prices is about forward passes,
and the backward pass through the math backend is a different computation
again. Extending to training is a separate measurement, not an extrapolation.
"""

from __future__ import annotations

from typing import Final

import torch
from torch.nn.attention import SDPBackend
from typing_extensions import TypedDict

from model_trainer.core.services.model.backends.gpt2.hf_gpt2 import create_gpt2_model
from model_trainer.core.services.model.gemm_timing import synchroniser
from model_trainer.core.services.model.probe_shapes import PROBE_SEED
from model_trainer.core.services.model.timing_harness import (
    MeasuredCost,
    backend_context,
    time_calls,
    timed_or_unfitted,
)
from model_trainer.core.types import TracedLMModelProto

#: Calls discarded before measuring. Two rather than the per-call
#: benchmark's three: a forward pass at these sizes takes milliseconds to
#: seconds, so warmup is expensive, and one pass is already enough to pay for
#: kernel selection across every layer.
FORWARD_WARMUP = 2

#: Passes per timed batch. ONE, against the per-call benchmark's twenty, and
#: the reason is the same reason that one is twenty: batching exists to
#: amortise a ~10 microsecond launch against a ~20 microsecond call. A
#: forward pass issues hundreds of launches internally and has already
#: amortised them, so batching would only multiply the wall clock.
FORWARD_INNER = 1

#: Timed batches. The median is reported.
FORWARD_BATCHES = 5

#: The real GPT-2 vocabulary. Used by every workload row, because the output
#: projection scales with it and attention's share of the pass moves with it.
GPT2_VOCAB = 50257


class ForwardCostShape(TypedDict):
    """One forward pass to time.

    Attributes:
        name: What to call it in a record.
        model_size: A key of :data:`~model_sizes.GPT2_MODEL_SIZES`.
        batch: Sequences in the batch.
        sequence_len: Tokens per sequence, and the model's ``n_positions``.
        vocab_size: Vocabulary the model predicts over.
    """

    name: str
    model_size: str
    batch: int
    sequence_len: int
    vocab_size: int


#: The sweep. Batch times length is held roughly constant within a size so
#: the length axis is not confounded with total work, and the sizes descend
#: in batch as they grow so each row stays inside a 16 GB card.
FORWARD_SHAPES: Final[tuple[ForwardCostShape, ...]] = (
    {
        "name": "small-b8-s128",
        "model_size": "small",
        "batch": 8,
        "sequence_len": 128,
        "vocab_size": GPT2_VOCAB,
    },
    {
        "name": "small-b8-s512",
        "model_size": "small",
        "batch": 8,
        "sequence_len": 512,
        "vocab_size": GPT2_VOCAB,
    },
    {
        "name": "small-b4-s1024",
        "model_size": "small",
        "batch": 4,
        "sequence_len": 1024,
        "vocab_size": GPT2_VOCAB,
    },
    {
        "name": "medium-b4-s512",
        "model_size": "medium",
        "batch": 4,
        "sequence_len": 512,
        "vocab_size": GPT2_VOCAB,
    },
    {
        "name": "medium-b2-s1024",
        "model_size": "medium",
        "batch": 2,
        "sequence_len": 1024,
        "vocab_size": GPT2_VOCAB,
    },
    {
        "name": "large-b2-s1024",
        "model_size": "large",
        "batch": 2,
        "sequence_len": 1024,
        "vocab_size": GPT2_VOCAB,
    },
    # The probe's own configuration, kept as one row so the ladder's regime
    # has an end-to-end number too. Labelled `gate` rather than given a
    # workload name, because it is not one.
    {
        "name": "gate-tiny-b1-s64",
        "model_size": "tiny",
        "batch": 1,
        "sequence_len": 64,
        "vocab_size": 512,
    },
)


def forward_model_and_input(
    shape: ForwardCostShape, device: str
) -> tuple[TracedLMModelProto, torch.Tensor]:
    """Build one row's model and its batch of token ids.

    Built through :func:`create_gpt2_model`, the same constructor the ladder,
    the trace and the gpt2 backend use. A second spelling of "a GPT-2" here
    would be a second definition free to drift from the one everything else
    measures.

    Args:
        shape: The row to build.
        device: Where to build it.

    Returns:
        ``(model in eval mode on device, input_ids of shape
        [batch, sequence_len])``.
    """
    torch.manual_seed(PROBE_SEED)
    model = create_gpt2_model(
        vocab_size=shape["vocab_size"],
        max_seq_len=shape["sequence_len"],
        model_size=shape["model_size"],
    ).to(device)
    model.eval()

    # Content-free ids, as the probe uses: this measures time, and real text
    # would tie the number to a tokenizer revision. Taken modulo the
    # vocabulary so a sequence longer than it still indexes real tokens --
    # the probe refuses that case instead, because there the VALUES matter.
    ids = torch.arange(shape["batch"] * shape["sequence_len"], dtype=torch.long, device=device)
    return model, (ids % shape["vocab_size"]).view(shape["batch"], shape["sequence_len"])


def release_row() -> None:
    """Hand a finished row's memory back before the next one is built.

    CALLED BETWEEN ROWS BECAUSE NOT CALLING IT CORRUPTED THE MEASUREMENT.
    Measured 2026-08-27 on an RTX 3090 Ti sharing the card with another
    process: sweeping the rows in sequence reported `small-b8-s512` at 436 ms
    under the dispatcher's own choice and 82 ms under the pinned math
    backend -- the pin apparently five times FASTER, which is not a thing
    that happens. Timing the same row in isolation gave 66.5 ms and 79.1 ms,
    stable to a millisecond over repeats in both orders.

    The difference is that during the sweep the PREVIOUS row's model was
    still referenced while the next one was being built, so two models were
    resident at once and the allocator thrashed through the first arm of the
    new row. Whichever arm ran first wore it.

    Safe to call unconditionally: ``empty_cache`` does nothing when CUDA was
    never initialised, so a cpu run needs no branch here and no branch means
    no arm a cpu test cannot reach.
    """
    torch.cuda.empty_cache()


def measure_forward(
    model: TracedLMModelProto, ids: torch.Tensor, device: str, backend: SDPBackend | None
) -> MeasuredCost:
    """Time one forward pass and read its peak allocation.

    Takes a built model rather than a shape so BOTH arms time the same
    weights: rebuilding between them would put a fresh random init into the
    comparison, and would pay for a 774-million-parameter construction twice.

    PEAK ALLOCATION HERE INCLUDES THE MODEL, which the per-call attention
    benchmark's did not. That makes the memory RATIO an understatement of the
    activation growth -- the weights sit in both arms and dilute it -- and it
    is the right thing to report anyway, because what a card has to hold is
    the whole pass and not the attention scores alone.

    Args:
        model: The model to run, already in eval mode on ``device``.
        ids: The input token ids.
        device: Device being timed.
        backend: The attention backend to force, or None for the
            dispatcher's choice.

    Returns:
        The cost.

    Raises:
        torch.cuda.OutOfMemoryError: When the device cannot hold the pass.
            Caught by :func:`~timing_harness.timed_or_unfitted`.
    """

    def run() -> None:
        with torch.no_grad():
            model.forward(input_ids=ids, labels=ids)

    with backend_context(backend):
        return time_calls(
            run,
            synchroniser(device),
            device,
            FORWARD_WARMUP,
            FORWARD_INNER,
            FORWARD_BATCHES,
        )


def time_forward(
    model: TracedLMModelProto, ids: torch.Tensor, device: str, backend: SDPBackend | None
) -> MeasuredCost | None:
    """Measure seconds and peak memory for one forward pass.

    Args:
        model: The model to run.
        ids: The input token ids.
        device: Device being timed.
        backend: The attention backend to force, or None.

    Returns:
        The cost, or None when the pass did not fit in device memory.
    """

    def run() -> MeasuredCost:
        return measure_forward(model, ids, device, backend)

    return timed_or_unfitted(run)


__all__ = [
    "FORWARD_BATCHES",
    "FORWARD_INNER",
    "FORWARD_SHAPES",
    "FORWARD_WARMUP",
    "GPT2_VOCAB",
    "ForwardCostShape",
    "forward_model_and_input",
    "measure_forward",
    "release_row",
    "time_forward",
]
