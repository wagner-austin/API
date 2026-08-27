"""The arithmetic a probe rung runs.

:mod:`probe_shapes` says which probes exist and what each result is called;
this runs one. The split keeps torch out of the callers that only ever name a
probe -- see that module's docstring for why that is worth a second file.

WHAT IT COMPUTES. A GPT-2 forward pass over a fixed token sequence, reported
as the language-modelling loss. Chosen over a synthetic matmul because it
exercises the kernels a real arm exercises -- attention, layernorm, the fused
paths torch selects per architecture -- which is where a stack change would
actually show. The weights come from the model's own deterministic init under
a fixed seed rather than from the hub, so a probe needs no network, no
HuggingFace cache and no bind beyond the image itself. That matters on a
compute node, which has none of those.

IT BUILDS MODELS THROUGH :func:`create_gpt2_model`, the same constructor the
gpt2 backend uses, at the shared sizes. A second spelling of "a small GPT-2"
here would be a second definition free to drift from the one the backend
trains, and then the probe would be checking a stack nothing else runs.
"""

from __future__ import annotations

import torch

from model_trainer.core.services.model.backends.gpt2.hf_gpt2 import create_gpt2_model
from model_trainer.core.services.model.probe_shapes import PROBE_SEED, ProbeShape
from model_trainer.core.types import TracedLMModelProto


def probe_model_and_input(
    device: str, shape: ProbeShape
) -> tuple[TracedLMModelProto, torch.Tensor]:
    """Build one rung's model and its input, ready to run.

    Split out of :func:`probe_forward_loss` so that the forward TRACE runs
    the same model on the same tokens under the same seed. A second builder
    beside this one would be a second definition of what a rung is, and the
    trace's business is to explain a number this function's caller produced --
    which it cannot do if the two set up differently.

    Args:
        device: Device to build on, ``"cuda"`` or ``"cpu"``. Passed to torch
            unexamined; an unusable device raises from torch, which names the
            problem better than a check here could.
        shape: The rung to build.

    Returns:
        ``(model, input_ids)``. The model is in eval mode and on ``device``;
        the input is the identity ``arange`` of the rung's length, batched to
        one sequence, on the same device.

    Raises:
        ValueError: If the shape's sequence is longer than its vocabulary. The
            input below is the identity ``arange``, so such a shape would
            index tokens the embedding does not have. Wrapping them with a
            modulo would be the alternative and is worse: it would silently
            change what "the input" means for long rungs while leaving short
            ones alone, and the length axis would stop being one axis.
    """
    if shape["sequence_len"] > shape["vocab_size"]:
        raise ValueError(
            f"probe sequence_len {shape['sequence_len']} exceeds "
            f"vocab_size {shape['vocab_size']}; the input would index absent tokens"
        )

    torch.manual_seed(PROBE_SEED)

    model = create_gpt2_model(
        vocab_size=shape["vocab_size"],
        max_seq_len=shape["sequence_len"],
        model_size=shape["model_size"],
    ).to(device)
    model.eval()

    # A fixed, content-free sequence: the probe checks arithmetic, not
    # language, and real text would tie the answer to a tokenizer revision.
    ids = torch.arange(shape["sequence_len"], dtype=torch.long, device=device).unsqueeze(0)
    return model, ids


def probe_forward_loss(device: str, shape: ProbeShape) -> float:
    """Compute one rung's value on one device.

    Determinism is NOT pinned here. Pinning is a process-global side effect
    that must happen before any CUDA work -- ``CUBLAS_WORKSPACE_CONFIG`` is
    read once when the cuBLAS handle is created, and constructing a model on
    cuda is enough to create it -- so it belongs to the caller, ahead of this
    call. A pin inside this function would already be too late.

    Args:
        device: Device to compute on, ``"cuda"`` or ``"cpu"``. Passed to torch
            unexamined; an unusable device raises from torch, which names the
            problem better than a check here could.
        shape: The rung to run.

    Returns:
        The language-modelling loss, as a float.

    Raises:
        ValueError: Propagated from :func:`probe_model_and_input` when the
            shape's sequence is longer than its vocabulary.
    """
    model, ids = probe_model_and_input(device, shape)

    with torch.no_grad():
        outputs = model.forward(input_ids=ids, labels=ids)

    return float(outputs.loss.item())


__all__ = ["probe_forward_loss", "probe_model_and_input"]
