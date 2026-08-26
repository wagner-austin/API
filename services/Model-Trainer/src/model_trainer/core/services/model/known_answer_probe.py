"""A cheap input with a known output, for checking an environment still computes.

An image that still builds is not an image that still computes what it used
to. :mod:`platform_core.known_answer` describes what a known answer IS and
what checking one can conclude; this module is an actual one, small enough to
run in the seconds before a job stages anything.

WHAT IT COMPUTES. A GPT-2 forward pass over a fixed token sequence, reported
as the language-modelling loss. Chosen over a synthetic matmul because it
exercises the kernels a real arm exercises -- attention, layernorm, the fused
paths torch selects per architecture -- which is where a stack change would
actually show. The weights come from the model's own deterministic init under
a fixed seed rather than from the hub, so the probe needs no network, no
HuggingFace cache and no bind beyond the image itself. That matters on a
compute node, which has none of those.

WHY THE SHAPE IS CONSTANT AND NOT A SET OF FLAGS. The number means nothing
apart from the input that produced it, and an input assembled from six flags
is one nobody can reproduce without also recovering the command line. The
constants below ARE the probe; changing one produces a different probe and
must change :const:`PROBE_LABEL` with it.

WHAT THIS PROBE CANNOT DO, WHICH IS THE PART WORTH READING. It is far too
small to detect a hardware difference. Measured 2026-08-25 in one
content-addressed image with determinism pinned, it returned
6.25127649307251 on all three of a Tesla V100 (sm_70), an A100 80GB (sm_80)
and an RTX 3090 Ti (sm_86) -- bit-identical across two GPU generations. On
the same two of those cards, full gpt2 scored over 2,627 real cloze items
agreed on every decision while NOT ONE item produced a bitwise-identical
score, differing by up to 1.2e-3.

Both of those runs had determinism pinned. Together they say that
``torch.use_deterministic_algorithms(True)`` makes a run reproduce ITSELF on
one card and does not make two cards agree, and that whether two cards agree
depends on the work -- this probe sits below the size where the disagreement
appears. So a probe that matches on a new card is evidence the STACK is
intact, and is not evidence the card reproduces a real workload. Reading it
as the latter is the failure this docstring exists to prevent. (The mechanism
is deliberately not claimed: the two runs differ in model size AND in input,
so attributing it to size alone would be inference rather than measurement.)
"""

from __future__ import annotations

PROBE_SEED = 42
PROBE_VOCAB_SIZE = 512
PROBE_SEQUENCE_LEN = 64
PROBE_EMBED_DIM = 128
PROBE_LAYERS = 2
PROBE_HEADS = 4

# Names the shape as well as the seed, because two probes differing only in
# width would otherwise register under one label and silently overwrite each
# other's expected value.
PROBE_LABEL = (
    f"gpt2-tiny-L{PROBE_LAYERS}-d{PROBE_EMBED_DIM}-"
    f"seed{PROBE_SEED}-len{PROBE_SEQUENCE_LEN}"
)

# Fixed rather than a flag. `experiment` is what makes two records comparable
# at all, so a probe run under a caller-supplied name could not be compared
# with the entry it was meant to check.
PROBE_EXPERIMENT = "environment-known-answer"

PROBE_OBSERVATION = "probe_loss"


def probe_forward_loss(device: str) -> float:
    """Compute the probe value on one device.

    Determinism is NOT pinned here. Pinning is a process-global side effect
    that must happen before any CUDA work -- ``CUBLAS_WORKSPACE_CONFIG`` is
    read once when the cuBLAS handle is created and a later assignment is
    accepted in silence -- so it belongs to the caller, ahead of this call.
    A pin inside this function would be too late for the handle that building
    the model creates.

    Args:
        device: Device to compute on, ``"cuda"`` or ``"cpu"``. Passed through
            to torch unexamined; an unusable device raises from torch, which
            names the problem better than a check here could.

    Returns:
        The language-modelling loss, as a float.

    Raises:
        ValueError: If the model returns no loss. It is asked for one by
            being given labels, so an absent loss means the model did not do
            what was asked and returning a fabricated number would be worse
            than stopping.
    """
    # Imported inside the function so that importing this module does not pull
    # torch into a process that only wanted to read the constants above.
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(PROBE_SEED)

    config = GPT2Config(
        vocab_size=PROBE_VOCAB_SIZE,
        n_positions=PROBE_SEQUENCE_LEN,
        n_embd=PROBE_EMBED_DIM,
        n_layer=PROBE_LAYERS,
        n_head=PROBE_HEADS,
    )
    model = GPT2LMHeadModel(config).to(device)
    model.eval()

    # A fixed, content-free sequence: the probe checks arithmetic, not
    # language, and real text would tie the answer to a tokenizer revision.
    ids = torch.arange(PROBE_SEQUENCE_LEN, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids=ids, labels=ids)

    loss = outputs.loss
    if loss is None:
        raise ValueError("Probe model returned no loss despite being given labels")
    return float(loss.item())


__all__ = [
    "PROBE_EMBED_DIM",
    "PROBE_EXPERIMENT",
    "PROBE_HEADS",
    "PROBE_LABEL",
    "PROBE_LAYERS",
    "PROBE_OBSERVATION",
    "PROBE_SEED",
    "PROBE_SEQUENCE_LEN",
    "PROBE_VOCAB_SIZE",
    "probe_forward_loss",
]
