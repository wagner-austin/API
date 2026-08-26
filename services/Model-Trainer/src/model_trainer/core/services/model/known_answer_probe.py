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

IT BUILDS THE MODEL THROUGH :func:`create_gpt2_model`, the same constructor
the gpt2 backend uses, at the shared ``"tiny"`` size. A second spelling of
"a small GPT-2" here would be a second definition free to drift from the one
the backend trains, and then the probe would be checking a stack nothing
else runs.

WHY THE SHAPE IS CONSTANT AND NOT A SET OF FLAGS. The number means nothing
apart from the input that produced it, and an input assembled from six flags
is one nobody can reproduce without also recovering the command line. The
constants below ARE the probe. :const:`PROBE_LABEL` is built FROM them,
including the dimensions read out of the shared size table, so a change to
either produces a new label rather than silently replacing the expected value
of a probe that no longer exists.

WHAT THIS PROBE CANNOT DO, WHICH IS THE PART WORTH READING. It is far too
small to detect a hardware difference. Measured 2026-08-25 in one
content-addressed image with determinism pinned, an earlier revision of this
probe returned an identical loss to all seventeen digits on each of a Tesla
V100 (sm_70), an A100 80GB (sm_80) and an RTX 3090 Ti (sm_86) -- across two
GPU generations. On two of those same cards, full gpt2 scored over 2,627 real
cloze items agreed on every decision while NOT ONE item produced a
bitwise-identical score, differing by up to 1.2e-3.

Both of those runs had determinism pinned. Together they say that
``torch.use_deterministic_algorithms(True)`` makes a run reproduce ITSELF on
one card and does not make two cards agree, and that whether two cards agree
depends on the work -- this probe sits below the size where the disagreement
appears. So a probe that matches on a new card is evidence the STACK is
intact, and is not evidence that card reproduces a real workload. Reading it
as the latter is the failure this docstring exists to prevent. (The mechanism
is deliberately not claimed: those two runs differ in model size AND in
input, so attributing it to size alone would be inference, not measurement.)
"""

from __future__ import annotations

import torch

from model_trainer.core.services.model.backends.gpt2.hf_gpt2 import create_gpt2_model
from model_trainer.core.services.model.model_sizes import GPT2_MODEL_SIZES

PROBE_SEED = 42
PROBE_VOCAB_SIZE = 512
PROBE_SEQUENCE_LEN = 64

# The shared size, not a local set of dimensions. "tiny" is 128 hidden over 2
# heads, which keeps the 64-wide head dimension every larger size uses, so the
# attention shapes stay representative instead of degenerate.
PROBE_MODEL_SIZE = "tiny"

_PROBE_DIMS = GPT2_MODEL_SIZES[PROBE_MODEL_SIZE]

# Every axis that changes the number appears in the label. A probe re-widened
# without renaming would otherwise register under this name and overwrite an
# expected value it cannot reproduce; built this way, it cannot.
PROBE_LABEL = (
    f"gpt2-{PROBE_MODEL_SIZE}"
    f"-L{_PROBE_DIMS['n_layer']}"
    f"-d{_PROBE_DIMS['hidden_size']}"
    f"-h{_PROBE_DIMS['n_head']}"
    f"-v{PROBE_VOCAB_SIZE}"
    f"-len{PROBE_SEQUENCE_LEN}"
    f"-seed{PROBE_SEED}"
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
    read once when the cuBLAS handle is created, and constructing a model on
    cuda is enough to create it -- so it belongs to the caller, ahead of this
    call. A pin inside this function would already be too late.

    Args:
        device: Device to compute on, ``"cuda"`` or ``"cpu"``. Passed to torch
            unexamined; an unusable device raises from torch, which names the
            problem better than a check here could.

    Returns:
        The language-modelling loss, as a float.
    """
    torch.manual_seed(PROBE_SEED)

    model = create_gpt2_model(
        vocab_size=PROBE_VOCAB_SIZE,
        max_seq_len=PROBE_SEQUENCE_LEN,
        model_size=PROBE_MODEL_SIZE,
    ).to(device)
    model.eval()

    # A fixed, content-free sequence: the probe checks arithmetic, not
    # language, and real text would tie the answer to a tokenizer revision.
    ids = torch.arange(PROBE_SEQUENCE_LEN, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        outputs = model.forward(input_ids=ids, labels=ids)

    return float(outputs.loss.item())


__all__ = [
    "PROBE_EXPERIMENT",
    "PROBE_LABEL",
    "PROBE_MODEL_SIZE",
    "PROBE_OBSERVATION",
    "PROBE_SEED",
    "PROBE_SEQUENCE_LEN",
    "PROBE_VOCAB_SIZE",
    "probe_forward_loss",
]
