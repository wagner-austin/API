"""Which probes exist, what identifies them, and how they ladder.

The shapes, not the arithmetic. :mod:`known_answer_probe` runs one of these;
this module says what there is to run and what each result is called.

WHY THE TWO ARE SEPARATE MODULES. Running a probe needs torch. Naming one
does not, and three callers only ever name them: the ladder report, which
reads finished records on a laptop; the registry gate, which compares stored
numbers; and ``cli/_test_hooks``, which is written to keep torch out of a
process that only wanted to parse a command line and print a usage error.
Importing a 2 GB stack to format a label is a cost with nothing behind it.

WHY A TABLE OF SHAPES AND NOT A SET OF FLAGS. A probe's number means nothing
apart from the input that produced it, and an input assembled from six flags
is one nobody can reproduce without also recovering the command line. So a
caller names a RUNG -- ``"tiny"``, ``"small"``, ``"tiny-len512"`` -- and the
rung is a constant. :func:`probe_label` builds each rung's label FROM its
shape, including dimensions read out of the shared size table, so a change to
either produces a new label rather than silently replacing the expected value
of a probe that no longer exists.

WHY THERE IS A LADDER AT ALL, WHICH IS THE PART WORTH READING. The gate rung
is far too small to detect a hardware difference. Measured 2026-08-25 in one
content-addressed image with determinism pinned, an earlier revision returned
an identical loss on each of a Tesla V100 (sm_70), an A100 80GB (sm_80) and an
RTX 3090 Ti (sm_86) -- across two GPU generations, agreeing to every digit a
double carries. On two of those same cards, full gpt2 scored over 2,627 real
cloze items agreed on every decision while NOT ONE item produced a
bitwise-identical score, differing by up to 1.2e-3.

Both runs had determinism pinned. Together they say that
``torch.use_deterministic_algorithms(True)`` makes a run reproduce ITSELF on
one card and does not make two cards agree, and that whether two cards agree
depends on the work. So a probe that matches on a new card is evidence the
STACK is intact, and is not evidence that card reproduces a real workload.
Reading it as the latter is the failure this docstring exists to prevent.

Those two runs differ in model size AND in input, so attributing the
disagreement to size alone would be inference rather than measurement. The
ladder exists to stop inferring it: :data:`PROBE_SHAPES` varies ONE axis at a
time from the gate rung, so a rung where agreement breaks names the axis that
broke it.
"""

from __future__ import annotations

from typing import Final

from typing_extensions import TypedDict

from model_trainer.core.services.model.model_sizes import GPT2_MODEL_SIZES

PROBE_SEED = 42
"""Seed for every rung.

Shared rather than per-rung because varying it answers nothing this module
asks. Two seeds give two unrelated models; the ladder is about whether one
model's arithmetic survives a change of card, so the seed is held fixed and
the shape is what moves.
"""


class ProbeShape(TypedDict):
    """One rung: a complete, reproducible probe input.

    Attributes:
        model_size: A key of :data:`GPT2_MODEL_SIZES`. The dimensions are read
            from there rather than restated, so a reshape there renames the
            rung instead of redefining it.
        sequence_len: Tokens in the input, and the model's ``n_positions``.
        vocab_size: Vocabulary the model predicts over. Must be at least
            ``sequence_len``; see
            :func:`~model_trainer.core.services.model.known_answer_probe.probe_forward_loss`.
    """

    model_size: str
    sequence_len: int
    vocab_size: int


class ProbeAxis(TypedDict):
    """One direction the ladder walks away from the gate rung.

    Attributes:
        name: What the axis varies.
        rungs: Rung names in increasing order, starting at :data:`GATE_RUNG`.
            Both axes start there on purpose: it is the one rung whose
            cross-card behaviour is already measured, so it is the origin
            every other rung is read against.
    """

    name: str
    rungs: tuple[str, ...]


#: The rung the environment gate runs, and the only rung ever registered as a
#: known answer. Named separately from the table because the gate's identity
#: is a decision, not a table position: the gate must stay cheap enough to run
#: before a job stages anything, and a larger rung being added must not
#: silently promote itself into that role.
GATE_RUNG = "tiny"

# The ladder. Every rung differs from the gate rung on exactly ONE axis, which
# is what lets a break be attributed. Reading down the first group, model size
# grows at the gate's length; reading down the second, length grows at the
# gate's size.
#
# `sequence_len` never exceeds `vocab_size`, which keeps the input the identity
# `arange` at every rung rather than a wrapped one -- see `probe_forward_loss`.
# That is why the length group stops at 512.
PROBE_SHAPES: Final[dict[str, ProbeShape]] = {
    "tiny": {"model_size": "tiny", "sequence_len": 64, "vocab_size": 512},
    "small": {"model_size": "small", "sequence_len": 64, "vocab_size": 512},
    "medium": {"model_size": "medium", "sequence_len": 64, "vocab_size": 512},
    "large": {"model_size": "large", "sequence_len": 64, "vocab_size": 512},
    "xl": {"model_size": "xl", "sequence_len": 64, "vocab_size": 512},
    "tiny-len128": {"model_size": "tiny", "sequence_len": 128, "vocab_size": 512},
    "tiny-len256": {"model_size": "tiny", "sequence_len": 256, "vocab_size": 512},
    "tiny-len512": {"model_size": "tiny", "sequence_len": 512, "vocab_size": 512},
}

#: The axes, and the order to read each one in. Declared rather than inferred
#: from the rung names, because "these rungs vary only model size" is a claim
#: about the shapes and is worth failing a test over -- the suite checks each
#: axis against :data:`PROBE_SHAPES` field by field, so a rung that quietly
#: moved two axes at once cannot sit here describing itself as one.
#:
#: ONE FIELD IS NOT ONE ARCHITECTURAL DIMENSION, and the size axis is where
#: that bites. ``model_size`` names three coupled numbers in
#: :data:`GPT2_MODEL_SIZES` -- ``medium`` to ``large`` moves hidden size
#: 1024->1280, layers 24->36 AND heads 16->20 together. So a break on this
#: axis locates the threshold on the size ladder and does NOT say whether
#: depth, width or head count carried it across.
#:
#: Separating them would need rungs the shared size table does not contain,
#: and adding probe-only dimensions is exactly what this module refuses: the
#: probe builds through the same constructor the gpt2 backend trains, so a
#: shape that exists only here would be a second definition free to drift
#: from the one anything else runs. The sequence-length axis has no such
#: problem, because ``sequence_len`` is one number.
PROBE_AXES: Final[tuple[ProbeAxis, ...]] = (
    {"name": "model-size", "rungs": ("tiny", "small", "medium", "large", "xl")},
    {"name": "sequence-length", "rungs": ("tiny", "tiny-len128", "tiny-len256", "tiny-len512")},
)

#: The field each axis is allowed to move. Paired with :data:`PROBE_AXES` by
#: name so the suite can check every rung on an axis against its origin: an
#: axis named here with no entry, or an entry naming no axis, fails rather
#: than quietly exempting itself from the one-axis-at-a-time rule.
PROBE_AXIS_FIELDS: Final[dict[str, str]] = {
    "model-size": "model_size",
    "sequence-length": "sequence_len",
}

# Fixed rather than a flag. `experiment` is what makes two records comparable
# at all, so a probe run under a caller-supplied name could not be compared
# with the entry it was meant to check.
PROBE_EXPERIMENT = "environment-known-answer"

PROBE_OBSERVATION = "probe_loss"


#: Every field a shape carries. Stated rather than derived so that
#: :func:`differing_shape_fields` and this constant can be checked against
#: ``ProbeShape`` itself by the suite: a fourth field added to the TypedDict
#: fails that check until the comparison learns to look at it, which is what
#: keeps a silently-ignored axis from existing.
SHAPE_FIELDS: Final[tuple[str, ...]] = ("model_size", "sequence_len", "vocab_size")


def differing_shape_fields(left: ProbeShape, right: ProbeShape) -> tuple[str, ...]:
    """Name the fields on which two shapes disagree.

    The enforcement primitive behind :data:`PROBE_AXES`: an axis claims its
    rungs move one field, and this is what makes the claim checkable rather
    than a comment.

    Written as explicit lookups rather than a loop over :data:`SHAPE_FIELDS`,
    for the same reason
    :func:`~platform_core.known_answer_registry.incomplete_axes` is. Indexing
    a TypedDict with a variable is not something the type checker can verify,
    and the only ways to write the loop are a ``type: ignore`` or a cast --
    both of which trade a checked access for an unchecked one to save two
    lines.

    Args:
        left: One shape.
        right: The other.

    Returns:
        The differing field names, in :data:`SHAPE_FIELDS` order; empty when
        the two shapes are identical.
    """
    differing: list[str] = []
    if left["model_size"] != right["model_size"]:
        differing.append("model_size")
    if left["sequence_len"] != right["sequence_len"]:
        differing.append("sequence_len")
    if left["vocab_size"] != right["vocab_size"]:
        differing.append("vocab_size")
    return tuple(differing)


def require_probe_shape(rung: str) -> ProbeShape:
    """Look up a rung by name.

    Args:
        rung: The rung name, a key of :data:`PROBE_SHAPES`.

    Returns:
        That rung's shape.

    Raises:
        KeyError: If no such rung exists, naming the ones that do. A bare dict
            index would raise the same class carrying only the missing key,
            and the answer to a mistyped rung is nearly always the list.
    """
    shape = PROBE_SHAPES.get(rung)
    if shape is None:
        raise KeyError(f"unknown probe rung {rung!r}; known rungs: {', '.join(PROBE_SHAPES)}")
    return shape


def probe_label(shape: ProbeShape) -> str:
    """Build the label that identifies a shape's measurement.

    Every axis that changes the number appears in the label. A probe
    re-widened without renaming would otherwise register under an existing
    name and overwrite an expected value it cannot reproduce; built this way,
    it cannot.

    The rung NAME is deliberately absent. Two rung names for one shape must
    not produce two labels: they would name one number twice, and a registry
    could then hold two entries that can never disagree.

    Args:
        shape: The rung to name.

    Returns:
        The label, e.g. ``gpt2-tiny-L2-d128-h2-v512-len64-seed42``.

    Raises:
        KeyError: If the shape names a model size the shared table lacks.
    """
    dims = GPT2_MODEL_SIZES[shape["model_size"]]
    return (
        f"gpt2-{shape['model_size']}"
        f"-L{dims['n_layer']}"
        f"-d{dims['hidden_size']}"
        f"-h{dims['n_head']}"
        f"-v{shape['vocab_size']}"
        f"-len{shape['sequence_len']}"
        f"-seed{PROBE_SEED}"
    )


#: The gate rung's label. Eight entries in the deployed registry carry this
#: exact string, so the suite asserts it as a literal: a refactor that renames
#: the gate rung strands every one of them as an answer to a probe that no
#: longer exists.
PROBE_LABEL = probe_label(PROBE_SHAPES[GATE_RUNG])


__all__ = [
    "GATE_RUNG",
    "PROBE_AXES",
    "PROBE_AXIS_FIELDS",
    "PROBE_EXPERIMENT",
    "PROBE_LABEL",
    "PROBE_OBSERVATION",
    "PROBE_SEED",
    "PROBE_SHAPES",
    "SHAPE_FIELDS",
    "ProbeAxis",
    "ProbeShape",
    "differing_shape_fields",
    "probe_label",
    "require_probe_shape",
]
