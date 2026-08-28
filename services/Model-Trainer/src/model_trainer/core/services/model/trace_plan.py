"""Which rungs a forward trace walks, and what each traced tensor is called.

The names, not the arithmetic. :mod:`forward_trace` runs a trace; this says
what there is to trace and how to read a finished record back. Separated from
it for the reason :mod:`probe_shapes` is separated from
:mod:`known_answer_probe`: naming a measurement must not require importing
torch, and the report command that reads these records runs on a laptop.

WHAT QUESTION THE TRACE ANSWERS. Removing cuBLASLt's split-K makes three
cards agree on all eight ISOLATED matmuls. It does not make them agree on the
model those matmuls came from: measured 2026-08-27 in image
``b002cffc``, driver 580.82.07, on a V100, an A30 and an A100 --

    rung           default                  CUBLASLT_WORKSPACE_SIZE=0
    tiny           all three agree          V100 alone, Ampere pair moved
    small          all three agree          all three agree
    medium         all three agree          all three agree
    large          A100 alone               all three agree
    xl             V100 alone               A30 alone
    tiny-len128    all three agree          all three agree
    tiny-len256    all three agree          all three agree
    tiny-len512    V100 alone               all three agree

Two rungs move on each card and it is a different pair on every card. So a
loss is too coarse to attribute: it is one number at the end of thousands of
kernels, and by the time it differs, everything about where has been summed
away. The trace digests every tensor that crosses a module boundary, in
execution order, so the FIRST differing one names the operation rather than
the model.

WHICH RUNGS, AND WHY THESE FOUR. :data:`TRACE_RUNGS` is a contrast, not a
sample. ``tiny`` is the rung that removing split-K BREAKS, and the one eight
registered known answers are keyed on. ``xl`` is the rung it fails to fix --
though what it does there is relocate the disagreement rather than leave it
alone, since the odd card is the V100 by default and the A30 with split-K
removed. ``large`` is the rung it DOES fix, so a mechanism proposed for the
other two has to explain why it does not fire here. ``medium`` is the rung
whose LOSS agrees on all three cards under both conditions.

``medium`` is emphatically NOT a control that must show no divergence. The
isolated-matmul work already found six of eight shapes producing three
different tensors on three cards INCLUDING at rungs whose reported loss is
bit-identical -- so tensors diverging inside a rung whose loss agrees is the
expected case, not an instrument fault. What ``medium`` is for is the
opposite reading: if its internals diverge and its loss does not, that
carries the "a reported threshold is where an existing disagreement grows
past one float32 ULP" result up from a single matmul to a whole model, on a
rung where the loss cannot show it.

Rungs are named rather than shapes restated, so a reshaped rung in
:data:`~model_trainer.core.services.model.probe_shapes.PROBE_SHAPES` changes
what this traces instead of silently tracing a shape that no longer exists.
"""

from __future__ import annotations

import hashlib
from typing import Final

from typing_extensions import TypedDict

#: Distinct from the ladder's and the gemm probe's. Two records may only be
#: differenced when they answer the same question, and "which tensor first
#: differs" is not "where does agreement break".
TRACE_EXPERIMENT = "forward-trace-attribution"

#: The rungs one trace walks, in order. See the module docstring for why
#: these four and not others.
TRACE_RUNGS: Final[tuple[str, ...]] = ("tiny", "medium", "large", "xl")

#: A tensor entering a module.
INPUT_KIND = "in"

#: A tensor leaving one.
OUTPUT_KIND = "out"

#: Suffix for the observation carrying a tensor's identity.
DIGEST_SUFFIX = "digest48"

#: Suffix for the observation carrying its magnitude.
SUM_SUFFIX = "sum"

#: What the rung's own reported loss is called. Recorded alongside the
#: tensors as the trace's CONTROL: it must equal the value the ladder
#: recorded for that rung on that card under that condition, and if it does
#: not, the hooks changed the arithmetic and nothing else in the record can
#: be read.
LOSS_NAME = "loss"

#: What the split-K condition is called in a record.
#:
#: WHY THIS EXISTS AT ALL, which is worth the paragraph. The condition under
#: study is ``CUBLASLT_WORKSPACE_SIZE``, and NOTHING in a
#: :class:`~platform_core.comparability.RunFingerprint` carries it -- the
#: determinism record holds ``CUBLAS_WORKSPACE_CONFIG``, a different
#: variable. So two runs differing ONLY in the variable the experiment
#: manipulates have byte-identical fingerprints, and
#: :func:`~platform_core.comparability.find_differences` reports them as
#: "identical configuration". The six ladder records from 2026-08-27 have
#: exactly that defect: which condition produced which is recoverable only
#: from the filename and the run document beside it, which is a check a
#: person has to remember to perform by eye.
#:
#: WHY AN OBSERVATION RATHER THAN A FINGERPRINT AXIS. Adding an axis changes
#: every fingerprint ever written, and all eight entries in the deployed
#: known-answer registry would then report ``configuration_differs`` against
#: every future probe -- the same cost
#: :func:`~model_trainer.cli.known_answer_probe.probe_determinism` declines
#: to pay for a thread count. An observation is free, is compared by
#: :func:`~platform_core.run_record.agree_across_runs` like any other number,
#: and makes a record say what it ran under.
#:
#: AND WHY IT STAYS ONE, NOW THAT THE DETERMINISM RECORD CARRIES THE VARIABLE
#: TOO. As of the split-K wiring, a run that REMOVED split-K says so in its
#: determinism record -- see
#: :data:`platform_ml.determinism.SPLIT_K_SETTING`. That did not make this
#: observation redundant, because the two report different facts. The record
#: says what the process ASKED FOR; this says what the environment ACTUALLY
#: HELD, read back. They come apart in the two cases that matter: a launcher
#: that exported the variable itself (which is how every arm of this
#: experiment is run, and which the record correctly does not claim credit
#: for), and a pin that arrived too late because something had already
#: created the cuBLASLt handle -- accepted in silence, and invisible to any
#: record built from intent. Request and actual are a pair, and dropping
#: either half is how a run comes to be described by what it meant to do.
WORKSPACE_NAME = "cublaslt_workspace_size"

#: The value :data:`WORKSPACE_NAME` carries when the variable is not set.
#:
#: A negative workspace size is not a setting cuBLASLt can be given, so this
#: cannot collide with a real one -- and in particular it cannot be confused
#: with ``0``, which is the whole intervention.
WORKSPACE_UNSET = -1.0

#: Zero-padding for the execution counter. A record sorts its observations by
#: name, so the counter has to sort as text in the order it ran -- five digits
#: covers 100,000 hook calls, and the largest rung here makes about 1,200.
STEP_DIGITS = 5

#: Field separator inside an observation name. A module path is dotted and a
#: class name is an identifier, so neither can contain this.
FIELD_SEPARATOR = "|"

#: How many fields a traced-tensor name has.
_TENSOR_FIELDS = 7


class TraceName(TypedDict):
    """One traced tensor's observation name, taken apart.

    Attributes:
        rung: Which rung produced it.
        step: The hook call that recorded it, counting from zero in execution
            order across the whole rung.
        kind: :data:`INPUT_KIND` or :data:`OUTPUT_KIND`.
        index: Which tensor of that hook call, for a module returning several.
        module_class: The class whose instance produced it, e.g. ``Conv1D``.
            In the name rather than only in a comment because it is the
            answer: two cards running different attention classes would show
            up here as observations that do not pair, rather than as agreement
            over the ones that happen to match.
        path: The module's dotted path, e.g. ``transformer.h.0.attn.c_proj``.
        suffix: :data:`DIGEST_SUFFIX` or :data:`SUM_SUFFIX`.
    """

    rung: str
    step: int
    kind: str
    index: int
    module_class: str
    path: str
    suffix: str


def trace_tensor_name(name: TraceName) -> str:
    """Build the observation name for one traced tensor.

    Args:
        name: The fields to render.

    Returns:
        e.g. ``tiny|00042|out|0|Conv1D|transformer.h.0.attn.c_attn|digest48``.
    """
    return FIELD_SEPARATOR.join(
        (
            name["rung"],
            f"{name['step']:0{STEP_DIGITS}d}",
            name["kind"],
            str(name["index"]),
            name["module_class"],
            name["path"],
            name["suffix"],
        )
    )


def trace_loss_name(rung: str) -> str:
    """Name a rung's reported loss.

    Two fields rather than seven, so :func:`parse_trace_name` can tell a loss
    from a tensor by shape alone.

    Args:
        rung: The rung.

    Returns:
        e.g. ``tiny|loss``.
    """
    return f"{rung}{FIELD_SEPARATOR}{LOSS_NAME}"


def parse_trace_name(name: str) -> TraceName | None:
    """Read a traced-tensor name back into its fields.

    Args:
        name: An observation name from a trace record.

    Returns:
        The fields, or None when the name is not a traced tensor -- which is
        how a loss observation, and anything else a future record carries, is
        skipped by a reader walking tensors.
    """
    fields = name.split(FIELD_SEPARATOR)
    if len(fields) != _TENSOR_FIELDS:
        return None
    rung, step, kind, index, module_class, path, suffix = fields
    if not step.isdigit() or not index.isdigit():
        return None
    return TraceName(
        rung=rung,
        step=int(step),
        kind=kind,
        index=int(index),
        module_class=module_class,
        path=path,
        suffix=suffix,
    )


def trace_label(rungs: tuple[str, ...]) -> str:
    """Build the label identifying a trace by the rungs it walked.

    Derived rather than a version constant someone must remember to bump, for
    the reason
    :func:`~model_trainer.cli.probe_ladder.ladder_label` is: a rung added,
    removed or reordered produces a different label, so two records that
    traced different rungs can never be mistaken for two runs of one trace.

    Args:
        rungs: The rung names, in the order they were walked.

    Returns:
        The label, e.g. ``forward-trace-4x1a2b3c4d5e6f``.

    Raises:
        ValueError: If a rung appears twice. Every observation is prefixed by
            its rung, so a repeated rung would collide with itself and the
            record would be refused later, further from the cause.
    """
    duplicated = sorted({rung for rung in rungs if rungs.count(rung) > 1})
    if duplicated:
        raise ValueError(f"a trace cannot walk one rung twice: {duplicated}")
    digest = hashlib.sha256(FIELD_SEPARATOR.join(rungs).encode("utf-8")).hexdigest()
    return f"forward-trace-{len(rungs)}x{digest[:12]}"


def describe_workspace(value: float) -> str:
    """Render the split-K condition a record ran under, for a log line.

    Args:
        value: The recorded :data:`WORKSPACE_NAME` observation.

    Returns:
        ``"unset"`` for :data:`WORKSPACE_UNSET`, otherwise the size. An
        absent setting renders as a word rather than as a negative number,
        because ``-1`` in a report reads as a fault instead of as the
        absence it records.
    """
    if value == WORKSPACE_UNSET:
        return "unset"
    return f"{value:.0f}"


__all__ = [
    "DIGEST_SUFFIX",
    "FIELD_SEPARATOR",
    "INPUT_KIND",
    "LOSS_NAME",
    "OUTPUT_KIND",
    "STEP_DIGITS",
    "SUM_SUFFIX",
    "TRACE_EXPERIMENT",
    "TRACE_RUNGS",
    "WORKSPACE_NAME",
    "WORKSPACE_UNSET",
    "TraceName",
    "describe_workspace",
    "parse_trace_name",
    "trace_label",
    "trace_loss_name",
    "trace_tensor_name",
]
