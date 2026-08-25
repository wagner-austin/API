"""Kernel-level numerical determinism for training runs.

Seeding an RNG makes the same numbers come out of the sampler. It says
nothing about the order a GPU accumulates a reduction in, and floating-point
addition is not associative, so the same seed on the same card can still
produce a different model on every run. Measured consequences of leaving that
uncontrolled are large enough to swamp the effects an experiment is trying to
read: GPU nondeterminism alone has been reported to produce double-digit
relative standard deviation in a reinforcement-learning score, while making
the reductions deterministic costs at most single-digit percent.

Two things follow, and this module exists to make both explicit rather than
inherited.

First, determinism has to be TURNED ON. Nothing in PyTorch does it by
default, and the defaults are not even uniform across the flags involved --
matmul TF32 has defaulted off since torch 1.12 while cuDNN TF32 defaults on.

Second, it has to be RECORDED. "Whatever this torch version happened to
default to" is not a specification, cannot be written into a provenance
block, and cannot be compared against a later run. So every entry point here
returns a :class:`DeterminismReport` describing what was actually applied,
for the caller to log or store beside the run.

What this module deliberately does NOT do:

* It does not seed anything. Seeding is the caller's, and is orthogonal --
  a seeded nondeterministic run and an unseeded deterministic one are both
  irreproducible, for different reasons.
* It does not promise reproducibility ACROSS different GPU architectures.
  That is not achievable: different cards select different kernels with
  different reduction trees. What it buys is reproducibility WITHIN one
  configuration, which is what lets a rerun be compared to its own earlier
  self, and what lets a cross-configuration difference be MEASURED against a
  known run rather than assumed to be zero.
* It does not offer a warn-only mode. PyTorch can be asked to warn instead of
  raising when an operation has no deterministic implementation; that
  silently returns a nondeterministic result while reporting success, which
  is the exact failure this module is here to prevent. An operation with no
  deterministic kernel is a hard error, and the traceback names the op.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

# PyTorch's documented setting for deterministic cuBLAS reductions. It is read
# by the CUDA runtime when the cuBLAS handle is created, which happens on
# first use, so it must be in the environment BEFORE any CUDA work -- setting
# it afterwards is accepted silently and has no effect.
#
# Imported rather than defined here: a job submitter writes the same variable
# into a batch script and must not depend on torch to know its value. The pair
# lives in platform_core so the two tiers cannot drift apart -- if they did,
# nothing would fail and the runs would simply stop being comparable.
from platform_core.determinism_env import (
    CUBLAS_DETERMINISTIC_WORKSPACE,
    CUBLAS_WORKSPACE_ENV_VAR,
)
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_str,
)
from typing_extensions import TypedDict

#: The stack this module pins. Named here rather than spelled at the call
#: site so a reader comparing two records knows the string is a constant and
#: not one producer's spelling.
TORCH_STACK = "torch"


#: Value spelling for a pinned boolean setting. Settings are strings because
#: the record must hold any stack's vocabulary -- a torch run pins
#: ``cudnn_benchmark``, a BLAS-bound one pins a thread count, an
#: arbitrary-precision one pins a mantissa width -- and a union of every
#: stack's value types would grow forever while comparing no better.
TRUE = "true"
FALSE = "false"

#: The stack name for a run that pinned nothing. Distinct from a pinned
#: record with no settings, which cannot occur: a stack that pins states what
#: it pinned. "Nothing was pinned" is a fact about a run and must be
#: recordable, because a run whose determinism is unknown and a run that was
#: deliberately left free are the same thing to a later comparison and both
#: differ from a pinned one.
UNPINNED_STACK = "none"


class DeterminismRecord(TypedDict):
    """What determinism was in force, and which stack put it there.

    Deliberately NOT a torch shape. Most of this monorepo's research is not
    torch -- gradient boosting, transliteration, metabolomics -- and a record
    with ``cudnn_benchmark`` in it cannot be filled by a numpy run, a Rust
    booster, or a future job with no GPU at all. Those runs still have a
    determinism posture worth recording, and a fingerprint whose determinism
    axis only one stack can populate makes every other stack's runs compare
    as though the question did not apply.

    Attributes:
        stack: What pinned these settings, e.g. ``"torch"``, or
            :const:`UNPINNED_STACK` when nothing did. Part of the record
            rather than inferred from the setting names, because two stacks
            may pin settings that share a name and mean different things.
        settings: The pinned settings as ``(name, value)`` pairs, sorted by
            name. Sorted at construction so two records describing the same
            posture are equal and render identically regardless of the order
            a producer emitted them in.
    """

    stack: str
    settings: tuple[tuple[str, str], ...]


def determinism_record(stack: str, settings: Mapping[str, str]) -> DeterminismRecord:
    """Build a record, putting the settings in canonical order.

    Args:
        stack: What pinned the settings.
        settings: The pinned settings by name.

    Returns:
        The record, with settings sorted by name.

    Raises:
        ValueError: When ``stack`` is empty. A record that cannot say what
            pinned it is not comparable with one that can -- and
            :const:`UNPINNED_STACK` is how "nothing did" is spelled, so an
            empty string carries no meaning the vocabulary lacks.
    """
    if stack == "":
        raise ValueError("stack must name what pinned these settings, or be UNPINNED_STACK")
    return DeterminismRecord(stack=stack, settings=tuple(sorted(settings.items())))


class CudnnBackendProtocol(Protocol):
    """The ``torch.backends.cudnn`` surface this module writes."""

    allow_tf32: bool
    deterministic: bool
    benchmark: bool


class MatmulBackendProtocol(Protocol):
    """The ``torch.backends.cuda.matmul`` surface this module writes."""

    allow_tf32: bool


class SetDeterministicAlgorithmsProtocol(Protocol):
    """``torch.use_deterministic_algorithms``."""

    def __call__(self, mode: bool) -> None: ...


class SetEnvProtocol(Protocol):
    """A writer for one process environment variable.

    A write-only seam rather than a mapping, for two reasons. Production
    passes ``os.putenv``, which reaches the real process environment that a
    C library's ``getenv`` reads -- the only environment cuBLAS consults.
    And the monorepo bans reading config out of ``os.environ``, correctly:
    configuration comes from the config layer. Writing a variable that a
    native library requires is a different act, and this Protocol keeps the
    two from being confused.

    Deliberately no read side. ``os.putenv`` does not update ``os.environ``,
    so a "did it get set?" helper built on the Python mapping would report
    False on a correctly configured process.

    Parameters are positional-only: ``os.putenv`` names them ``name`` and
    ``value``, and a Protocol that named them otherwise would reject the one
    implementation that matters.
    """

    def __call__(self, key: str, value: str, /) -> None: ...


def set_cublas_workspace(set_env: SetEnvProtocol) -> str:
    """Place the deterministic cuBLAS workspace setting in the environment.

    Must be called before any CUDA work in the process. Setting it afterwards
    is accepted without error and has no effect, which is precisely why this
    returns the value it wrote: a caller that records the return value
    records what the run actually had, and a caller that ignores it has at
    least been handed the evidence.

    Args:
        set_env: Writer for a process environment variable.

    Returns:
        The value written.
    """
    set_env(CUBLAS_WORKSPACE_ENV_VAR, CUBLAS_DETERMINISTIC_WORKSPACE)
    return CUBLAS_DETERMINISTIC_WORKSPACE


def apply_determinism(
    cudnn: CudnnBackendProtocol,
    matmul: MatmulBackendProtocol,
    set_deterministic_algorithms: SetDeterministicAlgorithmsProtocol,
    set_env: SetEnvProtocol,
) -> DeterminismRecord:
    """Put the process into deterministic mode and report what was applied.

    Takes the three leaf objects it writes rather than the ``torch`` module,
    for two reasons. It states exactly what this function touches, so the
    signature is the blast radius. And a Protocol whose member is itself a
    submodule cannot be satisfied by a module under a strict type checker,
    which would otherwise have forced a cast at the one call site that
    matters -- the production one.

    Order matters: the cuBLAS workspace variable is written first, because
    every later step may touch CUDA and the variable is only read once.

    Args:
        cudnn: The ``torch.backends.cudnn`` module, or a double.
        matmul: The ``torch.backends.cuda.matmul`` module, or a double.
        set_deterministic_algorithms: ``torch.use_deterministic_algorithms``.
        set_env: Writer for a process environment variable, e.g.
            ``os.putenv``.

    Returns:
        A :class:`DeterminismReport` describing the state now in force.
        Store it beside the run; a run whose determinism settings are unknown
        cannot be compared to one whose are.
    """
    workspace = set_cublas_workspace(set_env)

    matmul.allow_tf32 = False
    cudnn.allow_tf32 = False
    cudnn.deterministic = True
    cudnn.benchmark = False
    set_deterministic_algorithms(True)

    return determinism_record(
        TORCH_STACK,
        {
            "deterministic_algorithms": TRUE,
            "cublas_workspace_config": workspace,
            "matmul_tf32": FALSE,
            "cudnn_tf32": FALSE,
            "cudnn_deterministic": TRUE,
            "cudnn_benchmark": FALSE,
        },
    )


def encode_determinism_record(record: DeterminismRecord) -> JSONObject:
    """Encode a record for a run record or a structured log field.

    Args:
        record: The record to encode.

    Returns:
        A JSON object carrying the stack and the settings as a nested
        object. Nested rather than flattened so a setting can never collide
        with the ``stack`` key, whatever a future stack decides to name one.
    """
    return {
        "stack": record["stack"],
        "settings": dict(record["settings"]),
    }


def decode_determinism_record(value: JSONValue) -> DeterminismRecord:
    """Validate a JSON value as a determinism record.

    Args:
        value: The value to validate, typically from a stored run record.

    Returns:
        The validated record, with settings in canonical order.

    Raises:
        JSONTypeError: When ``value`` is not an object, the stack is absent
            or empty, ``settings`` is absent or not an object, or any
            setting value is not a string. A record that cannot say what
            pinned it, or that carries a setting whose value has to be
            guessed at, is not comparable with one that can.
    """
    obj = narrow_json_to_dict(value)
    stack = require_str(obj, "stack")
    if stack == "":
        raise JSONTypeError("Field 'stack' must name what pinned these settings")
    raw = require_dict(obj, "settings")
    settings: dict[str, str] = {}
    for name, setting in raw.items():
        if not isinstance(setting, str):
            raise JSONTypeError(f"Setting {name!r} must be a string, got {type(setting).__name__}")
        settings[name] = setting
    return determinism_record(stack, settings)


def render_determinism_record(record: DeterminismRecord) -> str:
    """Render a record as one stable comparison key.

    Args:
        record: The record to render.

    Returns:
        The stack and its settings in canonical order, so two runs with the
        same posture render byte-identically and a difference is legible
        without reading two nested objects side by side.
    """
    body = ",".join(f"{name}={value}" for name, value in record["settings"])
    return f"{record['stack']}[{body}]"


__all__ = [
    "CUBLAS_DETERMINISTIC_WORKSPACE",
    "CUBLAS_WORKSPACE_ENV_VAR",
    "FALSE",
    "TORCH_STACK",
    "TRUE",
    "UNPINNED_STACK",
    "CudnnBackendProtocol",
    "DeterminismRecord",
    "MatmulBackendProtocol",
    "SetDeterministicAlgorithmsProtocol",
    "SetEnvProtocol",
    "apply_determinism",
    "decode_determinism_record",
    "determinism_record",
    "encode_determinism_record",
    "render_determinism_record",
    "set_cublas_workspace",
]
