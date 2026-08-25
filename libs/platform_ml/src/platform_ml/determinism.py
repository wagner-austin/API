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
from platform_core.determinism_record import (
    FALSE,
    TRUE,
    DeterminismRecord,
    determinism_record,
)

#: The stack this module pins. Named here rather than spelled at the call
#: site so a reader comparing two records knows the string is a constant and
#: not one producer's spelling.
TORCH_STACK = "torch"


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


__all__ = [
    "CUBLAS_DETERMINISTIC_WORKSPACE",
    "CUBLAS_WORKSPACE_ENV_VAR",
    "TORCH_STACK",
    "CudnnBackendProtocol",
    "MatmulBackendProtocol",
    "SetDeterministicAlgorithmsProtocol",
    "SetEnvProtocol",
    "apply_determinism",
    "set_cublas_workspace",
]
