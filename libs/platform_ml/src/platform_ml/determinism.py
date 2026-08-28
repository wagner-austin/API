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
  What it buys is reproducibility WITHIN one configuration, which is what
  lets a rerun be compared to its own earlier self, and what lets a
  cross-configuration difference be MEASURED against a known run rather than
  assumed to be zero.

  That used to read "not achievable", flatly. It is narrower than that, and
  the correction is measured. Two controls here each buy a piece of it:

  * ``remove_split_k`` makes three cards produce bit-identical tensors on
    every probed GEMM shape, and an A100 and an A30 agree on 1,017 of a
    1.5-billion-parameter model's 1,018 traced tensors. It is free at a
    training step's row count.
  * ``math_attention`` makes attention bit-identical across a V100, an A30
    and an A100. It costs 1.3-1.6x peak memory on the probed shapes and
    grows with the square of sequence length.

  They are disjoint -- split-K removal moves not one of 72 measured attention
  digests -- so neither substitutes for the other, and having both does not
  make the promise whole either. What remains uncontrolled is everything that
  is neither a cuBLASLt matmul nor an SDPA call. Cross-card agreement is a
  property to establish per operation and per configuration, not one this
  module can promise or refuse wholesale.
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
    CUBLASLT_NO_SPLIT_K,
    CUBLASLT_WORKSPACE_ENV_VAR,
    SetEnvProtocol,
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


class SdpBackendsProtocol(Protocol):
    """The four attention-backend switches on ``torch.backends.cuda``.

    All four rather than only the one being turned on, because these are
    independent booleans and not a selector: enabling math while leaving the
    fused kernels enabled changes nothing at all, since the dispatcher still
    prefers a fused one. Restricting attention means disabling the other
    three.

    The names are torch's own, including ``mem_efficient`` where the backend
    enum spells it ``EFFICIENT_ATTENTION``. Renaming them for tidiness would
    make the Protocol unsatisfiable by the one module that has to satisfy it.
    """

    def enable_flash_sdp(self, enabled: bool, /) -> None: ...

    def enable_mem_efficient_sdp(self, enabled: bool, /) -> None: ...

    def enable_math_sdp(self, enabled: bool, /) -> None: ...

    def enable_cudnn_sdp(self, enabled: bool, /) -> None: ...


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


#: Setting name recorded when a run took split-K out of cuBLASLt's options.
#:
#: RECORDED ONLY WHEN IT WAS DONE, and absent otherwise. Absence is readable
#: as "whatever the library chose", exactly as an absent
#: :data:`TORCH_THREAD_SETTING` is readable as "whatever the machine chose",
#: and for the same practical reason: adding a key that every run carries
#: would change every fingerprint ever written, and the deployed
#: known-answer registry would report ``configuration_differs`` against every
#: future probe. A key present only on runs that did the thing costs the
#: registry nothing and still distinguishes the two postures, because a run
#: that removed split-K genuinely is a different configuration from one that
#: did not -- it computes different numbers.
SPLIT_K_SETTING = "cublaslt_split_k"

#: The value :data:`SPLIT_K_SETTING` carries. A single value rather than a
#: pair, because the setting is absent when it was not applied; a record
#: never has to be read as "present and false".
SPLIT_K_REMOVED = "removed"


def remove_cublaslt_split_k(set_env: SetEnvProtocol) -> None:
    """Take split-K out of cuBLASLt's algorithm choices for this process.

    Must be called before any CUDA work, and before
    :func:`set_cublas_workspace` has no bearing on it -- the two variables
    are read by two different libraries when each creates its own handle, and
    both handles are created on first use. Ordering between them is
    irrelevant; ordering against CUDA is everything.

    Unlike :func:`set_cublas_workspace` this returns nothing. That function
    returns what it wrote because the value is a SETTING with more than one
    admissible value, and a record has to say which one was in force. Here
    there is exactly one value that means anything -- see
    :data:`~platform_core.determinism_env.CUBLASLT_NO_SPLIT_K` -- so a
    returned string would be a constant handed back to its own caller.

    Args:
        set_env: Writer for a process environment variable.
    """
    set_env(CUBLASLT_WORKSPACE_ENV_VAR, CUBLASLT_NO_SPLIT_K)


#: Setting name recorded when a run restricted attention to the math kernel.
#:
#: Recorded only when it was done, for the same reason as
#: :data:`SPLIT_K_SETTING`: absence is readable, and a key on every record
#: would cost the known-answer registry for a posture most records do not
#: have.
ATTENTION_SETTING = "sdpa_backends"

#: The value :data:`ATTENTION_SETTING` carries. Names the surviving backend
#: rather than saying "restricted", because the useful question a reader has
#: is which kernel ran, and a future posture permitting two would be spelled
#: here rather than needing a second key.
ATTENTION_MATH_ONLY = "math"


def restrict_attention_to_math(sdp: SdpBackendsProtocol) -> None:
    """Leave the attention dispatcher no kernel but the math one.

    WHAT THIS BUYS, MEASURED. Pinning ``SDPBackend.MATH`` makes attention
    bit-identical across a V100, an A30 and an A100 -- which nothing else
    here does. Removing cuBLASLt's split-K moves not one of 72 measured
    attention digests, because scaled-dot-product attention is not a cuBLASLt
    call, so the two controls address disjoint halves of a model and neither
    substitutes for the other.

    WHAT IT COSTS, AND WHY THE COST IS NOT A CONSTANT. The math path
    materialises the full ``[batch, heads, seq, seq]`` score matrix; a fused
    kernel never does. So the penalty grows with the SQUARE of sequence
    length -- measured at 1.3-1.6x peak allocation on the probed shapes, and
    worse than that as sequences lengthen. Time is close to free at 1.0-1.2x.
    A table of seconds would report this as "slightly slower" right up to the
    point where it stops fitting on the card at all: on a 16 GB V100 it takes
    gpt2-medium from trains to does not fit. That is a real operational
    consequence of turning it on, not a caveat.

    WHY ALL FOUR SWITCHES. This is the persistent form of
    ``sdpa_kernel([SDPBackend.MATH])``, which is implemented as exactly these
    four calls -- read from torch 2.6.0's
    ``torch/nn/attention/__init__.py``, not inferred. Using the context
    manager instead would mean wrapping every forward pass in the codebase
    and silently losing the pin at whichever site someone forgot; these
    switches are process-global and take for every subsequent call.

    Args:
        sdp: The ``torch.backends.cuda`` module, or a double.
    """
    sdp.enable_flash_sdp(False)
    sdp.enable_mem_efficient_sdp(False)
    sdp.enable_cudnn_sdp(False)
    sdp.enable_math_sdp(True)


#: Setting name for the thread count a torch run resolved to.
#:
#: Distinct from the ``OMP_NUM_THREADS`` family that
#: :mod:`platform_core.determinism_cpu` records, because the two are
#: different mechanisms: those are read by a BLAS when it LOADS, while
#: ``torch.set_num_threads`` takes whenever it is called. A run pinned by one
#: has not had the other done to it, and one spelling for both would make
#: those two runs compare as equal.
TORCH_THREAD_SETTING = "torch_num_threads"


def with_torch_thread_count(record: DeterminismRecord, threads: int) -> DeterminismRecord:
    """Add the resolved thread count to a determinism record.

    Separate from :func:`apply_determinism` on purpose. Thread count is
    pinned whether or not determinism was requested -- a job pins it to use
    the machine well, not to be reproducible -- so it has to be recordable on
    an :const:`~platform_core.determinism_record.UNPINNED_STACK` record too.
    Folding it into ``apply_determinism`` would also change every fingerprint
    that function has ever produced, including the known answers already
    registered against it.

    Measured, on this stack: a 4096x4096 matmul at one thread and at eight
    differs in 865,498 of 16,777,216 elements. A record that omits the count
    describes two runs that cannot reproduce each other identically.

    Args:
        record: What the stack pinned, if anything.
        threads: The count the run RESOLVED to -- read back from torch, not
            the number requested.

    Returns:
        The same record with the thread count among its settings.
    """
    return determinism_record(
        record["stack"],
        {**dict(record["settings"]), TORCH_THREAD_SETTING: str(threads)},
    )


def apply_determinism(
    cudnn: CudnnBackendProtocol,
    matmul: MatmulBackendProtocol,
    set_deterministic_algorithms: SetDeterministicAlgorithmsProtocol,
    set_env: SetEnvProtocol,
    sdp: SdpBackendsProtocol,
    *,
    remove_split_k: bool,
    math_attention: bool,
) -> DeterminismRecord:
    """Put the process into deterministic mode and report what was applied.

    Takes the three leaf objects it writes rather than the ``torch`` module,
    for two reasons. It states exactly what this function touches, so the
    signature is the blast radius. And a Protocol whose member is itself a
    submodule cannot be satisfied by a module under a strict type checker,
    which would otherwise have forced a cast at the one call site that
    matters -- the production one.

    Order matters: both environment variables are written first, because
    every later step may touch CUDA and each is only read once.

    WHY THE TWO CONTROLS ARE REQUIRED ARGUMENTS WITH NO DEFAULT. Neither is
    "on" and "off" of one feature; each is the treatment and the control of a
    live experiment. A training run wants split-K gone and attention pinned,
    because together that is what makes its numbers agree across cards. The
    commands that MEASURE what either does must be able to run without it, or
    they can only ever observe the treated arm -- an instrument that imposes
    the intervention cannot measure it. Defaulting either argument would
    silently pick a posture for a caller who had not thought about it, and
    the failure would be a number that looks fine.

    THE TWO ARE INDEPENDENT AND NEITHER IMPLIES THE OTHER. Split-K governs
    cuBLASLt matmuls; attention does not go through cuBLASLt and not one of
    72 measured attention digests moves when split-K is removed. Conversely
    the math kernel says nothing about a linear layer. They are separate
    arguments because they are separate halves of a model, with separate
    costs -- split-K removal is free at a training step's row count, and the
    attention pin is not free at all.

    Args:
        cudnn: The ``torch.backends.cudnn`` module, or a double.
        matmul: The ``torch.backends.cuda.matmul`` module, or a double.
        set_deterministic_algorithms: ``torch.use_deterministic_algorithms``.
        set_env: Writer for a process environment variable, e.g.
            ``os.putenv``.
        sdp: The ``torch.backends.cuda`` module, or a double, carrying the
            four attention-backend switches.
        remove_split_k: Whether to take split-K out of cuBLASLt's options.
            True for a run whose numbers should be comparable across cards;
            False for a run measuring what that costs or what it changes.
            False writes nothing at all -- it does not write a "keep split-K"
            value -- so a launcher that exported the variable itself still
            governs, and the record says only that this call did not do it.
        math_attention: Whether to leave the attention dispatcher no kernel
            but the math one. True makes attention bit-identical across a
            V100, an A30 and an A100; it also costs 1.3-1.6x peak memory on
            the probed shapes and worse as sequences lengthen, because the
            math path materialises the whole score matrix. See
            :func:`restrict_attention_to_math` -- this is the one control
            here that can turn a run that fits into a run that does not.
            False touches none of the four switches, leaving whatever the
            process already had.

    Returns:
        A :class:`DeterminismReport` describing the state now in force.
        Store it beside the run; a run whose determinism settings are unknown
        cannot be compared to one whose are.
    """
    workspace = set_cublas_workspace(set_env)
    # Written and recorded in one branch so the two can never disagree: a
    # record claiming split-K was removed by a process that did not write the
    # variable would be the exact failure this record exists to prevent.
    split_k: dict[str, str] = {}
    if remove_split_k:
        remove_cublaslt_split_k(set_env)
        split_k = {SPLIT_K_SETTING: SPLIT_K_REMOVED}

    # Same one-branch discipline: applied and recorded together, so a record
    # claiming the math kernel can only come from a process that pinned it.
    # Unlike the environment variables this needs no particular ordering
    # against CUDA -- the switches are read per dispatch, not once at handle
    # creation -- but it sits with them so one function is one posture.
    attention: dict[str, str] = {}
    if math_attention:
        restrict_attention_to_math(sdp)
        attention = {ATTENTION_SETTING: ATTENTION_MATH_ONLY}

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
            **split_k,
            **attention,
        },
    )


__all__ = [
    "ATTENTION_MATH_ONLY",
    "ATTENTION_SETTING",
    "CUBLASLT_NO_SPLIT_K",
    "CUBLASLT_WORKSPACE_ENV_VAR",
    "CUBLAS_DETERMINISTIC_WORKSPACE",
    "CUBLAS_WORKSPACE_ENV_VAR",
    "SPLIT_K_REMOVED",
    "SPLIT_K_SETTING",
    "TORCH_STACK",
    "TORCH_THREAD_SETTING",
    "CudnnBackendProtocol",
    "MatmulBackendProtocol",
    "SdpBackendsProtocol",
    "SetDeterministicAlgorithmsProtocol",
    "SetEnvProtocol",
    "apply_determinism",
    "remove_cublaslt_split_k",
    "restrict_attention_to_math",
    "set_cublas_workspace",
    "with_torch_thread_count",
]
