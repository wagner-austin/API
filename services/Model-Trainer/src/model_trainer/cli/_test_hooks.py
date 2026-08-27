"""Hooks for the command-line entries - production defaults, tests override.

Production sets these to the real implementations at import. Tests replace
them with fakes before exercising the code under test, so there is no
conditional in the entry itself -- it calls the hook.

Only the two seams that need real weights and a real GPU are here. Everything
else in the scorer is pure and is exercised directly, because a fake in front
of pure code tests the fake.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from platform_core.determinism_record import DeterminismRecord

from model_trainer.core._hook_protocols_ml import PinTorchThreadsProto
from model_trainer.core.contracts.cloze import ClozeEvalResult, ClozeItem
from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.services.model.forward_cost import FORWARD_SHAPES, ForwardCostShape

# Safe at module scope where the others are not: `probe_shapes` is a table of
# TypedDicts and a label formatter, and imports no torch. That separation is
# the reason it is its own module -- see its docstring.
from model_trainer.core.services.model.gemm_shapes import GemmShape, timed_shapes
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES, ProbeShape
from model_trainer.core.services.model.sdpa_shapes import SdpaCostShape, cost_shapes
from model_trainer.core.services.model.trace_plan import TRACE_RUNGS
from model_trainer.core.services.model.train_cost import TRAIN_SHAPES


class LoadHubModelProto(Protocol):
    """Protocol for loading an untrained model straight from the hub."""

    def __call__(self, hub_model_id: str, /) -> PreparedLMModel:
        """Load the named model with nothing applied to it."""
        ...


class ScoreClozeProto(Protocol):
    """Protocol for the cloze scorer.

    Keyword-only, matching the real signature: the scorer takes five
    arguments whose order carries no meaning and would be easy to transpose.
    """

    def __call__(
        self,
        *,
        items: list[ClozeItem],
        model: PreparedLMModel,
        device: str,
        max_seq_len: int,
    ) -> ClozeEvalResult:
        """Score every item and report accuracy against the guessing baseline."""
        ...


class ApplyDeterminismProto(Protocol):
    """Protocol for the determinism pin.

    Behind a hook because it writes process-global torch state and the
    environment, which a test must be able to observe without a real CUDA
    stack.
    """

    def __call__(self) -> DeterminismRecord:
        """Pin kernel determinism and report what was actually applied."""
        ...


class LadderShapesProto(Protocol):
    """Protocol for the probe ladder the report command walks.

    Behind a hook for a different reason than the others: not because the real
    thing needs a GPU, but because it needs one to be REASONABLE. The ladder
    ends at a 1.5-billion-parameter model, which is the measurement's point
    and is not something to construct on a test runner. Tests install a
    two-rung ladder and exercise every line of the walk; the cluster runs the
    real one.
    """

    def __call__(self) -> Mapping[str, ProbeShape]:
        """Return the rungs to walk, in the order to walk them."""
        ...


class TraceRungsProto(Protocol):
    """Protocol for the rungs the forward trace walks.

    Behind a hook for the same reason the ladder's shapes are: the declared
    set ends at a 1.5-billion-parameter model, and tracing it digests about a
    hundred and seventy million floats. Tests install one tiny rung and walk
    every line; the cluster walks the real four.
    """

    def __call__(self) -> tuple[str, ...]:
        """Return the rung names to trace, in the order to trace them."""
        ...


class CostShapesProto(Protocol):
    """Protocol for the attention calls the cost sweep times.

    Behind a hook for the same reason the ladder's and the gemm benchmark's
    tables are, and more urgently: the declared sweep ends at eight sequences
    of 4096 tokens, where the math path allocates gigabytes and takes tens of
    milliseconds per call. Timing that on a test runner's CPU would not fail,
    it would HANG -- which is how a probe-ladder test once had to be killed
    after ten minutes.
    """

    def __call__(self) -> tuple[SdpaCostShape, ...]:
        """Return the calls to time, in order."""
        ...


def _default_cost_shapes() -> tuple[SdpaCostShape, ...]:
    """Production attention-cost sweep - used as default hook.

    Returns:
        The batch-by-length grid, then the ladder's own calls.
    """
    return cost_shapes()


class ForwardShapesProto(Protocol):
    """Protocol for the forward passes the end-to-end benchmark times.

    Behind a hook for the same reason the other sweeps are, and the reason
    bites hardest here: the declared rows build models up to 774 million
    parameters and run them over a 50,257-token vocabulary. On a test
    runner's CPU that is not slow, it is unusable.
    """

    def __call__(self) -> tuple[ForwardCostShape, ...]:
        """Return the passes to time, in order."""
        ...


def _default_forward_shapes() -> tuple[ForwardCostShape, ...]:
    """Production forward sweep - used as default hook.

    Returns:
        Every declared row, in table order.
    """
    return FORWARD_SHAPES


class TrainShapesProto(Protocol):
    """Protocol for the training steps the step benchmark times.

    Behind a hook for the same reason every other sweep is, and hardest here:
    a training step holds parameters, gradients and two AdamW moments, so the
    declared rows need about twelve gigabytes before a single activation.
    """

    def __call__(self) -> tuple[ForwardCostShape, ...]:
        """Return the steps to time, in order."""
        ...


def _default_train_shapes() -> tuple[ForwardCostShape, ...]:
    """Production training sweep - used as default hook.

    Returns:
        Every declared row, in table order.
    """
    return TRAIN_SHAPES


class EnvCublasltWorkspaceProto(Protocol):
    """Protocol for reading the split-K condition out of the environment.

    Behind a hook because it reads process-global configuration a test must
    be able to set both ways without a real cuBLASLt, and because reading it
    is the only way a record can say which condition produced it -- see
    :data:`~model_trainer.core.services.model.trace_plan.WORKSPACE_NAME`.
    """

    def __call__(self) -> str | None:
        """Return ``CUBLASLT_WORKSPACE_SIZE``, or None when it is not set."""
        ...


def _default_env_cublaslt_workspace() -> str | None:
    """Production reader for the split-K condition - used as default hook.

    Goes through ``config_test_hooks.get_env`` rather than ``os.environ``,
    which is the read this monorepo's env guard exists to stop.

    Returns:
        The variable's value, or None when unset or empty. Empty is treated
        as unset because cuBLASLt itself ignores an empty value, and a record
        claiming a condition its library did not apply would be worse than
        one saying it does not know.
    """
    from platform_core.config import config_test_hooks

    value = config_test_hooks.get_env("CUBLASLT_WORKSPACE_SIZE")
    return value if value else None


def _default_trace_rungs() -> tuple[str, ...]:
    """Production trace rungs - used as default hook.

    Returns:
        The declared contrast: the rung split-K removal breaks, the one it
        fails to fix, the one it fixes, and the one that never moves.
    """
    return TRACE_RUNGS


class BenchmarkShapesProto(Protocol):
    """Protocol for the shape table the split-K benchmark times.

    Behind a hook for the same reason the ladder's is: the real table is a GPU
    measurement. Timing is deliberately repetitive -- warmup plus several
    batches of many calls each -- so walking all 43 shapes on a test runner's
    CPU spent minutes producing numbers nobody will read.
    """

    def __call__(self) -> tuple[tuple[str, GemmShape], ...]:
        """Return the shapes to time, in order."""
        ...


def _default_benchmark_shapes() -> tuple[tuple[str, GemmShape], ...]:
    """Production benchmark shape table - used as default hook.

    Returns:
        The ladder's calls at one short sequence and again at a real batch.
    """
    return timed_shapes()


class RunBenchmarkChildProto(Protocol):
    """Protocol for spawning the benchmark's second-condition process.

    Behind a hook because the child is a real subprocess that needs a GPU and
    several seconds of timing. A test can drive the parent's plumbing --
    including the refusal when the child comes back with the wrong
    condition -- without paying for either.
    """

    def __call__(self, argv: list[str], variable: str, value: str, /) -> int:
        """Run the child with one variable set, and return its exit code."""
        ...


def _default_run_benchmark_child(argv: list[str], variable: str, value: str, /) -> int:
    """Production child spawner - used as default hook.

    ``os.putenv`` and an INHERITED environment, rather than building a full
    mapping to hand ``subprocess``. Two reasons, and the first is the monorepo
    rule: reading ``os.environ`` to assemble that mapping is the config read
    the env guard exists to stop, while ``putenv`` is a write -- the same
    distinction ``core/_test_hooks`` already relies on to set
    ``CUBLAS_WORKSPACE_CONFIG``. The second is that putenv reaches the real
    process environment, which is what a child inherits and what cuBLASLt's
    own getenv reads.

    Args:
        argv: The command line to run.
        variable: The variable to set for the child.
        value: What to set it to. This is the whole reason the child exists:
            ``CUBLASLT_WORKSPACE_SIZE`` is read once when the cuBLASLt handle
            is created, so a process that already has one cannot change
            condition -- measured, two calls with it set between them both
            still used split-K.

    Returns:
        The child's exit code.
    """
    import os
    import subprocess

    os.putenv(variable, value)
    return subprocess.run(argv, check=False).returncode


def _default_ladder_shapes() -> Mapping[str, ProbeShape]:
    """Production probe ladder - used as default hook.

    Returns:
        Every declared rung, in table order.
    """
    return PROBE_SHAPES


def _default_load_hub_model(hub_model_id: str, /) -> PreparedLMModel:
    """Production hub loader - used as default hook.

    Imported inside the function so that importing this module does not pull
    torch into a process that only wanted to parse a command line and print
    a usage error.

    Args:
        hub_model_id: HuggingFace model id, for example ``gpt2-medium``.

    Returns:
        The prepared model, with nothing applied to it.
    """
    from model_trainer.core.services.model.backends.hf_lm.io import (
        load_prepared_hf_lm_from_hub,
    )

    return load_prepared_hf_lm_from_hub(hub_model_id)


def _default_score_cloze(
    *,
    items: list[ClozeItem],
    model: PreparedLMModel,
    device: str,
    max_seq_len: int,
) -> ClozeEvalResult:
    """Production scorer - used as default hook.

    Unpacks the prepared model into the model and its encoder, which is the
    only reason this is not the scorer itself: the entry holds a
    PreparedLMModel and the scorer takes the two halves.

    Args:
        items: The cloze items to score.
        model: The prepared model and its encoder.
        device: Device to score on.
        max_seq_len: Token budget per item.

    Returns:
        The scored result.
    """
    from model_trainer.core.services.model.cloze import score_cloze_items

    return score_cloze_items(
        items=items,
        model=model.model,
        encoder=model.tok_for_dataset,
        device=device,
        max_seq_len=max_seq_len,
    )


def _default_apply_determinism() -> DeterminismRecord:
    """Production determinism pin - used as default hook.

    Delegates to the same hook the workers use, so a run scored from the
    command line and one scored through the queue pin identically. A second
    spelling here would be a second posture nobody noticed diverging.

    Returns:
        What was actually applied.
    """
    from model_trainer.core import _test_hooks as core_hooks

    return core_hooks.apply_determinism_hook()


def _default_pin_torch_threads(threads: int) -> int:
    """Production torch thread pin - used as default hook.

    Delegates to the worker's hook for the same reason
    :func:`_default_apply_determinism` does: a probe run from the command
    line and a job run through the queue must pin by the same call, or the
    two postures diverge without anyone noticing.

    Args:
        threads: Count to request.

    Returns:
        The count torch resolved to, which may differ from the request.
    """
    from model_trainer.core import _test_hooks as core_hooks

    return core_hooks.pin_torch_threads(threads)


load_hub_model: LoadHubModelProto = _default_load_hub_model

score_cloze: ScoreClozeProto = _default_score_cloze

apply_determinism_hook: ApplyDeterminismProto = _default_apply_determinism

pin_torch_threads: PinTorchThreadsProto = _default_pin_torch_threads

ladder_shapes: LadderShapesProto = _default_ladder_shapes

trace_rungs: TraceRungsProto = _default_trace_rungs

env_cublaslt_workspace: EnvCublasltWorkspaceProto = _default_env_cublaslt_workspace

cost_shapes_hook: CostShapesProto = _default_cost_shapes

forward_shapes: ForwardShapesProto = _default_forward_shapes

train_shapes: TrainShapesProto = _default_train_shapes

run_benchmark_child: RunBenchmarkChildProto = _default_run_benchmark_child

benchmark_shapes: BenchmarkShapesProto = _default_benchmark_shapes


__all__ = [
    "ApplyDeterminismProto",
    "BenchmarkShapesProto",
    "CostShapesProto",
    "EnvCublasltWorkspaceProto",
    "ForwardShapesProto",
    "LadderShapesProto",
    "LoadHubModelProto",
    "RunBenchmarkChildProto",
    "ScoreClozeProto",
    "TraceRungsProto",
    "TrainShapesProto",
    "apply_determinism_hook",
    "benchmark_shapes",
    "cost_shapes_hook",
    "env_cublaslt_workspace",
    "forward_shapes",
    "ladder_shapes",
    "load_hub_model",
    "pin_torch_threads",
    "run_benchmark_child",
    "score_cloze",
    "trace_rungs",
    "train_shapes",
]
