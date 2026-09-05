"""Which measurements a command walks - production tables, tests substitute.

Split from :mod:`model_trainer.cli._test_hooks` when that module passed the
600-line ceiling, and the seam is the one its own docstring already claimed:
that file is for "the seams that need real weights and a real GPU", and these
are not that. These are TABLES -- which rungs, which shapes, which plans --
and every one is behind a hook for the same reason, which is not that it needs
a card but that the real table IS a cluster measurement.

The declared ladder ends at a 1.5-billion-parameter model. The gemm sweep is
ninety-three shapes at up to N=4096. The attention sweep reaches eight
sequences of 4096 tokens, where the math path allocates gigabytes per call --
timing that on a test runner does not fail, it HANGS, which is how a probe
ladder test once had to be killed after ten minutes. Tests install a two- or
three-row table and walk every line; the cluster walks the real one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

# Safe at module scope for the same reason `probe_shapes` is: a table of
# TypedDicts, a digest and a label formatter, importing no torch. The arms it
# describes live in `cartridge_measurement`, which does.
from model_trainer.core.services.model.cartridge_plans import (
    CARTRIDGE_PLANS,
    COMPANION_SWEEP_PLANS,
    COMPOSITION_SWEEP_PLANS,
    CartridgePlan,
    CompanionSweepPlan,
    CompositionSweepPlan,
)
from model_trainer.core.services.model.cartridge_pool_plans import (
    BASE_LORA_SWEEP_PLANS,
    DIVERSE_COMPANION_SWEEP_PLANS,
    VARIED_COMPANION_SWEEP_PLANS,
    BaseLoraSweepPlan,
    VariedCompanionSweepPlan,
)
from model_trainer.core.services.model.cartridge_qa_plans import QA_PLANS, QaPlan
from model_trainer.core.services.model.forward_cost import FORWARD_SHAPES, ForwardCostShape
from model_trainer.core.services.model.gemm_shapes import (
    GemmShape,
    probed_shapes,
    timed_shapes,
)
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES, ProbeShape
from model_trainer.core.services.model.sdpa_shapes import SdpaCostShape, cost_shapes
from model_trainer.core.services.model.trace_plan import TRACE_RUNGS
from model_trainer.core.services.model.train_cost import TRAIN_SHAPES


class CartridgePlansProto(Protocol):
    """Protocol for the cartridge plan table.

    Behind a hook for the same reason :data:`ladder_shapes` is: every plan in
    the real table is minutes of GPU per arm, so a suite that could only reach
    that table would either run it or leave the entry uncovered.
    """

    def __call__(self) -> Mapping[str, CartridgePlan]:
        """Return every declared plan, in table order."""
        ...


class QaPlansProto(Protocol):
    """Protocol for the question-set plan table.

    Behind a hook for the reason :class:`CartridgePlansProto` is: the real
    plan trains one cartridge per seed over a 124-million-parameter base and
    scores three arms against it, so a suite that could only reach that table
    would either run it or leave this entry uncovered.
    """

    def __call__(self) -> Mapping[str, QaPlan]:
        """Return every declared question-set plan, in table order."""
        ...


class CompositionSweepPlansProto(Protocol):
    """Protocol for the composition-sweep plan table.

    Behind a hook for the reason :class:`CartridgePlansProto` is: the real
    table's one plan trains dozens of cartridges over a real GPT-2, so a
    suite that could only reach it would either run it or leave the entry
    uncovered.
    """

    def __call__(self) -> Mapping[str, CompositionSweepPlan]:
        """Return every declared plan, in table order."""
        ...


class CompanionSweepPlansProto(Protocol):
    """Protocol for the companion-sweep plan table.

    Behind a hook for the reason :class:`CompositionSweepPlansProto` is: the
    real table's one plan trains over a hundred cartridges over a real
    GPT-2, so a suite that could only reach it would either run it or leave
    the entry uncovered.
    """

    def __call__(self) -> Mapping[str, CompanionSweepPlan]:
        """Return every declared plan, in table order."""
        ...


class VariedCompanionSweepPlansProto(Protocol):
    """Protocol for the varied-count companion-sweep plan table.

    Behind a hook for the reason :class:`CompanionSweepPlansProto` is: the
    real table's one plan trains dozens of cartridges over a real GPT-2, so
    a suite that could only reach it would either run it or leave the entry
    uncovered.
    """

    def __call__(self) -> Mapping[str, VariedCompanionSweepPlan]:
        """Return every declared plan, in table order."""
        ...


class BaseLoraSweepPlansProto(Protocol):
    """Protocol for the base-LoRA sweep plan table.

    Behind a hook for the reason :class:`CompanionSweepPlansProto` is, and
    more so: the real table's one plan trains a pool, a LoRA and dozens of
    cartridges over a real GPT-2.
    """

    def __call__(self) -> Mapping[str, BaseLoraSweepPlan]:
        """Return every declared plan, in table order."""
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


class ProbedShapesProto(Protocol):
    """Protocol for the shape table the digest probe walks.

    Behind a hook since 2026-08-31, when the table grew the batched and
    crossover shapes: a single batched call is cheap on a GPU and the probe
    runs each twice, but ninety-three shapes at up to ``N=4096`` on a test
    runner's CPU -- digested through a per-element ``tolist`` -- is minutes
    per record, and the record tests build several. Tests install a
    three-shape table and exercise every line of the walk; the cluster walks
    the real one.
    """

    def __call__(self) -> tuple[tuple[str, GemmShape], ...]:
        """Return the shapes to digest, in order."""
        ...


def _default_probed_shapes() -> tuple[tuple[str, GemmShape], ...]:
    """Production digest-probe shape table - used as default hook.

    Returns:
        Every declared table: ladder, sweep grid, boundary bracket, batched
        twins and the batch-size sweep.
    """
    return probed_shapes()


def _default_cartridge_plans() -> Mapping[str, CartridgePlan]:
    """Production cartridge plan table - used as default hook.

    Returns:
        Every declared plan, in table order.
    """
    return CARTRIDGE_PLANS


def _default_qa_plans() -> Mapping[str, QaPlan]:
    """Production question-set plan table - used as default hook.

    Returns:
        Every declared plan, in table order.
    """
    return QA_PLANS


def _default_composition_sweep_plans() -> Mapping[str, CompositionSweepPlan]:
    """Production composition-sweep plan table - used as default hook.

    Returns:
        Every declared plan, in table order.
    """
    return COMPOSITION_SWEEP_PLANS


def _default_ladder_shapes() -> Mapping[str, ProbeShape]:
    """Production probe ladder - used as default hook.

    Returns:
        Every declared rung, in table order.
    """
    return PROBE_SHAPES


cartridge_plans: CartridgePlansProto = _default_cartridge_plans

composition_sweep_plans: CompositionSweepPlansProto = _default_composition_sweep_plans


def _default_companion_sweep_plans() -> Mapping[str, CompanionSweepPlan]:
    """Production companion-sweep plan table - used as default hook.

    Returns:
        Every declared plan, in table order.
    """
    return COMPANION_SWEEP_PLANS


companion_sweep_plans: CompanionSweepPlansProto = _default_companion_sweep_plans


def _default_varied_companion_sweep_plans() -> Mapping[str, VariedCompanionSweepPlan]:
    """Production varied-count plan table - used as default hook.

    Returns:
        Every declared plan, in table order.
    """
    return VARIED_COMPANION_SWEEP_PLANS


varied_companion_sweep_plans: VariedCompanionSweepPlansProto = _default_varied_companion_sweep_plans


def _default_diverse_companion_sweep_plans() -> Mapping[str, VariedCompanionSweepPlan]:
    """Production diverse-pool plan table - used as default hook.

    Returns:
        Every declared plan, in table order.
    """
    return DIVERSE_COMPANION_SWEEP_PLANS


#: Same protocol as the varied table: the plan SHAPE is shared and only the
#: pool's construction differs, which is a CLI concern, not a table one.
diverse_companion_sweep_plans: VariedCompanionSweepPlansProto = (
    _default_diverse_companion_sweep_plans
)


def _default_base_lora_sweep_plans() -> Mapping[str, BaseLoraSweepPlan]:
    """Production base-LoRA plan table - used as default hook.

    Returns:
        Every declared plan, in table order.
    """
    return BASE_LORA_SWEEP_PLANS


base_lora_sweep_plans: BaseLoraSweepPlansProto = _default_base_lora_sweep_plans

qa_plans: QaPlansProto = _default_qa_plans

ladder_shapes: LadderShapesProto = _default_ladder_shapes

trace_rungs: TraceRungsProto = _default_trace_rungs

cost_shapes_hook: CostShapesProto = _default_cost_shapes

forward_shapes: ForwardShapesProto = _default_forward_shapes

train_shapes: TrainShapesProto = _default_train_shapes

benchmark_shapes: BenchmarkShapesProto = _default_benchmark_shapes

probed_shapes_hook: ProbedShapesProto = _default_probed_shapes


__all__ = [
    "BaseLoraSweepPlansProto",
    "BenchmarkShapesProto",
    "CartridgePlansProto",
    "CompanionSweepPlansProto",
    "CompositionSweepPlansProto",
    "CostShapesProto",
    "ForwardShapesProto",
    "LadderShapesProto",
    "ProbedShapesProto",
    "QaPlansProto",
    "TraceRungsProto",
    "TrainShapesProto",
    "VariedCompanionSweepPlansProto",
    "base_lora_sweep_plans",
    "benchmark_shapes",
    "cartridge_plans",
    "companion_sweep_plans",
    "composition_sweep_plans",
    "cost_shapes_hook",
    "diverse_companion_sweep_plans",
    "forward_shapes",
    "ladder_shapes",
    "probed_shapes_hook",
    "qa_plans",
    "trace_rungs",
    "train_shapes",
    "varied_companion_sweep_plans",
]
