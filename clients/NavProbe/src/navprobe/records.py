"""Typed records for probe runs.

Every record is a ``TypedDict`` whose collection fields are tuples, so a
decoded record cannot be mutated in place by a later stage.

The shapes are declared here and their codecs live in
:mod:`navprobe.wireformat`, one ``encode_*``/``decode_*`` pair per record, with
every decoded field passing a ``require_*`` check. The split is deliberate:
:mod:`navprobe.rollout`, :mod:`navprobe.comparison`, and
:mod:`navprobe.experiment` all depend on the shapes and none of them depends on
the on-disk representation, so the format can change without touching them.
"""

from __future__ import annotations

from typing import TypedDict


class RunSpec(TypedDict):
    """The experimental condition a rollout was produced under.

    Attributes:
        label: Human-readable name of the condition, e.g. ``"fresh-process"``.
        seed: The seed pinned for the rollout.
        step_count: Number of steps the rollout was asked to take.
        world_count: Number of parallel worlds the simulator was configured
            with. Part of the condition because batched execution is the
            mechanism under test.
    """

    label: str
    seed: int
    step_count: int
    world_count: int


class StepRecord(TypedDict):
    """One step's digest within a rollout.

    Attributes:
        step_index: Zero-based position within the rollout.
        digest: Lowercase hexadecimal digest of the step's observation.
    """

    step_index: int
    digest: str


class RunRecord(TypedDict):
    """A complete rollout: its condition, its steps, and its run digest.

    Attributes:
        spec: The condition this rollout was produced under.
        steps: Per-step digests in step order.
        digest: Digest folding every step digest, in order.
    """

    spec: RunSpec
    steps: tuple[StepRecord, ...]
    digest: str


class ComparisonRecord(TypedDict):
    """The verdict from comparing two rollouts.

    Attributes:
        left_label: Condition label of the first rollout.
        right_label: Condition label of the second rollout.
        digests_match: Whether the two run digests are identical.
        first_divergent_step: Index of the earliest step whose digests differ,
            or ``None`` when every compared step agreed. Absence here means
            agreement, not an unknown.
        compared_step_count: Number of step positions actually compared, which
            is the shorter of the two rollouts.
    """

    left_label: str
    right_label: str
    digests_match: bool
    first_divergent_step: int | None
    compared_step_count: int


class TrialSpec(TypedDict):
    """The design of a determinism trial.

    Attributes:
        seed: The seed every repetition is pinned to. One seed is the whole
            point: repetitions under different seeds would diverge by design
            and prove nothing.
        step_count: Number of steps each repetition takes.
        repetitions: Number of independent rollouts to compare. Two is the
            minimum that can establish anything; a single rollout has nothing
            to disagree with.
    """

    seed: int
    step_count: int
    repetitions: int


class TrialRecord(TypedDict):
    """The outcome of a determinism trial.

    Deliberately flat. It is the row that goes in a results table, and the
    per-repetition detail it summarises is already persistable as individual
    run records, so nesting them here would store the same bytes twice.

    Attributes:
        spec: The trial design.
        world_count: Parallel worlds the simulator reported, carried because
            batch width is the variable under test in a sweep.
        reference_digest: Run digest of the first repetition, which every later
            repetition was compared against.
        deterministic: Whether every repetition matched the reference.
        first_divergent_step: Earliest step index at which any repetition left
            the reference, or ``None`` when none did.
    """

    spec: TrialSpec
    world_count: int
    reference_digest: str
    deterministic: bool
    first_divergent_step: int | None


class SceneSpec(TypedDict):
    """A parametrised lattice of spheres dropped into a walled box.

    The scene family every determinism sweep in this package is measured over.
    It is declared as data rather than written as XML per experiment so that a
    published result names a value of this type and anyone can rebuild the exact
    scene it was measured on.

    Whether bodies touch each other is *derived* from ``spacing`` and ``radius``
    rather than stored, because a stored flag could disagree with the geometry
    it describes. See :func:`navprobe.scenes.bodies_touch`.

    Attributes:
        body_count: Number of free spheres.
        lattice_width: Columns per row. A body count above
            ``lattice_width * lattice_width`` starts a second layer, so setting
            this to ``body_count`` produces a single row that never stacks.
        spacing: Centre-to-centre distance between neighbouring lattice sites.
        radius: Sphere radius.
        timestep: Simulation timestep.
    """

    body_count: int
    lattice_width: int
    spacing: float
    radius: float
    timestep: float


class DispersionRecord(TypedDict):
    """How far apart repeated rollouts of one configuration ended up.

    A digest comparison answers whether runs differ. This answers by how much,
    in the units of whatever the simulator observes — metres for a state
    observation, depth units for a rendered one. The instrument does not know
    which, and deliberately does not say.

    Attributes:
        repetitions: Number of rollouts compared.
        observation_length: Elements in each rollout's final observation.
        max_spread: Largest element-wise range across the repetitions.
        mean_spread: Mean element-wise range across the repetitions.
    """

    repetitions: int
    observation_length: int
    max_spread: float
    mean_spread: float


class ObservationRecord(TypedDict):
    """One rollout's final observation, kept so it can leave the process.

    A run record carries step digests, which is what makes it small and what
    makes it useless for asking *how far apart* two rollouts are. This carries
    the values themselves, for the one case that needs them: two configurations
    that cannot share a process — such as two MuJoCo-Warp devices, since Warp's
    device is global state — and so cannot be compared live.

    Only the final observation, not the whole stream. The stream is what the
    digest already summarises; the endpoint is what a magnitude is measured on.

    Attributes:
        label: Human-readable name of the environment that produced it.
        seed: The seed the rollout was pinned to.
        step_count: Steps taken before the observation was captured.
        values: The observation, in the simulator's own element order.
    """

    label: str
    seed: int
    step_count: int
    values: tuple[float, ...]


class DivergenceRecord(TypedDict):
    """How far apart two different configurations ended up.

    The sibling of :class:`DispersionRecord`, and the distinction is the point.
    Dispersion measures one configuration against itself across repetitions;
    divergence measures one configuration against *another* — a different
    device, backend, or library version — over a single rollout each.

    ``differing_elements`` is carried separately from the magnitudes because the
    two answer different questions. A large maximum over one element is a
    localised artefact; a small maximum over a third of them is a systematic
    difference, and only the count distinguishes those.

    Attributes:
        observation_length: Elements in each side's final observation.
        differing_elements: How many elements are not bit-identical.
        max_absolute_difference: Largest element-wise absolute difference.
        mean_absolute_difference: Mean absolute difference over the differing
            elements only, so it is not diluted by the ones that agree.
    """

    observation_length: int
    differing_elements: int
    max_absolute_difference: float
    mean_absolute_difference: float


class SweepEntry(TypedDict):
    """One scene's determinism verdict within a sweep.

    Attributes:
        scene: The scene the trial was run on.
        trial: The verdict for that scene.
    """

    scene: SceneSpec
    trial: TrialRecord


__all__ = [
    "ComparisonRecord",
    "DispersionRecord",
    "RunRecord",
    "RunSpec",
    "SceneSpec",
    "StepRecord",
    "SweepEntry",
    "TrialRecord",
    "TrialSpec",
]
