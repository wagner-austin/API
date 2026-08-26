"""What an experiment emits, and what it takes to subtract two of them.

The one shape every research run produces, whatever it ran. The submission
layer already generalises -- an hpc3 run document names any command for any
project -- and each experiment already runs however it likes. What was
missing was an obligation on what a command LEAVES BEHIND, so nothing tied a
number to the configuration that produced it or to the experiment it belongs
to.

WHY OBSERVATIONS ARE NAMED SCALARS. Comparability is about numbers someone
will subtract: a cloze accuracy, an AUC, a wall clock, a binding energy. The
experiment-specific payload -- 2,627 per-item cloze outcomes, a per-step
rollout trace, an assigned-formula table -- stays with the experiment and
appears here only as :attr:`RunRecord.payload_digest`. That lets this layer
check two runs for bit-identity without understanding a single byte of what
they produced, which is what keeps it usable by research it was not written
for.

WHY COMPARISON LIVES HERE AND NOT IN THE EXPERIMENT. Subtracting two numbers
is the step where comparability is either honoured or lost, and every
experiment that re-implements it re-decides whether to check. The ablation
subtracted arm B from arm A across a torch major version for weeks because
nothing owned this step.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

from platform_core.comparability import (
    Calibration,
    IdenticalVerdict,
    OffsetVerdict,
    RunFingerprint,
    UncalibratedVerdict,
    compare_configurations,
    decode_run_fingerprint,
    encode_run_fingerprint,
)
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_float,
    require_list,
    require_str,
)

#: What an unmeasured payload digest looks like. Empty means the run emitted
#: no payload worth hashing, not that hashing failed -- a run that produced
#: bytes and could not hash them must fail rather than record this.
NO_PAYLOAD = ""


class Observation(TypedDict):
    """One named number a run produced.

    Attributes:
        name: What was measured, e.g. ``"cloze_accuracy"`` or ``"auc"``.
            Stable across runs of one experiment, because it is what pairs
            two runs' numbers with each other.
        value: The number.
    """

    name: str
    value: float


class RunRecord(TypedDict):
    """One run's numbers, and everything needed to judge whether they compare.

    Attributes:
        experiment: What was run, e.g.
            ``"wiki-corpus-extraction-ablation"``. Two records from different
            experiments are not comparable at all, and this is what makes
            that checkable rather than assumed.
        label: Which run within the experiment, e.g. ``"armB-s42"``.
        fingerprint: The configuration it ran under.
        observations: The named numbers, sorted by name at construction so
            two records list them in one order.
        payload_digest: Digest of the experiment-specific output, or
            :const:`NO_PAYLOAD`. Lets two runs be checked for bit-identity
            without this layer understanding the payload.
    """

    experiment: str
    label: str
    fingerprint: RunFingerprint
    observations: tuple[Observation, ...]
    payload_digest: str


class ObservationDelta(TypedDict):
    """One observation's difference between two runs.

    Attributes:
        name: The observation both runs reported.
        left: The left run's value.
        right: The right run's value.
        difference: ``right - left``, with any calibrated offset already
            applied, so a reader subtracts nothing further.
    """

    name: str
    left: float
    right: float
    difference: float


class RunComparison(TypedDict):
    """Two runs whose configurations permit subtraction, and the differences.

    Attributes:
        kind: Discriminant.
        verdict: Why subtraction was permitted -- identical configurations,
            or differing ones fully covered by measured offsets. Carried
            rather than discarded so a reader can audit which correction was
            applied.
        deltas: One entry per observation both runs reported, in name order.
        unmatched: Observation names only one run reported. Reported rather
            than dropped: a metric that silently disappears between two runs
            is a finding, and a comparison that omits it looks complete.
    """

    kind: Literal["compared"]
    verdict: IdenticalVerdict | OffsetVerdict
    deltas: tuple[ObservationDelta, ...]
    unmatched: tuple[str, ...]


def _observation_name(observation: Observation) -> str:
    """Sort key putting observations in name order.

    A named function rather than a lambda because the lambda's parameter
    infers as ``Any`` under this repo's mypy settings, and an ``Any`` here
    would mean the sort key was unchecked.

    Args:
        observation: The observation to key.

    Returns:
        Its name.
    """
    return observation["name"]


def run_record(
    experiment: str,
    label: str,
    fingerprint: RunFingerprint,
    observations: tuple[Observation, ...],
    payload_digest: str,
) -> RunRecord:
    """Build a record, putting the observations in canonical order.

    Args:
        experiment: What was run.
        label: Which run within it.
        fingerprint: The configuration it ran under.
        observations: The named numbers, in any order.
        payload_digest: Digest of the experiment-specific output, or
            :const:`NO_PAYLOAD`.

    Returns:
        The record, observations sorted by name.

    Raises:
        ValueError: When the experiment or label is empty, or two
            observations share a name. An unnamed run cannot be paired with
            anything, and a duplicated observation name makes the pairing
            ambiguous -- silently keeping one of the two would decide which
            number a later contrast reads.
    """
    if experiment == "":
        raise ValueError("experiment must name what was run")
    if label == "":
        raise ValueError("label must name which run within the experiment")
    names = [o["name"] for o in observations]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise ValueError(f"observations must have distinct names; repeated: {duplicated}")
    return RunRecord(
        experiment=experiment,
        label=label,
        fingerprint=fingerprint,
        observations=tuple(sorted(observations, key=_observation_name)),
        payload_digest=payload_digest,
    )


def compare_run_records(
    left: RunRecord,
    right: RunRecord,
    calibrations: tuple[Calibration, ...],
) -> RunComparison | UncalibratedVerdict:
    """Difference two runs' observations, if their configurations allow it.

    The configuration is judged FIRST. When it does not permit subtraction,
    no differences are computed and none are reported -- returning numbers
    beside a "not comparable" note is how a caller ends up using them.

    Args:
        left: One run.
        right: The other.
        calibrations: Measured offsets available to bridge configuration
            differences.

    Returns:
        A :class:`RunComparison` when subtraction is permitted, otherwise the
        :class:`UncalibratedVerdict` naming the axes that prevent it.

    Raises:
        ValueError: When the two records name different experiments. That is
            a caller mistake rather than a data condition: no calibration
            bridges two different questions, so there is nothing to report a
            verdict about.
    """
    if left["experiment"] != right["experiment"]:
        raise ValueError(
            f"cannot compare runs from different experiments: "
            f"{left['experiment']!r} and {right['experiment']!r}"
        )

    verdict = compare_configurations(left["fingerprint"], right["fingerprint"], calibrations)
    if verdict["kind"] == "uncalibrated":
        return verdict

    offset = verdict["offset"] if verdict["kind"] == "offset" else 0.0
    by_name = {o["name"]: o["value"] for o in right["observations"]}
    deltas = tuple(
        ObservationDelta(
            name=o["name"],
            left=o["value"],
            right=by_name[o["name"]],
            difference=by_name[o["name"]] - o["value"] - offset,
        )
        for o in left["observations"]
        if o["name"] in by_name
    )
    paired = {d["name"] for d in deltas}
    seen = {o["name"] for o in left["observations"]} | set(by_name)
    unmatched = tuple(sorted(seen - paired))
    return RunComparison(kind="compared", verdict=verdict, deltas=deltas, unmatched=unmatched)


class ObservationAgreement(TypedDict):
    """One observation's values across a set of runs.

    Attributes:
        name: The observation every run in the set reported.
        values: That observation's value from each run, in the order the runs
            were given. Kept in full rather than reduced to a summary,
            because which run is the odd one out is the whole finding when a
            set of three or more disagrees.
        distinct: How many different values appear. One means bit-identical
            agreement across every run -- the only value that means agreement,
            since a difference of 1e-16 is still a difference and the reason
            this exists is that "close enough" was how a stack change got
            through.
        spread: Largest value minus smallest. Zero exactly when ``distinct``
            is one; carried alongside it because the SIZE of a disagreement
            says whether it is last-bit rounding or a different computation.
    """

    name: str
    values: tuple[float, ...]
    distinct: int
    spread: float


class RunAgreement(TypedDict):
    """Whether a set of runs computed the same numbers.

    Distinct from :class:`RunComparison`, which asks a different question and
    must not be reached for here. That one subtracts two runs whose
    configurations a calibration permits subtracting, and refuses when none
    does. This one asks whether N runs AGREE, which is exactly the measurement
    that would establish such a calibration -- so requiring one first would be
    circular, and it deliberately does not consult them.

    The corollary is that the caller owns the configuration question. This
    reports agreement between whatever it was handed; that the runs differ
    only on the axis under study is established with
    :func:`~platform_core.comparability.find_differences`, not here.

    Attributes:
        experiment: The experiment all the runs answer. They must agree on it,
            since agreement between answers to different questions is not a
            quantity.
        runs: How many runs were compared.
        shared: One entry per observation EVERY run reported, in name order.
        unmatched: Observation names some run reported and some did not.
            Reported rather than dropped: a ladder missing a rung agrees
            trivially over the rungs it kept, and a set of shared results that
            silently omitted it would read as complete.
    """

    experiment: str
    runs: int
    shared: tuple[ObservationAgreement, ...]
    unmatched: tuple[str, ...]


def agree_across_runs(records: tuple[RunRecord, ...]) -> RunAgreement:
    """Report whether several runs produced the same numbers.

    Args:
        records: The runs to compare, in the order the caller wants their
            values reported.

    Returns:
        The agreement: every observation all of them share, with its values
        and whether those values are identical.

    Raises:
        ValueError: If fewer than two records are given -- agreement is a
            property of a set and a set of one always has it, so answering
            would be reporting a fact about arithmetic rather than about the
            runs. Also if the records name different experiments, for the
            reason :class:`RunAgreement` gives.
    """
    if len(records) < 2:
        raise ValueError(f"agreement needs at least two runs, got {len(records)}")

    experiments = {record["experiment"] for record in records}
    if len(experiments) != 1:
        raise ValueError(
            "cannot judge agreement across different experiments: "
            + ", ".join(sorted(repr(name) for name in experiments))
        )

    by_name = [{o["name"]: o["value"] for o in record["observations"]} for record in records]
    every = set(by_name[0]).intersection(*by_name[1:])
    seen: set[str] = set()
    for values in by_name:
        seen |= set(values)

    shared = tuple(
        _observation_agreement(name, tuple(values[name] for values in by_name))
        for name in sorted(every)
    )
    return RunAgreement(
        experiment=records[0]["experiment"],
        runs=len(records),
        shared=shared,
        unmatched=tuple(sorted(seen - every)),
    )


def _observation_agreement(name: str, values: tuple[float, ...]) -> ObservationAgreement:
    """Judge one observation's values.

    Args:
        name: The observation's name.
        values: Its value from each run, in run order.

    Returns:
        The agreement entry for it.
    """
    return ObservationAgreement(
        name=name,
        values=values,
        distinct=len(set(values)),
        spread=max(values) - min(values),
    )


def encode_run_record(record: RunRecord) -> JSONObject:
    """Encode a record for the ledger.

    Args:
        record: The record to encode.

    Returns:
        A JSON object carrying the experiment, the label, the nested
        fingerprint, the observations as a list, and the payload digest.
    """
    return {
        "experiment": record["experiment"],
        "label": record["label"],
        "fingerprint": encode_run_fingerprint(record["fingerprint"]),
        "observations": [{"name": o["name"], "value": o["value"]} for o in record["observations"]],
        "payload_digest": record["payload_digest"],
    }


def decode_run_record(value: JSONValue) -> RunRecord:
    """Validate a JSON value as a run record.

    Args:
        value: The value to validate, typically read from the ledger.

    Returns:
        The validated record, observations in canonical order.

    Raises:
        JSONTypeError: When the value is not an object, any field is absent
            or mistyped, the nested fingerprint fails its own validation, or
            an observation is not an object with a string name and a numeric
            value.
        ValueError: When the experiment or label is empty, or two
            observations share a name.
    """
    obj = narrow_json_to_dict(value)
    raw = require_list(obj, "observations")
    observations: list[Observation] = []
    for entry in raw:
        item = narrow_json_to_dict(entry)
        name = require_str(item, "name")
        if name == "":
            raise JSONTypeError("Observation 'name' must say what was measured")
        observations.append(Observation(name=name, value=require_float(item, "value")))
    return run_record(
        experiment=require_str(obj, "experiment"),
        label=require_str(obj, "label"),
        fingerprint=decode_run_fingerprint(require_dict(obj, "fingerprint")),
        observations=tuple(observations),
        payload_digest=require_str(obj, "payload_digest"),
    )


__all__ = [
    "NO_PAYLOAD",
    "Observation",
    "ObservationAgreement",
    "ObservationDelta",
    "RunAgreement",
    "RunComparison",
    "RunRecord",
    "agree_across_runs",
    "compare_run_records",
    "decode_run_record",
    "encode_run_record",
    "run_record",
]
