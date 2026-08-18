"""Compare run records and produce a determinism verdict.

The verdict answers two questions that are deliberately kept separate: whether
the runs agree at all, and where they first stopped agreeing. A run digest
alone answers the first and is useless for diagnosis; the first divergent step
localises the failure to a point in the rollout, which is what distinguishes
"diverged immediately" from "drifted after four hundred steps".

Rollouts of different lengths are compared over their common prefix. That is
reported explicitly through ``compared_step_count`` rather than treated as an
error, because a shortened run is itself a finding worth localising.
"""

from __future__ import annotations

from navprobe import NavProbeError
from navprobe.records import ComparisonRecord, RunRecord


class ComparisonError(NavProbeError):
    """Two run records could not be meaningfully compared.

    Args:
        code: Stable identifier in the ``NP-COMPARE-<NNN>`` range.
        message: Human-readable description of what went wrong.
    """


def find_first_divergent_step(left: RunRecord, right: RunRecord) -> int | None:
    """Locate the earliest step index at which two rollouts disagree.

    Args:
        left: The first rollout.
        right: The second rollout.

    Returns:
        The index of the earliest differing step, or ``None`` when every step
        in the common prefix agreed.
    """
    for left_step, right_step in zip(left["steps"], right["steps"], strict=False):
        if left_step["digest"] != right_step["digest"]:
            return left_step["step_index"]
    return None


def compare_runs(left: RunRecord, right: RunRecord) -> ComparisonRecord:
    """Compare two rollouts and produce a verdict.

    The two rollouts must share a seed. Comparing different seeds would produce
    a divergence that says nothing about determinism, and reporting it as one
    would be the instrument manufacturing its own headline result.

    Args:
        left: The first rollout.
        right: The second rollout.

    Returns:
        The verdict, carrying both labels, whether the run digests match, the
        first divergent step if any, and how many steps were compared.

    Raises:
        ComparisonError: When the two rollouts were produced under different
            seeds, or when a run reports matching digests while its steps
            disagree, which means a digest was computed from something other
            than the recorded steps.
    """
    left_seed = left["spec"]["seed"]
    right_seed = right["spec"]["seed"]
    if left_seed != right_seed:
        raise ComparisonError(
            "NP-COMPARE-001",
            f"cannot compare rollouts at different seeds ({left_seed} and {right_seed}); "
            "a divergence between them would not be evidence of non-determinism",
        )

    digests_match = left["digest"] == right["digest"]
    first_divergent_step = find_first_divergent_step(left, right)
    compared_step_count = min(len(left["steps"]), len(right["steps"]))

    if digests_match and first_divergent_step is not None:
        raise ComparisonError(
            "NP-COMPARE-002",
            f"run digests match while step {first_divergent_step} differs; "
            "the run digest was not computed from these steps",
        )

    return ComparisonRecord(
        left_label=left["spec"]["label"],
        right_label=right["spec"]["label"],
        digests_match=digests_match,
        first_divergent_step=first_divergent_step,
        compared_step_count=compared_step_count,
    )


__all__ = ["ComparisonError", "compare_runs", "find_first_divergent_step"]
