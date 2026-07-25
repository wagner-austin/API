"""Summary statistics over repeated fit timings.

Pure functions with no I/O and no clock access, so the estimator choice is
directly testable from a fixed list of samples.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from .types import ERR_NO_TIMING_SAMPLES, TimingSummary


def summarize_timings(samples_s: Sequence[float]) -> TimingSummary:
    """Reduce repeated fit timings to the summary the manifest records.

    The canonical value is the median. A minimum would report the fastest
    repeat, but the first fits after an idle period run with full turbo
    headroom -- a different power regime rather than noise -- so a minimum
    systematically reports a cold-start outlier in place of the steady state
    that sustained training actually experiences.

    Args:
        samples_s: Timed repeats in seconds, in execution order. Must be
            non-empty.

    Returns:
        Summary carrying the canonical value alongside the full spread, so a
        reader can see whether a difference is resolvable or is noise.

    Raises:
        ValueError: If ``samples_s`` is empty, which means the caller
            requested zero timed repeats and there is nothing to summarise.
    """
    if len(samples_s) == 0:
        raise ValueError(
            f"[{ERR_NO_TIMING_SAMPLES}] Cannot summarise timings: no samples were recorded"
        )

    ordered = list(samples_s)
    median_s = statistics.median(ordered)
    return {
        "canonical_s": median_s,
        "min_s": min(ordered),
        "median_s": median_s,
        "mean_s": statistics.fmean(ordered),
        "max_s": max(ordered),
        "samples_s": ordered,
    }


__all__ = ["summarize_timings"]
