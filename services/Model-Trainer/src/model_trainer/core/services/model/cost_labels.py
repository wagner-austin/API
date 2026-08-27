"""What a cost measurement is called in a record.

Torch-free, because the reports that read these records run on a laptop.

Shared by the per-call attention benchmark and the end-to-end forward one so
that a single report can decide both. That matters more than saving a few
lines: the whole point of the forward measurement is to be read against the
per-call one, and two vocabularies would mean two readers, which would mean
two places for "is this ratio trustworthy" to be decided differently.
"""

from __future__ import annotations

#: The unforced arm -- whatever the dispatcher chose on its own.
DEFAULT_KEY = "default"

#: Suffix for the observation carrying seconds per call.
SECONDS_SUFFIX = "seconds"

#: Suffix for the slowest-minus-fastest batch, in seconds per call. Carried
#: beside every median because a median with an enormous spread is a number
#: that must not be compared with another one.
SPREAD_SUFFIX = "spread"

#: Suffix for peak CUDA bytes allocated during the timed run.
PEAK_SUFFIX = "peak_bytes"

#: Suffix for whether the work fitted in memory at all. An out-of-memory is
#: not a failed measurement -- it is the strongest cost result there is, so
#: it is recorded rather than raised.
FITTED_SUFFIX = "fitted"

#: How a boolean is carried in a record, which holds only numbers.
TRUE_VALUE = 1.0
FALSE_VALUE = 0.0


def labelled(prefix: str, backend: str, suffix: str) -> str:
    """Name one measurement of one thing under one backend.

    Args:
        prefix: What was measured, including its dimensions -- so a record
            read on its own still says what it timed, and a shape edited
            without renaming cannot quietly reuse an old name.
        backend: :data:`DEFAULT_KEY` or a backend key.
        suffix: Which measurement.

    Returns:
        e.g. ``cost-grid-b8-s2048-h12-d64|math|seconds``.
    """
    return f"{prefix}|{backend}|{suffix}"


__all__ = [
    "DEFAULT_KEY",
    "FALSE_VALUE",
    "FITTED_SUFFIX",
    "PEAK_SUFFIX",
    "SECONDS_SUFFIX",
    "SPREAD_SUFFIX",
    "TRUE_VALUE",
    "labelled",
]
