"""Tests for the fit-time summary statistics."""

from __future__ import annotations

import pytest

from covenant_ml.benchmarking.timing import summarize_timings
from covenant_ml.benchmarking.types import ERR_NO_TIMING_SAMPLES


def test_summary_reports_every_statistic() -> None:
    summary = summarize_timings([3.0, 1.0, 2.0])
    assert summary["min_s"] == 1.0
    assert summary["median_s"] == 2.0
    assert summary["mean_s"] == 2.0
    assert summary["max_s"] == 3.0
    assert summary["samples_s"] == [3.0, 1.0, 2.0]


def test_canonical_is_the_median_not_the_minimum() -> None:
    """A cold-start outlier must not become the canonical number.

    The first sample is 40% under the steady state, which is what a
    turbo-boosted first fit looks like. The median ignores it; a minimum
    would report it as the result.
    """
    summary = summarize_timings([0.49, 0.83, 0.85, 0.84, 0.86])
    assert summary["canonical_s"] == 0.84
    assert summary["min_s"] == 0.49


def test_single_sample_summarises_to_itself() -> None:
    summary = summarize_timings([1.25])
    assert summary["canonical_s"] == 1.25
    assert summary["min_s"] == 1.25
    assert summary["max_s"] == 1.25


def test_samples_are_copied_not_aliased() -> None:
    source = [1.0, 2.0]
    summary = summarize_timings(source)
    source.append(99.0)
    assert summary["samples_s"] == [1.0, 2.0]


def test_empty_samples_raise() -> None:
    with pytest.raises(ValueError, match=ERR_NO_TIMING_SAMPLES):
        summarize_timings([])
