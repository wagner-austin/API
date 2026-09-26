"""Tests for the optimize and explain jobs' progress vocabularies."""

from __future__ import annotations

from covenant_radar_api.worker.job_phases import ExplainJobStatus, OptimizePhase


def test_optimize_phases_are_their_wire_words_in_job_order() -> None:
    """The phases read as the words a progress event carries, in run order."""
    assert [str(m) for m in OptimizePhase] == [
        "loading_data",
        "feature_engineering",
        "optimizing",
        "saving",
    ]


def test_explain_statuses_are_their_wire_words_in_job_order() -> None:
    """The statuses read as the words a progress event carries, in run order."""
    assert [str(m) for m in ExplainJobStatus] == [
        "started",
        "loading_model",
        "loading_data",
        "computing",
        "complete",
    ]
