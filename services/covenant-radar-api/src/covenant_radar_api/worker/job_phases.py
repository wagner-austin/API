"""The progress vocabularies the optimize and explain jobs report.

The classification and regression variants of each job report the same
steps, so each vocabulary is declared once here and both variants import it.
"""

from __future__ import annotations

from enum import StrEnum


class OptimizePhase(StrEnum):
    """The step an optimize job has entered."""

    LOADING_DATA = "loading_data"
    FEATURE_ENGINEERING = "feature_engineering"
    OPTIMIZING = "optimizing"
    SAVING = "saving"


class ExplainJobStatus(StrEnum):
    """The step an explain job has reached."""

    STARTED = "started"
    LOADING_MODEL = "loading_model"
    LOADING_DATA = "loading_data"
    COMPUTING = "computing"
    COMPLETE = "complete"


__all__ = ["ExplainJobStatus", "OptimizePhase"]
