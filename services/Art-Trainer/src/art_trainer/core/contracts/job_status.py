"""The LoRA training job's status vocabulary.

The orchestrator writes QUEUED and the worker writes the rest into the job's
status key, and get_status reads it back; this is the one definition of the
words all three use.
"""

from __future__ import annotations

from enum import StrEnum


class LoraJobStatus(StrEnum):
    """Where a LoRA training job is, as its status key and status response spell it."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


__all__ = ["LoraJobStatus"]
