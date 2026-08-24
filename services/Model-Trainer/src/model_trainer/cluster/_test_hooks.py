"""The one seam the cluster entry point has.

Production points this at the real training job at import time; a test replaces
it to assert on what the entry point composed without running a trainer. The
entry point's whole job is wiring, so this is the only thing worth substituting
in it -- and a hook is how this repo does that, rather than rebinding an
attribute on a production module.
"""

from __future__ import annotations

from typing import Protocol

from platform_core.json_utils import JSONObject

from model_trainer.worker.train_job import process_train_job


class RunJobProto(Protocol):
    """Protocol for the training entry point the cluster root invokes."""

    def __call__(self, payload_raw: JSONObject) -> None:
        """Run one training job from its raw payload."""
        ...


run_job: RunJobProto = process_train_job


def reset_hooks() -> None:
    """Restore the production wiring.

    Called by tests after substituting, so one test's stand-in does not
    answer for every later test in the same worker.
    """
    global run_job
    run_job = process_train_job


__all__ = ["RunJobProto", "reset_hooks", "run_job"]
