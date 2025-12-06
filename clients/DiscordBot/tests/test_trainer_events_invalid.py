from __future__ import annotations

import logging

from platform_core.json_utils import dump_json_str
from platform_discord.trainer.handler import decode_trainer_event


def test_decode_progress_invalid_types_returns_none() -> None:
    bad = dump_json_str(
        {
            "type": "trainer.metrics.progress.v1",
            "job_id": "r",
            "user_id": 1,
            "epoch": "x",  # Invalid: should be int
            "total_epochs": 2,
            "step": 10,
            "loss": 1.0,
        }
    )
    assert decode_trainer_event(bad) is None


def test_decode_completed_invalid_returns_none() -> None:
    bad = dump_json_str(
        {
            "type": "trainer.metrics.completed.v1",
            "job_id": "r",
            "user_id": 1,
            "loss": "a",  # Invalid: should be float
            "perplexity": 2,
            "artifact_path": "/x",
        }
    )
    assert decode_trainer_event(bad) is None


def test_decode_failed_invalid_error_kind_returns_none() -> None:
    bad = dump_json_str(
        {
            "type": "trainer.job.failed.v1",
            "domain": "trainer",
            "job_id": "r",
            "user_id": 1,
            "error_kind": "oops",  # Invalid: must be "user" or "system"
            "message": "stop",
        }
    )
    assert decode_trainer_event(bad) is None


logger = logging.getLogger(__name__)
