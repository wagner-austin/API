from __future__ import annotations

import logging

from platform_core.json_utils import dump_json_str
from platform_core.trainer_metrics_events import (
    TrainerConfigV1,
    encode_trainer_metrics_event,
    make_config_event,
)
from platform_discord.trainer.handler import decode_trainer_event


def test_config_with_optional_fields_decodes() -> None:
    ev: TrainerConfigV1 = make_config_event(
        job_id="r",
        user_id=1,
        model_family="gpt2",
        model_size="small",
        total_epochs=3,
        queue="training",
        cpu_cores=8,
        memory_mb=2048,
        optimal_threads=4,
        optimal_workers=2,
        batch_size=2,
        learning_rate=5e-4,
    )
    out = decode_trainer_event(encode_trainer_metrics_event(ev))
    assert out is not None and out.get("optimal_threads") == 4


def test_invalid_payload_returns_none() -> None:
    assert decode_trainer_event("not json") is None
    assert decode_trainer_event("[]") is None
    bad = dump_json_str({"type": "trainer.metrics.config.v1"})
    assert decode_trainer_event(bad) is None


logger = logging.getLogger(__name__)
