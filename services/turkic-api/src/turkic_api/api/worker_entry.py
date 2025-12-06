from __future__ import annotations

from platform_core.job_events import default_events_channel
from platform_core.logging import get_logger, setup_logging
from platform_core.queues import TURKIC_QUEUE
from platform_workers.rq_harness import WorkerConfig, run_rq_worker

from turkic_api.api.config import settings_from_env
from turkic_api.api.logging_fields import LOG_EXTRA_FIELDS


def _build_config() -> WorkerConfig:
    settings = settings_from_env()
    return {
        "redis_url": settings["redis_url"],
        "queue_name": TURKIC_QUEUE,
        "events_channel": default_events_channel("turkic"),
    }


def main() -> None:
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="turkic-worker",
        instance_id=None,
        extra_fields=LOG_EXTRA_FIELDS,
    )
    logger = get_logger(__name__)
    cfg = _build_config()
    logger.info(
        "Starting RQ worker",
        extra={
            "redis_url": cfg["redis_url"],
            "queue_name": cfg["queue_name"],
            "events_channel": cfg["events_channel"],
        },
    )
    run_rq_worker(cfg)


if __name__ == "__main__":
    main()
