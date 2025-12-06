from __future__ import annotations

from platform_core.config import _require_env_str
from platform_core.job_events import default_events_channel
from platform_core.logging import get_logger, setup_logging
from platform_core.queues import DIGITS_QUEUE
from platform_workers.rq_harness import WorkerConfig, run_rq_worker


def _build_config() -> WorkerConfig:
    """Build worker configuration from environment variables."""
    redis_url = _require_env_str("REDIS_URL")
    return {
        "redis_url": redis_url,
        "queue_name": DIGITS_QUEUE,
        "events_channel": default_events_channel("digits"),
    }


def main() -> None:
    """Start the RQ worker for handwriting-ai digit training jobs."""
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="handwriting-worker",
        instance_id=None,
        extra_fields=None,
    )
    logger = get_logger(__name__)
    cfg = _build_config()
    logger.info(
        "Starting RQ worker",
        extra={
            "queue": cfg["queue_name"],
            "events_channel": cfg["events_channel"],
        },
    )
    run_rq_worker(cfg)


if __name__ == "__main__":
    main()
