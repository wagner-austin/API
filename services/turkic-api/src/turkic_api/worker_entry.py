"""RQ worker entry point for turkic-api background jobs."""

from __future__ import annotations

from typing import Protocol

from platform_core.job_events import default_events_channel
from platform_core.logging import get_logger, setup_logging
from platform_core.queues import TURKIC_QUEUE
from platform_workers.rq_harness import WorkerConfig, run_rq_worker

from turkic_api import _test_hooks
from turkic_api.api.config import settings_from_env
from turkic_api.api.logging_fields import LOG_EXTRA_FIELDS


class LoggerProtocol(Protocol):
    """Protocol for logger used in worker entry."""

    def info(self, message: str, *, extra: dict[str, str]) -> None: ...


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None: ...


def _get_default_runner() -> WorkerRunnerProtocol:
    """Get the default worker runner.

    Returns test_runner from _test_hooks if set (for testing), otherwise run_rq_worker.
    """
    if _test_hooks.test_runner is not None:
        return _test_hooks.test_runner
    return run_rq_worker


def _build_config() -> WorkerConfig:
    """Build worker configuration from settings."""
    settings = settings_from_env()
    return {
        "redis_url": settings["redis_url"],
        "queue_name": TURKIC_QUEUE,
        "events_channel": default_events_channel("turkic"),
    }


def _run_worker(
    config: WorkerConfig,
    logger: LoggerProtocol,
    runner: WorkerRunnerProtocol,
) -> None:
    """Run the worker with provided dependencies.

    Args:
        config: Worker configuration.
        logger: Logger for startup message.
        runner: Function to run the worker.
    """
    logger.info(
        "Starting RQ worker",
        extra={
            "redis_url": config["redis_url"],
            "queue_name": config["queue_name"],
            "events_channel": config["events_channel"],
        },
    )
    runner(config)


def main(
    config: WorkerConfig | None = None,
    logger: LoggerProtocol | None = None,
    runner: WorkerRunnerProtocol | None = None,
) -> None:
    """Start the RQ worker for turkic-api background jobs.

    Args:
        config: Worker configuration. If None, builds from settings.
        logger: Logger instance. If None, uses default logger after setup.
        runner: Worker runner function. If None, uses _get_default_runner().
    """
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="turkic-worker",
        instance_id=None,
        extra_fields=LOG_EXTRA_FIELDS,
    )
    resolved_logger: LoggerProtocol = logger if logger is not None else get_logger(__name__)
    resolved_config: WorkerConfig = config if config is not None else _build_config()
    resolved_runner: WorkerRunnerProtocol = runner if runner is not None else _get_default_runner()
    _run_worker(resolved_config, resolved_logger, resolved_runner)


if __name__ == "__main__":
    main()
