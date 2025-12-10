"""RQ worker entry point for data-bank-api background jobs."""

from __future__ import annotations

from typing import Protocol

from platform_core.config import _require_env_str
from platform_core.job_events import default_events_channel
from platform_core.logging import get_logger, setup_logging
from platform_core.queues import DATA_BANK_QUEUE
from platform_workers.rq_harness import WorkerConfig, run_rq_worker

from data_bank_api import _test_hooks


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
    """Build worker configuration from environment variables."""
    redis_url = _require_env_str("REDIS_URL")
    return {
        "redis_url": redis_url,
        "queue_name": DATA_BANK_QUEUE,
        "events_channel": default_events_channel("databank"),
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
            "queue": config["queue_name"],
            "events_channel": config["events_channel"],
        },
    )
    runner(config)


def main(
    config: WorkerConfig | None = None,
    logger: LoggerProtocol | None = None,
    runner: WorkerRunnerProtocol | None = None,
) -> None:
    """Start the RQ worker for data-bank-api background jobs.

    Args:
        config: Worker configuration. If None, builds from environment.
        logger: Logger instance. If None, uses default logger after setup.
        runner: Worker runner function. If None, uses _get_default_runner().
    """
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="data-bank-worker",
        instance_id=None,
        extra_fields=None,
    )
    resolved_logger: LoggerProtocol = logger if logger is not None else get_logger(__name__)
    resolved_config: WorkerConfig = config if config is not None else _build_config()
    resolved_runner: WorkerRunnerProtocol = runner if runner is not None else _get_default_runner()
    _run_worker(resolved_config, resolved_logger, resolved_runner)


if __name__ == "__main__":
    main()
