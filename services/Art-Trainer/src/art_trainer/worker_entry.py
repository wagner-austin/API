"""RQ worker entry point for art-trainer background jobs."""

from __future__ import annotations

from typing import Protocol

from platform_core.config import _require_env_str
from platform_core.logging import get_logger, setup_logging
from platform_workers.rq_harness import WorkerConfig

from art_trainer import _test_hooks


class LoggerProtocol(Protocol):
    """Protocol for logger used in worker entry."""

    def info(self, message: str, *, extra: dict[str, str]) -> None:
        """Log an info message.

        Args:
            message: Log message.
            extra: Extra log fields.
        """
        ...


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None:
        """Run the worker with the given config.

        Args:
            config: Worker configuration.
        """
        ...


def _build_config() -> WorkerConfig:
    """Build worker configuration from environment variables.

    Returns:
        Worker configuration.
    """
    redis_url = _require_env_str("REDIS_URL")
    return {
        "redis_url": redis_url,
        "queue_name": "art-trainer",
        "events_channel": "art-trainer:events",
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
    """Start the RQ worker for art-trainer background jobs.

    Args:
        config: Worker configuration. If None, builds from environment.
        logger: Logger instance. If None, uses default logger after setup.
        runner: Worker runner function. If None, uses the worker_runner hook.
    """
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="art-trainer-worker",
        instance_id=None,
        extra_fields=None,
    )
    resolved_logger: LoggerProtocol = logger if logger is not None else get_logger(__name__)
    resolved_config: WorkerConfig = config if config is not None else _build_config()
    resolved_runner: WorkerRunnerProtocol = (
        runner if runner is not None else _test_hooks.worker_runner
    )
    _run_worker(resolved_config, resolved_logger, resolved_runner)


if __name__ == "__main__":
    main()
