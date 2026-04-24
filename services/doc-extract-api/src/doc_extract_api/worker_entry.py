"""RQ worker entry point for doc-extract-api background jobs."""

from __future__ import annotations

from typing import Protocol

from platform_core.config import _require_env_str
from platform_core.logging import get_logger, setup_logging
from platform_workers.rq_harness import WorkerConfig, run_rq_worker

from doc_extract_api import _test_hooks

_QUEUE_NAME: str = "doc_extract"
_EVENTS_CHANNEL: str = "doc_extract:events"


class _LoggerProtocol(Protocol):
    """Protocol for logger used in worker entry."""

    def info(self, message: str, *, extra: dict[str, str]) -> None: ...


def _get_default_runner() -> _test_hooks.WorkerRunnerProtocol:
    """Get the default worker runner.

    Returns test_runner from _test_hooks if set (for testing),
    otherwise run_rq_worker.

    Returns:
        A worker runner function.
    """
    if _test_hooks.test_runner is not None:
        return _test_hooks.test_runner
    runner: _test_hooks.WorkerRunnerProtocol = run_rq_worker
    return runner


def _build_config() -> WorkerConfig:
    """Build worker configuration from environment variables.

    Returns:
        WorkerConfig with Redis URL, queue name, and events channel.
    """
    redis_url = _require_env_str("REDIS_URL")
    return WorkerConfig(
        redis_url=redis_url,
        queue_name=_QUEUE_NAME,
        events_channel=_EVENTS_CHANNEL,
    )


def _run_worker(
    config: WorkerConfig,
    logger: _LoggerProtocol,
    runner: _test_hooks.WorkerRunnerProtocol,
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
    logger: _LoggerProtocol | None = None,
    runner: _test_hooks.WorkerRunnerProtocol | None = None,
) -> None:
    """Start the RQ worker for doc-extract-api background jobs.

    Args:
        config: Worker configuration. If None, builds from environment.
        logger: Logger instance. If None, uses default logger after setup.
        runner: Worker runner function. If None, uses _get_default_runner().
    """
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="doc-extract-worker",
        instance_id=None,
        extra_fields=None,
    )
    resolved_logger: _LoggerProtocol = logger if logger is not None else get_logger(__name__)
    resolved_config: WorkerConfig = config if config is not None else _build_config()
    resolved_runner = runner if runner is not None else _get_default_runner()
    _run_worker(resolved_config, resolved_logger, resolved_runner)


if __name__ == "__main__":
    main()
