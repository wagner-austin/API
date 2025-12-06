"""Minimal subprocess test to see if child can log."""

import multiprocessing as mp

from platform_core.logging import get_logger


def child_func(msg: str) -> str:
    """Child process function."""
    from platform_core.logging import get_logger, setup_logging

    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="test",
        instance_id=None,
        extra_fields=None,
    )
    log = get_logger("handwriting_ai")
    log.info(f"CHILD_START: {msg}")
    return "OK"


if __name__ == "__main__":
    from platform_core.logging import get_logger, setup_logging

    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="test",
        instance_id=None,
        extra_fields=None,
    )
    log = get_logger("handwriting_ai")

    log.info("PARENT_START: About to spawn child")

    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=child_func, args=("Hello from child",))
    proc.start()
    proc.join(timeout=5)

    if proc.is_alive():
        log.info("PARENT_ERROR: Child timed out!")
        proc.terminate()
        proc.join()
    else:
        log.info(f"PARENT_DONE: Child exited with code {proc.exitcode}")
