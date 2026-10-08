"""Fixtures that keep the calibration machinery out of tests that are not about it.

Two things in the calibration path cost this package's suite its time budget
and, worse, its ability to finish (API board task 8bbe083b):

- ``runner._child_entry`` configures logging for a whole process, because it
  is the entry point of a spawned child. Called inside a pytest worker it
  leaves a QueueHandler feeding a pipe nobody reads, and the worker's exit
  then waits on the queue's blocked feeder thread forever.
  :func:`child_log_queue` gives such a test its queue and undoes the rest.
- ``train_with_config`` always calibrates, and the real calibration spawns
  one training subprocess per candidate, six to nine of them on the limits
  these tests use: 180 s for a two-epoch run on four images.
  :func:`fixed_calibration` binds the calibration hook to a fixed answer for
  tests whose subject is training, not calibration.

conftest.py re-exports both, which is what registers them as fixtures.

:data:`CHILD_HANG_BOUND_S` is the one timeout every test that spawns a real
calibration child gives a child it expects to finish.
"""

from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Generator
from multiprocessing.queues import Queue as MPQueue
from pathlib import Path
from typing import Final

import pytest

from handwriting_ai import _test_hooks
from handwriting_ai._hook_protocols_ml import ResourceLimitsDict
from handwriting_ai._hook_protocols_training import EffectiveConfig
from handwriting_ai.training.calibration.ds_spec import PreprocessSpec
from handwriting_ai.training.dataset import DataLoaderConfig

#: The budget timeout, in seconds, for a real spawned calibration child that
#: is expected to finish: a bound on a hung child, never on a slow one. The
#: child boots a fresh interpreter, imports torch and torchvision and builds
#: ResNet-18 before its dataset is read; on 2026-10-08 that took 5 s on an
#: idle hub, and under load the child first logged 38 s after its spawn and
#: had not finished at the 60 s the cleanup test then allowed, which failed a
#: healthy run (API board task 8bbe083b). The timeout path itself is tested
#: with a timeout of 0.
CHILD_HANG_BOUND_S: Final[float] = 120.0


@pytest.fixture()
def child_log_queue() -> Generator[MPQueue[logging.LogRecord], None, None]:
    """Give a test the log queue for running ``runner._child_entry`` in this process.

    ``_child_entry`` is the entry point of a spawned calibration child, a
    process that exits when it returns, so it configures logging for the whole
    process: the real ``setup_logging`` replaces the root logger's handlers
    and level, and the ``handwriting_ai`` logger gets a new level, loses its
    StreamHandlers and ``propagate``, and gains a QueueHandler feeding
    ``log_q``. Run inside a pytest worker instead, all of it outlives the
    test, and the QueueHandler feeds every later record into a pipe nobody
    reads. When the pipe fills, the queue's feeder thread blocks, and the
    worker's exit waits for it in ``multiprocessing.queues._finalize_join``
    forever: tests/test_calibration_child_runner.py printed "3 passed in
    11.54s" on 2026-10-08 and its pytest was still running nine minutes
    later.

    Teardown puts both loggers back as they were and releases the queue:
    ``cancel_join_thread`` lets this process exit without flushing records
    no test will read, and ``close`` ends the feeder.

    Yields:
        A spawn-context queue to pass as ``log_q``.
    """
    root = logging.getLogger()
    app = logging.getLogger("handwriting_ai")
    root_handlers = tuple(root.handlers)
    root_level = root.level
    app_handlers = tuple(app.handlers)
    app_level = app.level
    app_propagate = app.propagate
    queue: MPQueue[logging.LogRecord] = multiprocessing.get_context("spawn").Queue()
    yield queue
    root.handlers[:] = root_handlers
    root.setLevel(root_level)
    app.handlers[:] = app_handlers
    app.setLevel(app_level)
    app.propagate = app_propagate
    queue.cancel_join_thread()
    queue.close()


@pytest.fixture()
def fixed_calibration() -> None:
    """Bind the calibration hook to one thread, no workers and the requested batch.

    For tests that run ``train_with_config`` to exercise training, progress
    or loader lifecycle. The calibrator itself is tested in
    tests/test_calibrate.py and tests/test_calibration_*.py, and the spawned
    child in tests/test_calibration_runner_subprocess_integration.py; here
    the answer it would give is fixed so the run trains at once. The
    conftest autouse fixture restores the real hook after each test.
    """

    def _fixed(
        ds: PreprocessSpec,
        *,
        limits: ResourceLimitsDict,
        requested_batch_size: int,
        samples: int,
        cache_path: Path,
        ttl_seconds: int,
        force: bool,
    ) -> EffectiveConfig:
        _ = (ds, limits, samples, cache_path, ttl_seconds, force)
        batch_size = max(1, int(requested_batch_size))
        return {
            "intra_threads": 1,
            "interop_threads": None,
            "batch_size": batch_size,
            "loader_cfg": DataLoaderConfig(
                batch_size=batch_size,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                prefetch_factor=2,
            ),
        }

    _test_hooks.calibrate_input_pipeline = _fixed
