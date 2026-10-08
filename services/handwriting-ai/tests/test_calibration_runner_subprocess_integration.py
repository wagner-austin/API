from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path

import pytest
from PIL import Image
from tests._calibration_fixtures import CHILD_HANG_BOUND_S

from handwriting_ai import _test_hooks
from handwriting_ai.training.calibration._types import BudgetConfig, Candidate
from handwriting_ai.training.calibration.runner import SubprocessRunner
from handwriting_ai.training.dataset import PreprocessDataset
from handwriting_ai.training.train_config import default_train_config


class _FakeMNIST:
    def __init__(self, n: int = 8) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), 0), 0


def test_subprocess_runner_child_writes_result_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    """One real spawned child: its result reaches the parent, then nothing of it is left.

    The package's only test that spawns a calibration child which finishes,
    so it carries the whole contract of a clean run: the result file is read
    back, the child's logs reach the parent, and the result directory and the
    child are gone when ``run`` returns. Its sibling that asserted only the
    result, and the clean-run case of tests/test_calibration_temp_cleanup.py,
    each ran the identical spawn, one more fresh interpreter and torch import,
    and were merged into this one (API board task 8bbe083b).

    Without setup_logging() in _child_entry(), child processes would timeout
    silently because logging wasn't initialized, making all log statements
    no-ops; the lifecycle records asserted below are that regression's test.
    """
    # The real mkdtemp makes the result directory; the test only learns its path
    made: list[str] = []
    real_mkdtemp = _test_hooks.tempfile_mkdtemp

    def _recording_mkdtemp(prefix: str) -> str:
        path = real_mkdtemp(prefix)
        made.append(path)
        return path

    _test_hooks.tempfile_mkdtemp = _recording_mkdtemp
    before = {child.pid for child in multiprocessing.active_children()}
    base = _FakeMNIST(8)
    ds = PreprocessDataset(base, default_train_config(batch_size=4))
    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=2)
    budget = BudgetConfig(
        start_pct_max=99.0, abort_pct=99.0, timeout_s=CHILD_HANG_BOUND_S, max_failures=1
    )

    with caplog.at_level(logging.INFO, logger="handwriting_ai"):
        out = SubprocessRunner().run(ds, cand, samples=1, budget=budget)

    # The child's result file was written and read back
    assert out["ok"] and out["res"] is not None, "Child process failed or timed out"
    assert out["res"]["batch_size"] >= 1
    assert len(made) == 1 and Path(made[0]).name.startswith("calib_child_")
    assert not Path(made[0]).exists()
    assert {child.pid for child in multiprocessing.active_children()} == before

    # Verify child lifecycle logs were emitted through the application logger
    messages = [rec.message for rec in caplog.records if rec.name == "handwriting_ai"]
    assert any("calibration_child_started" in m for m in messages), (
        "Missing 'calibration_child_started' in logs. Got:\n" + "\n".join(messages)
    )
    assert any("calibration_child_complete" in m for m in messages), (
        "Missing 'calibration_child_complete' in logs. Got:\n" + "\n".join(messages)
    )
