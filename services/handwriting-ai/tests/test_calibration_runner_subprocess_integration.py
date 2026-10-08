from __future__ import annotations

import logging

import pytest
from PIL import Image

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
    """One real spawned child: its result file reaches the parent, and so do its logs.

    The package's only test that spawns the calibration child, so it carries
    both halves of the child's contract. Its sibling that asserted only the
    result ran the identical spawn, one more fresh interpreter and torch
    import, and was merged into this one (API board task 8bbe083b).

    Without setup_logging() in _child_entry(), child processes would timeout
    silently because logging wasn't initialized, making all log statements
    no-ops; the lifecycle records asserted below are that regression's test.
    """
    base = _FakeMNIST(8)
    ds = PreprocessDataset(base, default_train_config(batch_size=4))
    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=2)
    # 120s bounds a hung child without racing a healthy one -- the child
    # boots a fresh interpreter plus torch, measured at 8-15s under host
    # load, and a busy host must not fail this test by wall clock.
    budget = BudgetConfig(start_pct_max=99.0, abort_pct=99.0, timeout_s=120.0, max_failures=1)

    with caplog.at_level(logging.INFO, logger="handwriting_ai"):
        out = SubprocessRunner().run(ds, cand, samples=1, budget=budget)

    # The child's result file was written and read back
    assert out["ok"] and out["res"] is not None, "Child process failed or timed out"
    assert out["res"]["batch_size"] >= 1

    # Verify child lifecycle logs were emitted through the application logger
    messages = [rec.message for rec in caplog.records if rec.name == "handwriting_ai"]
    assert any("calibration_child_started" in m for m in messages), (
        "Missing 'calibration_child_started' in logs. Got:\n" + "\n".join(messages)
    )
    assert any("calibration_child_complete" in m for m in messages), (
        "Missing 'calibration_child_complete' in logs. Got:\n" + "\n".join(messages)
    )
