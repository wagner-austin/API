"""Result directories belong to one real calibration child run."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from handwriting_ai import _test_hooks
from handwriting_ai.training.calibration._types import BudgetConfig, Candidate
from handwriting_ai.training.calibration.ds_spec import BaseKind, PreprocessSpec
from handwriting_ai.training.calibration.runner import SubprocessRunner


@pytest.mark.parametrize(
    ("child_fails", "timeout_s", "succeeds"),
    [(False, 60.0, True), (True, 60.0, False), (False, 0.0, False)],
)
def test_result_directory_removed_after_child_exit(
    tmp_path: Path, child_fails: bool, timeout_s: float, succeeds: bool
) -> None:
    """Clean success, child failure and timeout leave neither child nor result files."""
    result_dir = tmp_path / "calib_child_owned"
    result_dir.mkdir()
    _test_hooks.tempfile_mkdtemp = lambda prefix: str(result_dir)
    before = {child.pid for child in multiprocessing.active_children()}
    candidate: Candidate = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 1,
    }
    budget: BudgetConfig = {
        "start_pct_max": 99.0,
        "abort_pct": 99.0,
        "timeout_s": timeout_s,
        "max_failures": 1,
    }
    spec: PreprocessSpec = {
        "base_kind": BaseKind.INLINE,
        "mnist": None,
        "inline": {"n": 2, "sleep_s": 0.0, "fail": child_fails},
        "augment": {
            "augment": False,
            "aug_rotate": 0.0,
            "aug_translate": 0.0,
            "noise_prob": 0.0,
            "noise_salt_vs_pepper": 0.5,
            "dots_prob": 0.0,
            "dots_count": 0,
            "dots_size_px": 1,
            "blur_sigma": 0.0,
            "morph": "none",
        },
    }
    outcome = SubprocessRunner().run(spec, candidate, samples=1, budget=budget)
    assert outcome["ok"] is succeeds
    assert not result_dir.exists()
    assert {child.pid for child in multiprocessing.active_children()} == before
    if timeout_s == 0.0:
        assert outcome["error"] == {
            "kind": "timeout",
            "message": "candidate timed out",
            "exit_code": None,
        }
