"""Result directories belong to one real calibration child run.

A child that finishes is tested, with the same cleanup assertions, in
tests/test_calibration_runner_subprocess_integration.py; this file holds the
two outcomes it does not reach, a child that fails and one that times out.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest
from tests._calibration_fixtures import CHILD_HANG_BOUND_S

from handwriting_ai import _test_hooks
from handwriting_ai.training.calibration._types import BudgetConfig, Candidate, CandidateError
from handwriting_ai.training.calibration.ds_spec import BaseKind, PreprocessSpec
from handwriting_ai.training.calibration.runner import SubprocessRunner

# The child's dataset raises on its first item, so the child exits 1.
_CHILD_FAILED: CandidateError = {
    "kind": "runtime",
    "message": "child exited code=1",
    "exit_code": 1,
}
_TIMED_OUT: CandidateError = {
    "kind": "timeout",
    "message": "candidate timed out",
    "exit_code": None,
}


@pytest.mark.parametrize(
    ("child_fails", "timeout_s", "error"),
    [(True, CHILD_HANG_BOUND_S, _CHILD_FAILED), (False, 0.0, _TIMED_OUT)],
)
def test_result_directory_removed_after_child_exit(
    tmp_path: Path, child_fails: bool, timeout_s: float, error: CandidateError
) -> None:
    """Child failure and timeout leave neither child nor result files."""
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
    assert outcome == {"ok": False, "res": None, "error": error}
    assert not result_dir.exists()
    assert {child.pid for child in multiprocessing.active_children()} == before
