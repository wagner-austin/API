from __future__ import annotations

from pathlib import Path

from PIL import Image

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import MemorySnapshotDict, PreprocessDatasetProtocol
from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.calibration.ds_spec import PreprocessSpec
from handwriting_ai.training.calibration.measure import CalibrationResult
from handwriting_ai.training.calibration.orchestrator import Orchestrator, OrchestratorConfig
from handwriting_ai.training.calibration.runner import BudgetConfig, CandidateOutcome
from handwriting_ai.training.dataset import AugmentConfig, PreprocessDataset

_TEST_CFG: AugmentConfig = {
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
    "morph_kernel_px": 1,
    "batch_size": 4,
}

_TEST_CFG_SMALL: AugmentConfig = {
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
    "morph_kernel_px": 1,
    "batch_size": 2,
}


class _FakeMNIST:
    def __init__(self, n: int = 8) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img = Image.new("L", (28, 28), 0)
        return img, 0


def _make_budget(start_pct: float, max_failures: int) -> BudgetConfig:
    """Create a BudgetConfig TypedDict."""
    return {
        "start_pct_max": start_pct,
        "abort_pct": 95.0,
        "timeout_s": 1.0,
        "max_failures": max_failures,
    }


def _make_cand(batch_size: int) -> Candidate:
    """Create a Candidate TypedDict."""
    return {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": batch_size,
    }


def _make_result(cand: Candidate) -> CalibrationResult:
    """Create a CalibrationResult TypedDict from a Candidate."""
    return {
        "intra_threads": cand["intra_threads"],
        "interop_threads": cand["interop_threads"],
        "num_workers": cand["num_workers"],
        "batch_size": cand["batch_size"],
        "samples_per_sec": 1.0,
        "p95_ms": 1.0,
    }


class _FailingRunner:
    def __init__(self, fails: int) -> None:
        self._fails = int(fails)
        self.calls = 0

    def run(
        self,
        ds: PreprocessDatasetProtocol | PreprocessSpec,
        cand: Candidate,
        samples: int,
        budget: BudgetConfig,
    ) -> CandidateOutcome:
        self.calls += 1
        if self._fails > 0:
            self._fails -= 1
            return {"ok": False, "res": None, "error": None}
        res = _make_result(cand)
        return {"ok": True, "res": res, "error": None}


def _make_snapshot(percent: float) -> MemorySnapshotDict:
    """Create a MemorySnapshot TypedDict for testing with given percent."""
    return {
        "main_process": {"pid": 1, "rss_bytes": 100 * 1024 * 1024},
        "workers": (),
        "cgroup_usage": {
            "usage_bytes": 500 * 1024 * 1024,
            "limit_bytes": 1024 * 1024 * 1024,
            "percent": percent,
        },
        "cgroup_breakdown": {
            "anon_bytes": 400 * 1024 * 1024,
            "file_bytes": 50 * 1024 * 1024,
            "kernel_bytes": 30 * 1024 * 1024,
            "slab_bytes": 20 * 1024 * 1024,
        },
    }


def test_orchestrator_preflight_and_abort(tmp_path: Path) -> None:
    ds = PreprocessDataset(_FakeMNIST(8), _TEST_CFG)
    cands = [_make_cand(2), _make_cand(4)]
    runner = _FailingRunner(fails=0)
    cfg: OrchestratorConfig = {
        "stage_a_budget": _make_budget(10.0, 2),
        "stage_b_budget": _make_budget(10.0, 2),
        "checkpoint_path": tmp_path / "ck.json",
    }
    orch = Orchestrator(runner, cfg)

    # Force preflight to fail by setting cgroup available and high memory
    def _cgroup_available() -> bool:
        return True

    def _high_mem() -> MemorySnapshotDict:
        return _make_snapshot(99.0)  # Way above the 10% threshold

    _test_hooks.is_cgroup_available = _cgroup_available
    _test_hooks.get_memory_snapshot = _high_mem

    res = orch.run_stage_a(ds, cands, samples=1)
    # Should have written a checkpoint after first failure, then aborted on second
    assert res == []
    assert (tmp_path / "ck.json").exists()


def test_orchestrator_resume_stage_a_and_b(tmp_path: Path) -> None:
    ds = PreprocessDataset(_FakeMNIST(8), _TEST_CFG)
    cands = [_make_cand(2), _make_cand(4)]
    runner = _FailingRunner(fails=0)
    cfg: OrchestratorConfig = {
        "stage_a_budget": _make_budget(99.0, 2),
        "stage_b_budget": _make_budget(99.0, 2),
        "checkpoint_path": tmp_path / "ck.json",
    }
    orch = Orchestrator(runner, cfg)

    # Set up hooks for normal operation
    _test_hooks.is_cgroup_available = lambda: False
    _test_hooks.get_memory_snapshot = lambda: _make_snapshot(50.0)

    # Write stage A checkpoint to skip first candidate
    from handwriting_ai.training.calibration.checkpoint import (
        CalibrationCheckpoint,
        CalibrationStage,
        write_checkpoint,
    )

    prior: CalibrationResult = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 2,
        "samples_per_sec": 1.0,
        "p95_ms": 1.0,
    }
    ck_a: CalibrationCheckpoint = {
        "stage": CalibrationStage.A,
        "index": 1,
        "results": [prior],
        "shortlist": None,
        "seed": None,
    }
    write_checkpoint(cfg["checkpoint_path"], ck_a)
    res_a = orch.run_stage_a(ds, cands, samples=1)
    # Prior plus one new result
    assert len(res_a) == 2

    # Now resume stage B with checkpoint
    ck_b: CalibrationCheckpoint = {
        "stage": CalibrationStage.B,
        "index": 1,
        "results": [prior],
        "shortlist": None,
        "seed": None,
    }
    write_checkpoint(cfg["checkpoint_path"], ck_b)
    res_b = orch.run_stage_b(ds, res_a, samples=1)
    assert res_b


def test_orchestrator_preflight_recovers_then_runs(tmp_path: Path) -> None:
    # First preflight returns False, then True to exercise continue path
    ds = PreprocessDataset(_FakeMNIST(8), _TEST_CFG)
    cands = [_make_cand(2)]

    class _Runner:
        def __init__(self) -> None:
            self.calls = 0

        def run(
            self,
            ds: PreprocessDatasetProtocol | PreprocessSpec,
            cand: Candidate,
            samples: int,
            budget: BudgetConfig,
        ) -> CandidateOutcome:
            self.calls += 1
            res = _make_result(cand)
            return {"ok": True, "res": res, "error": None}

    runner = _Runner()
    cfg: OrchestratorConfig = {
        "stage_a_budget": _make_budget(99.0, 2),
        "stage_b_budget": _make_budget(99.0, 2),
        "checkpoint_path": tmp_path / "ck.json",
    }
    orch = Orchestrator(runner, cfg)

    calls = {"i": 0}

    # First call fails (high memory), subsequent succeed (low memory)
    def _snap() -> MemorySnapshotDict:
        calls["i"] += 1
        if calls["i"] <= 2:  # First two checks fail (preflight + retry)
            return _make_snapshot(99.0)
        return _make_snapshot(10.0)  # Then succeed

    _test_hooks.is_cgroup_available = lambda: True
    _test_hooks.get_memory_snapshot = _snap

    out = orch.run_stage_a(ds, cands, samples=1)
    # One preflight failure (checkpoint) then success run
    assert len(out) == 1 and runner.calls == 1
    assert (tmp_path / "ck.json").exists()


def test_orchestrator_candidate_failure_aborts(tmp_path: Path) -> None:
    ds = PreprocessDataset(_FakeMNIST(4), _TEST_CFG_SMALL)
    cands = [_make_cand(2)]

    # Set up hooks for normal preflight
    _test_hooks.is_cgroup_available = lambda: False
    _test_hooks.get_memory_snapshot = lambda: _make_snapshot(50.0)

    class _FailRunner:
        def run(
            self,
            ds: PreprocessDatasetProtocol | PreprocessSpec,
            cand: Candidate,
            samples: int,
            budget: BudgetConfig,
        ) -> CandidateOutcome:
            from handwriting_ai.training.calibration.runner import CandidateError

            err: CandidateError = {
                "kind": "timeout",
                "message": "candidate timed out",
                "exit_code": None,
            }
            return {"ok": False, "res": None, "error": err}

    cfg: OrchestratorConfig = {
        "stage_a_budget": _make_budget(99.0, 1),
        "stage_b_budget": _make_budget(99.0, 1),
        "checkpoint_path": tmp_path / "ck.json",
    }
    orch = Orchestrator(_FailRunner(), cfg)
    res = orch.run_stage_a(ds, cands, samples=1)
    assert res == []


def test_orchestrator_preflight_ok_uses_cgroup_snapshot(tmp_path: Path) -> None:
    """Exercise _preflight_ok branch when cgroup metrics are available."""

    # Below threshold -> True
    _test_hooks.is_cgroup_available = lambda: True
    _test_hooks.get_memory_snapshot = lambda: _make_snapshot(25.0)

    cfg_below: OrchestratorConfig = {
        "stage_a_budget": _make_budget(50.0, 1),
        "stage_b_budget": _make_budget(50.0, 1),
        "checkpoint_path": tmp_path / "ck.json",
    }
    assert Orchestrator(_FailingRunner(0), cfg_below)._preflight_ok(50.0) is True

    # Above threshold -> False
    _test_hooks.get_memory_snapshot = lambda: _make_snapshot(75.0)
    cfg_above: OrchestratorConfig = {
        "stage_a_budget": _make_budget(50.0, 1),
        "stage_b_budget": _make_budget(50.0, 1),
        "checkpoint_path": tmp_path / "ck2.json",
    }
    assert Orchestrator(_FailingRunner(0), cfg_above)._preflight_ok(50.0) is False


def test_orchestrator_preflight_retry_succeeds(tmp_path: Path) -> None:
    """Test branch where first preflight fails but retry succeeds (line 96->131)."""
    ds = PreprocessDataset(_FakeMNIST(8), _TEST_CFG)
    cands = [_make_cand(2)]

    class _Runner:
        def __init__(self) -> None:
            self.calls = 0

        def run(
            self,
            ds: PreprocessDatasetProtocol | PreprocessSpec,
            cand: Candidate,
            samples: int,
            budget: BudgetConfig,
        ) -> CandidateOutcome:
            self.calls += 1
            res = _make_result(cand)
            return {"ok": True, "res": res, "error": None}

    runner = _Runner()
    cfg: OrchestratorConfig = {
        "stage_a_budget": _make_budget(50.0, 2),
        "stage_b_budget": _make_budget(50.0, 2),
        "checkpoint_path": tmp_path / "ck.json",
    }
    orch = Orchestrator(runner, cfg)

    check_calls = {"i": 0}

    # First preflight check fails, retry succeeds
    def _snap() -> MemorySnapshotDict:
        check_calls["i"] += 1
        if check_calls["i"] == 1:  # First check fails
            return _make_snapshot(99.0)
        return _make_snapshot(10.0)  # Retry succeeds

    _test_hooks.is_cgroup_available = lambda: True
    _test_hooks.get_memory_snapshot = _snap

    out = orch.run_stage_a(ds, cands, samples=1)
    # Should run the candidate after retry succeeds
    assert len(out) == 1
    assert runner.calls == 1
