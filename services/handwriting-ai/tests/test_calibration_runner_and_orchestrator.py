from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image
from platform_core.logging import get_logger

from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.calibration.ds_spec import (
    AugmentSpec,
    InlineSpec,
    PreprocessSpec,
)
from handwriting_ai.training.calibration.measure import CalibrationResult
from handwriting_ai.training.calibration.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
)
from handwriting_ai.training.calibration.runner import (
    BudgetConfig,
    CandidateOutcome,
    SubprocessRunner,
)
from handwriting_ai.training.dataset import PreprocessDataset
from handwriting_ai.training.train_config import TrainConfig

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


class _FakeMNIST:
    def __init__(self, n: int = 32, *, sleep_s: float = 0.0, fail: bool = False) -> None:
        self._n = n
        self._sleep = float(sleep_s)
        self._fail = bool(fail)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        if self._fail:
            raise RuntimeError("fail-item")
        if self._sleep > 0:
            time.sleep(self._sleep)
        img = Image.new("L", (28, 28), color=0)
        return img, int(idx % 10)


def _mk_ds(n: int = 32, *, sleep_s: float = 0.0, fail: bool = False) -> PreprocessDataset:
    base = _FakeMNIST(n=n, sleep_s=sleep_s, fail=fail)
    cfg: TrainConfig = {
        "data_root": Path("."),
        "out_dir": Path("."),
        "model_id": "test",
        "epochs": 1,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "seed": 42,
        "device": "cpu",
        "optim": "adamw",
        "scheduler": "none",
        "step_size": 10,
        "gamma": 0.5,
        "min_lr": 1e-5,
        "patience": 0,
        "min_delta": 0.0,
        "threads": 1,
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
        "morph_kernel_px": 3,
        "progress_every_epochs": 1,
        "progress_every_batches": 0,
        "calibrate": False,
        "calibration_samples": 0,
        "force_calibration": False,
        "memory_guard": False,
    }
    return PreprocessDataset(base, cfg)


def test_subprocess_runner_success() -> None:
    ds = _mk_ds(32)
    runner = SubprocessRunner()
    budget = BudgetConfig(start_pct_max=99.0, abort_pct=95.0, timeout_s=20.0, max_failures=2)
    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=8)
    out = runner.run(ds, cand, samples=1, budget=budget)
    assert out["ok"]
    if out["res"] is None:
        raise AssertionError("expected res")


def test_subprocess_runner_timeout() -> None:
    # Build inline spec with per-item sleep so child exceeds timeout
    spec = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=8, sleep_s=0.25, fail=False),
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    runner = SubprocessRunner()
    budget = BudgetConfig(start_pct_max=99.0, abort_pct=95.0, timeout_s=0.2, max_failures=1)
    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=4)
    out = runner.run(spec, cand, samples=1, budget=budget)
    assert not out["ok"] and out["error"] is not None
    assert out["error"]["kind"] == "timeout"


def test_subprocess_runner_runtime_error() -> None:
    spec = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=8, sleep_s=0.0, fail=True),
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    runner = SubprocessRunner()
    budget = BudgetConfig(start_pct_max=99.0, abort_pct=95.0, timeout_s=10.0, max_failures=1)
    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=4)
    out = runner.run(spec, cand, samples=1, budget=budget)
    assert not out["ok"] and out["error"] is not None
    assert out["error"]["kind"] in {"runtime", "oom", "timeout"}


def test_orchestrator_stage_flow_and_breaker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Dummy runner that fails once then succeeds
    class _DummyRunner:
        def __init__(self) -> None:
            self._calls = 0

        def run(
            self,
            ds: PreprocessDataset | PreprocessSpec,
            cand: Candidate,
            samples: int,
            budget: BudgetConfig,
        ) -> CandidateOutcome:
            self._calls += 1
            if self._calls == 1:
                return {"ok": False, "res": None, "error": None}
            res: CalibrationResult = {
                "intra_threads": cand["intra_threads"],
                "interop_threads": cand["interop_threads"],
                "num_workers": cand["num_workers"],
                "batch_size": cand["batch_size"],
                "samples_per_sec": 10.0,
                "p95_ms": 5.0,
            }
            return {"ok": True, "res": res, "error": None}

    ds = _mk_ds(16)
    cands: list[Candidate] = [
        {"intra_threads": 1, "interop_threads": None, "num_workers": 0, "batch_size": 4},
        {"intra_threads": 1, "interop_threads": None, "num_workers": 0, "batch_size": 8},
    ]
    cfg: OrchestratorConfig = {
        "stage_a_budget": {
            "start_pct_max": 99.0,
            "abort_pct": 95.0,
            "timeout_s": 2.0,
            "max_failures": 2,
        },
        "stage_b_budget": {
            "start_pct_max": 99.0,
            "abort_pct": 95.0,
            "timeout_s": 2.0,
            "max_failures": 2,
        },
        "checkpoint_path": tmp_path / "calib.ckpt.json",
    }
    orch = Orchestrator(_DummyRunner(), cfg)
    res_a = orch.run_stage_a(ds, cands, samples=1)
    assert res_a
    best = Orchestrator.select_best(res_a)
    assert best["batch_size"] in {4, 8}


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    from handwriting_ai.training.calibration.checkpoint import (
        CalibrationCheckpoint,
        CalibrationStage,
        read_checkpoint,
        write_checkpoint,
    )

    res = CalibrationResult(
        intra_threads=1,
        interop_threads=None,
        num_workers=0,
        batch_size=4,
        samples_per_sec=3.14,
        p95_ms=7.5,
    )
    ck = CalibrationCheckpoint(
        stage=CalibrationStage.A,
        index=1,
        results=[res],
        shortlist=None,
        seed=None,
    )
    path = tmp_path / "ck.json"
    write_checkpoint(path, ck)
    ck2 = read_checkpoint(path)
    assert ck2 is not None and ck2["stage"] == CalibrationStage.A
    assert len(ck2["results"]) == 1 and ck2["results"][0]["batch_size"] == 4


def test_try_read_result_handles_open_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from handwriting_ai.training.calibration.runner import SubprocessRunner as _Runner

    out_path = tmp_path / "result.txt"
    out_path.write_text("ok=1\n", encoding="utf-8")

    def _boom(
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
    ) -> None:
        if path == out_path.as_posix():
            raise OSError("boom")
        raise AssertionError("unexpected open path")

    import builtins

    monkeypatch.setattr(builtins, "open", _boom, raising=True)
    out = _Runner._try_read_result(out_path.as_posix(), exited=False, exit_code=None)
    assert out is None


def test_child_entry_flush_handles_handlers_without_flush(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Import here to avoid circulars at module import time
    from handwriting_ai.training.calibration.candidates import Candidate
    from handwriting_ai.training.calibration.ds_spec import AugmentSpec, InlineSpec, PreprocessSpec
    from handwriting_ai.training.calibration.measure import CalibrationResult
    from handwriting_ai.training.calibration.runner import _child_entry

    out_path = tmp_path / "child_result.txt"

    spec = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=0, sleep_s=0.0, fail=False),
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=4)

    # Stub heavy operations in child entry to keep the test fast and deterministic
    def _stub_build_ds(spec_in: PreprocessSpec) -> PreprocessDataset:
        _ = spec_in
        # Tiny in-memory dataset
        return _mk_ds(4)

    def _stub_measure(
        ds_in: PreprocessDataset,
        cand_in: Candidate,
        samples: int,
        on_improvement: Callable[[CalibrationResult], None] | None,
        enable_headroom: bool,
    ) -> CalibrationResult:
        _ = (ds_in, cand_in, samples, on_improvement, enable_headroom)
        return CalibrationResult(
            intra_threads=1,
            interop_threads=None,
            num_workers=0,
            batch_size=4,
            samples_per_sec=1.0,
            p95_ms=1.0,
        )

    monkeypatch.setattr(
        "handwriting_ai.training.calibration.runner._build_dataset_from_spec",
        _stub_build_ds,
        raising=True,
    )
    monkeypatch.setattr(
        "handwriting_ai.training.calibration.runner._measure_candidate_internal",
        _stub_measure,
        raising=True,
    )

    def _stub_emit_result_file(path: str, res: CalibrationResult) -> None:
        _ = (path, res)

    monkeypatch.setattr(
        "handwriting_ai.training.calibration.runner._emit_result_file",
        _stub_emit_result_file,
        raising=True,
    )

    # Attach a handler that deliberately lacks a usable flush attribute
    class _NoFlushHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    logger = get_logger("handwriting_ai")
    no_flush = _NoFlushHandler()
    logger.addHandler(no_flush)

    # Patch hasattr so the branch where a handler lacks a flush attribute is exercised.
    import builtins

    orig_hasattr = builtins.hasattr

    def _patched_hasattr(obj: logging.Handler | str | int | float | bool, name: str) -> bool:
        if obj is no_flush and name == "flush":
            return False
        return orig_hasattr(obj, name)

    monkeypatch.setattr("builtins.hasattr", _patched_hasattr, raising=True)

    log_q: mp.Queue[logging.LogRecord] = mp.get_context("spawn").Queue()

    _child_entry(out_path.as_posix(), spec, cand, samples=1, abort_pct=95.0, log_q=log_q)

    # Ensure our handler saw records and that queue forwarding occurred
    assert no_flush.records
    assert not log_q.empty()
