"""calibrate_input_pipeline: its two stages, its cache and its force flag.

The candidates are measured by ``_RankedRunner``, a CandidateRunner that
scores each candidate from its own fields, so the real Orchestrator,
checkpoint and cache run here while no training subprocess is spawned. The
spawn is tests/test_calibration_runner_subprocess_integration.py's subject;
spawning it from this file too cost these two tests 204 s on the hub on
2026-10-08, one fresh interpreter and torch import per candidate, which is
most of what put this package's make check past its five-minute budget
(API board task 8bbe083b).
"""

from __future__ import annotations

from pathlib import Path

from handwriting_ai import _test_hooks
from handwriting_ai._hook_protocols_ml import PreprocessDatasetProtocol
from handwriting_ai._hook_protocols_training import (
    CandidateRunnerProtocol,
    MemorySnapshotDict,
    OrchestratorProtocol,
)
from handwriting_ai.training.calibrate import _candidate_workers, calibrate_input_pipeline
from handwriting_ai.training.calibration._types import (
    BudgetConfig,
    Candidate,
    CandidateOutcome,
    OrchestratorConfig,
)
from handwriting_ai.training.calibration.cache import _read_cache
from handwriting_ai.training.calibration.ds_spec import (
    AugmentSpec,
    BaseKind,
    InlineSpec,
    PreprocessSpec,
)
from handwriting_ai.training.calibration.orchestrator import Orchestrator
from handwriting_ai.training.resources import ResourceLimits


class _RankedRunner:
    """Scores a candidate by its threads, then its workers, and records each call.

    The score makes the best candidate known in advance: the most intra-op
    threads, then the most loader workers. ``calls`` holds every
    ``(candidate, samples)`` the orchestrator asked for, in order.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Candidate, int]] = []

    def run(
        self,
        ds: PreprocessDatasetProtocol | PreprocessSpec,
        cand: Candidate,
        samples: int,
        budget: BudgetConfig,
    ) -> CandidateOutcome:
        _ = (ds, budget)
        self.calls.append((cand, samples))
        return {
            "ok": True,
            "res": {
                "intra_threads": cand["intra_threads"],
                "interop_threads": cand["interop_threads"],
                "num_workers": cand["num_workers"],
                "batch_size": cand["batch_size"],
                "samples_per_sec": float(cand["intra_threads"] * 10 + cand["num_workers"]),
                "p95_ms": 1.0,
            },
            "error": None,
        }


def _use_runner(ranked: _RankedRunner) -> None:
    """Route the calibrator's orchestrator to ``ranked`` on a fixed 4 GiB host at 50 percent.

    The calibrator builds a SubprocessRunner and hands it to the orchestrator
    factory; the factory here discards it for ``ranked``.
    """

    def _factory(
        *, runner: CandidateRunnerProtocol, config: OrchestratorConfig
    ) -> OrchestratorProtocol:
        _ = runner
        return Orchestrator(runner=ranked, config=config)

    def _snapshot() -> MemorySnapshotDict:
        return {
            "main_process": {"pid": 1, "rss_bytes": 100 * 1024 * 1024},
            "workers": (),
            "cgroup_usage": {
                "usage_bytes": 2 * 1024 * 1024 * 1024,
                "limit_bytes": 4 * 1024 * 1024 * 1024,
                "percent": 50.0,
            },
            "cgroup_breakdown": {
                "anon_bytes": 0,
                "file_bytes": 0,
                "kernel_bytes": 0,
                "slab_bytes": 0,
            },
        }

    def _cgroup() -> bool:
        return True

    _test_hooks.orchestrator_factory = _factory
    _test_hooks.get_memory_snapshot = _snapshot
    _test_hooks.is_cgroup_available = _cgroup


def _inline_spec() -> PreprocessSpec:
    aug = AugmentSpec(
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
    )
    return PreprocessSpec(
        base_kind=BaseKind.INLINE,
        mnist=None,
        inline=InlineSpec(n=8, sleep_s=0.0, fail=False),
        augment=aug,
    )


def test_calibrate_persists_and_reuses_cache(tmp_path: Path) -> None:
    runner = _RankedRunner()
    _use_runner(runner)
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=128 * 1024 * 1024,
        optimal_threads=1,
        optimal_workers=0,
        max_batch_size=64,
    )
    cache = tmp_path / "calibration.json"
    # First run measures both stages and writes the cache
    ec1 = calibrate_input_pipeline(
        _inline_spec(),
        limits=limits,
        requested_batch_size=8,
        samples=2,
        cache_path=cache,
        ttl_seconds=3600,
        force=False,
    )
    # Stage A: one thread and two threads, no workers under 2 GiB, 2 samples
    # each; stage B: the same two refined at twice the samples.
    assert [(c["intra_threads"], c["num_workers"], s) for c, s in runner.calls] == [
        (1, 0, 2),
        (2, 0, 2),
        (2, 0, 4),
        (1, 0, 4),
    ]
    assert ec1["intra_threads"] == 2
    assert ec1["loader_cfg"]["num_workers"] == 0
    assert ec1["batch_size"] == 8
    cached = _read_cache(cache)
    assert cached is not None and cached[1]["intra_threads"] == 2
    assert not cache.with_suffix(".ckpt.json").exists()
    # A different requested batch with the same host signature reuses the cache
    ec2 = calibrate_input_pipeline(
        _inline_spec(),
        limits=limits,
        requested_batch_size=32,
        samples=2,
        cache_path=cache,
        ttl_seconds=3600,
        force=False,
    )
    assert len(runner.calls) == 4
    assert ec2["intra_threads"] == ec1["intra_threads"]
    assert ec2["interop_threads"] == ec1["interop_threads"]
    assert ec2["batch_size"] == ec1["batch_size"] == 8
    assert ec2["loader_cfg"]["num_workers"] == ec1["loader_cfg"]["num_workers"]


def test_candidate_workers_enumeration() -> None:
    # Minimal sanity on worker enumeration; calibration decides, not heuristics
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=None,
        optimal_threads=1,
        optimal_workers=0,
        max_batch_size=None,
    )
    ws = _candidate_workers(limits)
    assert 0 in ws and 1 in ws


def test_calibrate_force_recomputes(tmp_path: Path) -> None:
    runner = _RankedRunner()
    _use_runner(runner)
    limits = ResourceLimits(
        cpu_cores=4,
        memory_bytes=None,
        optimal_threads=2,
        optimal_workers=1,
        max_batch_size=None,
    )
    cache = tmp_path / "cal.json"
    # A cache that is not one: reading it would raise, so force must skip it
    cache.write_text("{}", encoding="utf-8")
    ec = calibrate_input_pipeline(
        _inline_spec(),
        limits=limits,
        requested_batch_size=4,
        samples=2,
        cache_path=cache,
        ttl_seconds=1,
        force=True,
    )
    # Stage A: threads 2 and 4 by workers 0, 1 and 2; stage B: the top three
    assert len(runner.calls) == 6 + 3
    assert ec["intra_threads"] == 4
    assert ec["loader_cfg"]["num_workers"] == 2
    assert ec["loader_cfg"]["persistent_workers"] is True
    assert ec["loader_cfg"]["batch_size"] == 4
    cached = _read_cache(cache)
    assert cached is not None and cached[1]["num_workers"] == 2
