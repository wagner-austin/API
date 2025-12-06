from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger

from handwriting_ai.monitoring import get_memory_snapshot
from handwriting_ai.training.dataset import AugmentConfig, DataLoaderConfig
from handwriting_ai.training.resources import ResourceLimits
from handwriting_ai.training.runtime import EffectiveConfig

from .cache import _read_cache, _valid_cache, _write_cache
from .candidates import _generate_candidates
from .ds_spec import PreprocessSpec
from .measure import CalibrationResult
from .orchestrator import Orchestrator
from .runner import BudgetConfig, SubprocessRunner
from .signature import make_signature as _make_signature

_LOGGER = get_logger("handwriting_ai.calibration")


class CalibrationError(RuntimeError):
    """Raised when calibration cannot produce any viable candidate results."""

    pass


def _result_to_effective(res: CalibrationResult) -> EffectiveConfig:
    loader_cfg = DataLoaderConfig(
        batch_size=res["batch_size"],
        num_workers=res["num_workers"],
        pin_memory=False,
        persistent_workers=bool(res["num_workers"] > 0),
        prefetch_factor=2,
    )
    return {
        "intra_threads": res["intra_threads"],
        "interop_threads": res["interop_threads"],
        "batch_size": res["batch_size"],
        "loader_cfg": loader_cfg,
    }


def calibrate_input_pipeline(
    ds: PreprocessSpec,
    *,
    limits: ResourceLimits,
    requested_batch_size: int,
    samples: int,
    cache_path: Path,
    ttl_seconds: int,
    force: bool,
) -> EffectiveConfig:
    log = _LOGGER
    sig = _make_signature(limits)
    if not force:
        cached = _valid_cache(sig, _read_cache(cache_path), ttl_seconds)
        if cached is not None:
            return _result_to_effective(cached)

    _: AugmentConfig = {
        "batch_size": max(1, int(requested_batch_size)),
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
    }

    # Compute budgets based on observed environment and thresholds (subprocess-only)
    # Use current snapshot for start gate; use conservative aborts for <1GB tiers
    snap = get_memory_snapshot()
    mem_limit_mb = snap["cgroup_usage"]["limit_bytes"] // (1024 * 1024)
    stage_a_budget: BudgetConfig
    stage_b_budget: BudgetConfig
    if mem_limit_mb <= 1024:
        stage_a_budget = {
            "start_pct_max": 80.0,
            "abort_pct": 85.0,
            "timeout_s": 45.0,
            "max_failures": 2,
        }
        stage_b_budget = {
            "start_pct_max": 83.0,
            "abort_pct": 88.0,
            "timeout_s": 60.0,
            "max_failures": 2,
        }
    else:
        stage_a_budget = {
            "start_pct_max": 85.0,
            "abort_pct": 90.0,
            "timeout_s": 45.0,
            "max_failures": 2,
        }
        stage_b_budget = {
            "start_pct_max": 88.0,
            "abort_pct": 92.0,
            "timeout_s": 60.0,
            "max_failures": 2,
        }

    cands = _generate_candidates(limits, requested_batch_size)
    log.info("calibration_stage_a_start candidates=%d samples=%d", len(cands), samples)
    runner = SubprocessRunner()
    ckpt_path = cache_path.with_suffix(".ckpt.json")
    orch = Orchestrator(
        runner=runner,
        config={
            "stage_a_budget": stage_a_budget,
            "stage_b_budget": stage_b_budget,
            "checkpoint_path": ckpt_path,
        },
    )
    stage_a: list[CalibrationResult] = orch.run_stage_a(ds, cands, samples)
    log.info("calibration_stage_a_complete measured=%d", len(stage_a))
    if len(stage_a) == 0:
        log.error("calibration_failed_stage_a_no_results")
        # Clear checkpoint on total failure to avoid stale resume on retry
        ckpt_path.unlink(missing_ok=True)
        raise CalibrationError("calibration failed: no results in stage A")
    stage_a.sort(key=lambda r: (-r["samples_per_sec"], r["p95_ms"]))
    shortlist = stage_a[: min(3, len(stage_a))]

    samples_refine = max(1, samples * 2)
    log.info("calibration_stage_b_start shortlist=%d samples=%d", len(shortlist), samples_refine)
    refined: list[CalibrationResult] = orch.run_stage_b(ds, shortlist, samples_refine)
    log.info("calibration_stage_b_complete measured=%d", len(refined))
    if len(refined) == 0:
        log.error("calibration_failed_stage_b_no_results")
        # Clear checkpoint on total failure to avoid stale resume on retry
        ckpt_path.unlink(missing_ok=True)
        raise CalibrationError("calibration failed: no results in stage B")
    refined.sort(key=lambda r: (-r["samples_per_sec"], r["p95_ms"]))
    best = refined[0]

    # Emit a concise calibration report (top 3 + chosen)
    def _fmt(r: CalibrationResult) -> str:
        return (
            f"threads={r['intra_threads']} workers={r['num_workers']} bs={r['batch_size']} "
            f"sps={r['samples_per_sec']:.2f} p95={r['p95_ms']:.2f}"
        )

    top_str = ", ".join(_fmt(r) for r in refined[: min(3, len(refined))])
    log.info(f"calibration_report candidates={len(cands)} top=[{top_str}] chosen=({_fmt(best)})")

    _write_cache(cache_path, sig, best)
    # Clear checkpoint on success
    ckpt_path.unlink(missing_ok=True)
    return _result_to_effective(best)
