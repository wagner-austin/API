from __future__ import annotations

from collections.abc import Callable, Generator
from statistics import quantiles
from typing import Final, Protocol

import torch
from platform_core.logging import get_logger
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from handwriting_ai import _test_hooks
from handwriting_ai.monitoring import is_cgroup_available
from handwriting_ai.training.calibration._types import CalibrationResultDict
from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.dataset import DataLoaderConfig, PreprocessDataset
from handwriting_ai.training.optim import (
    build_optimizer_and_scheduler as _build_optim,
)
from handwriting_ai.training.optim import (
    default_optim_config,
)
from handwriting_ai.training.safety import get_memory_guard_config
from handwriting_ai.training.train_utils import _build_model as _build_train_model

HEADROOM_TARGET: Final[float] = 60.0
_LOGGER = get_logger("handwriting_ai")


class _BatchIterator(Protocol):
    """Iterator protocol for training batches."""

    def __iter__(self) -> _BatchIterator: ...
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]: ...


class _BatchIterable(Protocol):
    """Protocol for objects that can be iterated to produce training batches."""

    def __iter__(self) -> _BatchIterator: ...


# CalibrationResult is imported from _types to avoid circular imports.
# Re-export for backwards compatibility.
CalibrationResult = CalibrationResultDict


# Cache a single calibration model per process to avoid repeated large
# allocations and allocator fragmentation across successive measurements.
_CAL_MODEL: Module | None = None
_CAL_OPT: Optimizer | None = None


def _reset_calibration_state(*, reset_interop: bool = False) -> None:
    """Clear cached calibration resources to avoid cross-run memory growth."""
    global _CAL_MODEL, _CAL_OPT
    _CAL_MODEL = None
    _CAL_OPT = None
    if reset_interop:
        _test_hooks.interop_configured_setter(False)
    _test_hooks.gc_collect()
    _test_hooks.torch_cuda_empty_cache()


def _get_calibration_model() -> Module:
    global _CAL_MODEL
    model = _CAL_MODEL
    if model is None:
        model = _build_train_model()
        _CAL_MODEL = model
    return model


def _get_calibration_optimizer(model: Module) -> Optimizer:
    global _CAL_OPT
    opt = _CAL_OPT
    if opt is None:
        opt, _sch = _build_optim(model, default_optim_config())
        _CAL_OPT = opt
    return opt


def _configure_interop_threads_once(interp_threads: int | None) -> None:
    if _test_hooks.interop_configured_getter():
        return
    _test_hooks.torch_set_interop_threads(int(interp_threads) if interp_threads is not None else 1)
    _test_hooks.interop_configured_setter(True)


def _log_candidate_start(cand: Candidate, bs_lo: int, bs_hi: int) -> None:
    from handwriting_ai.monitoring import get_memory_snapshot

    snap = get_memory_snapshot()
    mem_pct = float(snap["cgroup_usage"]["percent"])
    main_mb = snap["main_process"]["rss_bytes"] // (1024 * 1024)
    workers_mb = sum(w["rss_bytes"] for w in snap["workers"]) // (1024 * 1024)
    cgroup_mb = snap["cgroup_usage"]["usage_bytes"] // (1024 * 1024)
    anon_mb = snap["cgroup_breakdown"]["anon_bytes"] // (1024 * 1024)
    file_mb = snap["cgroup_breakdown"]["file_bytes"] // (1024 * 1024)
    kernel_mb = snap["cgroup_breakdown"]["kernel_bytes"] // (1024 * 1024)
    slab_mb = snap["cgroup_breakdown"]["slab_bytes"] // (1024 * 1024)
    _LOGGER.info(
        "calibration_candidate_start threads=%d workers=%d bs_range=[%d,%d] mem_pct=%.1f "
        "cgroup_mb=%d main_rss_mb=%d workers_rss_mb=%d anon_mb=%d file_mb=%d "
        "kernel_mb=%d slab_mb=%d",
        cand["intra_threads"],
        cand["num_workers"],
        bs_lo,
        bs_hi,
        mem_pct,
        cgroup_mb,
        main_mb,
        workers_mb,
        anon_mb,
        file_mb,
        kernel_mb,
        slab_mb,
    )


def _log_cleanup_state(mid: int) -> None:
    from handwriting_ai.monitoring import get_memory_snapshot

    snap = get_memory_snapshot()
    mem_pct = float(snap["cgroup_usage"]["percent"])
    main_mb = snap["main_process"]["rss_bytes"] // (1024 * 1024)
    workers_mb = sum(w["rss_bytes"] for w in snap["workers"]) // (1024 * 1024)
    cgroup_mb = snap["cgroup_usage"]["usage_bytes"] // (1024 * 1024)
    anon_mb = snap["cgroup_breakdown"]["anon_bytes"] // (1024 * 1024)
    file_mb = snap["cgroup_breakdown"]["file_bytes"] // (1024 * 1024)
    kernel_mb = snap["cgroup_breakdown"]["kernel_bytes"] // (1024 * 1024)
    slab_mb = snap["cgroup_breakdown"]["slab_bytes"] // (1024 * 1024)
    _LOGGER.info(
        "calibration_cleanup_complete bs=%d mem_pct=%.1f cgroup_mb=%d main_rss_mb=%d "
        "workers_rss_mb=%d anon_mb=%d file_mb=%d kernel_mb=%d slab_mb=%d",
        mid,
        mem_pct,
        cgroup_mb,
        main_mb,
        workers_mb,
        anon_mb,
        file_mb,
        kernel_mb,
        slab_mb,
    )


def _log_candidate_complete(best_bs: int, best_sps: float) -> None:
    from handwriting_ai.monitoring import get_memory_snapshot

    snap = get_memory_snapshot()
    mem_pct = float(snap["cgroup_usage"]["percent"])
    main_mb = snap["main_process"]["rss_bytes"] // (1024 * 1024)
    workers_mb = sum(w["rss_bytes"] for w in snap["workers"]) // (1024 * 1024)
    cgroup_mb = snap["cgroup_usage"]["usage_bytes"] // (1024 * 1024)
    anon_mb = snap["cgroup_breakdown"]["anon_bytes"] // (1024 * 1024)
    file_mb = snap["cgroup_breakdown"]["file_bytes"] // (1024 * 1024)
    kernel_mb = snap["cgroup_breakdown"]["kernel_bytes"] // (1024 * 1024)
    slab_mb = snap["cgroup_breakdown"]["slab_bytes"] // (1024 * 1024)
    _LOGGER.info(
        "calibration_candidate_complete best_bs=%d sps=%.2f mem_pct=%.1f cgroup_mb=%d "
        "main_rss_mb=%d workers_rss_mb=%d anon_mb=%d file_mb=%d "
        "kernel_mb=%d slab_mb=%d",
        best_bs,
        best_sps,
        mem_pct,
        cgroup_mb,
        main_mb,
        workers_mb,
        anon_mb,
        file_mb,
        kernel_mb,
        slab_mb,
    )


def _expand_upper_bound_if_headroom(
    *,
    enable_headroom: bool,
    mid: int,
    bs_hi: int,
    initial_cap: int,
    ds_len: int,
    peak_pct: float,
) -> int:
    if not enable_headroom or mid < bs_hi:
        return bs_hi
    guard_cfg = get_memory_guard_config()
    thr = float(guard_cfg["threshold_percent"])
    has_headroom = bool(guard_cfg["enabled"] and thr > 0.0 and peak_pct <= HEADROOM_TARGET)
    if not has_headroom:
        return bs_hi
    cap_limit = min(int(ds_len), int(initial_cap + max(1, initial_cap // 2)))
    return min(cap_limit, max(mid + 1, int(mid * 2)))


def _search_best_batch_size(
    *,
    ds: _test_hooks.PreprocessDatasetProtocol,
    cand: Candidate,
    samples: int,
    device: torch.device,
    ds_len: int,
    initial_cap: int,
    enable_headroom: bool,
    model: Module,
    opt: Optimizer,
    on_improvement: Callable[[CalibrationResult], None] | None,
) -> tuple[int, float, float]:
    bs_lo = 1
    bs_hi = int(initial_cap)
    best_bs = 1
    best_sps: float = 0.0
    best_p95: float = 0.0

    while bs_lo <= bs_hi:
        mid = (bs_lo + bs_hi) // 2
        loader: _test_hooks.BatchIterableProtocol | None = None
        try:
            mid = min(mid, int(ds_len))
            cfg_try = DataLoaderConfig(
                batch_size=mid,
                num_workers=int(cand["num_workers"]),
                pin_memory=False,
                # Ensure workers terminate promptly between attempts to prevent
                # memory growth across measurements.
                persistent_workers=False,
                prefetch_factor=1,
            )
            loader = _test_hooks.safe_loader(ds, cfg_try)
            sps, p95, peak_pct, exceeded = _test_hooks.measure_training(
                int(ds_len),
                loader,
                samples,
                device=device,
                batch_size_hint=mid,
                model=model,
                opt=opt,
            )
            if not exceeded:
                best_bs, best_sps, best_p95 = mid, sps, p95
                _LOGGER.info(
                    "calibration_attempt_success bs=%d peak_pct=%.1f sps=%.2f new_range=[%d,%d]",
                    mid,
                    peak_pct,
                    sps,
                    mid + 1,
                    bs_hi,
                )
                if on_improvement is not None:
                    on_improvement(
                        {
                            "intra_threads": cand["intra_threads"],
                            "interop_threads": cand["interop_threads"],
                            "num_workers": cand["num_workers"],
                            "batch_size": mid,
                            "samples_per_sec": sps,
                            "p95_ms": p95,
                        }
                    )
                bs_lo = mid + 1
                new_hi = _expand_upper_bound_if_headroom(
                    enable_headroom=enable_headroom,
                    mid=mid,
                    bs_hi=bs_hi,
                    initial_cap=initial_cap,
                    ds_len=ds_len,
                    peak_pct=peak_pct,
                )
                if new_hi != bs_hi:
                    _LOGGER.info(
                        "calibration_headroom_expand from=%d to=%d peak_pct=%.1f",
                        bs_hi,
                        new_hi,
                        peak_pct,
                    )
                    bs_hi = new_hi
            else:
                _LOGGER.info(
                    "calibration_backoff reason=mem_threshold peak_pct=%.1f bs=%d range=[%d,%d]",
                    peak_pct,
                    mid,
                    bs_lo,
                    mid - 1,
                )
                bs_hi = mid - 1
        except (RuntimeError, MemoryError) as exc:
            _LOGGER.error(
                "calibration_backoff reason=exception exc=%s bs=%d error=%s",
                type(exc).__name__,
                mid,
                exc,
            )
            raise
        finally:
            if loader is not None:
                _shutdown_loader(loader)
                del loader
            _test_hooks.gc_collect()
            _join_active_children()
            _log_cleanup_state(mid)
    return best_bs, best_sps, best_p95


def _measure_loader(
    ds_len: int,
    loader: Generator[tuple[torch.Tensor, torch.Tensor], None, None],
    k: int,
    *,
    batch_size_hint: int,
) -> tuple[float, float]:
    import time as _t

    it = iter(loader)
    first = next(it, None)
    if first is None:
        return 0.0, 0.0
    n_batches = max(1, min(max(1, k), max(1, ds_len // max(1, batch_size_hint))))
    times: list[float] = []
    samples = 0
    start = _t.perf_counter()
    # Measure first batch
    t0 = _t.perf_counter()
    _x, y = first
    samples += int(y.shape[0])
    times.append((_t.perf_counter() - t0) * 1000.0)
    seen = 1
    # Measure subsequent batches up to n_batches
    while seen < n_batches:
        nxt = next(it, None)
        if nxt is None:
            break
        t1 = _t.perf_counter()
        _x2, y2 = nxt
        samples += int(y2.shape[0])
        times.append((_t.perf_counter() - t1) * 1000.0)
        seen += 1
    total_s = _t.perf_counter() - start
    p95 = max(times) if len(times) >= 2 else (times[0] if times else 0.0)
    if samples <= 0:
        samples = int(batch_size_hint) * n_batches
    sps = float(samples) / total_s if total_s > 0 else 0.0
    return sps, p95


def _resolve_worker_context(
    num_workers: int,
) -> str | None:
    """Resolve a stable multiprocessing context method for DataLoader workers.

    In a spawned calibration child, prefer 'forkserver' on POSIX where
    available to avoid forking a threaded process; otherwise use 'spawn'.
    On Windows, always 'spawn'. Returns None when num_workers <= 0.

    Returns the method name as a string (not a context object) since DataLoader
    accepts strings directly for multiprocessing_context.
    """
    if int(num_workers) <= 0:
        return None
    if _test_hooks.os_name == "nt":
        return "spawn"
    methods = set(_test_hooks.mp_get_all_start_methods())
    return "forkserver" if "forkserver" in methods else ("spawn" if "spawn" in methods else None)


def _safe_loader(
    ds: _test_hooks.PreprocessDatasetProtocol,
    cfg: _test_hooks.DataLoaderConfigProtocol,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    ctx_method = _resolve_worker_context(int(cfg["num_workers"]))
    # Import PreprocessDataset here to check type at runtime
    from handwriting_ai.training.dataset import PreprocessDataset

    if not isinstance(ds, PreprocessDataset):
        raise TypeError(f"Expected PreprocessDataset, got {type(ds).__name__}")
    return DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        prefetch_factor=(int(cfg["prefetch_factor"]) if cfg["num_workers"] > 0 else None),
        persistent_workers=(bool(cfg["persistent_workers"]) if cfg["num_workers"] > 0 else False),
        multiprocessing_context=ctx_method,
    )


def _shutdown_loader(loader: _test_hooks.BatchIterableProtocol) -> None:
    """Forcefully shut down DataLoader workers to prevent lingering processes.

    This function handles both real DataLoader objects (which have _iterator)
    and test fakes (which don't need shutdown).
    """
    # Delegate to hook for cleanup - allows tests to use fakes
    _test_hooks.shutdown_loader(loader)


def _join_active_children() -> None:
    """Join any active multiprocessing children to release resources promptly.

    Repeatedly scan for active children and request termination with bounded
    joins to ensure workers exit within the current process lifetime.
    """
    # Bounded attempts to ensure deterministic cleanup without sleeping.
    max_passes = 8
    attempt = 0
    while attempt < max_passes:
        attempt += 1
        alive_after_pass = False
        children = _test_hooks.mp_active_children()
        if not children:
            break
        for child in children:
            if not child.is_alive():
                continue
            child.join(timeout=0.25)
            if child.is_alive():
                child.terminate()
                child.join(timeout=0.25)
                if child.is_alive():
                    alive_after_pass = True
        if not alive_after_pass and not _test_hooks.mp_active_children():
            break


def _measure_training(
    ds_len: int,
    loader: _BatchIterable,
    k: int,
    *,
    device: torch.device,
    batch_size_hint: int,
    model: Module,
    opt: Optimizer,
) -> tuple[float, float, float, bool]:
    """Run k training steps and measure throughput/latency/memory.

    Returns: (samples_per_sec, p95_ms, peak_percent, exceeded_threshold)
    """
    import time as _t

    # Ensure train mode for consistent calibration behavior
    model.train()

    # Warm-up a single batch to initialize optimizer state sizes (no exceptions)
    it = iter(loader)
    first = next(it, None)
    if first is None:
        _LOGGER.info("calibration_no_samples")
        return 0.0, 0.0, 0.0, False
    first_batch: tuple[torch.Tensor, torch.Tensor] = first
    x0: torch.Tensor = first_batch[0]
    y0: torch.Tensor = first_batch[1]
    x0 = x0.to(device)
    y0 = y0.to(device)
    opt.zero_grad(set_to_none=True)
    logits0: torch.Tensor = model(x0)
    loss0: torch.Tensor = torch.nn.functional.cross_entropy(logits0, y0)
    torch.autograd.backward((loss0,))
    opt.step()
    del x0, y0, logits0, loss0

    # Now measure k batches
    times: list[float] = []
    samples = 0
    exceeded = False
    peak_pct: float = 0.0
    thr = float(get_memory_guard_config()["threshold_percent"])
    n_batches = max(1, min(max(1, k), max(1, ds_len // max(1, batch_size_hint))))
    start = _t.perf_counter()
    batch: tuple[torch.Tensor, torch.Tensor]
    for seen, batch in enumerate(it, start=1):
        t0 = _t.perf_counter()
        x: torch.Tensor = batch[0]
        y: torch.Tensor = batch[1]
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        logits: torch.Tensor = model(x)
        loss: torch.Tensor = torch.nn.functional.cross_entropy(logits, y)
        torch.autograd.backward((loss,))
        opt.step()
        dt_ms = (_t.perf_counter() - t0) * 1000.0
        times.append(dt_ms)
        samples += int(y.shape[0])
        del x, y, logits, loss
        # Memory tracking: use configured guard threshold consistently
        from handwriting_ai.monitoring import get_memory_snapshot

        pct = float(get_memory_snapshot()["cgroup_usage"]["percent"])
        if pct > peak_pct:
            peak_pct = pct
        # Enforce calibration backoff under two conditions:
        # 1) threshold <= 0.0 explicitly forces backoff (used in tests), or
        # 2) running under cgroup limits and usage exceeds threshold.
        if (thr <= 0.0) or (is_cgroup_available() and pct >= thr):
            exceeded = True
        if seen >= n_batches:
            break
    total_s = _t.perf_counter() - start
    if len(times) >= 2:
        pcts = quantiles(times, n=20)
        p95 = pcts[18]
    else:
        p95 = times[0] if times else 0.0
    if samples <= 0:
        samples = int(batch_size_hint) * n_batches
    sps = float(samples) / total_s if total_s > 0 else 0.0
    del it
    return sps, p95, peak_pct, exceeded


def _measure_candidate(
    ds: PreprocessDataset,
    cand: Candidate,
    samples: int,
    on_improvement: Callable[[CalibrationResult], None] | None = None,
) -> CalibrationResult:
    """Public entry that measures a candidate with headroom expansion enabled."""
    return _measure_candidate_internal(
        ds,
        cand,
        samples,
        on_improvement=on_improvement,
        enable_headroom=True,
    )


def _measure_candidate_internal(
    ds: _test_hooks.PreprocessDatasetProtocol,
    cand: Candidate,
    samples: int,
    on_improvement: Callable[[CalibrationResult], None] | None,
    *,
    enable_headroom: bool,
) -> CalibrationResult:
    """Measure a candidate using real training steps with binary search for safe batch size."""
    torch.set_num_threads(int(cand["intra_threads"]))
    _configure_interop_threads_once(cand["interop_threads"])
    device = torch.device("cpu")

    # Build model once per process (cached) to avoid allocator churn and
    # stabilize RSS across repeated calibration attempts.
    model = _get_calibration_model()
    opt = _get_calibration_optimizer(model)

    # Calibration operates on CPU. No backend toggles are required for
    # determinism here; rely on model/optimizer reuse within this function
    # and explicit DataLoader cleanup between attempts.

    # Upper bound batch size by dataset length and log initial state
    ds_len = len(ds)
    initial_cap = max(1, min(int(cand["batch_size"]), int(ds_len)))
    _log_candidate_start(cand, 1, initial_cap)

    best_bs, best_sps, best_p95 = _search_best_batch_size(
        ds=ds,
        cand=cand,
        samples=samples,
        device=device,
        ds_len=int(ds_len),
        initial_cap=int(initial_cap),
        enable_headroom=enable_headroom,
        model=model,
        opt=opt,
        on_improvement=on_improvement,
    )

    # Finished search; log result and finalize resources
    _log_candidate_complete(best_bs, best_sps)

    # Final cleanup: aggressively join workers; retain cached model/optimizer
    # to avoid cross-run allocator churn and stabilize RSS across runs.
    _join_active_children()

    return {
        "intra_threads": cand["intra_threads"],
        "interop_threads": cand["interop_threads"],
        "num_workers": cand["num_workers"],
        "batch_size": best_bs,
        "samples_per_sec": best_sps,
        "p95_ms": best_p95,
    }


__all__ = [
    "_configure_interop_threads_once",
    "_get_calibration_optimizer",
    "_measure_candidate",
    "_measure_candidate_internal",
    "_reset_calibration_state",
]
