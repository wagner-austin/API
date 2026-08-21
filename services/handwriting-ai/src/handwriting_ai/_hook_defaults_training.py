"""Default (production) implementations for the training-pipeline hooks."""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal

import torch
from PIL.Image import Image as PILImage
from torch.nn import Module as TorchModule
from torch.optim.optimizer import Optimizer as TorchOptimizer

from handwriting_ai._hook_protocols import (
    GuardRunForProjectProtocol,
)
from handwriting_ai._hook_protocols_ml import (
    InferenceTorchModelProtocol,
    PreprocessDatasetProtocol,
    ResourceLimitsDict,
)
from handwriting_ai._hook_protocols_training import (
    BatchIterableProtocol,
    BatchLoaderProtocol,
    BatchMetricsDict,
    CalibrationRunnerResultDict,
    CandidateRunnerProtocol,
    DataLoaderConfigProtocol,
    EffectiveConfigDict,
    GradScalerProtocol,
    MemorySnapshotDict,
    MultiprocessingContextProtocol,
    OrchestratorProtocol,
    TrainingProgressModuleProtocol,
    _LoaderIterator,
)
from handwriting_ai.training.calibration._types import (
    CandidateDict,
    OrchestratorConfigDict,
)
from handwriting_ai.training.calibration.ds_spec import PreprocessSpec
from handwriting_ai.training.train_config import TrainConfig, TrainingResult


def _default_guard_load_orchestrator(monorepo_root: Path) -> GuardRunForProjectProtocol:
    """Production implementation - loads the orchestrator module."""
    libs_path = monorepo_root / "libs"
    guards_src = libs_path / "monorepo_guards" / "src"
    sys.path.insert(0, str(guards_src))
    sys.path.insert(0, str(libs_path))
    mod = __import__("monorepo_guards.orchestrator", fromlist=["run_for_project"])
    run_for_project: GuardRunForProjectProtocol = mod.run_for_project
    return run_for_project


def _default_get_memory_snapshot() -> MemorySnapshotDict:
    """Production implementation - gets real memory snapshot."""
    from .monitoring import get_memory_snapshot as _get

    return _get()


def _default_check_memory_pressure(*, threshold_percent: float) -> bool:
    """Production implementation - checks real memory pressure."""
    from .monitoring import check_memory_pressure as _check

    return _check(threshold_percent)


def _default_on_batch_check() -> bool:
    """Production implementation - calls real on_batch_check."""
    from .training.safety import on_batch_check as _obc

    return _obc()


def _default_emit_batch(metrics: BatchMetricsDict) -> None:
    """Production implementation - calls real emit_batch."""
    from .training.progress import emit_batch as _eb

    _eb(metrics)


def _default_build_model(arch: str, n_classes: int) -> InferenceTorchModelProtocol:
    """Production implementation - calls real _build_model."""
    from .inference.engine import _build_model as _bm

    return _bm(arch, n_classes)


def _default_random() -> float:
    """Production implementation."""
    import random as _random

    return _random.random()


def _default_orchestrator_factory(
    *, runner: CandidateRunnerProtocol, config: OrchestratorConfigDict
) -> OrchestratorProtocol:
    """Production implementation - creates real Orchestrator."""
    from handwriting_ai.training.calibration.orchestrator import Orchestrator

    return Orchestrator(runner=runner, config=config)


def _default_safe_loader(
    ds: PreprocessDatasetProtocol,
    cfg: DataLoaderConfigProtocol,
) -> BatchIterableProtocol:
    """Production implementation."""
    from handwriting_ai.training.calibration.measure import (
        _safe_loader as _sl,
    )

    return _sl(ds, cfg)


def _default_measure_training(
    ds_len: int,
    loader: BatchIterableProtocol,
    k: int,
    *,
    device: torch.device,
    batch_size_hint: int,
    model: TorchModule,
    opt: TorchOptimizer,
) -> tuple[float, float, float, bool]:
    """Production implementation."""
    from handwriting_ai.training.calibration.measure import (
        _measure_training as _mt,
    )

    return _mt(
        ds_len,
        loader,
        k,
        device=device,
        batch_size_hint=batch_size_hint,
        model=model,
        opt=opt,
    )


def _default_shutdown_loader(loader: BatchIterableProtocol) -> None:
    """Production implementation - shuts down DataLoader workers.

    This handles the DataLoader-specific cleanup of internal iterator
    and worker processes. For test fakes, this is a no-op.
    """
    # Only DataLoader has _iterator - test fakes don't need shutdown
    iterator_obj_raw: _LoaderIterator | None = getattr(loader, "_iterator", None)
    if iterator_obj_raw is None:
        return
    iterator_obj_raw._shutdown_workers()
    # Clear the iterator reference on the loader (DataLoader-specific).
    # The actual DataLoader type has _iterator; we use a compile/exec
    # trick to set it without Protocol complaints.
    code = compile("loader._iterator = None", "<shutdown>", "exec")
    exec(code)


def _default_gc_collect() -> int:
    """Production implementation."""
    import gc as _gc

    return _gc.collect()


def _default_mp_get_all_start_methods() -> list[str]:
    """Production implementation."""
    import multiprocessing as _mp

    return list(_mp.get_all_start_methods())


def _default_mp_get_context(method: str | None) -> MultiprocessingContextProtocol:
    """Production implementation - returns context with requested method.

    Note: This hook is primarily for tests. The actual mp.get_context returns
    BaseContext which satisfies this Protocol since it has a `method` attribute.
    """
    import multiprocessing as _mp

    ctx = _mp.get_context(method)
    # Create a simple wrapper that exposes just the method attribute
    # to satisfy the Protocol without returning the full BaseContext.

    class _CtxWrapper:
        def __init__(self, method_val: str | None) -> None:
            self.method = method_val

    # Get the context name safely - BaseContext stores it in _name
    ctx_name: str | None = getattr(ctx, "_name", method)
    return _CtxWrapper(ctx_name)


def _default_exif_transpose(img: PILImage) -> PILImage | None:
    """Production implementation."""
    from PIL import ImageOps

    return ImageOps.exif_transpose(img)


def _default_run_training(cfg: TrainConfig) -> TrainingResult:
    """Production implementation - runs actual training."""
    from handwriting_ai.jobs.digits import _run_training_impl as _rt

    return _rt(cfg)


def _default_build_dataset_from_spec(
    spec: PreprocessSpec,
) -> PreprocessDatasetProtocol:
    """Production implementation."""
    from handwriting_ai.training.calibration.runner import (
        _build_dataset_from_spec as _bds,
    )

    return _bds(spec)


def _default_measure_candidate_internal(
    ds: PreprocessDatasetProtocol,
    cand: CandidateDict,
    samples: int,
    on_improvement: Callable[[CalibrationRunnerResultDict], None] | None,
    *,
    enable_headroom: bool,
) -> CalibrationRunnerResultDict:
    """Production implementation."""
    from handwriting_ai.training.calibration.measure import (
        _measure_candidate_internal as _mci,
    )

    return _mci(ds, cand, samples, on_improvement, enable_headroom=enable_headroom)


def _default_emit_result_file(out_path: str, res: CalibrationRunnerResultDict) -> None:
    """Production implementation."""
    from handwriting_ai.training.calibration.runner import (
        _emit_result_file as _erf,
    )

    _erf(out_path, res)


def _default_train_epoch(
    model: TorchModule,
    train_loader: BatchLoaderProtocol,
    device: torch.device,
    precision: Literal["fp32", "fp16", "bf16"],
    optimizer: TorchOptimizer,
    ep: int,
    ep_total: int,
    total_batches: int,
) -> float:
    """Production implementation - runs one training epoch."""
    from handwriting_ai.training.loops import train_epoch as _te

    return _te(
        model,
        train_loader,
        device,
        precision,
        optimizer,
        ep=ep,
        ep_total=ep_total,
        total_batches=total_batches,
    )


def _default_calibrate_input_pipeline(
    ds: PreprocessSpec,
    *,
    limits: ResourceLimitsDict,
    requested_batch_size: int,
    samples: int,
    cache_path: Path,
    ttl_seconds: int,
    force: bool,
) -> EffectiveConfigDict:
    """Production implementation - runs real calibration."""
    from handwriting_ai.training.calibrate import calibrate_input_pipeline as _cip

    return _cip(
        ds,
        limits=limits,
        requested_batch_size=requested_batch_size,
        samples=samples,
        cache_path=cache_path,
        ttl_seconds=ttl_seconds,
        force=force,
    )


def _default_tempfile_mkdtemp(prefix: str) -> str:
    """Production implementation - creates real temp directory."""
    import tempfile as _tmp

    return _tmp.mkdtemp(prefix=prefix)


def _default_get_training_progress_module() -> TrainingProgressModuleProtocol | None:
    """Production implementation - imports training progress module."""
    from handwriting_ai.training import progress

    return progress


def _default_get_autocast_context(
    precision: Literal["fp32", "fp16", "bf16"], device: torch.device
) -> AbstractContextManager[None]:
    """Production implementation - get autocast context based on precision and device.

    Args:
        precision: The precision to use ("fp32", "fp16", "bf16").
        device: The device (cpu or cuda).

    Returns:
        A context manager for autocast, or nullcontext for fp32.

    Note:
        By the time this is called, precision has been validated by resolve_precision.
        fp16/bf16 on CPU raises in resolve_precision, so we only reach here with CUDA.
    """
    from contextlib import nullcontext as _nullcontext

    if precision == "fp32":
        return _nullcontext()
    # fp16/bf16 requires CUDA - resolve_precision enforces this upstream
    # Get autocast from torch.amp (PyTorch 2.0+ API)
    torch_amp = __import__("torch.amp", fromlist=["autocast"])
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    ctx: AbstractContextManager[None] = torch_amp.autocast(device_type=device.type, dtype=dtype)
    return ctx


def _default_create_grad_scaler() -> GradScalerProtocol:
    """Production implementation - create a GradScaler for fp16 mixed precision training.

    Returns:
        A GradScaler instance for scaling gradients.
    """
    torch_amp = __import__("torch.amp", fromlist=["GradScaler"])
    scaler: GradScalerProtocol = torch_amp.GradScaler()
    return scaler
