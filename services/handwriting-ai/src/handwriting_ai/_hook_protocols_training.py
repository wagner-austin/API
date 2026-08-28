"""Training-pipeline hook protocols (loaders, batches, calibration, memory guard)."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Literal, Protocol, TypedDict

import torch
from PIL.Image import Image as PILImage
from torch.nn import Module as TorchModule
from torch.optim.optimizer import Optimizer as TorchOptimizer

from handwriting_ai._hook_protocols_ml import (
    InferenceTorchModelProtocol,
    PreprocessDatasetProtocol,
    ResourceLimitsDict,
)
from handwriting_ai.training.calibration._types import (
    BudgetConfig,
    CalibrationResult,
    Candidate,
    CandidateOutcome,
    OrchestratorConfig,
)
from handwriting_ai.training.calibration.ds_spec import PreprocessSpec
from handwriting_ai.training.progress import (
    BatchProgressEmitter,
    BestEmitter,
    EpochEmitter,
)
from handwriting_ai.training.train_config import TrainConfig, TrainingResult


class CgroupMemoryUsageDict(TypedDict):
    """Cgroup-level memory usage (what the kernel OOM killer sees)."""

    usage_bytes: int
    limit_bytes: int
    percent: float


class CgroupMemoryBreakdownDict(TypedDict):
    """Detailed memory breakdown from cgroup memory.stat."""

    anon_bytes: int
    file_bytes: int
    kernel_bytes: int
    slab_bytes: int


class ProcessMemoryDict(TypedDict):
    """Per-process memory information."""

    pid: int
    rss_bytes: int


class MemorySnapshotDict(TypedDict):
    """Complete memory snapshot including process, cgroup, and worker data."""

    main_process: ProcessMemoryDict
    workers: tuple[ProcessMemoryDict, ...]
    cgroup_usage: CgroupMemoryUsageDict
    cgroup_breakdown: CgroupMemoryBreakdownDict


class GetMemorySnapshotProtocol(Protocol):
    """Protocol for get_memory_snapshot function."""

    def __call__(self) -> MemorySnapshotDict: ...


class CheckMemoryPressureProtocol(Protocol):
    """Protocol for check_memory_pressure function."""

    def __call__(self, *, threshold_percent: float) -> bool: ...


class OnBatchCheckProtocol(Protocol):
    """Protocol for on_batch_check function."""

    def __call__(self) -> bool: ...


class BatchMetricsDict(TypedDict):
    """Single source of truth for batch progress metrics."""

    epoch: int
    total_epochs: int
    batch: int
    total_batches: int
    batch_loss: float
    batch_acc: float
    avg_loss: float
    samples_per_sec: float
    main_rss_mb: int
    workers_rss_mb: int
    worker_count: int
    cgroup_usage_mb: int
    cgroup_limit_mb: int
    cgroup_pct: float
    anon_mb: int
    file_mb: int


class EmitBatchProtocol(Protocol):
    """Protocol for emit_batch function."""

    def __call__(self, metrics: BatchMetricsDict) -> None: ...


class BuildModelProtocol(Protocol):
    """Protocol for _build_model function."""

    def __call__(self, arch: str, n_classes: int) -> InferenceTorchModelProtocol: ...


class RandomFloatProtocol(Protocol):
    """Protocol for random.random."""

    def __call__(self) -> float: ...


class RandomRandintProtocol(Protocol):
    """Protocol for random.randint."""

    def __call__(self, a: int, b: int) -> int: ...


class RandomUniformProtocol(Protocol):
    """Protocol for random.uniform."""

    def __call__(self, a: float, b: float) -> float: ...


class DataLoaderConfigProtocol(Protocol):
    """Protocol for DataLoaderConfig to avoid circular imports.

    Mirrors the interface of handwriting_ai.training.dataset.DataLoaderConfig
    without importing it.
    """

    def __getitem__(self, key: str) -> int | bool: ...


class CandidateRunnerProtocol(Protocol):
    """Protocol for calibration candidate runner.

    Matches the CandidateRunner Protocol from runner.py.
    """

    def run(
        self,
        ds: PreprocessDatasetProtocol | PreprocessSpec,
        cand: Candidate,
        samples: int,
        budget: BudgetConfig,
    ) -> CandidateOutcome: ...


class OrchestratorProtocol(Protocol):
    """Protocol for calibration orchestrator."""

    def __init__(self, *, runner: CandidateRunnerProtocol, config: OrchestratorConfig) -> None: ...

    def run_stage_a(
        self,
        ds: PreprocessDatasetProtocol | PreprocessSpec,
        cands: list[Candidate],
        samples: int,
    ) -> list[CalibrationResult]: ...

    def run_stage_b(
        self,
        ds: PreprocessDatasetProtocol | PreprocessSpec,
        shortlist: list[CalibrationResult],
        samples: int,
    ) -> list[CalibrationResult]: ...


class OrchestratorFactoryProtocol(Protocol):
    """Protocol for orchestrator factory."""

    def __call__(
        self, *, runner: CandidateRunnerProtocol, config: OrchestratorConfig
    ) -> OrchestratorProtocol: ...


class BatchIteratorProtocol(Protocol):
    """Protocol for batch iterator."""

    def __iter__(self) -> BatchIteratorProtocol: ...

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]: ...


class BatchIterableProtocol(Protocol):
    """Protocol for batch iterable matching measure._BatchIterable."""

    def __iter__(self) -> BatchIteratorProtocol: ...


class SafeLoaderProtocol(Protocol):
    """Protocol for _safe_loader function."""

    def __call__(
        self,
        ds: PreprocessDatasetProtocol,
        cfg: DataLoaderConfigProtocol,
    ) -> BatchIterableProtocol: ...


class MeasureTrainingProtocol(Protocol):
    """Protocol for _measure_training function."""

    def __call__(
        self,
        ds_len: int,
        loader: BatchIterableProtocol,
        k: int,
        *,
        device: torch.device,
        batch_size_hint: int,
        model: TorchModule,
        opt: TorchOptimizer,
    ) -> tuple[float, float, float, bool]: ...


class ShutdownLoaderProtocol(Protocol):
    """Protocol for shutdown_loader function."""

    def __call__(self, loader: BatchIterableProtocol) -> None: ...


class _LoaderIterator(Protocol):
    """Protocol for DataLoader internal iterator with shutdown capability."""

    def _shutdown_workers(self) -> None: ...


class GcCollectProtocol(Protocol):
    """Protocol for gc.collect function."""

    def __call__(self) -> int: ...


class MultiprocessingChildProtocol(Protocol):
    """Protocol for multiprocessing child process."""

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    def terminate(self) -> None: ...


class MultiprocessingActiveChildrenProtocol(Protocol):
    """Protocol for multiprocessing.active_children."""

    def __call__(self) -> list[MultiprocessingChildProtocol]: ...


class MultiprocessingProcessProtocol(Protocol):
    """Protocol for multiprocessing.Process."""

    daemon: bool

    def start(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...

    def is_alive(self) -> bool: ...

    def kill(self) -> None: ...

    def terminate(self) -> None: ...

    @property
    def exitcode(self) -> int | None: ...


class MpGetAllStartMethodsProtocol(Protocol):
    """Protocol for multiprocessing.get_all_start_methods."""

    def __call__(self) -> list[str]: ...


class MultiprocessingContextProtocol(Protocol):
    """Protocol for multiprocessing context (returned by mp.get_context).

    This is a generic protocol that captures the interface used by tests,
    allowing fakes to be simpler than actual BaseContext. Used where tests
    need to provide simple fake contexts. The `method` attribute is what
    tests typically check.
    """

    method: str | None


class MpGetContextProtocol(Protocol):
    """Protocol for multiprocessing.get_context.

    Note: Tests that need to fake this hook for simple cases (like
    test_calibration_measure_context.py) can use MultiprocessingContextProtocol.
    Production code and runner.py use the actual BaseContext from mp.get_context().
    """

    def __call__(self, method: str | None) -> MultiprocessingContextProtocol: ...


class ExifTransposeProtocol(Protocol):
    """Protocol for ImageOps.exif_transpose."""

    def __call__(self, img: PILImage) -> PILImage | None: ...


class RunTrainingProtocol(Protocol):
    """Protocol for _run_training function."""

    def __call__(self, cfg: TrainConfig) -> TrainingResult: ...


class CalibrationRunnerResultDict(TypedDict):
    """Result from calibration measurement."""

    intra_threads: int
    interop_threads: int | None
    num_workers: int
    batch_size: int
    samples_per_sec: float
    p95_ms: float


class BuildDatasetFromSpecProtocol(Protocol):
    """Protocol for _build_dataset_from_spec."""

    def __call__(self, spec: PreprocessSpec) -> PreprocessDatasetProtocol: ...


class MeasureCandidateInternalProtocol(Protocol):
    """Protocol for _measure_candidate_internal."""

    def __call__(
        self,
        ds: PreprocessDatasetProtocol,
        cand: Candidate,
        samples: int,
        on_improvement: Callable[[CalibrationRunnerResultDict], None] | None,
        *,
        enable_headroom: bool,
    ) -> CalibrationRunnerResultDict: ...


class EmitResultFileProtocol(Protocol):
    """Protocol for _emit_result_file."""

    def __call__(self, out_path: str, res: CalibrationRunnerResultDict) -> None: ...


class BatchLoaderProtocol(Protocol):
    """Protocol for data loaders that yield (tensor, tensor) batches."""

    def __iter__(self) -> BatchIteratorProtocol: ...
    def __len__(self) -> int: ...


class TrainEpochProtocol(Protocol):
    """Protocol for _train_epoch function."""

    def __call__(
        self,
        model: TorchModule,
        train_loader: BatchLoaderProtocol,
        device: torch.device,
        precision: Literal["fp32", "fp16", "bf16"],
        optimizer: TorchOptimizer,
        ep: int,
        ep_total: int,
        total_batches: int,
    ) -> float: ...


class EffectiveConfig(TypedDict):
    """Mirror of EffectiveConfig (training/runtime.py) to avoid circular import."""

    intra_threads: int
    interop_threads: int | None
    batch_size: int
    loader_cfg: DataLoaderConfigProtocol


class CalibrateInputPipelineProtocol(Protocol):
    """Protocol for calibrate_input_pipeline function."""

    def __call__(
        self,
        ds: PreprocessSpec,
        *,
        limits: ResourceLimitsDict,
        requested_batch_size: int,
        samples: int,
        cache_path: Path,
        ttl_seconds: int,
        force: bool,
    ) -> EffectiveConfig: ...


class TempfileMkdtempProtocol(Protocol):
    """Protocol for tempfile.mkdtemp."""

    def __call__(self, prefix: str) -> str: ...


class OtsuBinarizeProtocol(Protocol):
    """Protocol for _otsu_binarize function."""

    def __call__(self, gray: PILImage) -> PILImage: ...


class MemoryGuardConfigDict(TypedDict):
    """Memory guard configuration dict."""

    enabled: bool
    threshold_percent: float
    required_consecutive: int


class GetMemoryGuardConfigProtocol(Protocol):
    """Protocol for get_memory_guard_config function."""

    def __call__(self) -> MemoryGuardConfigDict: ...


class TrainingProgressModuleProtocol(Protocol):
    """Protocol for the training progress module interface.

    This matches the actual functions exported by handwriting_ai.training.progress.
    """

    def set_batch_emitter(self, emitter: BatchProgressEmitter | None) -> None: ...

    def set_epoch_emitter(self, emitter: EpochEmitter | None) -> None: ...

    def set_best_emitter(self, emitter: BestEmitter | None) -> None: ...


class GetTrainingProgressModuleProtocol(Protocol):
    """Protocol for get_training_progress_module function."""

    def __call__(self) -> TrainingProgressModuleProtocol | None: ...


class GradScalerProtocol(Protocol):
    """Protocol for torch.amp.GradScaler."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor: ...

    def unscale_(self, optimizer: TorchOptimizer) -> None: ...

    def step(self, optimizer: TorchOptimizer) -> None: ...

    def update(self) -> None: ...


class AutocastContextProtocol(Protocol):
    """Protocol for autocast context manager."""

    def __enter__(self) -> None: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...


class GetAutocastContextProtocol(Protocol):
    """Protocol for get_autocast_context function."""

    def __call__(
        self, precision: Literal["fp32", "fp16", "bf16"], device: torch.device
    ) -> AbstractContextManager[None]: ...


class CreateGradScalerProtocol(Protocol):
    """Protocol for create_grad_scaler function."""

    def __call__(self) -> GradScalerProtocol: ...
