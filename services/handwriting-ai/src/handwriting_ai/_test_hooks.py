"""Test hooks for handwriting-ai - allows injecting test dependencies.

This module provides hooks for dependency injection in tests. Production code
sets hooks to real implementations at startup; tests set them to fakes.

Hooks are module-level callables that production code calls directly. Tests
assign fake implementations before running the code under test.

Usage in production code:
    from handwriting_ai import _test_hooks
    root = _test_hooks.guard_find_monorepo_root(start)

Usage in tests:
    from handwriting_ai import _test_hooks
    _test_hooks.guard_find_monorepo_root = lambda start: Path("/fake/root")
"""

from __future__ import annotations

# Standard library imports
import sys
import threading
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from types import ModuleType, TracebackType
from typing import BinaryIO, Literal, Protocol, Self, TextIO, TypedDict

# Third-party imports
import psutil
import torch
from PIL.Image import Image as PILImage

# Platform imports
from platform_core.config import HandwritingAiSettings
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.job_events import JobDomain
from platform_core.json_utils import JSONValue
from platform_core.logging import (
    LogFormat,
    LogLevel,
    QueueHandlerFactory,
    QueueListenerFactory,
    load_queue_handler_factory,
    load_queue_listener_factory,
    stdlib_logging,
)
from platform_workers.job_context import JobContext
from platform_workers.redis import (
    RedisStrProto,
    _RedisBytesClient,
    redis_for_kv,
    redis_raw_for_rq,
)
from platform_workers.rq_harness import RQClientQueue, WorkerConfig, rq_queue
from torch.nn import Module as TorchModule
from torch.optim.optimizer import Optimizer as TorchOptimizer

from handwriting_ai.inference.types import PredictOutput

# Local imports - these modules do NOT import _test_hooks at module level
from handwriting_ai.training.calibration._types import (
    BudgetConfigDict,
    CalibrationResultDict,
    CandidateDict,
    CandidateOutcomeDict,
    OrchestratorConfigDict,
)
from handwriting_ai.training.calibration.ds_spec import PreprocessSpec
from handwriting_ai.training.progress import (
    BatchProgressEmitter,
    BestEmitter,
    EpochEmitter,
)
from handwriting_ai.training.train_config import TrainConfig, TrainingResult


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None:
        """Run the worker with the given config."""
        ...


# =============================================================================
# Guard script hooks
# =============================================================================


class GuardRunForProjectProtocol(Protocol):
    """Protocol for run_for_project function from monorepo_guards."""

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int:
        """Run guards for a project."""
        ...


class GuardFindMonorepoRootProtocol(Protocol):
    """Protocol for _find_monorepo_root function."""

    def __call__(self, start: Path) -> Path:
        """Find the monorepo root from a starting path."""
        ...


class GuardLoadOrchestratorProtocol(Protocol):
    """Protocol for _load_orchestrator function."""

    def __call__(self, monorepo_root: Path) -> GuardRunForProjectProtocol:
        """Load the orchestrator module and return run_for_project."""
        ...


def _default_guard_find_monorepo_root(start: Path) -> Path:
    """Production implementation - finds monorepo root by climbing directories."""
    current = start
    while True:
        if (current / "libs").is_dir():
            return current
        if current.parent == current:
            raise RuntimeError("monorepo root with 'libs' directory not found")
        current = current.parent


def _default_guard_load_orchestrator(monorepo_root: Path) -> GuardRunForProjectProtocol:
    """Production implementation - loads the orchestrator module."""
    libs_path = monorepo_root / "libs"
    guards_src = libs_path / "monorepo_guards" / "src"
    sys.path.insert(0, str(guards_src))
    sys.path.insert(0, str(libs_path))
    mod = __import__("monorepo_guards.orchestrator", fromlist=["run_for_project"])
    run_for_project: GuardRunForProjectProtocol = mod.run_for_project
    return run_for_project


# =============================================================================
# RQ and queue hooks (used by api/dependencies.py and jobs/digits.py)
#
# For Redis, use the centralized platform_workers.testing.hooks system.
# Tests should set platform_workers.testing.hooks.load_redis_str_module and
# platform_workers.testing.hooks.load_redis_bytes_module.
# =============================================================================


class KVStoreFactoryProtocol(Protocol):
    """Protocol for key-value store factory (returns RedisStrProto)."""

    def __call__(self, url: str) -> RedisStrProto:
        """Create a string-based KV store client from URL."""
        ...


class QueueConnFactoryProtocol(Protocol):
    """Protocol for queue connection factory (returns _RedisBytesClient)."""

    def __call__(self, url: str) -> _RedisBytesClient:
        """Create a bytes-based connection for queue operations from URL."""
        ...


class QueueFactoryProtocol(Protocol):
    """Protocol for job queue factory (returns RQClientQueue)."""

    def __call__(self, name: str, connection: _RedisBytesClient) -> RQClientQueue:
        """Create a job queue from name and connection."""
        ...


# =============================================================================
# Artifact store hooks (used by jobs/digits.py)
# =============================================================================


class ArtifactStoreProtocol(Protocol):
    """Protocol for ArtifactStore - allows injecting fakes for testing."""

    def upload_artifact(
        self,
        dir_path: Path,
        *,
        artifact_name: str,
        request_id: str,
    ) -> FileUploadResponse:
        """Upload a directory as a tarball artifact."""
        ...

    def download_artifact(
        self,
        file_id: str,
        *,
        dest_dir: Path,
        request_id: str,
        expected_root: str,
    ) -> Path:
        """Download and extract a tarball artifact."""
        ...


class ArtifactStoreFactoryProtocol(Protocol):
    """Protocol for ArtifactStore factory."""

    def __call__(self, api_url: str, api_key: str) -> ArtifactStoreProtocol:
        """Create an ArtifactStore instance."""
        ...


def _default_artifact_store_factory(api_url: str, api_key: str) -> ArtifactStoreProtocol:
    """Production implementation - creates real ArtifactStore."""
    from platform_core.data_bank_client import DataBankClient
    from platform_ml import ArtifactStore

    client = DataBankClient(api_url, api_key, timeout_seconds=600.0)
    return ArtifactStore(client)


# =============================================================================
# Module-level hooks
# =============================================================================

# Module-level injectable runner for testing.
# Tests set this BEFORE running worker_entry as __main__.
# Because this is a separate module, it persists across runpy.run_module.
test_runner: WorkerRunnerProtocol | None = None

# Hook for guard find_monorepo_root. Tests can override to return fake paths.
guard_find_monorepo_root: GuardFindMonorepoRootProtocol = _default_guard_find_monorepo_root

# Hook for guard load_orchestrator. Tests can override to return fake orchestrators.
guard_load_orchestrator: GuardLoadOrchestratorProtocol = _default_guard_load_orchestrator

# Hook for key-value store factory. Tests can override with FakeRedis.
# Note: For Redis module hooks, use platform_workers.testing.hooks instead.
redis_factory: KVStoreFactoryProtocol = redis_for_kv

# Hook for queue connection factory. Tests can override with fakes.
rq_conn: QueueConnFactoryProtocol = redis_raw_for_rq

# Hook for queue factory. Tests can override with fake queues.
rq_queue_factory: QueueFactoryProtocol = rq_queue

# Hook for ArtifactStore factory. Tests can override to inject fakes.
artifact_store_factory: ArtifactStoreFactoryProtocol = _default_artifact_store_factory


# =============================================================================
# Monitoring hooks (used by monitoring.py)
# =============================================================================


class VirtualMemoryResultProtocol(Protocol):
    """Protocol for psutil.virtual_memory() result (a NamedTuple with total and used)."""

    @property
    def total(self) -> int: ...

    @property
    def used(self) -> int: ...


class MemoryInfoProtocol(Protocol):
    """Protocol for psutil.Process.memory_info() result.

    Note: rss is int | str to allow tests to inject bad data for
    testing runtime isinstance defensive checks.
    """

    @property
    def rss(self) -> int | str: ...


class PsutilProcessProtocol(Protocol):
    """Protocol for psutil.Process - what monitoring code needs."""

    @property
    def pid(self) -> int: ...

    def memory_info(self) -> MemoryInfoProtocol: ...

    def children(self, recursive: bool = False) -> Sequence[PsutilProcessProtocol]: ...


class ProcessFactoryProtocol(Protocol):
    """Protocol for psutil.Process factory."""

    def __call__(self, pid: int | None = None) -> PsutilProcessProtocol: ...


class VirtualMemoryFactoryProtocol(Protocol):
    """Protocol for psutil.virtual_memory factory."""

    def __call__(self) -> VirtualMemoryResultProtocol: ...


class CpuCountFactoryProtocol(Protocol):
    """Protocol for psutil.cpu_count factory."""

    def __call__(self, *, logical: bool = True) -> int | None: ...


class GetPidProtocol(Protocol):
    """Protocol for os.getpid."""

    def __call__(self) -> int: ...


def _default_psutil_process(pid: int | None = None) -> PsutilProcessProtocol:
    """Production implementation - returns real psutil.Process."""
    return psutil.Process(pid)


def _default_psutil_virtual_memory() -> VirtualMemoryResultProtocol:
    """Production implementation - returns real psutil.virtual_memory()."""
    return psutil.virtual_memory()


def _default_psutil_cpu_count(*, logical: bool = True) -> int | None:
    """Production implementation - returns real psutil.cpu_count()."""
    import psutil as _psutil

    return _psutil.cpu_count(logical=logical)


def _default_os_getpid() -> int:
    """Production implementation - returns real os.getpid()."""
    import os as _os

    return _os.getpid()


# Hook for psutil.Process factory. Tests can override with fake processes.
# Use Callable for flexibility - tests may need to inject fakes with different
# return types to test runtime isinstance checks.
psutil_process: Callable[[int | None], PsutilProcessProtocol] = _default_psutil_process

# Hook for psutil.virtual_memory. Tests can override with fake memory stats.
psutil_virtual_memory: VirtualMemoryFactoryProtocol = _default_psutil_virtual_memory

# Hook for psutil.cpu_count. Tests can override with fake CPU counts.
psutil_cpu_count: CpuCountFactoryProtocol = _default_psutil_cpu_count

# Hook for os.getpid. Tests can override with fake PIDs.
os_getpid: GetPidProtocol = _default_os_getpid

# Cgroup path hooks - tests can override with temp paths.
cgroup_mem_current: Path = Path("/sys/fs/cgroup/memory.current")
cgroup_mem_max: Path = Path("/sys/fs/cgroup/memory.max")
cgroup_mem_stat: Path = Path("/sys/fs/cgroup/memory.stat")


# =============================================================================
# Resource detection hooks (used by training/resources.py)
# =============================================================================


class ReadTextFileProtocol(Protocol):
    """Protocol for reading text files."""

    def __call__(self, path: Path) -> str:
        """Read text content from a file path."""
        ...


def _default_read_text_file(path: Path) -> str:
    """Production implementation - reads text file."""
    return path.read_text(encoding="utf-8").strip()


# Hook for _read_text_file. Tests can override with fake file reading.
read_text_file: ReadTextFileProtocol = _default_read_text_file

# Cgroup CPU path hook - tests can override with temp paths.
cgroup_cpu_max: Path = Path("/sys/fs/cgroup/cpu.max")


class OsCpuCountProtocol(Protocol):
    """Protocol for os.cpu_count."""

    def __call__(self) -> int | None:
        """Return number of CPUs."""
        ...


def _default_os_cpu_count() -> int | None:
    """Production implementation - returns real os.cpu_count()."""
    import os as _os

    return _os.cpu_count()


# Hook for os.cpu_count. Tests can override with fake CPU counts.
os_cpu_count: OsCpuCountProtocol = _default_os_cpu_count


class TorchSetInteropThreadsProtocol(Protocol):
    """Protocol for torch.set_num_interop_threads."""

    def __call__(self, nthreads: int) -> None:
        """Set number of interop threads."""
        ...


def _default_torch_set_interop_threads(nthreads: int) -> None:
    """Production implementation - sets real interop threads.

    Catches RuntimeError if parallel work has already started, since
    torch.set_num_interop_threads must be called before any parallel ops.
    In tests, this commonly happens when multiple tests run in sequence.
    """
    import torch as _torch
    from platform_core.logging import get_logger

    try:
        _torch.set_num_interop_threads(nthreads)
    except RuntimeError as exc:
        # Parallel work already started - interop threads already configured.
        # This is expected in test environments or when multiple calibrations run.
        get_logger("handwriting_ai").debug(
            "torch_set_interop_threads_skipped reason=parallel_work_started exc=%s",
            exc,
        )


# Hook for torch.set_num_interop_threads. Tests can override with fake or error.
torch_set_interop_threads: TorchSetInteropThreadsProtocol = _default_torch_set_interop_threads


class TorchHasSetNumInteropThreadsProtocol(Protocol):
    """Protocol for hasattr(torch, 'set_num_interop_threads')."""

    def __call__(self) -> bool: ...


def _default_torch_has_set_num_interop_threads() -> bool:
    """Production implementation - checks if torch has set_num_interop_threads."""
    import torch as _torch

    return hasattr(_torch, "set_num_interop_threads")


# Hook for hasattr(torch, 'set_num_interop_threads'). Tests can override.
torch_has_set_num_interop_threads: TorchHasSetNumInteropThreadsProtocol = (
    _default_torch_has_set_num_interop_threads
)


class TorchHasGetNumInteropThreadsProtocol(Protocol):
    """Protocol for hasattr(torch, 'get_num_interop_threads')."""

    def __call__(self) -> bool: ...


def _default_torch_has_get_num_interop_threads() -> bool:
    """Production implementation - checks if torch has get_num_interop_threads."""
    import torch as _torch

    return hasattr(_torch, "get_num_interop_threads")


# Hook for hasattr(torch, 'get_num_interop_threads'). Tests can override.
torch_has_get_num_interop_threads: TorchHasGetNumInteropThreadsProtocol = (
    _default_torch_has_get_num_interop_threads
)


class TorchGetNumInteropThreadsProtocol(Protocol):
    """Protocol for torch.get_num_interop_threads."""

    def __call__(self) -> int: ...


def _default_torch_get_num_interop_threads() -> int:
    """Production implementation - gets real interop threads count."""
    import torch as _torch

    return _torch.get_num_interop_threads()


# Hook for torch.get_num_interop_threads. Tests can override.
torch_get_num_interop_threads: TorchGetNumInteropThreadsProtocol = (
    _default_torch_get_num_interop_threads
)


class TorchCudaIsAvailableProtocol(Protocol):
    """Protocol for torch.cuda.is_available."""

    def __call__(self) -> bool: ...


class TorchCudaCurrentDeviceProtocol(Protocol):
    """Protocol for torch.cuda.current_device."""

    def __call__(self) -> int: ...


class TorchCudaMemoryProtocol(Protocol):
    """Protocol for torch.cuda memory functions."""

    def __call__(self, device: int) -> int: ...


# Hooks for torch.cuda functions
torch_cuda_is_available: TorchCudaIsAvailableProtocol = torch.cuda.is_available
torch_cuda_current_device: TorchCudaCurrentDeviceProtocol = torch.cuda.current_device
torch_cuda_memory_allocated: TorchCudaMemoryProtocol = torch.cuda.memory_allocated
torch_cuda_memory_reserved: TorchCudaMemoryProtocol = torch.cuda.memory_reserved
torch_cuda_max_memory_allocated: TorchCudaMemoryProtocol = torch.cuda.max_memory_allocated


class TorchCudaEmptyCacheProtocol(Protocol):
    """Protocol for torch.cuda.empty_cache."""

    def __call__(self) -> None: ...


# Hook for torch.cuda.empty_cache. Tests can override.
torch_cuda_empty_cache: TorchCudaEmptyCacheProtocol = torch.cuda.empty_cache


# -----------------------------------------------------------------------------
# Calibration state hooks (for _INTEROP_CONFIGURED state)
# -----------------------------------------------------------------------------

# Internal state for interop threads configuration tracking.
# This is managed via hooks so tests can reset it between runs.
_INTEROP_CONFIGURED: bool = False


class InteropConfiguredGetterProtocol(Protocol):
    """Protocol for getting _INTEROP_CONFIGURED state."""

    def __call__(self) -> bool: ...


class InteropConfiguredSetterProtocol(Protocol):
    """Protocol for setting _INTEROP_CONFIGURED state."""

    def __call__(self, value: bool) -> None: ...


def _default_interop_configured_getter() -> bool:
    """Production implementation - reads module state."""
    return _INTEROP_CONFIGURED


def _default_interop_configured_setter(value: bool) -> None:
    """Production implementation - writes module state."""
    global _INTEROP_CONFIGURED
    _INTEROP_CONFIGURED = value


# Hooks for _INTEROP_CONFIGURED state. Tests can override.
interop_configured_getter: InteropConfiguredGetterProtocol = _default_interop_configured_getter
interop_configured_setter: InteropConfiguredSetterProtocol = _default_interop_configured_setter


class ResourceLimitsDict(TypedDict):
    """Resource limits returned by detect_resource_limits."""

    cpu_cores: int
    memory_bytes: int | None
    optimal_threads: int
    optimal_workers: int
    max_batch_size: int | None


class DetectResourceLimitsProtocol(Protocol):
    """Protocol for detect_resource_limits."""

    def __call__(self) -> ResourceLimitsDict:
        """Detect resource limits."""
        ...


def _default_detect_resource_limits() -> ResourceLimitsDict:
    """Production implementation - detects real resource limits."""
    from .training.resources import detect_resource_limits as _detect

    return _detect()


# Hook for detect_resource_limits. Tests can override with fake limits.
detect_resource_limits: DetectResourceLimitsProtocol = _default_detect_resource_limits


# =============================================================================
# PIL Image hooks (used by api/routes/read.py)
# =============================================================================


class PILImageOpenProtocol(Protocol):
    """Protocol for PIL.Image.open."""

    def __call__(self, fp: BinaryIO) -> PILImage: ...


def _default_pil_image_open(fp: BinaryIO) -> PILImage:
    """Production implementation - opens image using PIL."""
    from PIL import Image

    return Image.open(fp)


# Hook for PIL.Image.open. Tests can override with fake image loading.
pil_image_open: PILImageOpenProtocol = _default_pil_image_open


# =============================================================================
# Preprocessing hooks (used by api/routes/read.py, preprocess.py)
# =============================================================================

# Note: PreprocessOptions and PreprocessOutput are TypedDicts defined in their
# respective modules. We define compatible TypedDicts here to avoid circular
# imports (preprocess.py imports _test_hooks).


class PreprocessOptionsDict(TypedDict):
    """Options for preprocessing (matches preprocess.PreprocessOptions)."""

    invert: bool | None
    center: bool
    visualize: bool
    visualize_max_kb: int


class PreprocessOutputDict(TypedDict):
    """Output from preprocessing (matches inference.types.PreprocessOutput)."""

    tensor: torch.Tensor
    visual_png: bytes | None


class RunPreprocessProtocol(Protocol):
    """Protocol for run_preprocess function."""

    def __call__(self, img: PILImage, opts: PreprocessOptionsDict) -> PreprocessOutputDict: ...


class PreprocessSignatureProtocol(Protocol):
    """Protocol for preprocess_signature function."""

    def __call__(self) -> str: ...


def _default_run_preprocess(img: PILImage, opts: PreprocessOptionsDict) -> PreprocessOutputDict:
    """Production implementation - runs actual preprocessing."""
    from .preprocess import run_preprocess as _run_preprocess

    return _run_preprocess(img, opts)


def _default_preprocess_signature() -> str:
    """Production implementation - returns actual signature."""
    from .preprocess import preprocess_signature as _preprocess_signature

    return _preprocess_signature()


# Hook for run_preprocess. Tests can override with fake preprocessing.
run_preprocess: RunPreprocessProtocol = _default_run_preprocess

# Hook for preprocess_signature. Tests can override with fake signature.
preprocess_signature: PreprocessSignatureProtocol = _default_preprocess_signature


class PrincipalAngleConfidenceProtocol(Protocol):
    """Protocol for _principal_angle_confidence function."""

    def __call__(self, img: PILImage, width: int, height: int) -> tuple[float, float] | None: ...


def _default_principal_angle_confidence(
    img: PILImage, width: int, height: int
) -> tuple[float, float] | None:
    """Production implementation - computes angle confidence."""
    from .preprocess import _principal_angle_confidence as _pac

    return _pac(img, width, height)


# Hook for _principal_angle_confidence. Tests can override with fake results.
principal_angle_confidence: PrincipalAngleConfidenceProtocol = _default_principal_angle_confidence


# =============================================================================
# Model/state dict hooks (used by api/routes/admin.py, inference/engine.py)
# =============================================================================


class LoadStateDictFileProtocol(Protocol):
    """Protocol for loading state dict from file."""

    def __call__(self, path: Path) -> dict[str, torch.Tensor]: ...


class ValidateStateDictProtocol(Protocol):
    """Protocol for validating state dict."""

    def __call__(self, sd: dict[str, torch.Tensor], arch: str, n_classes: int) -> None: ...


def _default_load_state_dict_file(path: Path) -> dict[str, torch.Tensor]:
    """Production implementation - loads state dict from file."""
    result: dict[str, torch.Tensor] = torch.load(path, map_location="cpu", weights_only=True)
    return result


def _default_validate_state_dict(sd: dict[str, torch.Tensor], arch: str, n_classes: int) -> None:
    """Production implementation - validates state dict structure."""
    from .inference.engine import _validate_state_dict as _validate

    _validate(sd, arch, n_classes)


# Hook for loading state dict from file. Tests can override with fake loading.
load_state_dict_file: LoadStateDictFileProtocol = _default_load_state_dict_file

# Hook for validating state dict. Tests can override with fake validation.
validate_state_dict: ValidateStateDictProtocol = _default_validate_state_dict


# =============================================================================
# Inference engine hooks (used by inference/engine.py)
# =============================================================================


# Override for submit_predict. If set, called instead of actual inference.
submit_predict_override: Callable[[torch.Tensor], Future[PredictOutput]] | None = None

# Override for download_remote. If set, called instead of actual download.
download_remote_override: Callable[[Path, Path], None] | None = None


# =============================================================================
# Worker hooks (used by worker_entry.py)
# =============================================================================


class LoggerProtocol(Protocol):
    """Minimal protocol for logging.Logger."""

    def info(self, msg: str, *args: str) -> None: ...

    def error(self, msg: str, *args: str) -> None: ...


class RunWorkerProtocol(Protocol):
    """Protocol for run_worker function."""

    def __call__(
        self,
        config: WorkerConfig,
        logger: LoggerProtocol,
        runner: WorkerRunnerProtocol,
    ) -> None: ...


def _default_run_worker(
    config: WorkerConfig,
    logger: LoggerProtocol,
    runner: WorkerRunnerProtocol,
) -> None:
    """Production implementation - runs actual worker."""
    runner(config)


# Hook for run_worker. Tests can override to intercept worker startup.
run_worker: RunWorkerProtocol = _default_run_worker


# =============================================================================
# Threading hooks (used by api/routes/admin.py, inference/engine.py)
# =============================================================================


class ThreadProtocol(Protocol):
    """Protocol for threading.Thread."""

    def start(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...


class EventProtocol(Protocol):
    """Protocol for threading.Event."""

    def set(self) -> None: ...

    def wait(self, timeout: float | None = None) -> bool: ...

    def is_set(self) -> bool: ...


class ThreadTargetProtocol(Protocol):
    """Protocol for thread target callable - callable with no args."""

    def __call__(self) -> None: ...


class ThreadFactoryProtocol(Protocol):
    """Protocol for threading.Thread constructor."""

    def __call__(
        self,
        *,
        target: ThreadTargetProtocol,
        daemon: bool = True,
        name: str | None = None,
    ) -> ThreadProtocol: ...


class EventFactoryProtocol(Protocol):
    """Protocol for threading.Event constructor."""

    def __call__(self) -> EventProtocol: ...


def _default_thread_factory(
    *,
    target: ThreadTargetProtocol,
    daemon: bool = True,
    name: str | None = None,
) -> ThreadProtocol:
    """Production implementation - creates real thread."""
    return threading.Thread(target=target, daemon=daemon, name=name)


def _default_event_factory() -> EventProtocol:
    """Production implementation - creates real event."""
    return threading.Event()


# Hook for creating threads. Tests can override with fake threads.
thread_factory: ThreadFactoryProtocol = _default_thread_factory

# Hook for creating events. Tests can override with fake events.
event_factory: EventFactoryProtocol = _default_event_factory


# =============================================================================
# Import hooks (used by inference/engine.py)
# =============================================================================


class ImportModuleProtocol(Protocol):
    """Protocol for importlib.import_module."""

    def __call__(self, name: str, package: str | None = None) -> ModuleType: ...


def _default_import_module(name: str, package: str | None = None) -> ModuleType:
    """Production implementation - imports actual module."""
    import importlib

    return importlib.import_module(name, package)


# Hook for importing modules. Tests can override with fake imports.
import_module: ImportModuleProtocol = _default_import_module


# =============================================================================
# Safety/monitoring hooks (used by training/safety.py)
# =============================================================================


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


def _default_get_memory_snapshot() -> MemorySnapshotDict:
    """Production implementation - gets real memory snapshot."""
    from .monitoring import get_memory_snapshot as _get

    return _get()


def _default_check_memory_pressure(*, threshold_percent: float) -> bool:
    """Production implementation - checks real memory pressure."""
    from .monitoring import check_memory_pressure as _check

    return _check(threshold_percent)


# Hook for get_memory_snapshot. Tests can override with fake snapshots.
get_memory_snapshot: GetMemorySnapshotProtocol = _default_get_memory_snapshot

# Hook for check_memory_pressure. Tests can override with fake pressure checks.
check_memory_pressure: CheckMemoryPressureProtocol = _default_check_memory_pressure


class IsCgroupAvailableProtocol(Protocol):
    """Protocol for is_cgroup_available function."""

    def __call__(self) -> bool: ...


def _default_is_cgroup_available() -> bool:
    """Production implementation - checks actual cgroup availability."""
    from .monitoring import is_cgroup_available as _ica

    return _ica()


# Hook for is_cgroup_available. Tests can override with fake availability.
is_cgroup_available: IsCgroupAvailableProtocol = _default_is_cgroup_available


class OnBatchCheckProtocol(Protocol):
    """Protocol for on_batch_check function."""

    def __call__(self) -> bool: ...


def _default_on_batch_check() -> bool:
    """Production implementation - calls real on_batch_check."""
    from .training.safety import on_batch_check as _obc

    return _obc()


# Hook for on_batch_check. Tests can override with fake checks.
on_batch_check: OnBatchCheckProtocol = _default_on_batch_check


class LoggerInstanceProtocol(Protocol):
    """Protocol for logger instance returned by get_logger.

    Note: setLevel is intentionally omitted as it's not used through this Protocol.
    Code that needs setLevel uses the actual logger directly.
    """

    def info(
        self,
        msg: str,
        *args: float | int | str | Path | BaseException,
        extra: dict[str, str | int | float | bool | None] | None = None,
    ) -> None: ...

    def warning(
        self,
        msg: str,
        *args: float | int | str | Path | BaseException,
        extra: dict[str, str | int | float | bool | None] | None = None,
    ) -> None: ...

    def error(
        self,
        msg: str,
        *args: float | int | str | Path | BaseException,
        extra: dict[str, str | int | float | bool | None] | None = None,
    ) -> None: ...

    def debug(
        self,
        msg: str,
        *args: float | int | str | Path | BaseException,
        extra: dict[str, str | int | float | bool | None] | None = None,
    ) -> None: ...


class GetLoggerProtocol(Protocol):
    """Protocol for get_logger function."""

    def __call__(self, name: str) -> LoggerInstanceProtocol: ...


def _default_get_logger(name: str) -> LoggerInstanceProtocol:
    """Production implementation - calls real get_logger."""
    from platform_core.logging import get_logger as _gl

    return _gl(name)


# Hook for get_logger. Tests can override with fake loggers.
get_logger: GetLoggerProtocol = _default_get_logger


# =============================================================================
# Time hooks (used by training/calibration/runner.py)
# =============================================================================


class PerfCounterProtocol(Protocol):
    """Protocol for time.perf_counter."""

    def __call__(self) -> float: ...


def _default_perf_counter() -> float:
    """Production implementation."""
    import time as _time

    return _time.perf_counter()


# Hook for time.perf_counter. Tests can override with fake timing.
perf_counter: PerfCounterProtocol = _default_perf_counter


# =============================================================================
# OS hooks (used by training/calibration/runner.py)
# =============================================================================


class OsAccessProtocol(Protocol):
    """Protocol for os.access."""

    def __call__(self, path: str, mode: int) -> bool: ...


def _default_os_access(path: str, mode: int) -> bool:
    """Production implementation."""
    import os as _os

    return _os.access(path, mode)


# Hook for os.access. Tests can override with fake access checks.
os_access: OsAccessProtocol = _default_os_access

# Hook for os.name. Tests can override to simulate different platforms.
os_name: str = "posix"  # Default, will be set at import time


def _init_os_name() -> None:
    """Initialize os_name from actual system at module load time."""
    import os as _os

    global os_name
    os_name = _os.name


_init_os_name()


# =============================================================================
# Progress hooks (used by training/loops.py)
# =============================================================================


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


def _default_emit_batch(metrics: BatchMetricsDict) -> None:
    """Production implementation - calls real emit_batch."""
    from .training.progress import emit_batch as _eb

    _eb(metrics)


# Hook for emit_batch. Tests can override with fake emitters.
emit_batch: EmitBatchProtocol = _default_emit_batch


# =============================================================================
# Inference hooks (used by inference/engine.py)
# =============================================================================


class LoadStateResultProtocol(Protocol):
    """Protocol for load_state_dict return value."""

    @property
    def missing_keys(self) -> tuple[str, ...] | Sequence[str]: ...

    @property
    def unexpected_keys(self) -> tuple[str, ...] | Sequence[str]: ...


class InferenceTorchModelProtocol(Protocol):
    """Protocol for torch model used in inference."""

    def eval(self) -> Self: ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> LoadStateResultProtocol: ...

    def train(self, mode: bool = True) -> Self: ...

    def state_dict(self) -> dict[str, torch.Tensor]: ...

    def parameters(self) -> Sequence[torch.nn.Parameter]: ...


class BuildModelProtocol(Protocol):
    """Protocol for _build_model function."""

    def __call__(self, arch: str, n_classes: int) -> InferenceTorchModelProtocol: ...


def _default_build_model(arch: str, n_classes: int) -> InferenceTorchModelProtocol:
    """Production implementation - calls real _build_model."""
    from .inference.engine import _build_model as _bm

    return _bm(arch, n_classes)


# Hook for _build_model. Tests can override with fake model builders.
build_model: BuildModelProtocol = _default_build_model


# =============================================================================
# Random hooks (used by training/augment.py)
# =============================================================================


class RandomFloatProtocol(Protocol):
    """Protocol for random.random."""

    def __call__(self) -> float: ...


class RandomRandintProtocol(Protocol):
    """Protocol for random.randint."""

    def __call__(self, a: int, b: int) -> int: ...


class RandomUniformProtocol(Protocol):
    """Protocol for random.uniform."""

    def __call__(self, a: float, b: float) -> float: ...


def _default_random() -> float:
    """Production implementation."""
    import random as _random

    return _random.random()


def _default_randint(a: int, b: int) -> int:
    """Production implementation."""
    import random as _random

    return _random.randint(a, b)


def _default_uniform(a: float, b: float) -> float:
    """Production implementation."""
    import random as _random

    return _random.uniform(a, b)


# Hook for random.random. Tests can override with deterministic values.
random_random: RandomFloatProtocol = _default_random

# Hook for random.randint. Tests can override with deterministic values.
random_randint: RandomRandintProtocol = _default_randint

# Hook for random.uniform. Tests can override with deterministic values.
random_uniform: RandomUniformProtocol = _default_uniform


# =============================================================================
# Calibration hooks (used by training/calibration/calibrator.py)
# =============================================================================


class AugmentKnobsDict(TypedDict):
    """TypedDict mirroring _AugmentKnobs from dataset.py to avoid circular imports."""

    enable: bool
    rotate_deg: float
    translate_frac: float
    noise_prob: float
    noise_salt_vs_pepper: float
    dots_prob: float
    dots_count: int
    dots_size_px: int
    blur_sigma: float
    morph_mode: str
    morph_kernel_px: int


class PreprocessDatasetProtocol(Protocol):
    """Protocol for PreprocessDataset to avoid circular imports.

    Mirrors the interface of handwriting_ai.training.dataset.PreprocessDataset
    without importing it.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...

    @property
    def knobs(self) -> AugmentKnobsDict:
        """Expose augmentation knobs for runtime access."""
        ...


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
        cand: CandidateDict,
        samples: int,
        budget: BudgetConfigDict,
    ) -> CandidateOutcomeDict: ...


class OrchestratorProtocol(Protocol):
    """Protocol for calibration orchestrator."""

    def __init__(
        self, *, runner: CandidateRunnerProtocol, config: OrchestratorConfigDict
    ) -> None: ...

    def run_stage_a(
        self,
        ds: PreprocessDatasetProtocol | PreprocessSpec,
        cands: list[CandidateDict],
        samples: int,
    ) -> list[CalibrationResultDict]: ...

    def run_stage_b(
        self,
        ds: PreprocessDatasetProtocol | PreprocessSpec,
        shortlist: list[CalibrationResultDict],
        samples: int,
    ) -> list[CalibrationResultDict]: ...


class OrchestratorFactoryProtocol(Protocol):
    """Protocol for orchestrator factory."""

    def __call__(
        self, *, runner: CandidateRunnerProtocol, config: OrchestratorConfigDict
    ) -> OrchestratorProtocol: ...


def _default_orchestrator_factory(
    *, runner: CandidateRunnerProtocol, config: OrchestratorConfigDict
) -> OrchestratorProtocol:
    """Production implementation - creates real Orchestrator."""
    from handwriting_ai.training.calibration.orchestrator import Orchestrator

    return Orchestrator(runner=runner, config=config)


# Hook for Orchestrator factory. Tests can override to inject fakes.
orchestrator_factory: OrchestratorFactoryProtocol = _default_orchestrator_factory


# -----------------------------------------------------------------------------
# Calibration measure hooks (_safe_loader, _measure_training)
# -----------------------------------------------------------------------------


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


# Hook for _safe_loader. Tests can override to inject fake loaders.
safe_loader: SafeLoaderProtocol = _default_safe_loader


class ShutdownLoaderProtocol(Protocol):
    """Protocol for shutdown_loader function."""

    def __call__(self, loader: BatchIterableProtocol) -> None: ...


class _LoaderIterator(Protocol):
    """Protocol for DataLoader internal iterator with shutdown capability."""

    def _shutdown_workers(self) -> None: ...


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


# Hook for shutdown_loader. Tests can override to use fakes that don't need cleanup.
shutdown_loader: ShutdownLoaderProtocol = _default_shutdown_loader

# Hook for _measure_training. Tests can override to stub training measurement.
measure_training: MeasureTrainingProtocol = _default_measure_training


class GcCollectProtocol(Protocol):
    """Protocol for gc.collect function."""

    def __call__(self) -> int: ...


def _default_gc_collect() -> int:
    """Production implementation."""
    import gc as _gc

    return _gc.collect()


# Hook for gc.collect. Tests can override to track/stub garbage collection.
gc_collect: GcCollectProtocol = _default_gc_collect


# =============================================================================
# Multiprocessing hooks (used by training/calibration/measure.py)
# =============================================================================


class MultiprocessingChildProtocol(Protocol):
    """Protocol for multiprocessing child process."""

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    def terminate(self) -> None: ...


class MultiprocessingActiveChildrenProtocol(Protocol):
    """Protocol for multiprocessing.active_children."""

    def __call__(self) -> list[MultiprocessingChildProtocol]: ...


def _default_mp_active_children() -> list[MultiprocessingChildProtocol]:
    """Production implementation."""
    import multiprocessing as _mp

    return list(_mp.active_children())


# Hook for multiprocessing.active_children. Tests can override with fakes.
mp_active_children: MultiprocessingActiveChildrenProtocol = _default_mp_active_children


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


# Hooks for multiprocessing start methods and context
mp_get_all_start_methods: MpGetAllStartMethodsProtocol = _default_mp_get_all_start_methods
mp_get_context: MpGetContextProtocol = _default_mp_get_context


# =============================================================================
# PIL preprocessing hooks (used by preprocess.py)
# =============================================================================


class ExifTransposeProtocol(Protocol):
    """Protocol for ImageOps.exif_transpose."""

    def __call__(self, img: PILImage) -> PILImage | None: ...


def _default_exif_transpose(img: PILImage) -> PILImage | None:
    """Production implementation."""
    from PIL import ImageOps

    return ImageOps.exif_transpose(img)


# Hook for ImageOps.exif_transpose. Tests can override with fake.
exif_transpose: ExifTransposeProtocol = _default_exif_transpose


class PrincipalAngleProtocol(Protocol):
    """Protocol for _principal_angle function."""

    def __call__(self, img: PILImage, width: int, height: int) -> float | None: ...


def _default_principal_angle(img: PILImage, width: int, height: int) -> float | None:
    """Production implementation."""
    from .preprocess import _principal_angle as _pa

    return _pa(img, width, height)


# Hook for _principal_angle. Tests can override with fake angle values.
principal_angle: PrincipalAngleProtocol = _default_principal_angle


# =============================================================================
# Digits job hooks (used by jobs/digits.py)
# =============================================================================


class RunTrainingProtocol(Protocol):
    """Protocol for _run_training function."""

    def __call__(self, cfg: TrainConfig) -> TrainingResult: ...


class LoadSettingsProtocol(Protocol):
    """Protocol for _load_settings function."""

    def __call__(self, *, create_dirs: bool = True) -> HandwritingAiSettings: ...


def _default_run_training(cfg: TrainConfig) -> TrainingResult:
    """Production implementation - runs actual training."""
    from handwriting_ai.jobs.digits import _run_training_impl as _rt

    return _rt(cfg)


def _default_load_settings(*, create_dirs: bool = True) -> HandwritingAiSettings:
    """Production implementation - loads actual settings."""
    from handwriting_ai.config import load_settings as _ls

    return _ls(create_dirs=create_dirs)


# Hook for _run_training. Tests can override with fake training.
run_training: RunTrainingProtocol = _default_run_training

# Hook for _load_settings. Tests can override with fake settings.
load_settings: LoadSettingsProtocol = _default_load_settings


# =============================================================================
# Job context hooks (used by jobs/digits.py)
# =============================================================================


class JobContextProtocol(Protocol):
    """Protocol for JobContext returned by make_job_context."""

    def publish_started(self) -> None: ...

    def publish_progress(
        self, progress: int, message: str | None = None, *, payload: JSONValue | None = None
    ) -> None: ...

    def publish_completed(self, result_id: str, result_bytes: int) -> None: ...

    def publish_failed(self, error_kind: str, message: str) -> None: ...


class MakeJobContextProtocol(Protocol):
    """Protocol for make_job_context function.

    Returns JobContext or None. None is used in tests to verify
    job completion without a publisher.
    """

    def __call__(
        self,
        *,
        redis: RedisStrProto,
        domain: JobDomain,
        events_channel: str,
        job_id: str,
        user_id: int,
        queue_name: str,
    ) -> JobContext | None: ...


def _default_make_job_context(
    *,
    redis: RedisStrProto,
    domain: JobDomain,
    events_channel: str,
    job_id: str,
    user_id: int,
    queue_name: str,
) -> JobContext:
    """Production implementation - creates real JobContext."""
    from platform_workers.job_context import make_job_context as _mjc

    return _mjc(
        redis=redis,
        domain=domain,
        events_channel=events_channel,
        job_id=job_id,
        user_id=user_id,
        queue_name=queue_name,
    )


# Hook for make_job_context. Tests can override with fake contexts.
make_job_context: MakeJobContextProtocol = _default_make_job_context


# =============================================================================
# Calibration runner hooks (used by training/calibration/runner.py)
# =============================================================================


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
        cand: CandidateDict,
        samples: int,
        on_improvement: Callable[[CalibrationRunnerResultDict], None] | None,
        *,
        enable_headroom: bool,
    ) -> CalibrationRunnerResultDict: ...


class EmitResultFileProtocol(Protocol):
    """Protocol for _emit_result_file."""

    def __call__(self, out_path: str, res: CalibrationRunnerResultDict) -> None: ...


class SetupLoggingProtocol(Protocol):
    """Protocol for setup_logging."""

    def __call__(
        self,
        *,
        level: LogLevel,
        format_mode: LogFormat,
        service_name: str,
        instance_id: str | None,
        extra_fields: list[str] | None,
    ) -> None: ...


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


def _default_runner_setup_logging(
    *,
    level: LogLevel,
    format_mode: LogFormat,
    service_name: str,
    instance_id: str | None,
    extra_fields: list[str] | None,
) -> None:
    """Production implementation."""
    from platform_core.logging import setup_logging as _sl

    _sl(
        level=level,
        format_mode=format_mode,
        service_name=service_name,
        instance_id=instance_id,
        extra_fields=extra_fields,
    )


# Hooks for calibration runner functions
build_dataset_from_spec: BuildDatasetFromSpecProtocol = _default_build_dataset_from_spec
measure_candidate_internal: MeasureCandidateInternalProtocol = _default_measure_candidate_internal
emit_result_file: EmitResultFileProtocol = _default_emit_result_file
runner_setup_logging: SetupLoggingProtocol = _default_runner_setup_logging


# =============================================================================
# File open hook (used by training/calibration/runner.py)
# =============================================================================


class FileOpenProtocol(Protocol):
    """Protocol for file open function for text mode."""

    def __call__(
        self,
        file: str | Path,
        encoding: str = "utf-8",
    ) -> TextIO: ...


def _default_file_open(
    file: str | Path,
    encoding: str = "utf-8",
) -> TextIO:
    """Production implementation - opens file in text mode using builtin open.

    Note: Caller is responsible for closing the returned file handle.
    """
    return Path(file).open(encoding=encoding)


# Hook for file open. Tests can override with fake file operations.
file_open: FileOpenProtocol = _default_file_open


# =============================================================================
# Subprocess runner hooks (used by training/calibration/runner.py)
# =============================================================================


# CandidateErrorDict and CandidateOutcomeDict are imported from _types.py above.

# Also import CandidateErrorDict for use in TryReadResultProtocol


class TryReadResultProtocol(Protocol):
    """Protocol for _try_read_result static method."""

    def __call__(
        self, out_path: str, *, exited: bool, exit_code: int | None
    ) -> CandidateOutcomeDict | None: ...


# =============================================================================
# Calibration cache time hooks (used by training/calibration/cache.py)
# =============================================================================


class NowTsProtocol(Protocol):
    """Protocol for _now_ts function."""

    def __call__(self) -> float: ...


def _default_now_ts() -> float:
    """Production implementation - returns current timestamp."""
    import time as _time

    return _time.time()


# Hook for _now_ts. Tests can override with fake timestamps.
now_ts: NowTsProtocol = _default_now_ts


# =============================================================================
# Path stat hook (used by inference/engine.py)
# =============================================================================


class StatResultProtocol(Protocol):
    """Protocol for os.stat_result - minimal interface for file stats."""

    @property
    def st_mtime(self) -> float: ...

    @property
    def st_size(self) -> int: ...


class PathStatProtocol(Protocol):
    """Protocol for Path.stat function."""

    def __call__(self, path: Path, *, follow_symlinks: bool = True) -> StatResultProtocol: ...


def _default_path_stat(path: Path, *, follow_symlinks: bool = True) -> StatResultProtocol:
    """Production implementation - calls real Path.stat."""
    return path.stat(follow_symlinks=follow_symlinks)


# Hook for Path.stat. Tests can override with fake stat results.
path_stat: PathStatProtocol = _default_path_stat


# =============================================================================
# State dict type guard hooks (used by inference/engine.py)
# =============================================================================


class IsWrappedStateDictProtocol(Protocol):
    """Protocol for _is_wrapped_state_dict type guard."""

    def __call__(
        self, value: dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]]
    ) -> bool: ...


class IsFlatStateDictProtocol(Protocol):
    """Protocol for _is_flat_state_dict type guard."""

    def __call__(
        self, value: dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]]
    ) -> bool: ...


def _default_is_wrapped_state_dict(
    value: dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]],
) -> bool:
    """Production implementation - checks if state dict is wrapped."""
    return set(value.keys()) == {"state_dict"}


def _default_is_flat_state_dict(
    value: dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]],
) -> bool:
    """Production implementation - checks if state dict is flat."""
    return not _default_is_wrapped_state_dict(value)


# Hook for _is_wrapped_state_dict. Tests can override with fake checks.
is_wrapped_state_dict: IsWrappedStateDictProtocol = _default_is_wrapped_state_dict

# Hook for _is_flat_state_dict. Tests can override with fake checks.
is_flat_state_dict: IsFlatStateDictProtocol = _default_is_flat_state_dict


# =============================================================================
# Training hooks (used by training/mnist_train.py)
# =============================================================================


class LogSystemInfoProtocol(Protocol):
    """Protocol for log_system_info function."""

    def __call__(self) -> None: ...


def _default_log_system_info() -> None:
    """Production implementation - logs system info."""
    from handwriting_ai.monitoring import log_system_info as _lsi

    _lsi()


# Hook for log_system_info. Tests can override to avoid monitoring calls.
log_system_info: LogSystemInfoProtocol = _default_log_system_info


class LimitThreadPoolsProtocol(Protocol):
    """Protocol for limit_thread_pools function."""

    def __call__(self, *, limits: int) -> AbstractContextManager[None]: ...


@contextmanager
def _default_limit_thread_pools(*, limits: int) -> Generator[None, None, None]:
    """Production implementation - limits thread pools."""
    from handwriting_ai.training.threadpool import limit_thread_pools as _ltp

    with _ltp(limits=limits):
        yield


# Hook for limit_thread_pools. Tests can override with fake context managers.
limit_thread_pools: LimitThreadPoolsProtocol = _default_limit_thread_pools


class TorchModelProtocol(Protocol):
    """Protocol for torch.nn.Module used in training."""

    def train(self, mode: bool = True) -> TorchModelProtocol: ...

    def eval(self) -> TorchModelProtocol: ...

    def parameters(self) -> Generator[torch.Tensor, None, None]: ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


# Import torch types for train_epoch


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


# Hook for _train_epoch. Tests can override with fake training.
train_epoch: TrainEpochProtocol = _default_train_epoch


# Note: calibrate_input_pipeline uses DataLoaderConfig and PreprocessSpec.
# DataLoaderConfig is referenced by Protocol to avoid circular imports.
# ResourceLimitsDict is defined locally to avoid circular imports.


class EffectiveConfigDict(TypedDict):
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
    ) -> EffectiveConfigDict: ...


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


# Hook for calibrate_input_pipeline. Tests can override with fake calibration.
calibrate_input_pipeline: CalibrateInputPipelineProtocol = _default_calibrate_input_pipeline


# =============================================================================
# Calibration runner hooks (used by training/calibration/runner.py)
# =============================================================================


class TempfileMkdtempProtocol(Protocol):
    """Protocol for tempfile.mkdtemp."""

    def __call__(self, prefix: str) -> str: ...


def _default_tempfile_mkdtemp(prefix: str) -> str:
    """Production implementation - creates real temp directory."""
    import tempfile as _tmp

    return _tmp.mkdtemp(prefix=prefix)


# Hook for tempfile.mkdtemp. Tests can override to control temp dirs.
tempfile_mkdtemp: TempfileMkdtempProtocol = _default_tempfile_mkdtemp


# =============================================================================
# Queue handler/listener hooks (used by training/calibration/runner.py)
# =============================================================================
# Types and factory loaders are in platform_core.logging (imported at top).

# Hooks for queue handler/listener factories.
# Production: set to real implementations at module load.
# Tests: override with fakes before running code under test.
queue_handler_factory: QueueHandlerFactory = load_queue_handler_factory()
queue_listener_factory: QueueListenerFactory = load_queue_listener_factory()


# =============================================================================
# PIL histogram hook (used by preprocess.py)
# =============================================================================


class PILHistogramProtocol(Protocol):
    """Protocol for PIL Image histogram method."""

    def __call__(self, img: PILImage) -> list[int]: ...


def _default_pil_histogram(img: PILImage) -> list[int]:
    """Production implementation - calls PIL histogram."""
    return img.histogram()


# Hook for PIL histogram. Tests can override to control histogram values.
pil_histogram: PILHistogramProtocol = _default_pil_histogram


# =============================================================================
# Otsu binarize hook (used by preprocess.py _center_on_square)
# =============================================================================


class OtsuBinarizeProtocol(Protocol):
    """Protocol for _otsu_binarize function."""

    def __call__(self, gray: PILImage) -> PILImage: ...


# This will be set by preprocess.py at module load time to avoid circular imports.
# Tests can override to return fake images for testing pix=None branch.
otsu_binarize: OtsuBinarizeProtocol


# =============================================================================
# Memory guard config hook (used by digits.py)
# =============================================================================


class MemoryGuardConfigDict(TypedDict):
    """Memory guard configuration dict."""

    enabled: bool
    threshold_percent: float
    required_consecutive: int


class GetMemoryGuardConfigProtocol(Protocol):
    """Protocol for get_memory_guard_config function."""

    def __call__(self) -> MemoryGuardConfigDict: ...


def _default_get_memory_guard_config() -> MemoryGuardConfigDict:
    """Production implementation - calls real get_memory_guard_config."""
    from .training.safety import get_memory_guard_config

    return get_memory_guard_config()


# Hook for memory guard config. Tests can override to return fake configs.
get_memory_guard_config: GetMemoryGuardConfigProtocol = _default_get_memory_guard_config


# =============================================================================
# Training progress module hook (used by jobs/digits.py)
# =============================================================================


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


def _default_get_training_progress_module() -> TrainingProgressModuleProtocol | None:
    """Production implementation - imports training progress module."""
    from handwriting_ai.training import progress

    return progress


# Hook for training progress module. Tests can override to return None.
get_training_progress_module: GetTrainingProgressModuleProtocol = (
    _default_get_training_progress_module
)


# ---------------------------------------------------------------------------
# Test data injection functions for runtime type validation tests
# ---------------------------------------------------------------------------
# These functions are called from test fakes to inject data that would not
# pass static type checking but is needed to test runtime validation.
# The return types are annotated as dict[str, torch.Tensor] but the actual
# runtime values do NOT match - this is intentional for testing.


class InjectBadStateDictListProtocol(Protocol):
    """Protocol for function that injects bad state dict (list instead of dict)."""

    def __call__(self) -> dict[str, torch.Tensor]: ...


class InjectBadStateDictValuesProtocol(Protocol):
    """Protocol for function that injects bad state dict (int values instead of Tensor)."""

    def __call__(self) -> dict[str, torch.Tensor]: ...


def _inject_bad_state_dict_list() -> dict[str, torch.Tensor]:
    """Return a list disguised as a state dict for testing runtime validation.

    The return type annotation says dict but we return a list at runtime.
    This tests the 'state_dict() did not return a dict' validation.
    """
    # Use compile/exec to execute code that mypy cannot track
    # This is intentional - we need to inject bad data to test runtime validation
    namespace: dict[str, dict[str, torch.Tensor]] = {}
    code = compile("namespace['result'] = [1, 2, 3]", "<test>", "exec")
    exec(code)
    return namespace["result"]


def _inject_bad_state_dict_values() -> dict[str, torch.Tensor]:
    """Return a dict with int values instead of Tensor for testing runtime validation.

    The return type annotation says dict[str, Tensor] but we return int values.
    This tests the 'invalid state dict entry from model' validation.
    """
    # Use compile/exec to execute code that mypy cannot track
    # This is intentional - we need to inject bad data to test runtime validation
    result: dict[str, torch.Tensor] = {}
    code = compile("result['fc.weight'] = 5", "<test>", "exec")
    exec(code)
    return result


class InjectBadStateDictNonStringKeyProtocol(Protocol):
    """Protocol for function that injects state dict with non-string key."""

    def __call__(self) -> dict[str, torch.Tensor]: ...


def _inject_bad_state_dict_non_string_key() -> dict[str, torch.Tensor]:
    """Return a dict with int key instead of str for testing runtime validation.

    The return type annotation says dict[str, Tensor] but we return int keys.
    This tests the 'state_dict key must be str' validation.
    """
    # Use compile/exec to execute code that mypy cannot track
    # This is intentional - we need to inject bad data to test runtime validation
    result: dict[str, torch.Tensor] = {}
    t = torch.zeros(1)
    code = compile("result[123] = t", "<test>", "exec")
    globs: dict[str, dict[str, torch.Tensor] | torch.Tensor] = {"result": result, "t": t}
    exec(code, globs)
    return result


# Hooks for test data injection - default to functions that return bad data.
# These are ONLY for testing runtime validation and should never be called
# in production code.
inject_bad_state_dict_list: InjectBadStateDictListProtocol = _inject_bad_state_dict_list
inject_bad_state_dict_values: InjectBadStateDictValuesProtocol = _inject_bad_state_dict_values
inject_bad_state_dict_non_string_key: InjectBadStateDictNonStringKeyProtocol = (
    _inject_bad_state_dict_non_string_key
)


# =============================================================================
# Image Protocol for testing pix=None branches in preprocess.py
# =============================================================================


class _PixelAccessProtocol(Protocol):
    """Protocol for PIL pixel access."""

    def __getitem__(self, xy: tuple[int, int]) -> int: ...


class FakeImageForPrincipalAngleProtocol(Protocol):
    """Protocol for fake images used in _principal_angle tests.

    This is the minimal interface that _principal_angle and
    _principal_angle_confidence need from an image.
    """

    def load(self) -> _PixelAccessProtocol | None:
        """Return pixel access or None to test defensive branch."""
        ...


class _FakeImageReturnsNoneFromLoad:
    """Fake image that returns None from load() for testing defensive branches.

    This tests the `if pix is None: return None` path in _principal_angle
    and _principal_angle_confidence.
    """

    def load(self) -> _PixelAccessProtocol | None:
        return None


def inject_fake_image_as_pil() -> PILImage:
    """Inject a fake image that returns None from load() as PILImage type.

    Uses compile/exec to bypass static type checking. The return type
    annotation says PILImage but we return _FakeImageReturnsNoneFromLoad.
    This tests the defensive `if pix is None: return None` branches.
    """
    fake = _FakeImageReturnsNoneFromLoad()
    namespace: dict[str, PILImage] = {}
    code = compile("namespace['result'] = fake", "<test>", "exec")
    globs: dict[str, dict[str, PILImage] | _FakeImageReturnsNoneFromLoad] = {
        "namespace": namespace,
        "fake": fake,
    }
    exec(code, globs)
    return namespace["result"]


# =============================================================================
# No-flush handler injection for testing hasattr(h, "flush") branch in runner.py
# =============================================================================


class _MinimalHandler:
    """Handler-like object without flush attribute for testing.

    This class does NOT inherit from logging.Handler so it lacks the flush
    attribute. Used to test the defensive hasattr check in _child_entry.
    """

    level: int

    def __init__(self) -> None:
        self.level = stdlib_logging.DEBUG

    def handle(self, record: stdlib_logging.LogRecord) -> bool:
        """Handle a log record (no-op). Required by logging internals."""
        _ = record
        return True


def inject_no_flush_handler(log: stdlib_logging.Logger) -> None:
    """Inject a handler without flush attribute into a logger.

    Uses compile/exec to bypass static type checking. The log.handlers
    list expects logging.Handler but we inject _MinimalHandler which
    lacks the flush attribute. This tests the defensive hasattr branch.
    """
    handler = _MinimalHandler()
    code = compile("log.handlers.append(handler)", "<test>", "exec")
    globs: dict[str, stdlib_logging.Logger | _MinimalHandler] = {"log": log, "handler": handler}
    exec(code, globs)


# =============================================================================
# Mixed-precision training hooks (used by training/loops.py)
# =============================================================================


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


# Hook for get_autocast_context. Tests can override to inject fakes.
get_autocast_context: GetAutocastContextProtocol = _default_get_autocast_context

# Hook for create_grad_scaler. Tests can override to inject fakes.
create_grad_scaler: CreateGradScalerProtocol = _default_create_grad_scaler
