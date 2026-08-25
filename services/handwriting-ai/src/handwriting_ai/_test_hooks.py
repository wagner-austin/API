"""Test hooks for handwriting-ai - allows injecting test dependencies.

This module provides hooks for dependency injection in tests. Production code
sets hooks to real implementations at startup; tests set them to fakes.

Hooks are module-level callables that production code calls directly. Tests
assign fake implementations before running the code under test.

Usage in production code:
    from handwriting_ai import _test_hooks
    queue = _test_hooks.rq_queue_factory(name, connection=conn)

Usage in tests:
    from handwriting_ai import _test_hooks
    _test_hooks.rq_queue_factory = lambda name, *, connection: FakeQueue()
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from PIL.Image import Image as PILImage
from platform_core.logging import (
    QueueHandlerFactory,
    QueueListenerFactory,
    load_queue_handler_factory,
    load_queue_listener_factory,
    stdlib_logging,
)
from platform_workers.redis import (
    redis_for_kv,
    redis_raw_for_rq,
)
from platform_workers.rq_harness import rq_queue, run_rq_worker

from handwriting_ai._hook_defaults import (
    _default_artifact_store_factory,
    _default_file_open,
    _default_get_logger,
    _default_is_cgroup_available,
    _default_load_settings,
    _default_log_system_info,
    _default_make_job_context,
    _default_mp_active_children,
    _default_now_ts,
    _default_os_access,
    _default_path_stat,
    _default_perf_counter,
    _default_randint,
    _default_run_worker,
    _default_runner_setup_logging,
    _default_uniform,
)
from handwriting_ai._hook_defaults_ml import (
    _default_detect_resource_limits,
    _default_download_remote,
    _default_is_flat_state_dict,
    _default_is_wrapped_state_dict,
    _default_load_state_dict_file,
    _default_make_inference_pool,
    _default_pil_histogram,
    _default_pil_image_open,
    _default_preprocess_signature,
    _default_principal_angle,
    _default_principal_angle_confidence,
    _default_run_preprocess,
    _default_validate_state_dict,
)
from handwriting_ai._hook_defaults_system import (
    _default_event_factory,
    _default_import_module,
    _default_limit_thread_pools,
    _default_os_cpu_count,
    _default_os_getpid,
    _default_psutil_cpu_count,
    _default_psutil_process,
    _default_psutil_virtual_memory,
    _default_read_text_file,
    _default_thread_factory,
    _default_torch_get_num_interop_threads,
    _default_torch_has_get_num_interop_threads,
    _default_torch_has_set_num_interop_threads,
)
from handwriting_ai._hook_defaults_training import (
    _default_build_dataset_from_spec,
    _default_build_model,
    _default_calibrate_input_pipeline,
    _default_check_memory_pressure,
    _default_create_grad_scaler,
    _default_emit_batch,
    _default_emit_result_file,
    _default_exif_transpose,
    _default_gc_collect,
    _default_get_autocast_context,
    _default_get_memory_snapshot,
    _default_get_training_progress_module,
    _default_measure_candidate_internal,
    _default_measure_training,
    _default_mp_get_all_start_methods,
    _default_mp_get_context,
    _default_on_batch_check,
    _default_orchestrator_factory,
    _default_random,
    _default_run_training,
    _default_safe_loader,
    _default_shutdown_loader,
    _default_tempfile_mkdtemp,
    _default_train_epoch,
)
from handwriting_ai._hook_protocols import (
    ArtifactStoreFactoryProtocol,
    FileOpenProtocol,
    GetLoggerProtocol,
    InjectBadStateDictListProtocol,
    InjectBadStateDictNonStringKeyProtocol,
    InjectBadStateDictValuesProtocol,
    IsCgroupAvailableProtocol,
    KVStoreFactoryProtocol,
    LoadSettingsProtocol,
    LogSystemInfoProtocol,
    MakeJobContextProtocol,
    NowTsProtocol,
    OsAccessProtocol,
    PathStatProtocol,
    PerfCounterProtocol,
    QueueConnFactoryProtocol,
    QueueFactoryProtocol,
    RunWorkerProtocol,
    SetupLoggingProtocol,
    WorkerRunnerProtocol,
)

# Explicit re-exports: the hook surface consumers annotate and restore against.
from handwriting_ai._hook_protocols import ArtifactStoreProtocol as ArtifactStoreProtocol
from handwriting_ai._hook_protocols import JobContextProtocol as JobContextProtocol
from handwriting_ai._hook_protocols import LoggerInstanceProtocol as LoggerInstanceProtocol
from handwriting_ai._hook_protocols_ml import AugmentKnobsDict as AugmentKnobsDict
from handwriting_ai._hook_protocols_ml import (
    DetectResourceLimitsProtocol,
    DownloadRemoteProtocol,
    InteropConfiguredGetterProtocol,
    InteropConfiguredSetterProtocol,
    IsFlatStateDictProtocol,
    IsWrappedStateDictProtocol,
    LoadStateDictFileProtocol,
    MakeInferencePoolProtocol,
    PILHistogramProtocol,
    PILImageOpenProtocol,
    PreprocessSignatureProtocol,
    PrincipalAngleConfidenceProtocol,
    PrincipalAngleProtocol,
    RunPreprocessProtocol,
    TorchCudaCurrentDeviceProtocol,
    TorchCudaEmptyCacheProtocol,
    TorchCudaIsAvailableProtocol,
    TorchCudaMemoryProtocol,
    ValidateStateDictProtocol,
    _PixelAccessProtocol,
)
from handwriting_ai._hook_protocols_ml import (
    InferenceTorchModelProtocol as InferenceTorchModelProtocol,
)
from handwriting_ai._hook_protocols_ml import PreprocessDatasetProtocol as PreprocessDatasetProtocol
from handwriting_ai._hook_protocols_ml import PreprocessOptionsDict as PreprocessOptionsDict
from handwriting_ai._hook_protocols_ml import PreprocessOutputDict as PreprocessOutputDict
from handwriting_ai._hook_protocols_ml import ResourceLimitsDict as ResourceLimitsDict
from handwriting_ai._hook_protocols_system import (
    CpuCountFactoryProtocol,
    EventFactoryProtocol,
    GetPidProtocol,
    ImportModuleProtocol,
    LimitThreadPoolsProtocol,
    OsCpuCountProtocol,
    PsutilProcessProtocol,
    ReadTextFileProtocol,
    ThreadFactoryProtocol,
    TorchGetNumInteropThreadsProtocol,
    TorchHasGetNumInteropThreadsProtocol,
    TorchHasSetNumInteropThreadsProtocol,
    TorchSetInteropThreadsProtocol,
    VirtualMemoryFactoryProtocol,
)
from handwriting_ai._hook_protocols_system import EventProtocol as EventProtocol
from handwriting_ai._hook_protocols_system import ThreadProtocol as ThreadProtocol
from handwriting_ai._hook_protocols_system import ThreadTargetProtocol as ThreadTargetProtocol
from handwriting_ai._hook_protocols_training import BatchIterableProtocol as BatchIterableProtocol
from handwriting_ai._hook_protocols_training import BatchLoaderProtocol as BatchLoaderProtocol
from handwriting_ai._hook_protocols_training import BatchMetricsDict as BatchMetricsDict
from handwriting_ai._hook_protocols_training import (
    BuildDatasetFromSpecProtocol,
    BuildModelProtocol,
    CalibrateInputPipelineProtocol,
    CheckMemoryPressureProtocol,
    CreateGradScalerProtocol,
    EmitBatchProtocol,
    EmitResultFileProtocol,
    ExifTransposeProtocol,
    GcCollectProtocol,
    GetAutocastContextProtocol,
    GetMemoryGuardConfigProtocol,
    GetMemorySnapshotProtocol,
    GetTrainingProgressModuleProtocol,
    MeasureCandidateInternalProtocol,
    MeasureTrainingProtocol,
    MemoryGuardConfigDict,
    MpGetAllStartMethodsProtocol,
    MpGetContextProtocol,
    MultiprocessingActiveChildrenProtocol,
    OnBatchCheckProtocol,
    OrchestratorFactoryProtocol,
    OtsuBinarizeProtocol,
    RandomFloatProtocol,
    RandomRandintProtocol,
    RandomUniformProtocol,
    RunTrainingProtocol,
    SafeLoaderProtocol,
    ShutdownLoaderProtocol,
    TempfileMkdtempProtocol,
    TrainEpochProtocol,
)
from handwriting_ai._hook_protocols_training import (
    CalibrationRunnerResultDict as CalibrationRunnerResultDict,
)
from handwriting_ai._hook_protocols_training import (
    CandidateRunnerProtocol as CandidateRunnerProtocol,
)
from handwriting_ai._hook_protocols_training import (
    DataLoaderConfigProtocol as DataLoaderConfigProtocol,
)
from handwriting_ai._hook_protocols_training import EffectiveConfigDict as EffectiveConfigDict
from handwriting_ai._hook_protocols_training import MemorySnapshotDict as MemorySnapshotDict
from handwriting_ai._hook_protocols_training import (
    MultiprocessingChildProtocol as MultiprocessingChildProtocol,
)
from handwriting_ai._hook_protocols_training import OrchestratorProtocol as OrchestratorProtocol


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


def _default_interop_configured_getter() -> bool:
    """Production implementation - reads module state."""
    return _INTEROP_CONFIGURED


def _default_interop_configured_setter(value: bool) -> None:
    """Production implementation - writes module state."""
    global _INTEROP_CONFIGURED
    _INTEROP_CONFIGURED = value


def _default_get_memory_guard_config() -> MemoryGuardConfigDict:
    """Production implementation - calls real get_memory_guard_config."""
    from .training.safety import get_memory_guard_config

    return get_memory_guard_config()


worker_runner: WorkerRunnerProtocol = run_rq_worker

redis_factory: KVStoreFactoryProtocol = redis_for_kv

rq_conn: QueueConnFactoryProtocol = redis_raw_for_rq

rq_queue_factory: QueueFactoryProtocol = rq_queue

artifact_store_factory: ArtifactStoreFactoryProtocol = _default_artifact_store_factory

psutil_process: Callable[[int | None], PsutilProcessProtocol] = _default_psutil_process

psutil_virtual_memory: VirtualMemoryFactoryProtocol = _default_psutil_virtual_memory

psutil_cpu_count: CpuCountFactoryProtocol = _default_psutil_cpu_count

os_getpid: GetPidProtocol = _default_os_getpid

cgroup_mem_current: Path = Path("/sys/fs/cgroup/memory.current")

cgroup_mem_max: Path = Path("/sys/fs/cgroup/memory.max")

cgroup_mem_stat: Path = Path("/sys/fs/cgroup/memory.stat")

read_text_file: ReadTextFileProtocol = _default_read_text_file

cgroup_cpu_max: Path = Path("/sys/fs/cgroup/cpu.max")

os_cpu_count: OsCpuCountProtocol = _default_os_cpu_count

torch_set_interop_threads: TorchSetInteropThreadsProtocol = _default_torch_set_interop_threads

torch_has_set_num_interop_threads: TorchHasSetNumInteropThreadsProtocol = (
    _default_torch_has_set_num_interop_threads
)

torch_has_get_num_interop_threads: TorchHasGetNumInteropThreadsProtocol = (
    _default_torch_has_get_num_interop_threads
)

torch_get_num_interop_threads: TorchGetNumInteropThreadsProtocol = (
    _default_torch_get_num_interop_threads
)

torch_cuda_is_available: TorchCudaIsAvailableProtocol = torch.cuda.is_available

torch_cuda_current_device: TorchCudaCurrentDeviceProtocol = torch.cuda.current_device

torch_cuda_memory_allocated: TorchCudaMemoryProtocol = torch.cuda.memory_allocated

torch_cuda_memory_reserved: TorchCudaMemoryProtocol = torch.cuda.memory_reserved

torch_cuda_max_memory_allocated: TorchCudaMemoryProtocol = torch.cuda.max_memory_allocated

torch_cuda_empty_cache: TorchCudaEmptyCacheProtocol = torch.cuda.empty_cache

_INTEROP_CONFIGURED: bool = False

interop_configured_getter: InteropConfiguredGetterProtocol = _default_interop_configured_getter

interop_configured_setter: InteropConfiguredSetterProtocol = _default_interop_configured_setter

detect_resource_limits: DetectResourceLimitsProtocol = _default_detect_resource_limits

pil_image_open: PILImageOpenProtocol = _default_pil_image_open

run_preprocess: RunPreprocessProtocol = _default_run_preprocess

preprocess_signature: PreprocessSignatureProtocol = _default_preprocess_signature

principal_angle_confidence: PrincipalAngleConfidenceProtocol = _default_principal_angle_confidence

load_state_dict_file: LoadStateDictFileProtocol = _default_load_state_dict_file

validate_state_dict: ValidateStateDictProtocol = _default_validate_state_dict

make_inference_pool: MakeInferencePoolProtocol = _default_make_inference_pool

download_remote: DownloadRemoteProtocol = _default_download_remote

run_worker: RunWorkerProtocol = _default_run_worker

thread_factory: ThreadFactoryProtocol = _default_thread_factory

event_factory: EventFactoryProtocol = _default_event_factory

import_module: ImportModuleProtocol = _default_import_module

get_memory_snapshot: GetMemorySnapshotProtocol = _default_get_memory_snapshot

check_memory_pressure: CheckMemoryPressureProtocol = _default_check_memory_pressure

is_cgroup_available: IsCgroupAvailableProtocol = _default_is_cgroup_available

on_batch_check: OnBatchCheckProtocol = _default_on_batch_check

get_logger: GetLoggerProtocol = _default_get_logger

perf_counter: PerfCounterProtocol = _default_perf_counter

os_access: OsAccessProtocol = _default_os_access

os_name: str = "posix"  # Default, will be set at import time


def _init_os_name() -> None:
    """Initialize os_name from actual system at module load time."""
    import os as _os

    global os_name
    os_name = _os.name


_init_os_name()

emit_batch: EmitBatchProtocol = _default_emit_batch

build_model: BuildModelProtocol = _default_build_model

random_random: RandomFloatProtocol = _default_random

random_randint: RandomRandintProtocol = _default_randint

random_uniform: RandomUniformProtocol = _default_uniform

orchestrator_factory: OrchestratorFactoryProtocol = _default_orchestrator_factory

safe_loader: SafeLoaderProtocol = _default_safe_loader

shutdown_loader: ShutdownLoaderProtocol = _default_shutdown_loader

measure_training: MeasureTrainingProtocol = _default_measure_training

gc_collect: GcCollectProtocol = _default_gc_collect

mp_active_children: MultiprocessingActiveChildrenProtocol = _default_mp_active_children

mp_get_all_start_methods: MpGetAllStartMethodsProtocol = _default_mp_get_all_start_methods

mp_get_context: MpGetContextProtocol = _default_mp_get_context

exif_transpose: ExifTransposeProtocol = _default_exif_transpose

principal_angle: PrincipalAngleProtocol = _default_principal_angle

run_training: RunTrainingProtocol = _default_run_training

load_settings: LoadSettingsProtocol = _default_load_settings

make_job_context: MakeJobContextProtocol = _default_make_job_context

build_dataset_from_spec: BuildDatasetFromSpecProtocol = _default_build_dataset_from_spec

measure_candidate_internal: MeasureCandidateInternalProtocol = _default_measure_candidate_internal

emit_result_file: EmitResultFileProtocol = _default_emit_result_file

runner_setup_logging: SetupLoggingProtocol = _default_runner_setup_logging

file_open: FileOpenProtocol = _default_file_open

now_ts: NowTsProtocol = _default_now_ts

path_stat: PathStatProtocol = _default_path_stat

is_wrapped_state_dict: IsWrappedStateDictProtocol = _default_is_wrapped_state_dict

is_flat_state_dict: IsFlatStateDictProtocol = _default_is_flat_state_dict

log_system_info: LogSystemInfoProtocol = _default_log_system_info

limit_thread_pools: LimitThreadPoolsProtocol = _default_limit_thread_pools

train_epoch: TrainEpochProtocol = _default_train_epoch

calibrate_input_pipeline: CalibrateInputPipelineProtocol = _default_calibrate_input_pipeline

tempfile_mkdtemp: TempfileMkdtempProtocol = _default_tempfile_mkdtemp

queue_handler_factory: QueueHandlerFactory = load_queue_handler_factory()

queue_listener_factory: QueueListenerFactory = load_queue_listener_factory()

pil_histogram: PILHistogramProtocol = _default_pil_histogram

otsu_binarize: OtsuBinarizeProtocol

get_memory_guard_config: GetMemoryGuardConfigProtocol = _default_get_memory_guard_config

get_training_progress_module: GetTrainingProgressModuleProtocol = (
    _default_get_training_progress_module
)


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


inject_bad_state_dict_list: InjectBadStateDictListProtocol = _inject_bad_state_dict_list

inject_bad_state_dict_values: InjectBadStateDictValuesProtocol = _inject_bad_state_dict_values

inject_bad_state_dict_non_string_key: InjectBadStateDictNonStringKeyProtocol = (
    _inject_bad_state_dict_non_string_key
)


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


get_autocast_context: GetAutocastContextProtocol = _default_get_autocast_context

create_grad_scaler: CreateGradScalerProtocol = _default_create_grad_scaler
