"""Hooks for container factories - production defaults, tests override.

Production code initializes these to real implementations at module level.
Tests replace them with fakes before exercising the code under test.
No conditionals needed - just call the hook directly.
"""

from __future__ import annotations

from platform_core.determinism_record import DeterminismRecord
from platform_core.logging import get_logger
from platform_ml import apply_determinism
from platform_workers.redis import (
    redis_for_kv,
    redis_raw_for_rq,
)

from model_trainer.core._hook_defaults import (
    _default_artifact_store,
    _default_corpus_cache_cleanup_service_factory,
    _default_corpus_fetcher_factory,
    _default_count_model_parameters,
    _default_cuda_device_name,
    _default_cuda_driver_version,
    _default_cuda_is_available,
    _default_datetime_utcnow_iso,
    _default_dump_json_str,
    _default_env_git_commit,
    _default_freeze_embeddings,
    _default_get_directory_size_bytes,
    _default_httpx_client_factory,
    _default_load_gpt2_model,
    _default_load_prepared_gpt2_from_handle,
    _default_load_settings,
    _default_load_tokenizer_for_training,
    _default_load_wandb_module,
    _default_model_dir,
    _default_os_scandir,
    _default_os_utime,
    _default_path_iterdir,
    _default_path_unlink,
    _default_random_factory,
    _default_random_getstate,
    _default_random_setstate,
    _default_rq_queue,
    _default_rq_retry,
    _default_sample_token,
    _default_service_container_from_settings,
    _default_shutil_disk_usage,
    _default_shutil_rmtree,
    _default_shutil_which,
    _default_split_corpus,
    _default_spm_decode_ids,
    _default_spm_encode_ids,
    _default_spm_require_cli,
    _default_spm_train,
    _default_time_monotonic,
    _default_time_sleep,
    _default_time_wall_clock,
    _default_tokenizer_cleanup_service_factory,
    _default_tokenizer_enqueue,
    _default_torch_cuda_get_rng_state_all,
    _default_torch_cuda_max_memory_allocated,
    _default_torch_cuda_reset_peak_memory_stats,
    _default_torch_cuda_set_rng_state_all,
    _default_torch_device,
)
from model_trainer.core._hook_protocols import (
    ArtifactStoreFactoryProto,
    CountModelParametersProto,
    DatetimeUtcnowIsoProto,
    DumpJsonStrProto,
    EnvGitCommitProto,
    GetDirectorySizeBytesProto,
    GpuMaxMemoryAllocatedProto,
    GpuResetPeakMemoryStatsProto,
    HttpxClientFactoryProto,
    KVStoreFactoryProto,
    LoadSettingsProto,
    OsScandirProto,
    OsUtimeProto,
    PathIterdirProto,
    PathUnlinkProto,
    PkgVersionProto,
    RandomFactoryProto,
    RandomGetstateProto,
    RandomSetstateProto,
    RQConnectionFactoryProto,
    RQQueueFactoryProto,
    RQRetryFactoryProto,
    ServiceContainerFactoryProto,
    ShutilDiskUsageProto,
    ShutilRmtreeProto,
    ShutilWhichProto,
    TimeMonotonicProto,
    TimeSleepProto,
    TimeWallClockProto,
    TorchDeviceProto,
)
from model_trainer.core._hook_protocols_ml import (
    ApplyDeterminismProto,
    CorpusCacheCleanupServiceFactoryProto,
    CorpusFetcherFactoryProto,
    CudaDeviceNameProto,
    CudaDriverVersionProto,
    CudaIsAvailableProto,
    FreezeEmbeddingsProto,
    LoadGpt2ModelProto,
    LoadPreparedGpt2FromHandleProto,
    LoadTokenizerProto,
    LoadWandbModuleProto,
    ModelDirProto,
    SampleTokenProto,
    SplitCorpusProto,
    SpmDecodeIdsProto,
    SpmEncodeIdsProto,
    SpmRequireCliProto,
    SpmTrainProto,
    TokenizerCleanupServiceFactoryProto,
    TokenizerEnqueueHookProto,
    TorchCudaGetRngStateAllProto,
    TorchCudaMaxMemoryAllocatedProto,
    TorchCudaResetPeakMemoryStatsProto,
    TorchCudaSetRngStateAllProto,
)


def _default_pkg_version(name: str) -> str:
    """Production pkg_version - used as default hook."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        _log.debug("Package %s not found, returning 'unknown'", name)
        return "unknown"


def _default_apply_determinism() -> DeterminismRecord:
    """Production determinism pin - used as default hook.

    Imported inside the function so that merely importing this module does
    not pull torch into a process that only wanted a Redis handle.
    """
    import os

    import torch
    import torch.backends.cuda
    import torch.backends.cudnn

    # os.putenv rather than os.environ: it reaches the real process
    # environment that cuBLAS's getenv reads, and it is a write rather than
    # the config read the monorepo's env guard exists to stop.
    return apply_determinism(
        torch.backends.cudnn,
        torch.backends.cuda.matmul,
        torch.use_deterministic_algorithms,
        os.putenv,
    )


def _default_gpu_max_memory_allocated() -> int:
    """Production torch.cuda.max_memory_allocated - used as default hook."""
    if not cuda_is_available():
        return 0
    return torch_cuda_max_memory_allocated()


def _default_gpu_reset_peak_memory_stats() -> None:
    """Production torch.cuda.reset_peak_memory_stats - used as default hook."""
    if cuda_is_available():
        torch_cuda_reset_peak_memory_stats()


_log = get_logger(__name__)

kv_store_factory: KVStoreFactoryProto = redis_for_kv

rq_connection_factory: RQConnectionFactoryProto = redis_raw_for_rq

rq_queue_factory: RQQueueFactoryProto = _default_rq_queue

rq_retry_factory: RQRetryFactoryProto = _default_rq_retry

load_settings: LoadSettingsProto = _default_load_settings

artifact_store_factory: ArtifactStoreFactoryProto = _default_artifact_store

service_container_from_settings: ServiceContainerFactoryProto = (
    _default_service_container_from_settings
)

corpus_fetcher_factory: CorpusFetcherFactoryProto = _default_corpus_fetcher_factory

load_tokenizer_for_training: LoadTokenizerProto = _default_load_tokenizer_for_training

httpx_client_factory: HttpxClientFactoryProto = _default_httpx_client_factory

apply_determinism_hook: ApplyDeterminismProto = _default_apply_determinism

cuda_is_available: CudaIsAvailableProto = _default_cuda_is_available

cuda_device_name: CudaDeviceNameProto = _default_cuda_device_name

cuda_driver_version: CudaDriverVersionProto = _default_cuda_driver_version

env_git_commit: EnvGitCommitProto = _default_env_git_commit

pkg_version: PkgVersionProto = _default_pkg_version

model_dir: ModelDirProto = _default_model_dir

split_corpus: SplitCorpusProto = _default_split_corpus

freeze_embeddings: FreezeEmbeddingsProto = _default_freeze_embeddings

shutil_which: ShutilWhichProto = _default_shutil_which

spm_require_cli: SpmRequireCliProto = _default_spm_require_cli

spm_train: SpmTrainProto = _default_spm_train

spm_encode_ids: SpmEncodeIdsProto = _default_spm_encode_ids

random_factory: RandomFactoryProto = _default_random_factory

shutil_rmtree: ShutilRmtreeProto = _default_shutil_rmtree

os_utime: OsUtimeProto = _default_os_utime

load_wandb_module: LoadWandbModuleProto = _default_load_wandb_module

load_gpt2_model: LoadGpt2ModelProto = _default_load_gpt2_model

sample_token: SampleTokenProto = _default_sample_token

spm_decode_ids: SpmDecodeIdsProto = _default_spm_decode_ids

os_scandir: OsScandirProto = _default_os_scandir

shutil_disk_usage: ShutilDiskUsageProto = _default_shutil_disk_usage

path_unlink: PathUnlinkProto = _default_path_unlink

time_sleep: TimeSleepProto = _default_time_sleep

path_iterdir: PathIterdirProto = _default_path_iterdir

corpus_cache_cleanup_service_factory: CorpusCacheCleanupServiceFactoryProto = (
    _default_corpus_cache_cleanup_service_factory
)

tokenizer_cleanup_service_factory: TokenizerCleanupServiceFactoryProto = (
    _default_tokenizer_cleanup_service_factory
)

dump_json_str: DumpJsonStrProto = _default_dump_json_str

tokenizer_enqueue: TokenizerEnqueueHookProto = _default_tokenizer_enqueue

load_prepared_gpt2_from_handle: LoadPreparedGpt2FromHandleProto = (
    _default_load_prepared_gpt2_from_handle
)

torch_cuda_max_memory_allocated: TorchCudaMaxMemoryAllocatedProto = (
    _default_torch_cuda_max_memory_allocated
)

torch_cuda_reset_peak_memory_stats: TorchCudaResetPeakMemoryStatsProto = (
    _default_torch_cuda_reset_peak_memory_stats
)

time_monotonic: TimeMonotonicProto = _default_time_monotonic

time_wall_clock: TimeWallClockProto = _default_time_wall_clock

datetime_utcnow_iso: DatetimeUtcnowIsoProto = _default_datetime_utcnow_iso

gpu_max_memory_allocated: GpuMaxMemoryAllocatedProto = _default_gpu_max_memory_allocated

gpu_reset_peak_memory_stats: GpuResetPeakMemoryStatsProto = _default_gpu_reset_peak_memory_stats

count_model_parameters: CountModelParametersProto = _default_count_model_parameters

get_directory_size_bytes: GetDirectorySizeBytesProto = _default_get_directory_size_bytes

torch_device: TorchDeviceProto = _default_torch_device

random_getstate: RandomGetstateProto = _default_random_getstate

random_setstate: RandomSetstateProto = _default_random_setstate

torch_cuda_get_rng_state_all: TorchCudaGetRngStateAllProto = _default_torch_cuda_get_rng_state_all

torch_cuda_set_rng_state_all: TorchCudaSetRngStateAllProto = _default_torch_cuda_set_rng_state_all
