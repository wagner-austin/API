"""Hooks for container factories - production defaults, tests override.

Production code initializes these to real implementations at module level.
Tests replace them with fakes before exercising the code under test.
No conditionals needed - just call the hook directly.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC
from pathlib import Path
from types import TracebackType
from typing import Protocol

import httpx
import torch
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.json_utils import _JSONInputValue as JSONInputValue
from platform_core.logging import get_logger
from platform_ml.testing import (
    WandbModuleProtocol as WandbModuleLike,
)
from platform_workers.redis import (
    RedisStrProto,
    _RedisBytesClient,
    redis_for_kv,
    redis_raw_for_rq,
)
from platform_workers.rq_harness import RQClientQueue, RQRetryLike, rq_queue, rq_retry

# Import tokenizer schema types for protocol definitions
# (import at top to avoid circular imports)
from model_trainer.api.schemas.tokenizers import (
    TokenizerTrainRequest,
    TokenizerTrainResponse,
)
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.dataset import CorpusSplit, DatasetConfig
from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.services.registries import ModelRegistry
from model_trainer.core.types import LMModelProto, TorchStateValue

# ============================================================================
# Training infrastructure hooks
# ============================================================================


class CudaIsAvailableProto(Protocol):
    """Protocol for cuda_is_available hook."""

    def __call__(self) -> bool:
        """Check if CUDA is available."""
        ...


class CudaDeviceNameProto(Protocol):
    """Protocol for cuda_device_name hook."""

    def __call__(self) -> str:
        """Get the CUDA device 0 model name. Callers gate on cuda_is_available."""
        ...


class EnvGitCommitProto(Protocol):
    """Protocol for env_git_commit hook."""

    def __call__(self) -> str | None:
        """Read the build-stamped GIT_COMMIT variable, None when unset or empty."""
        ...


class PkgVersionProto(Protocol):
    """Protocol for pkg_version hook."""

    def __call__(self, name: str) -> str:
        """Get package version by name."""
        ...


class ModelDirProto(Protocol):
    """Protocol for model_dir hook."""

    def __call__(self, settings: Settings, run_id: str) -> Path:
        """Get model directory path."""
        ...


class SplitCorpusProto(Protocol):
    """Protocol for the split_corpus hook."""

    def __call__(self, cfg: DatasetConfig) -> CorpusSplit:
        """Partition a corpus into disjoint train/validation/test lines."""
        ...


class FreezeEmbeddingsProto(Protocol):
    """Protocol for freeze_embeddings hook."""

    def __call__(self, model: LMModelProto) -> None:
        """Freeze embedding parameters in model."""
        ...


class ShutilWhichProto(Protocol):
    """Protocol for shutil_which hook."""

    def __call__(self, cmd: str) -> str | None:
        """Find command on PATH, return path or None."""
        ...


# ============================================================================
# SentencePiece backend hooks
# ============================================================================


class SpmRequireCliProto(Protocol):
    """Protocol for spm_require_cli hook."""

    def __call__(self) -> None:
        """Check that SentencePiece CLI is available."""
        ...


class SpmTrainProto(Protocol):
    """Protocol for spm_train hook."""

    def __call__(self, files: list[str], *, model_prefix: str, vocab_size: int) -> None:
        """Train a SentencePiece model."""
        ...


class SpmEncodeIdsProto(Protocol):
    """Protocol for spm_encode_ids hook."""

    def __call__(self, model_path: str, text: str) -> list[int]:
        """Encode text to token IDs using SentencePiece model."""
        ...


class CorpusFetcherProto(Protocol):
    """Protocol for CorpusFetcher."""

    def fetch(self, file_id: str) -> Path:
        """Fetch a corpus file from the data bank API."""
        ...


class CorpusFetcherFactoryProto(Protocol):
    """Protocol for CorpusFetcher factory."""

    def __call__(self, api_url: str, api_key: str, cache_dir: Path) -> CorpusFetcherProto:
        """Create CorpusFetcher instance."""
        ...


class LoadTokenizerProto(Protocol):
    """Protocol for load_tokenizer_for_training."""

    def __call__(self, settings: Settings, tokenizer_id: str) -> TokenizerHandle:
        """Load tokenizer from artifacts directory."""
        ...


class HttpxClientFactoryProto(Protocol):
    """Protocol for httpx.Client factory.

    Tests inject fake transports by returning httpx.Client(transport=MockTransport(...)).
    Production returns httpx.Client(timeout=timeout_seconds).
    """

    def __call__(self, *, timeout_seconds: float = 30.0) -> httpx.Client:
        """Create httpx.Client instance."""
        ...


class KVStoreFactoryProto(Protocol):
    """Protocol for redis_for_kv factory."""

    def __call__(self, url: str) -> RedisStrProto:
        """Create Redis client from URL."""
        ...


class RQConnectionFactoryProto(Protocol):
    """Protocol for redis_raw_for_rq factory."""

    def __call__(self, url: str) -> _RedisBytesClient:
        """Create Redis RQ client from URL."""
        ...


class RQQueueFactoryProto(Protocol):
    """Protocol for rq_queue factory."""

    def __call__(self, name: str, connection: _RedisBytesClient) -> RQClientQueue:
        """Create RQ queue from name and connection."""
        ...


class RQRetryFactoryProto(Protocol):
    """Protocol for rq_retry factory."""

    def __call__(self, *, max_retries: int, intervals: list[int]) -> RQRetryLike:
        """Create RQ retry from max_retries and intervals."""
        ...


class LoadSettingsProto(Protocol):
    """Protocol for load_settings factory."""

    def __call__(self) -> Settings:
        """Load settings."""
        ...


class ArtifactStoreProto(Protocol):
    """Protocol for ArtifactStore."""

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


class ArtifactStoreFactoryProto(Protocol):
    """Protocol for ArtifactStore factory."""

    def __call__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 600.0,
    ) -> ArtifactStoreProto:
        """Create ArtifactStore instance."""
        ...


class ServiceContainerProto(Protocol):
    """Protocol for ServiceContainer."""

    @property
    def settings(self) -> Settings:
        """Get settings."""
        ...

    @property
    def redis(self) -> RedisStrProto:
        """Get Redis client."""
        ...

    @property
    def model_registry(self) -> ModelRegistry:
        """Get model registry."""
        ...


class ServiceContainerFactoryProto(Protocol):
    """Protocol for ServiceContainer.from_settings factory."""

    def __call__(self, settings: Settings) -> ServiceContainerProto:
        """Create ServiceContainer from settings."""
        ...


def _default_rq_queue(name: str, connection: _RedisBytesClient) -> RQClientQueue:
    """Production rq_queue - used as default hook."""
    return rq_queue(name, connection)


def _default_rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
    """Production rq_retry - used as default hook."""
    return rq_retry(max_retries=max_retries, intervals=intervals)


def _default_load_settings() -> Settings:
    """Production load_settings - used as default hook."""
    from model_trainer.core.config.settings import load_settings as _load

    return _load()


def _default_artifact_store(
    base_url: str,
    api_key: str,
    *,
    timeout_seconds: float = 600.0,
) -> ArtifactStoreProto:
    """Production ArtifactStore - used as default hook."""
    from platform_core.data_bank_client import DataBankClient
    from platform_ml import ArtifactStore

    client = DataBankClient(base_url, api_key, timeout_seconds=timeout_seconds)
    return ArtifactStore(client)


def _default_service_container_from_settings(settings: Settings) -> ServiceContainerProto:
    """Production ServiceContainer.from_settings - used as default hook."""
    from model_trainer.core.services.container import ServiceContainer

    return ServiceContainer.from_settings(settings)


def _default_corpus_fetcher_factory(
    api_url: str, api_key: str, cache_dir: Path
) -> CorpusFetcherProto:
    """Production CorpusFetcher factory - used as default hook."""
    from model_trainer.core.services.data.corpus_fetcher import CorpusFetcher

    return CorpusFetcher(api_url, api_key, cache_dir)


def _default_load_tokenizer_for_training(settings: Settings, tokenizer_id: str) -> TokenizerHandle:
    """Production load_tokenizer_for_training - used as default hook."""
    from model_trainer.worker.job_utils import (
        load_tokenizer_for_training as _load_tok,
    )

    return _load_tok(settings, tokenizer_id)


def _default_httpx_client_factory(*, timeout_seconds: float = 30.0) -> httpx.Client:
    """Production httpx.Client factory - used as default hook."""
    return httpx.Client(timeout=timeout_seconds)


# ============================================================================
# Training infrastructure default implementations
# ============================================================================


def _default_cuda_is_available() -> bool:
    """Production cuda_is_available - used as default hook."""
    import torch

    return torch.cuda.is_available()


def _default_cuda_device_name() -> str:
    """Production cuda_device_name - used as default hook.

    Callers gate on the run's device being "cuda" (which _setup_device has
    already proven available); repeating the check here would hide a caller
    that forgot the gate. Calling this initialises a CUDA context in the
    process, which is exactly why cpu-device runs must not reach it.
    """
    import torch

    return torch.cuda.get_device_name(0)


def _default_env_git_commit() -> str | None:
    """Production env_git_commit - used as default hook.

    The deployed image carries no .git directory, so `git rev-parse` inside
    the container can never answer; the build bakes the commit into the
    GIT_COMMIT environment variable instead. An empty or unset variable is
    None: the manifest records that the commit was not stamped rather than
    an empty string that reads as a value. Reads through the platform config
    env hook, the one sanctioned environment accessor.
    """
    from platform_core.config import config_test_hooks

    value = config_test_hooks.get_env("GIT_COMMIT")
    return value if value else None


_log = get_logger(__name__)


def _default_pkg_version(name: str) -> str:
    """Production pkg_version - used as default hook."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        _log.debug("Package %s not found, returning 'unknown'", name)
        return "unknown"


def _default_model_dir(settings: Settings, run_id: str) -> Path:
    """Production model_dir - used as default hook."""
    from model_trainer.core.infra.paths import model_dir as _model_dir

    return _model_dir(settings, run_id)


def _default_split_corpus(cfg: DatasetConfig) -> CorpusSplit:
    """Production split_corpus - used as default hook.

    Args:
        cfg: Dataset configuration with corpus path and split ratios.

    Returns:
        The three disjoint partitions, as corpus lines.
    """
    from model_trainer.core.services.training.dataset_builder import (
        split_corpus as _split,
    )

    return _split(cfg)


def _default_freeze_embeddings(model: LMModelProto) -> None:
    """Production freeze_embeddings - used as default hook."""
    from model_trainer.core.services.training.base_trainer import (
        _freeze_embeddings as _freeze,
    )

    _freeze(model)


def _default_shutil_which(cmd: str) -> str | None:
    """Production shutil_which - used as default hook."""
    import shutil

    return shutil.which(cmd)


def _default_spm_require_cli() -> None:
    """Production spm_require_cli - used as default hook."""
    from platform_ml import sentencepiece as spm

    spm.require_module()


def _default_spm_train(files: list[str], *, model_prefix: str, vocab_size: int) -> None:
    """Production spm_train - used as default hook."""
    from model_trainer.core.services.tokenizer.spm_backend import (
        _spm_train as _real_spm_train,
    )

    _real_spm_train(files, model_prefix=model_prefix, vocab_size=vocab_size)


def _default_spm_encode_ids(model_path: str, text: str) -> list[int]:
    """Production spm_encode_ids - used as default hook."""
    from model_trainer.core.services.tokenizer.spm_backend import (
        _spm_encode_ids as _real_spm_encode_ids,
    )

    return _real_spm_encode_ids(model_path, text)


# Factory hooks - initialized to production implementations.
# Tests replace these with fakes before calling container code.
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

# Training infrastructure hooks
cuda_is_available: CudaIsAvailableProto = _default_cuda_is_available
cuda_device_name: CudaDeviceNameProto = _default_cuda_device_name
env_git_commit: EnvGitCommitProto = _default_env_git_commit
pkg_version: PkgVersionProto = _default_pkg_version
model_dir: ModelDirProto = _default_model_dir
split_corpus: SplitCorpusProto = _default_split_corpus
freeze_embeddings: FreezeEmbeddingsProto = _default_freeze_embeddings
shutil_which: ShutilWhichProto = _default_shutil_which

# SentencePiece backend hooks
spm_require_cli: SpmRequireCliProto = _default_spm_require_cli
spm_train: SpmTrainProto = _default_spm_train
spm_encode_ids: SpmEncodeIdsProto = _default_spm_encode_ids


# ============================================================================
# Additional hooks for testing edge cases
# ============================================================================


class RandomFactoryProto(Protocol):
    """Protocol for random.Random factory."""

    def __call__(self, seed: int) -> RandomLikeProto: ...


class RandomLikeProto(Protocol):
    """Protocol for random.Random-like objects."""

    def randint(self, a: int, b: int) -> int: ...


class ShutilRmtreeProto(Protocol):
    """Protocol for shutil.rmtree hook."""

    def __call__(self, path: Path | str) -> None: ...


class LoadWandbModuleProto(Protocol):
    """Protocol for wandb module loader."""

    def __call__(self) -> WandbModuleLike: ...


# WandbModuleLike, WandbRunLike, WandbConfigLike imported from platform_ml.testing


class LoadGpt2ModelProto(Protocol):
    """Protocol for load_gpt2_model hook."""

    def __call__(self, path: str) -> LMModelProto: ...


class Gpt2ModelLike(Protocol):
    """Protocol for GPT2-like models (for tests that need config.n_positions)."""

    @property
    def config(self) -> Gpt2ConfigLike: ...


class Gpt2ConfigLike(Protocol):
    """Protocol for GPT2 config-like objects (for tests that need n_positions)."""

    @property
    def n_positions(self) -> int: ...


class SampleTokenProto(Protocol):
    """Protocol for _sample_token hook in generation."""

    def __call__(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int: ...


class SpmDecodeIdsProto(Protocol):
    """Protocol for spm_decode_ids hook."""

    def __call__(self, model_path: str, ids: list[int]) -> str: ...


def _default_random_factory(seed: int) -> RandomLikeProto:
    """Production random.Random factory."""
    import random

    return random.Random(seed)


def _default_shutil_rmtree(path: Path | str) -> None:
    """Production shutil.rmtree."""
    import shutil

    shutil.rmtree(path)


def _default_load_wandb_module() -> WandbModuleLike:
    """Production wandb module loader."""
    from platform_ml.wandb_publisher import _load_wandb_module as _load

    return _load()


def _default_load_gpt2_model(path: str) -> LMModelProto:
    """Production load_gpt2_model."""
    from model_trainer.core.services.model.backends.gpt2.hf_gpt2 import (
        load_gpt2_model as _load,
    )

    return _load(path)


def _default_sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    """Production _sample_token."""
    from model_trainer.core.services.model.backends.char_lstm.generate import (
        _sample_token as _sample,
    )

    return _sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)


def _default_spm_decode_ids(model_path: str, ids: list[int]) -> str:
    """Production spm_decode_ids."""
    from model_trainer.core.services.tokenizer.spm_backend import (
        _spm_decode_ids as _decode,
    )

    return _decode(model_path, ids)


# Additional hooks
random_factory: RandomFactoryProto = _default_random_factory
shutil_rmtree: ShutilRmtreeProto = _default_shutil_rmtree
load_wandb_module: LoadWandbModuleProto = _default_load_wandb_module
load_gpt2_model: LoadGpt2ModelProto = _default_load_gpt2_model
sample_token: SampleTokenProto = _default_sample_token
spm_decode_ids: SpmDecodeIdsProto = _default_spm_decode_ids


# ============================================================================
# Standard library hooks for testing error cases
# ============================================================================


class OsScandirProto(Protocol):
    """Protocol for os.scandir hook."""

    def __call__(self, path: str) -> ScandirIterator: ...


class ScandirIterator(Protocol):
    """Protocol for os.scandir context manager."""

    def __enter__(self) -> ScandirIteratorContext: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None: ...


class ScandirIteratorContext(Protocol):
    """Protocol for iterating os.scandir entries."""

    def __iter__(self) -> ScandirIteratorContext: ...
    def __next__(self) -> DirEntryProto: ...


class DirEntryProto(Protocol):
    """Protocol for os.DirEntry-like objects."""

    @property
    def path(self) -> str: ...
    def is_file(self) -> bool: ...
    def stat(self) -> StatResultProto: ...


class StatResultProto(Protocol):
    """Protocol for stat result."""

    @property
    def st_size(self) -> int: ...
    @property
    def st_atime(self) -> float: ...
    @property
    def st_mtime(self) -> float: ...


class DiskUsageProto(Protocol):
    """Protocol for shutil.disk_usage result."""

    @property
    def total(self) -> int: ...
    @property
    def used(self) -> int: ...
    @property
    def free(self) -> int: ...


class ShutilDiskUsageProto(Protocol):
    """Protocol for shutil.disk_usage hook."""

    def __call__(self, path: str) -> DiskUsageProto: ...


class PathUnlinkProto(Protocol):
    """Protocol for Path.unlink hook."""

    def __call__(self, path: Path) -> None: ...


class TimeSleepProto(Protocol):
    """Protocol for time.sleep hook."""

    def __call__(self, seconds: float) -> None: ...


class PathIterdirProto(Protocol):
    """Protocol for Path.iterdir hook."""

    def __call__(self, path: Path) -> PathIterator: ...


class PathIterator(Protocol):
    """Protocol for Path iterator."""

    def __iter__(self) -> PathIterator: ...
    def __next__(self) -> Path: ...


def _default_os_scandir(path: str) -> ScandirIterator:
    """Production os.scandir - used as default hook."""
    import os

    return os.scandir(path)


def _default_shutil_disk_usage(path: str) -> DiskUsageProto:
    """Production shutil.disk_usage - used as default hook."""
    import shutil

    return shutil.disk_usage(path)


def _default_path_unlink(path: Path) -> None:
    """Production Path.unlink - used as default hook."""
    path.unlink()


def _default_time_sleep(seconds: float) -> None:
    """Production time.sleep - used as default hook."""
    import time

    time.sleep(seconds)


def _default_path_iterdir(path: Path) -> PathIterator:
    """Production Path.iterdir - used as default hook."""
    return path.iterdir()


# Standard library hooks
os_scandir: OsScandirProto = _default_os_scandir
shutil_disk_usage: ShutilDiskUsageProto = _default_shutil_disk_usage
path_unlink: PathUnlinkProto = _default_path_unlink
time_sleep: TimeSleepProto = _default_time_sleep
path_iterdir: PathIterdirProto = _default_path_iterdir


# ============================================================================
# Cleanup service hooks for maintenance
# ============================================================================


class CorpusCacheCleanupResultProto(Protocol):
    """Protocol for CorpusCacheCleanupResult-like objects."""

    @property
    def deleted_files(self) -> int: ...
    @property
    def bytes_freed(self) -> int: ...


class CorpusCacheCleanupServiceProto(Protocol):
    """Protocol for CorpusCacheCleanupService-like objects."""

    def clean(self) -> CorpusCacheCleanupResultProto: ...


class CorpusCacheCleanupServiceFactoryProto(Protocol):
    """Protocol for CorpusCacheCleanupService factory."""

    def __call__(self, *, settings: Settings) -> CorpusCacheCleanupServiceProto: ...


class TokenizerCleanupResultProto(Protocol):
    """Protocol for TokenizerCleanupResult-like objects."""

    @property
    def deleted_tokenizers(self) -> int: ...
    @property
    def bytes_freed(self) -> int: ...


class TokenizerCleanupServiceProto(Protocol):
    """Protocol for TokenizerCleanupService-like objects."""

    def clean(self) -> TokenizerCleanupResultProto: ...


class TokenizerCleanupServiceFactoryProto(Protocol):
    """Protocol for TokenizerCleanupService factory."""

    def __call__(self, *, settings: Settings) -> TokenizerCleanupServiceProto: ...


def _default_corpus_cache_cleanup_service_factory(
    *, settings: Settings
) -> CorpusCacheCleanupServiceProto:
    """Production CorpusCacheCleanupService factory."""
    from model_trainer.core.services.data.corpus_cache_cleanup import (
        CorpusCacheCleanupService,
    )

    return CorpusCacheCleanupService(settings=settings)


def _default_tokenizer_cleanup_service_factory(
    *, settings: Settings
) -> TokenizerCleanupServiceProto:
    """Production TokenizerCleanupService factory."""
    from model_trainer.core.services.tokenizer.tokenizer_cleanup import (
        TokenizerCleanupService,
    )

    return TokenizerCleanupService(settings=settings)


# Cleanup service hooks
corpus_cache_cleanup_service_factory: CorpusCacheCleanupServiceFactoryProto = (
    _default_corpus_cache_cleanup_service_factory
)
tokenizer_cleanup_service_factory: TokenizerCleanupServiceFactoryProto = (
    _default_tokenizer_cleanup_service_factory
)


# ============================================================================
# JSON serialization hooks for testing error paths
# ============================================================================


class DumpJsonStrProto(Protocol):
    """Protocol for dump_json_str hook."""

    def __call__(self, value: JSONInputValue, *, compact: bool = True) -> str: ...


def _default_dump_json_str(value: JSONInputValue, *, compact: bool = True) -> str:
    """Production dump_json_str - used as default hook."""
    from platform_core.json_utils import dump_json_str as _dump

    return _dump(value, compact=compact)


# JSON hooks
dump_json_str: DumpJsonStrProto = _default_dump_json_str


# ============================================================================
# Orchestrator hooks for testing edge cases
# ============================================================================


class TokenizerOrchestratorProto(Protocol):
    """Protocol for TokenizerOrchestrator-like objects."""

    def enqueue_training(self, req: TokenizerTrainRequest) -> TokenizerTrainResponse | None: ...


class TokenizerEnqueueHookProto(Protocol):
    """Protocol for the tokenizer enqueue seam.

    None is a real answer here, not an unset hook: the orchestrator returns
    None when the enqueue fails, and the route turns that into a 500. A test
    rebinds this to return None on demand to reach that path.
    """

    def __call__(
        self,
        orchestrator: TokenizerOrchestratorProto,
        req: TokenizerTrainRequest,
    ) -> TokenizerTrainResponse | None: ...


def _default_tokenizer_enqueue(
    orchestrator: TokenizerOrchestratorProto,
    req: TokenizerTrainRequest,
) -> TokenizerTrainResponse | None:
    """Enqueue tokenizer training through the orchestrator.

    Args:
        orchestrator: Orchestrator to enqueue through.
        req: Validated training request.

    Returns:
        The enqueue response, or None if the enqueue failed.
    """
    return orchestrator.enqueue_training(req)


# Hook for the tokenizer enqueue. Tests rebind it to reach the failure path.
tokenizer_enqueue: TokenizerEnqueueHookProto = _default_tokenizer_enqueue


# ============================================================================
# GPT2 backend hooks for testing
# ============================================================================


class LoadPreparedGpt2FromHandleProto(Protocol):
    """Protocol for load_prepared_gpt2_from_handle hook."""

    def __call__(
        self, artifact_path: str, tokenizer: TokenizerHandle | None
    ) -> PreparedLMModel: ...


class PreparedLMModelProto(Protocol):
    """Protocol for PreparedLMModel-like objects returned by load hooks."""

    @property
    def model(self) -> LMModelProto: ...
    @property
    def tokenizer_id(self) -> str: ...
    @property
    def eos_id(self) -> int: ...
    @property
    def pad_id(self) -> int: ...
    @property
    def max_seq_len(self) -> int: ...


def _default_load_prepared_gpt2_from_handle(
    artifact_path: str, tokenizer: TokenizerHandle | None
) -> PreparedLMModel:
    """Production load_prepared_gpt2_from_handle - used as default hook."""
    from model_trainer.core.services.model.backends.gpt2.io import (
        load_prepared_gpt2_from_handle as _load,
    )

    return _load(artifact_path, tokenizer)


# GPT2 backend hooks
load_prepared_gpt2_from_handle: LoadPreparedGpt2FromHandleProto = (
    _default_load_prepared_gpt2_from_handle
)


# ============================================================================
# Training metrics hooks for timing, memory, and model info
# ============================================================================


class TimeMonotonicProto(Protocol):
    """Protocol for time.monotonic hook."""

    def __call__(self) -> float:
        """Return monotonic time in seconds.

        Returns:
            Current monotonic clock value.
        """
        ...


class TimeWallClockProto(Protocol):
    """Protocol for the time.time hook.

    Distinct from :class:`TimeMonotonicProto` because heartbeats are stamped
    with wall-clock time and compared across processes: the worker writes the
    stamp and the API reads it, and two processes share no monotonic epoch.
    """

    def __call__(self) -> float:
        """Return seconds since the Unix epoch.

        Returns:
            Current wall-clock value.
        """
        ...


class DatetimeUtcnowIsoProto(Protocol):
    """Protocol for getting current UTC time as ISO 8601 string."""

    def __call__(self) -> str:
        """Return current UTC time as ISO 8601 string.

        Returns:
            ISO 8601 formatted timestamp (e.g., '2024-01-15T10:30:00').
        """
        ...


class GpuMaxMemoryAllocatedProto(Protocol):
    """Protocol for torch.cuda.max_memory_allocated hook."""

    def __call__(self) -> int:
        """Return peak GPU memory allocated in bytes.

        Returns:
            Peak memory in bytes, or 0 if CUDA not available.
        """
        ...


class GpuResetPeakMemoryStatsProto(Protocol):
    """Protocol for torch.cuda.reset_peak_memory_stats hook."""

    def __call__(self) -> None:
        """Reset peak memory tracking stats."""
        ...


class CountModelParametersProto(Protocol):
    """Protocol for counting model parameters."""

    def __call__(self, model: LMModelProto) -> int:
        """Count total trainable parameters in model.

        Args:
            model: The language model.

        Returns:
            Total number of trainable parameters.
        """
        ...


class GetDirectorySizeBytesProto(Protocol):
    """Protocol for calculating directory size on disk."""

    def __call__(self, path: Path) -> int:
        """Calculate total size of directory contents in bytes.

        Args:
            path: Directory path.

        Returns:
            Total size in bytes.
        """
        ...


class TorchCudaMaxMemoryAllocatedProto(Protocol):
    """Protocol for torch.cuda.max_memory_allocated hook."""

    def __call__(self) -> int:
        """Return peak GPU memory allocated in bytes.

        Returns:
            Peak memory in bytes.
        """
        ...


class TorchCudaResetPeakMemoryStatsProto(Protocol):
    """Protocol for torch.cuda.reset_peak_memory_stats hook."""

    def __call__(self) -> None:
        """Reset peak memory tracking stats."""
        ...


def _default_torch_cuda_max_memory_allocated() -> int:
    """Production torch.cuda.max_memory_allocated - used as default hook.

    A thin adapter over torch. Whether CUDA is present is the caller's
    question, and `_default_gpu_max_memory_allocated` already asks it through
    the `cuda_is_available` hook before delegating here; repeating the check
    added a branch that no caller can reach and that no machine with a GPU can
    execute.

    Returns:
        Peak GPU memory allocated in bytes.
    """
    return torch.cuda.max_memory_allocated()


def _default_torch_cuda_reset_peak_memory_stats() -> None:
    """Production torch.cuda.reset_peak_memory_stats - used as default hook.

    A thin adapter over torch; `_default_gpu_reset_peak_memory_stats` owns the
    availability check.
    """
    torch.cuda.reset_peak_memory_stats()


# Lower-level torch.cuda hooks (used by gpu_max_memory_allocated/gpu_reset_peak_memory_stats)
torch_cuda_max_memory_allocated: TorchCudaMaxMemoryAllocatedProto = (
    _default_torch_cuda_max_memory_allocated
)
torch_cuda_reset_peak_memory_stats: TorchCudaResetPeakMemoryStatsProto = (
    _default_torch_cuda_reset_peak_memory_stats
)


def _default_time_monotonic() -> float:
    """Production time.monotonic - used as default hook."""
    import time

    return time.monotonic()


def _default_time_wall_clock() -> float:
    """Production time.time - used as default hook.

    Returns:
        Seconds since the Unix epoch.
    """
    import time

    return time.time()


def _default_datetime_utcnow_iso() -> str:
    """Production datetime.utcnow ISO format - used as default hook."""
    from datetime import datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")


def _default_gpu_max_memory_allocated() -> int:
    """Production torch.cuda.max_memory_allocated - used as default hook."""
    if not cuda_is_available():
        return 0
    return torch_cuda_max_memory_allocated()


def _default_gpu_reset_peak_memory_stats() -> None:
    """Production torch.cuda.reset_peak_memory_stats - used as default hook."""
    if cuda_is_available():
        torch_cuda_reset_peak_memory_stats()


def _default_count_model_parameters(model: LMModelProto) -> int:
    """Production count_model_parameters - used as default hook."""
    total = 0
    for param in model.parameters():
        total += param.numel()
    return total


def _default_get_directory_size_bytes(path: Path) -> int:
    """Production get_directory_size_bytes - used as default hook."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


class TorchCudaGetRngStateAllProto(Protocol):
    """Protocol for the ``torch.cuda.get_rng_state_all`` hook."""

    def __call__(self) -> list[torch.Tensor]:
        """Return every CUDA device's generator state."""
        ...


class TorchCudaSetRngStateAllProto(Protocol):
    """Protocol for the ``torch.cuda.set_rng_state_all`` hook."""

    def __call__(self, states: list[torch.Tensor]) -> None:
        """Restore every CUDA device's generator state.

        Args:
            states: States previously returned by
                ``torch.cuda.get_rng_state_all``.
        """
        ...


def _default_torch_cuda_get_rng_state_all() -> list[torch.Tensor]:
    """Production torch.cuda.get_rng_state_all - used as default hook.

    A thin adapter over torch; the checkpoint capture owns the
    availability check through the ``cuda_is_available`` hook.
    """
    return list(torch.cuda.get_rng_state_all())


def _default_torch_cuda_set_rng_state_all(states: list[torch.Tensor]) -> None:
    """Production torch.cuda.set_rng_state_all - used as default hook.

    Args:
        states: States previously returned by
            ``torch.cuda.get_rng_state_all``.
    """
    torch.cuda.set_rng_state_all(states)


class RandomGetstateProto(Protocol):
    """Protocol for the ``random.getstate`` hook."""

    def __call__(self) -> tuple[TorchStateValue, ...]:
        """Return the current python RNG state tuple."""
        ...


class RandomSetstateProto(Protocol):
    """Protocol for the ``random.setstate`` hook."""

    def __call__(self, state: tuple[TorchStateValue, ...]) -> None:
        """Restore a python RNG state tuple.

        Args:
            state: State tuple previously returned by ``random.getstate``.
        """
        ...


def _default_random_getstate() -> tuple[TorchStateValue, ...]:
    """Production random.getstate - used as default hook."""
    random_mod = __import__("random")
    fn: Callable[[], tuple[TorchStateValue, ...]] = random_mod.getstate
    return fn()


def _default_random_setstate(state: tuple[TorchStateValue, ...]) -> None:
    """Production random.setstate - used as default hook.

    Args:
        state: State tuple previously returned by ``random.getstate``.
    """
    random_mod = __import__("random")
    fn: Callable[[tuple[TorchStateValue, ...]], None] = random_mod.setstate
    fn(state)


class TorchDeviceProto(Protocol):
    """Protocol for torch.device creation hook."""

    def __call__(self, device_str: str) -> torch.device:
        """Create a torch.device from string.

        Args:
            device_str: Device string ('cpu' or 'cuda').

        Returns:
            The torch device.
        """
        ...


def _default_torch_device(device_str: str) -> torch.device:
    """Production torch.device - used as default hook."""
    return torch.device(device_str)


# Training metrics hooks
time_monotonic: TimeMonotonicProto = _default_time_monotonic
time_wall_clock: TimeWallClockProto = _default_time_wall_clock
datetime_utcnow_iso: DatetimeUtcnowIsoProto = _default_datetime_utcnow_iso
gpu_max_memory_allocated: GpuMaxMemoryAllocatedProto = _default_gpu_max_memory_allocated
gpu_reset_peak_memory_stats: GpuResetPeakMemoryStatsProto = _default_gpu_reset_peak_memory_stats
count_model_parameters: CountModelParametersProto = _default_count_model_parameters
get_directory_size_bytes: GetDirectorySizeBytesProto = _default_get_directory_size_bytes
torch_device: TorchDeviceProto = _default_torch_device

# RNG hooks (used by training checkpoint capture/restore)
random_getstate: RandomGetstateProto = _default_random_getstate
random_setstate: RandomSetstateProto = _default_random_setstate
torch_cuda_get_rng_state_all: TorchCudaGetRngStateAllProto = _default_torch_cuda_get_rng_state_all
torch_cuda_set_rng_state_all: TorchCudaSetRngStateAllProto = _default_torch_cuda_set_rng_state_all
