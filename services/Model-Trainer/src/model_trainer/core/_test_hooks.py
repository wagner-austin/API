"""Hooks for container factories - production defaults, tests override.

Production code initializes these to real implementations at module level.
Tests replace them with fakes before exercising the code under test.
No conditionals needed - just call the hook directly.
"""

from __future__ import annotations

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
from model_trainer.core.contracts.dataset import DatasetConfig
from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.services.registries import ModelRegistry
from model_trainer.core.types import LMModelProto

# ============================================================================
# Training infrastructure hooks
# ============================================================================


class CudaIsAvailableProto(Protocol):
    """Protocol for cuda_is_available hook."""

    def __call__(self) -> bool:
        """Check if CUDA is available."""
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


class SplitCorpusFilesProto(Protocol):
    """Protocol for split_corpus_files hook."""

    def __call__(self, cfg: DatasetConfig) -> tuple[list[str], list[str], list[str]]:
        """Split corpus files into train/val/test sets."""
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


def _default_split_corpus_files(
    cfg: DatasetConfig,
) -> tuple[list[str], list[str], list[str]]:
    """Production split_corpus_files - used as default hook."""
    from model_trainer.core.services.training.dataset_builder import (
        split_corpus_files as _split,
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
pkg_version: PkgVersionProto = _default_pkg_version
model_dir: ModelDirProto = _default_model_dir
split_corpus_files: SplitCorpusFilesProto = _default_split_corpus_files
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
    """Protocol for tokenizer enqueue hook.

    When set, this hook is called in place of normal enqueue_training logic.
    Return None to simulate orchestrator failure that triggers API 500.
    """

    def __call__(
        self,
        orchestrator: TokenizerOrchestratorProto,
        req: TokenizerTrainRequest,
    ) -> TokenizerTrainResponse | None: ...


# Tokenizer orchestrator hook - None means use default behavior
tokenizer_enqueue_hook: TokenizerEnqueueHookProto | None = None


# ============================================================================
# GPT2 backend hooks for testing
# ============================================================================


class LoadPreparedGpt2FromHandleProto(Protocol):
    """Protocol for load_prepared_gpt2_from_handle hook."""

    def __call__(self, artifact_path: str, tokenizer: TokenizerHandle) -> PreparedLMModel: ...


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
    artifact_path: str, tokenizer: TokenizerHandle
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
# Guard script hooks for testing
# ============================================================================


class FindMonorepoRootProto(Protocol):
    """Protocol for _find_monorepo_root hook."""

    def __call__(self, start: Path) -> Path: ...


class RunForProjectProto(Protocol):
    """Protocol for run_for_project hook."""

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int: ...


class LoadOrchestratorProto(Protocol):
    """Protocol for _load_orchestrator hook."""

    def __call__(self, monorepo_root: Path) -> RunForProjectProto: ...


# Guard hooks - None means use default behavior (production implementation)
guard_find_monorepo_root: FindMonorepoRootProto | None = None
guard_load_orchestrator: LoadOrchestratorProto | None = None
