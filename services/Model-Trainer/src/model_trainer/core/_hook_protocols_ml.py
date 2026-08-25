"""ML-side hook protocols (tokenizers, models, corpus, CUDA).

Runtime protocols: :mod:`model_trainer.core._hook_protocols`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch
from platform_ml import DeterminismRecord
from platform_ml.testing import (
    WandbModuleProtocol as WandbModuleLike,
)

from model_trainer.api.schemas.tokenizers import (
    TokenizerTrainRequest,
    TokenizerTrainResponse,
)
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.dataset import CorpusSplit, DatasetConfig
from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.types import LMModelProto


class ApplyDeterminismProto(Protocol):
    """Protocol for the apply_determinism hook.

    Behind a hook because it writes process-global torch state and the
    environment, which a test must be able to observe without a real CUDA
    stack and without leaking settings into the rest of the suite.
    """

    def __call__(self) -> DeterminismRecord:
        """Pin kernel determinism and report what was actually applied."""
        ...


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


class CudaDriverVersionProto(Protocol):
    """Protocol for the cuda_driver_version hook.

    Named separately from the card because the same card under two drivers
    can select different kernels, so a number reproduced on one and not the
    other is a driver difference rather than a broken image.
    """

    def __call__(self) -> str:
        """Get the NVIDIA driver version. Callers gate on cuda_is_available."""
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


class LoadWandbModuleProto(Protocol):
    """Protocol for wandb module loader."""

    def __call__(self) -> WandbModuleLike: ...


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


class LoadPreparedGpt2FromHandleProto(Protocol):
    """Protocol for load_prepared_gpt2_from_handle hook."""

    def __call__(
        self, artifact_path: str, tokenizer: TokenizerHandle | None
    ) -> PreparedLMModel: ...


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
