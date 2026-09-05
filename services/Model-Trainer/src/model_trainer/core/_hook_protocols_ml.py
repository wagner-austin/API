"""ML-side hook protocols (tokenizers, models, corpus, CUDA).

Runtime protocols: :mod:`model_trainer.core._hook_protocols`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import torch
from platform_core.determinism_record import DeterminismRecord
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
from model_trainer.core.contracts.strategy_names import StrategyName
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.types import LMModelProto


class ApplyDeterminismProto(Protocol):
    """Protocol for the apply_determinism hook.

    Behind a hook because it writes process-global torch state and the
    environment, which a test must be able to observe without a real CUDA
    stack and without leaking settings into the rest of the suite.
    """

    def __call__(self, *, remove_split_k: bool, math_attention: bool) -> DeterminismRecord:
        """Pin kernel determinism and report what was actually applied.

        Both arguments are required with no default, and keyword-only,
        because each is the treatment and the control of a live experiment
        rather than an on/off switch -- see
        :func:`platform_ml.determinism.apply_determinism`. Training runs pass
        True for both; the commands that measure what either control does
        pass False, because an instrument that imposes the intervention
        cannot observe it.

        Args:
            remove_split_k: Whether to take split-K out of cuBLASLt's
                options, making matmuls agree across cards. Free at a
                training step's row count.
            math_attention: Whether to leave the attention dispatcher no
                kernel but the math one, making attention agree across cards.
                NOT free: 1.3-1.6x peak memory on the probed shapes, growing
                with the square of sequence length, because the math path
                materialises the whole score matrix.

        Returns:
            What was actually applied, for the run record.
        """
        ...


class PinTorchThreadsProto(Protocol):
    """Protocol for the torch intra-op thread pin.

    Separate from :class:`ApplyDeterminismProto` because it is a different
    lever with a different failure mode. The determinism settings are torch
    state and take whenever they are written; the thread count used to be
    pinned through ``OMP_NUM_THREADS``/``MKL_NUM_THREADS``, which a BLAS
    reads when it LOADS -- so a worker that set them after importing torch
    set nothing. ``torch.set_num_threads`` is a runtime call and does take.
    """

    def __call__(self, threads: int) -> int:
        """Pin the intra-op thread count and report what the process has.

        Args:
            threads: The count to request.

        Returns:
            The count torch reports AFTERWARDS, read back rather than
            assumed. A request and a resolved value are different facts, and
            only the second describes the run.
        """
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


class SdpaCudaEligibilityProto(Protocol):
    """Protocol for the sdpa_cuda_eligibility hook.

    A seam rather than a direct call because asking is not free of side
    effects: torch 2.7's ``can_use_cudnn_attention`` initializes the CUDA
    context even when the operands live on the CPU, which is fatal on a
    host whose driver cannot satisfy the runtime. Callers gate on the
    operands being CUDA tensors; the hook exists so a test can prove the
    gate holds without owning such a host.
    """

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> dict[str, bool]:
        """Ask torch which fused CUDA backends could serve this call."""
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


class ReloadShippedWeightsProto(Protocol):
    """Protocol for the reload_shipped_weights hook."""

    def __call__(
        self,
        prepared: PreparedLMModel,
        model_family: Literal["gpt2", "llama", "qwen", "char_lstm", "hf_lm"],
        path: str,
    ) -> None:
        """Load a saved artifact into the live model, in place."""
        ...


class FreezeEmbeddingsProto(Protocol):
    """Protocol for freeze_embeddings hook."""

    def __call__(self, model: LMModelProto) -> None:
        """Freeze embedding parameters in model."""
        ...


class EnableGradientCheckpointingProto(Protocol):
    """Protocol for enable_gradient_checkpointing hook."""

    def __call__(self, model: LMModelProto, strategy: StrategyName) -> bool:
        """Enable activation checkpointing when the strategy supports it."""
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
