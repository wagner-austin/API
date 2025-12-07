from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Protocol, TypedDict

from platform_ml.wandb_publisher import WandbPublisher

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.encoding import Encoder
from model_trainer.core.types import LMModelProto


class ScoreConfig(TypedDict):
    """Configuration for scoring (computing loss/perplexity on text)."""

    text: str | None
    path: str | None
    detail_level: Literal["summary", "per_char"]
    top_k: int | None
    seed: int | None


class GenerateConfig(TypedDict):
    """Configuration for text generation."""

    prompt_text: str | None
    prompt_path: str | None
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    stop_on_eos: bool
    stop_sequences: Sequence[str]
    seed: int | None
    num_return_sequences: int


class EarlyStoppingState(TypedDict):
    """Mutable state for early stopping tracking."""

    best_val_loss: float
    epochs_no_improve: int


class ValidationMetrics(TypedDict):
    """Metrics from a validation pass."""

    val_loss: float
    val_ppl: float


class GradientMetrics(TypedDict):
    """Gradient statistics for logging."""

    grad_norm: float


class ModelTrainConfig(TypedDict):
    """Configuration for model training.

    Single unified config type used by all backends.
    """

    model_family: Literal["gpt2", "llama", "qwen", "char_lstm"]
    model_size: str
    max_seq_len: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    tokenizer_id: str
    corpus_path: str
    holdout_fraction: float
    seed: int
    pretrained_run_id: str | None
    freeze_embed: bool
    gradient_clipping: float
    optimizer: Literal["adamw", "adam", "sgd"]
    device: Literal["cpu", "cuda"]
    precision: Literal["fp32", "fp16", "bf16"]
    data_num_workers: int
    data_pin_memory: bool
    early_stopping_patience: int
    test_split_ratio: float
    finetune_lr_cap: float


class TrainOutcome(TypedDict):
    """Result of model training."""

    loss: float
    perplexity: float
    steps: int
    out_dir: str
    cancelled: bool
    test_loss: float | None
    test_perplexity: float | None
    best_val_loss: float | None
    early_stopped: bool


class EvalOutcome(TypedDict):
    """Result of model evaluation."""

    loss: float
    perplexity: float


class ScoreOutcome(TypedDict):
    """Result of scoring text with a model."""

    loss: float
    perplexity: float
    surprisal: Sequence[float] | None
    topk: Sequence[Sequence[tuple[str, float]]] | None
    tokens: Sequence[str] | None


class GenerateOutcome(TypedDict):
    """Result of text generation."""

    outputs: Sequence[str]
    steps: int
    eos_terminated: Sequence[bool]


class ModelArtifact(TypedDict):
    """Reference to a saved model artifact."""

    out_dir: str


class BackendCapabilities(TypedDict):
    """Declares what operations a backend supports.

    Used for capability discovery and validation before invoking backend methods.
    """

    supports_train: bool
    supports_evaluate: bool
    supports_score: bool
    supports_generate: bool
    supports_distributed: bool
    supported_sizes: tuple[str, ...]


class PreparedLMModel:
    """Unified prepared language model with tokenizer for training and inference.

    This replaces the separate GPT2Prepared and CharLSTMPrepared classes.
    All language model backends use this same type.
    """

    model: LMModelProto
    tokenizer_id: str
    eos_id: int
    pad_id: int
    max_seq_len: int
    tok_for_dataset: Encoder

    def __init__(
        self: PreparedLMModel,
        *,
        model: LMModelProto,
        tokenizer_id: str,
        eos_id: int,
        pad_id: int,
        max_seq_len: int,
        tok_for_dataset: Encoder,
    ) -> None:
        self.model = model
        self.tokenizer_id = tokenizer_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.tok_for_dataset = tok_for_dataset


class ModelBackend(Protocol):
    """Protocol for model backend implementations."""

    def name(self: ModelBackend) -> str: ...

    def capabilities(self: ModelBackend) -> BackendCapabilities: ...

    def prepare(
        self: ModelBackend,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedLMModel: ...

    def save(
        self: ModelBackend,
        prepared: PreparedLMModel,
        out_dir: str,
    ) -> ModelArtifact: ...

    def load(
        self: ModelBackend,
        artifact_path: str,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedLMModel: ...

    def train(
        self: ModelBackend,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        heartbeat: Callable[[float], None],
        cancelled: Callable[[], bool],
        prepared: PreparedLMModel,
        progress: (
            Callable[[int, int, float, float, float, float, float | None, float | None], None]
            | None
        ) = None,
        wandb_publisher: WandbPublisher | None = None,
    ) -> TrainOutcome: ...

    def evaluate(
        self: ModelBackend,
        *,
        run_id: str,
        cfg: ModelTrainConfig,
        settings: Settings,
    ) -> EvalOutcome: ...

    def score(
        self: ModelBackend,
        *,
        prepared: PreparedLMModel,
        cfg: ScoreConfig,
        settings: Settings,
    ) -> ScoreOutcome: ...

    def generate(
        self: ModelBackend,
        *,
        prepared: PreparedLMModel,
        cfg: GenerateConfig,
        settings: Settings,
    ) -> GenerateOutcome: ...
