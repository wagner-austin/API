from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Protocol, TypedDict

from platform_core.determinism_record import DeterminismRecord
from platform_ml.wandb_publisher import WandbPublisher

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.strategy_names import StrategyName
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


class LoraConfig(TypedDict):
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning.

    LoRA reduces trainable parameters by decomposing weight updates into
    low-rank matrices, enabling efficient fine-tuning of large models.
    """

    enabled: bool
    r: int  # LoRA rank (typically 8, 16, 32, 64)
    lora_alpha: int  # Scaling factor (often equal to r)
    lora_dropout: float  # Dropout probability for LoRA layers
    target_modules: tuple[str, ...]  # Module names to apply LoRA (e.g., q_proj, v_proj)
    bias: Literal["none", "all", "lora_only"]


class QuantizationConfig(TypedDict):
    """Configuration for model quantization via bitsandbytes.

    Enables 4-bit or 8-bit quantization to reduce memory footprint
    while maintaining model quality through NF4/FP4 data types.

    EVERY FIELD IS REQUIRED, INCLUDING THE TWO THAT HAVE UPSTREAM DEFAULTS.
    ``transformers.BitsAndBytesConfig`` defaults ``bnb_4bit_quant_type`` to
    ``"fp4"`` and ``bnb_4bit_compute_dtype`` to ``torch.float32``. The QLoRA
    paper measures FP4 as roughly one percentage point behind the 16-bit LoRA
    baseline where NF4 recovers it, and pairs 4-bit storage with a 16-bit
    compute type. Inheriting either default therefore selects an arm the
    paper reports as worse, silently. A run's data type is part of what the
    run IS, so it is stated rather than defaulted.
    """

    load_in_4bit: bool
    load_in_8bit: bool
    bnb_4bit_compute_dtype: Literal["float16", "bfloat16", "float32"]
    bnb_4bit_quant_type: Literal["nf4", "fp4"]
    # Quantizes the quantization constants themselves. The paper reports this
    # saving about 0.37 bits per parameter and uses it in every experiment, so
    # a config that cannot express it cannot express the published arm.
    bnb_4bit_use_double_quant: bool


class GgufExportConfig(TypedDict):
    """Configuration for GGUF export of LoRA adapters.

    GGUF format enables direct compatibility with llama.cpp for efficient
    inference. Only valid for LoRA-based fine-tuning strategies.

    Attributes:
        enabled: Whether GGUF export is enabled after training.
        output_type: Output precision format for the GGUF file.
    """

    enabled: bool
    output_type: Literal["f32", "f16", "bf16", "q8_0"]


class ModelTrainConfig(TypedDict):
    """Configuration for model training.

    Single unified config type used by all backends.
    The 'hf_lm' backend supports any HuggingFace causal LM model with
    optional LoRA fine-tuning strategies.

    For hf_lm models, tokenizer_id is optional (None) because the tokenizer
    is loaded from hub_model_id. For other backends, tokenizer_id is required.
    """

    model_family: Literal["gpt2", "llama", "qwen", "char_lstm", "hf_lm"]
    model_size: str
    max_seq_len: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    tokenizer_id: str | None  # None for hf_lm (uses HF tokenizer from hub_model_id)
    corpus_path: str
    # Whether corpus_path is read as stripped lines or as JSONL documents.
    # Carried on the resolved config rather than inferred from the path's
    # suffix, because the suffix is a naming convention and this decides
    # what the model is actually shown.
    corpus_format: Literal["lines", "documents"]
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
    # When set, the part of each corpus line up to and including the first
    # occurrence of this string is fed to the model as context but excluded
    # from the loss. Used for metadata-conditioned corpora, where the marker
    # should inform prediction without being a prediction target. None means
    # every token is a target.
    loss_mask_prefix_separator: str | None
    # Pluggable fine-tuning strategy (like covenant_ml backends)
    finetuning_strategy: StrategyName
    # Strategy-specific configuration (None = not used by current strategy)
    hub_model_id: str | None  # HuggingFace model ID for pretrained loading
    lora: LoraConfig | None  # Config for lora/qlora strategies
    quantization: QuantizationConfig | None  # Config for qlora strategy
    gguf_export: GgufExportConfig | None  # Config for GGUF export (lora strategies only)


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

    For HuggingFace LM backend with finetuning strategies, optional fields
    store adapter metadata for save/load operations.

    Attributes:
        model: The language model instance.
        tokenizer_id: Identifier for custom tokenizer, or None for HF LM models
            that use the tokenizer from hub_model_id.
        eos_id: End-of-sequence token ID.
        pad_id: Padding token ID.
        max_seq_len: Maximum sequence length for training.
        tok_for_dataset: Encoder for dataset preparation.
        strategy_name: Fine-tuning strategy name (for HF LM models).
        hub_model_id: HuggingFace model ID (for HF LM models).
        is_peft: Whether the model uses PEFT adapters.
        quantization: Quantization the model was loaded under, or None.
    """

    model: LMModelProto
    tokenizer_id: str | None  # None for hf_lm (uses HF tokenizer from hub_model_id)
    eos_id: int
    pad_id: int
    max_seq_len: int
    tok_for_dataset: Encoder
    # Optional HF LM adapter metadata
    strategy_name: str | None
    hub_model_id: str | None
    is_peft: bool
    # The quantization this model was LOADED under, or None when it was not
    # quantized. Carried so a saved run records it: an adapter trained against
    # 4-bit weights and reloaded onto 16-bit ones is a different model, and
    # nothing else on disk would say which it was.
    quantization: QuantizationConfig | None

    def __init__(
        self: PreparedLMModel,
        *,
        model: LMModelProto,
        tokenizer_id: str | None,
        eos_id: int,
        pad_id: int,
        max_seq_len: int,
        tok_for_dataset: Encoder,
        strategy_name: str | None = None,
        hub_model_id: str | None = None,
        is_peft: bool = False,
        quantization: QuantizationConfig | None = None,
    ) -> None:
        """Initialize a prepared language model.

        Args:
            model: The language model instance.
            tokenizer_id: Identifier for custom tokenizer, or None for HF LM.
            eos_id: End-of-sequence token ID.
            pad_id: Padding token ID.
            max_seq_len: Maximum sequence length.
            tok_for_dataset: Encoder for dataset preparation.
            strategy_name: Fine-tuning strategy name.
            hub_model_id: HuggingFace model ID.
            is_peft: Whether the model uses PEFT adapters.
            quantization: Quantization the model was loaded under, or None.
        """
        self.model = model
        self.tokenizer_id = tokenizer_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.tok_for_dataset = tok_for_dataset
        self.strategy_name = strategy_name
        self.hub_model_id = hub_model_id
        self.is_peft = is_peft
        self.quantization = quantization


class ModelBackend(Protocol):
    """Protocol for model backend implementations.

    Backends implement training, evaluation, and inference for language models.
    The tokenizer parameter is optional for hf_lm backend (uses HF tokenizer
    from hub_model_id), but required for other backends.
    """

    def name(self: ModelBackend) -> str: ...

    def capabilities(self: ModelBackend) -> BackendCapabilities: ...

    def prepare(
        self: ModelBackend,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle | None,
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
        tokenizer: TokenizerHandle | None,
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
        resume: bool,
        progress: (
            Callable[[int, int, float, float, float, float, float | None, float | None], None]
            | None
        ) = None,
        wandb_publisher: WandbPublisher | None = None,
        determinism: DeterminismRecord,
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
