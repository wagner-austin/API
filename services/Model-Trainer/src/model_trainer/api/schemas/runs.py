from __future__ import annotations

from typing import Literal, NotRequired

from platform_core.comparability import RunFingerprint
from typing_extensions import TypedDict

from model_trainer.core.contracts.cloze import ClozeItemOutcome
from model_trainer.core.contracts.strategy_names import StrategyName


class LoraConfigRequest(TypedDict, total=True):
    """API request schema for LoRA configuration.

    Maps to core LoraConfig TypedDict. All fields required at API level
    with defaults applied in validators.

    Attributes:
        enabled: Whether LoRA is enabled.
        r: LoRA rank (typically 8, 16, 32, 64).
        lora_alpha: Scaling factor (often equal to r).
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: Module names to apply LoRA.
        bias: Bias handling mode.
    """

    enabled: bool
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    bias: Literal["none", "all", "lora_only"]


class CartridgeConfigRequest(TypedDict, total=True):
    """API request schema for cartridge configuration.

    Maps to core CartridgeConfig TypedDict.

    Attributes:
        enabled: Whether the cartridge strategy is enabled.
        num_slots: Prefix positions to train.
        init_seed: Seed for the initial draw, so a run is reproducible.
    """

    enabled: bool
    num_slots: int
    init_seed: int


class QuantizationConfigRequest(TypedDict, total=True):
    """API request schema for quantization configuration.

    Maps to core QuantizationConfig TypedDict.

    Attributes:
        load_in_4bit: Whether to load model in 4-bit precision.
        load_in_8bit: Whether to load model in 8-bit precision.
        bnb_4bit_compute_dtype: Compute dtype for 4-bit operations.
        bnb_4bit_quant_type: Quantization type (nf4 or fp4).
        bnb_4bit_use_double_quant: Whether to quantize the quantization
            constants. Required, not defaulted: the QLoRA paper uses it in
            every experiment.
    """

    load_in_4bit: bool
    load_in_8bit: bool
    bnb_4bit_compute_dtype: Literal["float16", "bfloat16", "float32"]
    bnb_4bit_quant_type: Literal["nf4", "fp4"]
    bnb_4bit_use_double_quant: bool


class GgufExportConfigRequest(TypedDict, total=True):
    """API request schema for GGUF export configuration.

    Maps to core GgufExportConfig TypedDict. GGUF export is only valid
    for LoRA-based fine-tuning strategies (lora, qlora).

    Attributes:
        enabled: Whether GGUF export is enabled after training.
        output_type: Output precision format for the GGUF file.
    """

    enabled: bool
    output_type: Literal["f32", "f16", "bf16", "q8_0"]


class TrainRequest(TypedDict, total=True):
    """Request to start model training.

    Supports both legacy backends (gpt2, char_lstm) and the new hf_lm backend
    for HuggingFace models with optional LoRA fine-tuning.

    Attributes:
        model_family: Model architecture. Use 'hf_lm' for HuggingFace models.
        corpus_format: How the corpus divides into training units -- 'lines'
            for stripped text lines, 'documents' for JSONL records taken
            verbatim. Required with no default: a source-code corpus read as
            lines loses its indentation, and a caller that omitted the field
            would be given that silently.
        hub_model_id: HuggingFace model ID (required when model_family='hf_lm').
        finetuning_strategy: Strategy for fine-tuning (full, lora, qlora).
        lora: LoRA configuration (required for lora/qlora strategies).
        cartridge: Cartridge configuration (required for the cartridge strategy).
        quantization: Quantization config (required for qlora strategy).
    """

    model_family: Literal["gpt2", "llama", "qwen", "char_lstm", "hf_lm"]
    model_size: str
    max_seq_len: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    corpus_file_id: str
    corpus_format: Literal["lines", "documents"]
    tokenizer_id: str | None  # Optional for hf_lm (uses HF tokenizer from hub_model_id)
    holdout_fraction: float
    seed: int
    pretrained_run_id: str | None
    freeze_embed: bool
    gradient_clipping: float
    optimizer: Literal["adamw", "adam", "sgd"]
    user_id: int
    device: Literal["cpu", "cuda", "auto"]
    precision: Literal["fp32", "fp16", "bf16", "auto"]
    # Data loading knobs (optional at API layer; resolved in worker)
    data_num_workers: NotRequired[int | None]
    data_pin_memory: NotRequired[bool | None]
    early_stopping_patience: int
    test_split_ratio: float
    finetune_lr_cap: float
    loss_mask_prefix_separator: str | None
    # HuggingFace LM backend fields
    hub_model_id: str | None
    finetuning_strategy: StrategyName
    lora: LoraConfigRequest | None
    cartridge: CartridgeConfigRequest | None
    quantization: QuantizationConfigRequest | None
    gguf_export: GgufExportConfigRequest | None


class TrainResponse(TypedDict, total=True):
    run_id: str
    job_id: str


class RunStatusResponse(TypedDict, total=True):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    last_heartbeat_ts: float | None
    message: str | None
    # Traceable code for why a run is not healthy, so a caller can tell a
    # training failure from the machine disappearing underneath one. None on a
    # run that has nothing wrong with it.
    error: str | None


class EvaluateRequest(TypedDict, total=True):
    split: Literal["validation", "test"]
    path_override: NotRequired[str | None]


class EvaluateResponse(TypedDict, total=True):
    run_id: str
    split: str
    status: Literal["queued", "running", "completed", "failed"]
    loss: float | None
    perplexity: float | None
    artifact_path: str | None


class CancelResponse(TypedDict, total=True):
    # `dequeued` means the job was still pending and was removed, so the run
    # is already terminal. `cancellation-requested` means a worker holds it
    # and will stop at its next cancellation check.
    status: Literal["cancellation-requested", "dequeued"]


class ArtifactPointerResponse(TypedDict, total=True):
    storage: str
    file_id: str


class ScoreRequest(TypedDict, total=True):
    """Request to score text with a trained model."""

    text: str | None
    path: str | None
    detail_level: Literal["summary", "per_char"]
    top_k: int | None
    seed: int | None


class ScoreResponse(TypedDict, total=True):
    """Response from scoring text."""

    request_id: str
    status: Literal["queued", "running", "completed", "failed"]
    loss: float | None
    perplexity: float | None
    surprisal: list[float] | None
    topk: list[list[tuple[str, float]]] | None
    tokens: list[str] | None


class ClozeRequest(TypedDict, total=True):
    """Request to score a cloze item set with a trained model."""

    items_file_id: str
    max_seq_len: int


class ClozeResponse(TypedDict, total=True):
    """Response from cloze evaluation.

    ``chance`` is the accuracy uniform guessing reaches on the same candidate
    counts. It is reported alongside ``accuracy`` because accuracy alone is not
    interpretable: a four-way item set floors at 25% before any knowledge.

    ``outcomes`` carries one record per item once the job completes. Two runs
    scored on the same item set can then be compared item by item, which an
    aggregate count cannot support.
    """

    request_id: str
    status: Literal["queued", "running", "completed", "failed"]
    total: int | None
    correct: int | None
    accuracy: float | None
    chance: float | None
    outcomes: list[ClozeItemOutcome] | None
    fingerprint: RunFingerprint | None


class BaselineClozeRequest(TypedDict, total=True):
    """Request to score an untrained hub model on a cloze item set.

    Names a model rather than a run, because the point is to measure what a
    model scores having never been trained here.
    """

    hub_model_id: str
    items_file_id: str
    max_seq_len: int
    device: str


class BaselineClozeResponse(TypedDict, total=True):
    """Response from scoring an untrained model.

    Identified by the model and the item set rather than by a request id.
    Those two fields fully determine the measurement, so a caller that asks the
    same question twice gets the same record instead of two that could
    disagree.
    """

    hub_model_id: str
    items_file_id: str
    status: Literal["queued", "running", "completed", "failed"]
    total: int | None
    correct: int | None
    accuracy: float | None
    chance: float | None
    outcomes: list[ClozeItemOutcome] | None
    fingerprint: RunFingerprint | None


class GenerateRequest(TypedDict, total=True):
    """Request to generate text from a trained model."""

    prompt_text: str | None
    prompt_path: str | None
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    stop_on_eos: bool
    stop_sequences: list[str]
    seed: int | None
    num_return_sequences: int


class GenerateResponse(TypedDict, total=True):
    """Response from text generation."""

    request_id: str
    status: Literal["queued", "running", "completed", "failed"]
    outputs: list[str] | None
    steps: int | None
    eos_terminated: list[bool] | None


class ChatMessage(TypedDict, total=True):
    """A single message in a conversation."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(TypedDict, total=True):
    """Request to send a chat message."""

    message: str
    session_id: str | None
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float


class ChatResponse(TypedDict, total=True):
    """Response from chat endpoint."""

    session_id: str
    status: Literal["queued", "running", "completed", "failed"]
    request_id: str
    response: str | None


class ChatHistoryResponse(TypedDict, total=True):
    """Response containing conversation history."""

    session_id: str
    run_id: str
    messages: list[ChatMessage]
    created_at: str


class ProgressResponse(TypedDict, total=True):
    """Detailed training progress for real-time monitoring.

    Attributes:
        run_id: Unique identifier for the training run.
        phase: Current phase (queued, tokenization, training, validation, etc.).
        epoch: Current epoch number (0-indexed during training).
        total_epochs: Total number of epochs configured.
        step: Current step number within the epoch.
        total_steps: Total steps per epoch (0 if unknown).
        train_loss: Current training loss value.
        train_ppl: Current training perplexity.
        grad_norm: Current gradient norm value.
        samples_per_sec: Training throughput in samples per second.
        val_loss: Validation loss from last validation (None if not run yet).
        val_ppl: Validation perplexity from last validation (None if not run yet).
        updated_at: ISO 8601 timestamp of last update.
    """

    run_id: str
    phase: Literal[
        "queued",
        "tokenization",
        "training",
        "validation",
        "test",
        "saving",
        "exporting",
        "uploading",
        "completed",
        "failed",
        "cancelled",
    ]
    epoch: int
    total_epochs: int
    step: int
    total_steps: int
    train_loss: float
    train_ppl: float
    grad_norm: float
    samples_per_sec: float
    val_loss: float | None
    val_ppl: float | None
    updated_at: str
