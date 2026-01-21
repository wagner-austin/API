from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict


class EvalCache(TypedDict):
    status: Literal["queued", "running", "completed", "failed"]
    split: str
    loss: float | None
    ppl: float | None
    artifact: str | None


class TrainingManifestVersions(TypedDict):
    torch: str
    transformers: str
    tokenizers: str
    datasets: str


class TrainingManifestSystem(TypedDict):
    cpu_count: int
    platform: str
    platform_release: str
    machine: str


class TrainingManifestTiming(TypedDict):
    """Timing information for training run.

    Attributes:
        training_duration_sec: Total training time in seconds.
        started_at: ISO 8601 timestamp when training began.
        completed_at: ISO 8601 timestamp when training finished.
    """

    training_duration_sec: float
    started_at: str
    completed_at: str


class TrainingManifestPerformance(TypedDict):
    """Performance metrics from training run.

    Attributes:
        peak_gpu_memory_mb: Maximum GPU memory used in megabytes (None if CPU).
        avg_samples_per_sec: Average throughput during training.
        total_tokens_processed: Total tokens seen during training.
    """

    peak_gpu_memory_mb: float | None
    avg_samples_per_sec: float
    total_tokens_processed: int


class TrainingManifestModelInfo(TypedDict):
    """Model metadata captured after training.

    Attributes:
        param_count: Total trainable parameters in the model.
        model_size_mb: Size of saved model on disk in megabytes.
        vocab_size: Tokenizer vocabulary size.
    """

    param_count: int
    model_size_mb: float
    vocab_size: int


class TrainingManifest(TypedDict):
    """Training manifest with all configuration and results.

    For hf_lm models, tokenizer_id may be None because the HF tokenizer
    is loaded from hub_model_id.
    """

    run_id: str
    model_family: str
    model_size: str
    epochs: int
    batch_size: int
    max_seq_len: int
    steps: int
    loss: float
    learning_rate: float
    tokenizer_id: str | None  # None for hf_lm (uses HF tokenizer from hub_model_id)
    corpus_path: str
    holdout_fraction: float
    optimizer: str
    freeze_embed: bool
    gradient_clipping: float
    seed: int
    pretrained_run_id: str | None
    versions: TrainingManifestVersions
    system: TrainingManifestSystem
    git_commit: str | None
    device: str
    precision: str
    early_stopping_patience: int
    test_split_ratio: float
    finetune_lr_cap: float
    test_loss: float | None
    test_perplexity: float | None
    best_val_loss: float | None
    early_stopped: bool
    # New fields for feature roadmap
    timing: TrainingManifestTiming
    performance: TrainingManifestPerformance
    model_info: TrainingManifestModelInfo


class TrainingManifestConfig(TypedDict):
    """Configuration section of training manifest.

    For hf_lm models, tokenizer_id may be None because the HF tokenizer
    is loaded from hub_model_id.
    """

    model_family: str
    model_size: str
    max_seq_len: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    tokenizer_id: str | None  # None for hf_lm (uses HF tokenizer from hub_model_id)
    corpus_path: str
    holdout_fraction: float
    seed: int
    pretrained_run_id: str | None
    freeze_embed: bool
    gradient_clipping: float
    optimizer: str
    device: str
    precision: str
    early_stopping_patience: int
    test_split_ratio: float
    finetune_lr_cap: float


class TrainingManifestFull(TypedDict):
    """Full training manifest with embedded config block.

    For hf_lm models, tokenizer_id may be None because the HF tokenizer
    is loaded from hub_model_id.
    """

    run_id: str
    model_family: str
    model_size: str
    epochs: int
    batch_size: int
    max_seq_len: int
    steps: int
    loss: float
    learning_rate: float
    tokenizer_id: str | None  # None for hf_lm (uses HF tokenizer from hub_model_id)
    corpus_path: str
    holdout_fraction: float
    optimizer: str
    freeze_embed: bool
    gradient_clipping: float
    seed: int
    pretrained_run_id: str | None
    versions: TrainingManifestVersions
    system: TrainingManifestSystem
    git_commit: str | None
    config: TrainingManifestConfig
    device: str
    precision: str
    early_stopping_patience: int
    test_split_ratio: float
    finetune_lr_cap: float
    test_loss: float | None
    test_perplexity: float | None
    best_val_loss: float | None
    early_stopped: bool
    # New fields for feature roadmap
    timing: TrainingManifestTiming
    performance: TrainingManifestPerformance
    model_info: TrainingManifestModelInfo
